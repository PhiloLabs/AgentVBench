"""Edit scorer (universal, 50 pts) — whole-video, parallelized.

Compares `fixed.mp4` against `source.mp4` across the entire runtime, not
just the injection window. Designed to catch both:
  - bad in-window fixes (the originally-scored case), and
  - off-region tampering (any region the agent shouldn't have touched).

Pipeline
--------
1. Split [0, duration] into `chunk_duration_s` segments (default 10s).
2. Dispatch each segment to a worker via ProcessPoolExecutor. Each worker
   independently samples video at `sample_fps` (default 10 fps); for each
   sampled fixed-frame it searches ±LAG_BAND_SAMPLES sample-steps in source
   for the best match (dhash pre-rank → SSIM on top-K), then multiplies the
   match by a linear lag penalty so a smaller phase offset earns a higher
   score. Audio per chunk: lag-constrained xcorr peak.
3. Aggregate chunks by sample/duration-weighted mean.

Formula
-------
    per_frame  = max_{|k|≤BAND} ssim(F_i, S_{i+k}) * sharp_ratio * chroma_ratio
                                * (1 − LAG_PENALTY_PER_SAMPLE × |k|)
    video_sub  = 0.6 * max_points * weighted_mean(per_frame)
    audio_sub  = 0.4 * max_points * duration_weighted_mean(chunk.xcorr)
    score      = video_sub + audio_sub

`profile` is accepted (ROADMAP contract) and used only for duration + GT
injection window reporting — scoring itself is whole-video.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_FPS = 10.0
CHUNK_DURATION_S = 10.0
AUDIO_SR = 48000
# 36-pt video budget, 24-pt audio budget at the Run #5 default max_points=60.
VIDEO_SHARE = 0.6
AUDIO_SHARE = 0.4
# dhash short-circuit: frames whose 64-bit dhash differs by <= this many bits
# are assumed visually identical (SSIM=1.0) and skip the full computation.
HASH_THRESHOLD = 4
# Deduction weights. Score starts at the full per-component budget and loses
# W_IN * (1 - signal_in) for damage inside the GT injection window ("missed
# repair") and W_OUT * (1 - signal_out) for damage outside it ("tampered what
# you shouldn't"). In-window is weighted heavier so a 2.6s in-window defect
# dominates the score over 250s of untouched content.
W_IN = 2.5
W_OUT = 1.0
# Audio xcorr peak search window AND the linear-penalty zero-credit point.
# Within ±AUDIO_MAX_LAG_MS the peak value is weighted by
# `max(0, 1 - |lag|/max_lag)` — lag=0 earns full credit, |lag|=max_lag earns
# 0, intermediate lags are linearly graded. Beyond the band the peak is
# never considered (translation-invariant xcorr would otherwise return ~1.0
# for any large delay since the matching content exists at some lag).
# Run #5 calibration: 150ms gives codec realities a softer landing (opus
# ~25ms priming → 0.83× credit, ffmpeg concat/atrim ~30ms → 0.80×, model
# re-encode shifts up to ~50ms → 0.67×) while still zeroing v03-class swaps
# (peak lag >1.8s) and v07-style large desyncs (>150ms). Pre-Run-#5 default
# was 50ms with a hard cutoff (binary in/out, no smoothing); previous
# Run #5 value was 100ms.
AUDIO_MAX_LAG_MS = 150.0
# Per-side deduction cap. At 1.0 a fully-collapsed signal_in (or signal_out)
# zeros that side of the budget entirely — combined with `FLOOR_FRAC = 0.0`,
# this means a fully-failed modality scores 0 with no participation credit.
# Was 0.85 through Run #5; raised to 1.0 on 2026-04-30 so do-nothing-but-
# report agents land near the floor instead of pocketing 15% of each modality.
# Set `floor_frac` per-call if a future variant needs the old soft-saturation.
MAX_DEDUCT_FRAC = 1.0
FLOOR_FRAC = 0.0
# Chroma signal (closes the blind-spot that grayscale SSIM has for pure hue /
# saturation / white-balance shifts — see PROJECT.md §6.3). Per-frame:
#   chroma_ratio = max(0, 1 − mean|Lab_{a,b}(fixed) − Lab_{a,b}(source)| / CHROMA_NORM)
# and is multiplied into `val` alongside ssim + sharp_ratio. On color-untouched
# variants (v01-v03, v05) Lab a/b matches → chroma_ratio ≈ 1.0 → no regression.
# CHROMA_NORM=25 is a pilot default calibrated for v04 (aggressive teal/warm
# pushes land around 20-30 mean abs a/b distance); re-calibrate after first
# v04 real-fix bracket.
CHROMA_NORM = 25.0
# Lag-tolerant SSIM (video analogue of AUDIO_MAX_LAG_MS). For each fixed-frame
# sampled at sample_fps, search ±LAG_BAND_FRAMES *native* source frames for
# the best match (full SSIM on each candidate), and multiply by a linear lag
# penalty: best earns 1.0×, every native frame of lag costs LAG_PENALTY_PER_FRAME.
# Lag is searched at native-frame resolution (not sample-step) so the scorer
# can detect a 1-frame drift even when sampling decimates by 2× — the v06
# "duplicate removed but everything after is shifted by one native frame"
# pattern lives entirely sub-sample-step. Calibration: BAND=10 native frames
# at 24 fps covers ±~0.42 s of drift (wide enough to absorb ffmpeg trim/concat
# + libopus priming artifacts, narrow enough to still penalize a swap), and
# PENALTY_PER_FRAME=0.03 puts a 1-frame ding at 3 %, a 5-frame ding at 15 %,
# and a 10-frame ding at 30 %. Setting LAG_BAND_FRAMES=0 reproduces the legacy
# strict frame-aligned behavior.
LAG_BAND_FRAMES = 10
LAG_PENALTY_PER_FRAME = 0.03
# Inside the band we use dhash hamming as a cheap pre-rank and only run full
# SSIM on the top-K candidates plus k=0. Without forcing k=0 a near-identical
# zero-lag frame can be tied/edged out in dhash space (64-bit fingerprint, lots
# of ties on similar frames) and never get scored, letting a penalized neighbor
# win against a candidate we never actually evaluated. K=3 + k=0 = up to 4
# SSIMs per sample, vs 2*BAND+1 = 21 if we evaluated the whole band. Set
# LAG_TOPK=0 to disable the pre-rank and evaluate everything in the band
# (slower, used for ground-truth re-scoring / calibration).
LAG_TOPK = 3


def _deduct_sub(
    signal_in: float | None,
    signal_out: float | None,
    budget: float,
    w_in: float,
    w_out: float,
    max_deduct_frac: float = MAX_DEDUCT_FRAC,
    floor_frac: float = FLOOR_FRAC,
) -> tuple[float, float, float]:
    """Return (sub_score, deduct_in, deduct_out). Missing side contributes 0.

    Per-side deduction is capped so a single bad side can't zero out the
    modality, and a small floor is applied so any attempt gets partial credit.
    """
    cap = max_deduct_frac * budget
    floor = floor_frac * budget
    deduct_in = 0.0
    deduct_out = 0.0
    if signal_in is not None:
        deduct_in = min(cap, w_in * budget * max(0.0, 1.0 - float(signal_in)))
    if signal_out is not None:
        deduct_out = min(cap, w_out * budget * max(0.0, 1.0 - float(signal_out)))
    return max(floor, budget - deduct_in - deduct_out), deduct_in, deduct_out


def _injection_window(profile: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve a single GT injection window for the edit-rubric in/out partition.

    Three profile shapes:
      - Run #5 single defect: `profile.injection = {start_s, end_s, ...}`.
      - Run #5 v03 scene swap: `profile.injection.scene_1/scene_4` carry the
        outer span (gapped swaps).
      - Run #6+ combined defects: `profile.injections = [<inj1>, <inj2>, ...]`.
        Returns the union span `(min start, max end)` so the chunk scorer's
        in/out partition still works as one boolean per sample. Per-window
        edit scoring is a future extension — for now any sample inside any
        defect window counts as in-window.
    """
    try:
        if "injection" in profile:
            inj = profile["injection"]
            if inj.get("variant") == "v03_scene_swap":
                return float(inj["scene_1"]["original_start_s"]), float(inj["scene_4"]["original_end_s"])
            return float(inj["start_s"]), float(inj["end_s"])
        injections = profile.get("injections")
        if isinstance(injections, list) and injections:
            starts = [float(x["start_s"]) for x in injections]
            ends = [float(x["end_s"]) for x in injections]
            return min(starts), max(ends)
        return None
    except (KeyError, TypeError, ValueError):
        return None


def _extract_audio_wav(path: str, start: float, duration: float, sr: int) -> np.ndarray:
    """Mono float32 samples at `sr` Hz for [start, start+duration]."""
    if duration <= 0:
        return np.zeros(0, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", path,
            "-ac", "1",
            "-ar", str(sr),
            "-f", "wav",
            "-acodec", "pcm_s16le",
            tmp.name,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extract failed: {res.stderr[-400:]}")
        with wave.open(tmp.name, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sw = wf.getsampwidth()
    if sw != 2:
        raise RuntimeError(f"unexpected sample width {sw}")
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _dhash_bits(gray: np.ndarray, size: int = 8) -> np.ndarray:
    """64-bit dhash packed as a uint8[8]. Cheap perceptual fingerprint."""
    import cv2
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = (resized[:, 1:] > resized[:, :-1]).astype(np.uint8)
    return np.packbits(diff.flatten())


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


def _score_chunk_worker(
    fixed_path: str,
    source_path: str,
    start_s: float,
    end_s: float,
    sample_fps: float,
    audio_sr: int,
    hash_threshold: int,
    gt_window: tuple[float, float] | None,
    audio_max_lag_ms: float,
    chroma_norm: float,
    lag_band_frames: int = LAG_BAND_FRAMES,
    lag_penalty_per_frame: float = LAG_PENALTY_PER_FRAME,
    lag_topk: int = LAG_TOPK,
) -> dict[str, Any]:
    """Independent per-chunk scorer, pickleable for ProcessPoolExecutor.

    If `gt_window` is given, samples are partitioned by timestamp into
    in-window vs out-of-window stats. Audio is split by time-slicing the
    extracted wav buffer. If None, everything is treated as out-of-window.

    Video matching is lag-tolerant at native-frame resolution: pass 1 buffers
    every source native frame in the chunk (gray, lap, lab a/b, dhash); pass 2
    decodes fixed at native fps and, every skip_f-th frame, dhash-ranks the
    candidates in [i−lag_band_frames, i+lag_band_frames], runs full SSIM on
    the top-lag_topk plus a forced k=0 candidate, and picks the lag that
    maximizes
        ssim * sharp_ratio * chroma_ratio * (1 − lag_penalty_per_frame × |lag|).
    Setting lag_band_frames=0 reproduces the legacy strict frame-aligned
    comparison; lag_topk=0 evaluates every candidate in the band (slow).
    """
    try:
        import cv2
        from scipy.signal import correlate
        from skimage.metrics import structural_similarity as ssim

        duration = float(end_s) - float(start_s)
        if duration <= 0:
            return {"start_s": start_s, "end_s": end_s, "error": "zero-length chunk"}

        # --- Video SSIM on sampled frames (two-pass, lag-tolerant) ---
        cap_s = cv2.VideoCapture(source_path)
        cap_f = cv2.VideoCapture(fixed_path)
        if not cap_s.isOpened() or not cap_f.isOpened():
            cap_s.release(); cap_f.release()
            return {"start_s": start_s, "end_s": end_s, "error": "cv2 open failed"}

        try:
            src_w = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))
            native_fps_s = cap_s.get(cv2.CAP_PROP_FPS) or 24.0
            native_fps_f = cap_f.get(cv2.CAP_PROP_FPS) or native_fps_s
            skip_s = max(1, int(round(native_fps_s / sample_fps)))
            skip_f = max(1, int(round(native_fps_f / sample_fps)))
            n_wanted = max(2, int(round(duration * sample_fps)))
            n_frames_s = int(round(duration * native_fps_s))
            n_frames_f = int(round(duration * native_fps_f))

            gt_start = gt_window[0] if gt_window else None
            gt_end = gt_window[1] if gt_window else None

            # --- Pass 1: buffer EVERY native source frame in the chunk so the
            # lag search can resolve sub-sample-step drift. Memory cost is
            # bounded by chunk length × native fps × (gray + lab a/b uint8) —
            # ~336 MB per 10 s chunk at 720p × 24 fps. Per-frame fingerprints
            # (lap, dhash) are computed once here, reused across every fixed
            # sample whose lag band touches this source frame.
            cap_s.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
            src_frames: list[dict[str, Any]] = []
            for i in range(n_frames_s):
                ok, fr_s = cap_s.read()
                if not ok:
                    break
                g_s = cv2.cvtColor(fr_s, cv2.COLOR_BGR2GRAY)
                lab_ab_s = cv2.cvtColor(fr_s, cv2.COLOR_BGR2LAB)[..., 1:3].copy()
                lap_s = float(cv2.Laplacian(g_s, cv2.CV_64F).var())
                src_frames.append({
                    "gray": g_s,
                    "lab_ab": lab_ab_s,
                    "lap": lap_s,
                    "dhash": _dhash_bits(g_s),
                })

            # --- Pass 2: iterate fixed at native fps, sample every skip_f-th
            # frame, lag-search against the native source buffer in
            # [i − band, i + band]. Lag is in native frames, so a 1-native-frame
            # offset (~42 ms at 24 fps) is detected even when sample_fps
            # decimates by 2× and would otherwise hide it sub-step.
            cap_f.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
            vals_in: list[float] = []
            vals_out: list[float] = []
            lags_in: list[int] = []
            lags_out: list[int] = []
            n_hash_skipped = 0
            for i in range(n_frames_f):
                ok, fr_f = cap_f.read()
                if not ok:
                    break
                if i % skip_f != 0:
                    continue
                if not src_frames:
                    break
                if fr_f.shape[:2] != (src_h, src_w):
                    fr_f = cv2.resize(fr_f, (src_w, src_h), interpolation=cv2.INTER_AREA)
                g_f = cv2.cvtColor(fr_f, cv2.COLOR_BGR2GRAY)
                lab_ab_f = cv2.cvtColor(fr_f, cv2.COLOR_BGR2LAB)[..., 1:3]
                lap_f = float(cv2.Laplacian(g_f, cv2.CV_64F).var())

                lo = max(0, i - lag_band_frames)
                hi = min(len(src_frames), i + lag_band_frames + 1)
                if lo >= hi:
                    continue

                # Pick which band positions to score with full SSIM. lag_topk<=0
                # means "evaluate every candidate in the band" (correct, slow).
                # Otherwise rank candidates by dhash hamming (cheap) and keep
                # the top-K — but ALWAYS append k=0 (= the i-th source frame,
                # if present in the band), because near-identical frames can
                # have their k=0 dhash tied or beaten by neighbors and we'd
                # otherwise never score the zero-lag candidate.
                band_keys = range(lo, hi)
                if 0 < lag_topk and (hi - lo) > lag_topk + 1:
                    dhash_f = _dhash_bits(g_f)
                    ranked = sorted(band_keys, key=lambda k: _hamming(dhash_f, src_frames[k]["dhash"]))
                    eval_keys = list(ranked[:lag_topk])
                    if lo <= i < hi and i not in eval_keys:
                        eval_keys.append(i)
                else:
                    eval_keys = list(band_keys)

                best_val = -1.0
                best_lag = 0
                best_ham: int | None = None
                for k in eval_keys:
                    src = src_frames[k]
                    s = float(ssim(src["gray"], g_f, data_range=255))
                    sharp_ratio = min(1.0, lap_f / src["lap"]) if src["lap"] > 1.0 else 1.0
                    dab = float(np.mean(np.abs(
                        src["lab_ab"].astype(np.int16) - lab_ab_f.astype(np.int16)
                    )))
                    chroma_ratio = max(0.0, 1.0 - dab / max(chroma_norm, 1e-6))
                    lag = k - i
                    lag_pen = max(0.0, 1.0 - lag_penalty_per_frame * abs(lag))
                    v = s * sharp_ratio * chroma_ratio * lag_pen
                    if v > best_val:
                        best_val = v
                        best_lag = lag
                        best_ham = _hamming(_dhash_bits(g_f), src["dhash"])

                if best_ham is not None and best_ham <= hash_threshold and best_lag == 0:
                    n_hash_skipped += 1

                ts = start_s + i / native_fps_f
                if gt_start is not None and gt_start <= ts <= gt_end:
                    vals_in.append(best_val)
                    lags_in.append(best_lag)
                else:
                    vals_out.append(best_val)
                    lags_out.append(best_lag)
                if len(vals_in) + len(vals_out) >= n_wanted:
                    break

            n_video_in = len(vals_in)
            n_video_out = len(vals_out)
            ssim_in = float(np.mean(vals_in)) if vals_in else None
            ssim_out = float(np.mean(vals_out)) if vals_out else None
            if ssim_in is not None:
                ssim_in = max(0.0, min(1.0, ssim_in))
            if ssim_out is not None:
                ssim_out = max(0.0, min(1.0, ssim_out))
            mean_abs_lag_in = float(np.mean(np.abs(lags_in))) if lags_in else None
            mean_abs_lag_out = float(np.mean(np.abs(lags_out))) if lags_out else None
            # Free the source buffer before the audio block — keeps peak RSS
            # down when scoring runs many chunks in parallel.
            src_frames.clear()
        finally:
            cap_f.release(); cap_s.release()

        # --- Audio xcorr, split by GT window overlap within this chunk ---
        a_full = _extract_audio_wav(fixed_path, start_s, duration, audio_sr)
        b_full = _extract_audio_wav(source_path, start_s, duration, audio_sr)
        n = min(a_full.size, b_full.size)
        a_full = a_full[:n]; b_full = b_full[:n]

        def _xcorr(a: np.ndarray, b: np.ndarray) -> dict | None:
            """Cross-correlation peak with linear lag penalty.

            Peak searched within ±audio_max_lag_ms; the peak's normalized
            value is then multiplied by `max(0, 1 - |lag|/max_lag)`.
            lag=0 → full credit, |lag|=max_lag → 0, intermediate lags
            linearly graded so a tighter alignment scores strictly higher
            than a looser one inside the same band.

            Returns a dict with the final score plus the raw factors
            (`peak_norm`, `peak_lag_ms`) so future re-calibrations of
            `audio_max_lag_ms` (or the penalty shape) can be recomputed
            from stored data without re-extracting audio. `score` is the
            clamped product `peak_norm × penalty` in [0, 1].
            """
            n = min(a.size, b.size)
            if n == 0:
                return None
            a = a[:n]; b = b[:n]
            na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
            if na == 0.0 and nb == 0.0:
                return {"score": 1.0, "peak_norm": 1.0, "peak_lag_ms": 0.0}
            denom = na * nb
            if denom <= 0.0:
                return {"score": 0.0, "peak_norm": 0.0, "peak_lag_ms": None}
            xc = correlate(a, b, mode="full", method="fft")
            center = n - 1                                     # zero-lag index
            max_lag = max(1, int(round(audio_max_lag_ms * 1e-3 * audio_sr)))
            lo = max(0, center - max_lag)
            hi = min(len(xc), center + max_lag + 1)
            window = xc[lo:hi]
            if window.size == 0:
                return {"score": 0.0, "peak_norm": 0.0, "peak_lag_ms": None}
            rel_idx = int(np.argmax(window))
            peak_lag_samples = (lo + rel_idx) - center
            peak_norm = float(window[rel_idx]) / denom
            penalty = max(0.0, 1.0 - abs(peak_lag_samples) / float(max_lag))
            score = max(0.0, min(1.0, peak_norm * penalty))
            return {
                "score": score,
                "peak_norm": peak_norm,
                "peak_lag_ms": peak_lag_samples * 1000.0 / float(audio_sr),
            }

        if gt_start is not None and end_s > gt_start and start_s < gt_end:
            ow_start = max(start_s, gt_start)
            ow_end = min(end_s, gt_end)
            i0 = int(round((ow_start - start_s) * audio_sr))
            i1 = int(round((ow_end - start_s) * audio_sr))
            i0 = max(0, min(n, i0))
            i1 = max(0, min(n, i1))
            a_in = a_full[i0:i1]; b_in = b_full[i0:i1]
            a_out_parts = [a_full[:i0], a_full[i1:]]
            b_out_parts = [b_full[:i0], b_full[i1:]]
            a_out = np.concatenate(a_out_parts) if any(p.size for p in a_out_parts) else np.zeros(0, dtype=np.float32)
            b_out = np.concatenate(b_out_parts) if any(p.size for p in b_out_parts) else np.zeros(0, dtype=np.float32)
            audio_dur_in = (ow_end - ow_start)
            audio_dur_out = duration - audio_dur_in
            xc_in = _xcorr(a_in, b_in)
            xc_out = _xcorr(a_out, b_out)
        else:
            audio_dur_in = 0.0
            audio_dur_out = duration
            xc_in = None
            xc_out = _xcorr(a_full, b_full)

        def _g(d, k):
            return d[k] if d is not None else None

        return {
            "start_s": float(start_s),
            "end_s": float(end_s),
            "duration_s": float(duration),
            "n_video_in": int(n_video_in),
            "n_video_out": int(n_video_out),
            "n_hash_skipped": int(n_hash_skipped),
            "video_ssim_in": ssim_in,
            "video_ssim_out": ssim_out,
            "video_lag_in_frames": mean_abs_lag_in,
            "video_lag_out_frames": mean_abs_lag_out,
            "audio_xcorr_in": _g(xc_in, "score"),
            "audio_xcorr_out": _g(xc_out, "score"),
            "audio_peak_norm_in": _g(xc_in, "peak_norm"),
            "audio_peak_norm_out": _g(xc_out, "peak_norm"),
            "audio_peak_lag_ms_in": _g(xc_in, "peak_lag_ms"),
            "audio_peak_lag_ms_out": _g(xc_out, "peak_lag_ms"),
            "audio_dur_in": float(audio_dur_in),
            "audio_dur_out": float(audio_dur_out),
        }
    except Exception as e:
        return {
            "start_s": float(start_s),
            "end_s": float(end_s),
            "error": f"chunk failed: {type(e).__name__}: {e}",
        }


def _build_chunks(duration: float, chunk_s: float) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    t = 0.0
    while t < duration - 1e-6:
        chunks.append((t, min(t + chunk_s, duration)))
        t += chunk_s
    return chunks


def score_edit(
    fixed_mp4: Path,
    source_mp4: Path,
    profile: dict[str, Any],
    *,
    max_points: int = 60,
    sample_fps: float = SAMPLE_FPS,
    chunk_duration_s: float = CHUNK_DURATION_S,
    max_workers: int | None = None,
    hash_threshold: int = HASH_THRESHOLD,
    w_in: float = W_IN,
    w_out: float = W_OUT,
    video_w_in: float | None = None,
    video_w_out: float | None = None,
    audio_w_in: float | None = None,
    audio_w_out: float | None = None,
    audio_max_lag_ms: float = AUDIO_MAX_LAG_MS,
    chroma_norm: float = CHROMA_NORM,
    lag_band_frames: int = LAG_BAND_FRAMES,
    lag_penalty_per_frame: float = LAG_PENALTY_PER_FRAME,
    lag_topk: int = LAG_TOPK,
    video_share: float = VIDEO_SHARE,
    audio_share: float = AUDIO_SHARE,
) -> dict[str, Any]:
    # Per-modality overrides fall back to the unified w_in / w_out when None.
    # task.json `scoring` blocks can target one modality without touching the
    # other (e.g. video_w_out=1.0 to soften "false touch" video damage while
    # leaving audio at the default w_out).
    v_w_in  = w_in  if video_w_in  is None else float(video_w_in)
    v_w_out = w_out if video_w_out is None else float(video_w_out)
    a_w_in  = w_in  if audio_w_in  is None else float(audio_w_in)
    a_w_out = w_out if audio_w_out is None else float(audio_w_out)
    fixed = str(Path(fixed_mp4))
    source = str(Path(source_mp4))

    try:
        duration = float(profile["format"]["duration_s"])
    except (KeyError, TypeError, ValueError) as e:
        return {"score": 0, "max": max_points, "error": f"bad profile: {e}"}

    chunks = _build_chunks(duration, chunk_duration_s)
    if not chunks:
        return {"score": 0, "max": max_points, "error": "zero-duration video"}

    if max_workers is None:
        max_workers = min(len(chunks), os.cpu_count() or 1)
    max_workers = max(1, int(max_workers))

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            gt_win = _injection_window(profile)
            futures = [
                ex.submit(
                    _score_chunk_worker,
                    fixed, source, start, end, sample_fps, AUDIO_SR, hash_threshold, gt_win,
                    audio_max_lag_ms, chroma_norm,
                    lag_band_frames, lag_penalty_per_frame, lag_topk,
                )
                for start, end in chunks
            ]
            results = [f.result() for f in futures]
    except Exception as e:
        return {"score": 0, "max": max_points, "error": f"pool dispatch failed: {e}"}

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if not valid:
        return {
            "score": 0,
            "max": max_points,
            "error": f"all {len(chunks)} chunks failed",
            "chunk_errors": errors,
        }

    # Sample-count-weighted SSIM, duration-weighted xcorr, split by window.
    def _weighted(sig_key: str, w_key: str) -> float | None:
        num = 0.0; den = 0.0
        for r in valid:
            s = r.get(sig_key)
            w = r.get(w_key) or 0
            if s is None or w <= 0:
                continue
            num += float(s) * float(w)
            den += float(w)
        return (num / den) if den > 0 else None

    ssim_in = _weighted("video_ssim_in", "n_video_in")
    ssim_out = _weighted("video_ssim_out", "n_video_out")
    lag_in = _weighted("video_lag_in_frames", "n_video_in")
    lag_out = _weighted("video_lag_out_frames", "n_video_out")
    xcorr_in = _weighted("audio_xcorr_in", "audio_dur_in")
    xcorr_out = _weighted("audio_xcorr_out", "audio_dur_out")

    video_budget = max_points * video_share
    audio_budget = max_points * audio_share
    video_sub, v_deduct_in, v_deduct_out = _deduct_sub(ssim_in, ssim_out, video_budget, v_w_in, v_w_out)
    audio_sub, a_deduct_in, a_deduct_out = _deduct_sub(xcorr_in, xcorr_out, audio_budget, a_w_in, a_w_out)

    inj = _injection_window(profile)
    return {
        "score": float(video_sub + audio_sub),
        "max": max_points,
        "window_s": [float(inj[0]), float(inj[1])] if inj else None,
        "video": {
            "sub": float(video_sub),
            "ssim_in": float(ssim_in) if ssim_in is not None else None,
            "ssim_out": float(ssim_out) if ssim_out is not None else None,
            "deduct_in": float(v_deduct_in),
            "deduct_out": float(v_deduct_out),
            "mean_abs_lag_frames_in": float(lag_in) if lag_in is not None else None,
            "mean_abs_lag_frames_out": float(lag_out) if lag_out is not None else None,
            "w_in": float(v_w_in),
            "w_out": float(v_w_out),
        },
        "audio": {
            "sub": float(audio_sub),
            "xcorr_in": float(xcorr_in) if xcorr_in is not None else None,
            "xcorr_out": float(xcorr_out) if xcorr_out is not None else None,
            "deduct_in": float(a_deduct_in),
            "deduct_out": float(a_deduct_out),
            "w_in": float(a_w_in),
            "w_out": float(a_w_out),
        },
        # Backward-compat top-level fields for Stage 2 consumers.
        "video_sub": float(video_sub),
        "audio_sub": float(audio_sub),
        "video_ssim_mean": float(ssim_out) if ssim_out is not None else 0.0,  # sample-mean outside window
        "audio_xcorr_peak": float(xcorr_out) if xcorr_out is not None else 0.0,
        "scoring": {
            "mode": "whole_video_deduction",
            "sample_fps": float(sample_fps),
            "chunk_duration_s": float(chunk_duration_s),
            "n_chunks": len(chunks),
            "n_workers": int(max_workers),
            "n_failed_chunks": len(errors),
            "hash_threshold": int(hash_threshold),
            "n_hash_skipped": int(sum(r.get("n_hash_skipped", 0) for r in valid)),
            "n_samples_total": int(sum((r.get("n_video_in") or 0) + (r.get("n_video_out") or 0) for r in valid)),
            "w_in": float(w_in),
            "w_out": float(w_out),
            "video_budget": float(video_budget),
            "audio_budget": float(audio_budget),
            "audio_max_lag_ms": float(audio_max_lag_ms),
            "chroma_norm": float(chroma_norm),
            "lag_band_frames": int(lag_band_frames),
            "lag_penalty_per_frame": float(lag_penalty_per_frame),
            "lag_topk": int(lag_topk),
        },
        "chunks": valid,
    }
