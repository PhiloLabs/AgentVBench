"""Pure-Python check kinds for cutbench rubrics.

Each kind is a deterministic check: takes a video path + params, returns
(passed: bool, evidence: str). No LLM calls.

Usage:
    from verifiers.cutbench.kinds import run_kind
    passed, evidence = run_kind("ffprobe_duration_range",
                                video_path="/tmp/final.mp4",
                                params={"min": 59.9, "max": 60.1})
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


# ---- ffprobe primitives (cached per video_path) -----------------------------

_PROBE_CACHE: dict[str, dict] = {}


def _ffprobe_full(video_path: str) -> dict:
    """Run ffprobe -show_format -show_streams and cache the JSON result."""
    if video_path in _PROBE_CACHE:
        return _PROBE_CACHE[video_path]
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams",
         "-print_format", "json", video_path],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        return {"error": out.stderr.strip()[:300]}
    data = json.loads(out.stdout)
    _PROBE_CACHE[video_path] = data
    return data


def _video_stream(probe: dict) -> dict:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    return {}


def _audio_streams(probe: dict) -> list[dict]:
    return [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]


def _format_duration(probe: dict) -> float | None:
    f = probe.get("format", {})
    d = f.get("duration")
    return float(d) if d is not None else None


def _parse_fps(rate: str) -> float:
    """Parse 'num/den' rational to float."""
    if not rate:
        return 0.0
    if "/" in rate:
        n, d = rate.split("/", 1)
        return float(n) / float(d) if float(d) != 0 else 0.0
    return float(rate)


# ---- Kind implementations ---------------------------------------------------

def _kind_ffprobe_duration_range(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {min: float, max: float}"""
    p = _ffprobe_full(video_path)
    d = _format_duration(p)
    if d is None:
        return False, "ffprobe: no format duration"
    lo, hi = float(params["min"]), float(params["max"])
    return (lo <= d <= hi, f"duration={d:.3f}s, range=[{lo}, {hi}]")


def _kind_ffprobe_resolution_eq(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {width: int, height: int}"""
    p = _ffprobe_full(video_path)
    v = _video_stream(p)
    w, h = v.get("width"), v.get("height")
    want_w, want_h = int(params["width"]), int(params["height"])
    return (w == want_w and h == want_h, f"resolution={w}x{h}, want={want_w}x{want_h}")


def _kind_ffprobe_resolution_in(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {options: [[w,h], ...]} — pass iff (w,h) matches any option"""
    p = _ffprobe_full(video_path)
    v = _video_stream(p)
    w, h = v.get("width"), v.get("height")
    options = [tuple(o) for o in params.get("options", [])]
    return ((w, h) in options, f"resolution={w}x{h}, options={options}")


def _kind_ffprobe_video_codec_in(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {codecs: [str, ...]}"""
    p = _ffprobe_full(video_path)
    v = _video_stream(p)
    codec = v.get("codec_name", "").lower()
    allowed = [c.lower() for c in params["codecs"]]
    return (codec in allowed, f"video codec={codec}, allowed={allowed}")


def _kind_ffprobe_audio_codec_in(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {codecs: [str, ...]}"""
    p = _ffprobe_full(video_path)
    a = _audio_streams(p)
    if not a:
        return False, "no audio streams"
    codec = a[0].get("codec_name", "").lower()
    allowed = [c.lower() for c in params["codecs"]]
    return (codec in allowed, f"audio codec={codec}, allowed={allowed}")


def _kind_ffprobe_format_in(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {substrings: [str, ...]} — format_name contains any"""
    p = _ffprobe_full(video_path)
    fmt = p.get("format", {}).get("format_name", "").lower()
    needles = [s.lower() for s in params["substrings"]]
    ok = any(n in fmt for n in needles)
    return (ok, f"format_name={fmt}, needles={needles}")


def _kind_ffprobe_fps_range(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {min: float, max: float (optional)}"""
    p = _ffprobe_full(video_path)
    v = _video_stream(p)
    fps = _parse_fps(v.get("r_frame_rate", "0/1"))
    lo = float(params["min"])
    hi = float(params.get("max", 1e9))
    return (lo <= fps <= hi, f"fps={fps:.3f}, range=[{lo}, {hi}]")


def _kind_ffprobe_audio_streams(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {min: int (optional), max: int (optional), eq: int (optional)}"""
    p = _ffprobe_full(video_path)
    n = len(_audio_streams(p))
    if "eq" in params:
        return (n == int(params["eq"]), f"audio_streams={n}, want={params['eq']}")
    lo = int(params.get("min", 0))
    hi = int(params.get("max", 1_000_000))
    return (lo <= n <= hi, f"audio_streams={n}, range=[{lo}, {hi}]")


def _kind_ffprobe_audio_channels_eq(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {channels: int}"""
    p = _ffprobe_full(video_path)
    a = _audio_streams(p)
    if not a:
        return False, "no audio streams"
    ch = a[0].get("channels", 0)
    want = int(params["channels"])
    return (ch == want, f"channels={ch}, want={want}")


def _kind_ffmpeg_audio_loudness_min(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {threshold_db: float (e.g. -50)}
    Pass iff mean_volume from `volumedetect` is above the threshold.
    """
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", video_path,
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, timeout=180,
    )
    # volumedetect logs to stderr like: [Parsed_volumedetect_0...] mean_volume: -23.4 dB
    m = re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB", out.stderr)
    if not m:
        return False, f"could not parse mean_volume from ffmpeg output"
    mean_db = float(m.group(1))
    th = float(params["threshold_db"])
    return (mean_db > th, f"mean_volume={mean_db}dB, threshold={th}dB")


def _kind_ffmpeg_blackdetect_max_run(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {max_seconds: float, exclude_last_seconds: float (default 0), pix_th: float (default 0.10)}
    Pass iff no black-frame stretch longer than max_seconds, ignoring the last
    `exclude_last_seconds` of runtime (intended for end title cards).
    """
    pix_th = float(params.get("pix_th", 0.10))
    max_run = float(params["max_seconds"])
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", video_path,
         "-vf", f"blackdetect=d={max_run/2}:pix_th={pix_th}",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    p = _ffprobe_full(video_path)
    runtime = _format_duration(p) or 0.0
    cutoff = runtime - float(params.get("exclude_last_seconds", 0.0))
    # Lines like: [blackdetect @ 0x7f...] black_start:1.5 black_end:2.5 black_duration:1.0
    runs = re.findall(
        r"black_start:([0-9.]+)\s*black_end:([0-9.]+)\s*black_duration:([0-9.]+)",
        out.stderr,
    )
    long_runs = []
    for s, e, d in runs:
        s, e, d = float(s), float(e), float(d)
        if e <= cutoff and d > max_run:
            long_runs.append(f"{s:.2f}-{e:.2f}s (dur={d:.2f}s)")
    if long_runs:
        return False, f"black runs >{max_run}s before {cutoff:.2f}s: {long_runs[:3]}"
    return True, f"no black runs >{max_run}s before {cutoff:.2f}s ({len(runs)} short runs)"


def _kind_ffmpeg_freeze_max_run(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {max_seconds: float, exclude_last_seconds: float (default 0)}
    Pass iff no static-frame stretch longer than max_seconds, ignoring last N.
    Uses ffmpeg freezedetect.
    """
    max_run = float(params["max_seconds"])
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", video_path,
         "-vf", f"freezedetect=n=0.003:d={max_run/2}",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    p = _ffprobe_full(video_path)
    runtime = _format_duration(p) or 0.0
    cutoff = runtime - float(params.get("exclude_last_seconds", 0.0))
    starts = re.findall(r"freeze_start:\s*([0-9.]+)", out.stderr)
    ends = re.findall(r"freeze_end:\s*([0-9.]+)", out.stderr)
    long_runs = []
    for s, e in zip(starts, ends):
        s, e = float(s), float(e)
        d = e - s
        if e <= cutoff and d > max_run:
            long_runs.append(f"{s:.2f}-{e:.2f}s (dur={d:.2f}s)")
    if long_runs:
        return False, f"freeze runs >{max_run}s before {cutoff:.2f}s: {long_runs[:3]}"
    return True, f"no freeze runs >{max_run}s before {cutoff:.2f}s"


def _kind_ffmpeg_brightness_at_times(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {times: [float, ...], min_brightness: float (0-255), min_passing: int (default = all)}
    Pass iff at least min_passing of the sampled frames have mean brightness > min_brightness.
    """
    times = params["times"]
    th = float(params["min_brightness"])
    min_passing = int(params.get("min_passing", len(times)))
    passed_count = 0
    samples = []
    for t in times:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-ss", str(t), "-i", video_path,
             "-vframes", "1",
             "-vf", "format=gray,signalstats", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        m = re.search(r"YAVG:([0-9.]+)", out.stderr)
        if m:
            avg = float(m.group(1))
            samples.append(f"t={t}s YAVG={avg:.1f}")
            if avg > th:
                passed_count += 1
    return (passed_count >= min_passing,
            f"{passed_count}/{len(times)} frames with brightness>{th} ({samples})")


# ---- Specialty kinds (require extra deps) -----------------------------------

def _kind_scenedetect_median_shot_duration(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {min: float (optional), max: float (optional)} — median shot duration in seconds"""
    from scenedetect import SceneManager, open_video, ContentDetector

    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27))
    sm.detect_scenes(video=video)
    scenes = sm.get_scene_list()
    durations = [(end - start).get_seconds() for start, end in scenes]
    if not durations:
        return False, "no scenes detected"
    durations.sort()
    median = durations[len(durations) // 2]
    lo = float(params.get("min", 0))
    hi = float(params.get("max", 1e9))
    return (lo <= median <= hi,
            f"median_shot_duration={median:.2f}s, range=[{lo}, {hi}], n_scenes={len(scenes)}")


def _kind_pytesseract_substring_match(
    video_path: str, params: dict,
) -> tuple[bool, str]:
    """params: {substrings: [str, ...], case_insensitive: bool, sample_fps: float, mode: 'any'|'all'}"""
    import pytesseract
    from PIL import Image
    import tempfile, os
    from pathlib import Path

    sample_fps = float(params.get("sample_fps", 1.0))
    case_ins = bool(params.get("case_insensitive", True))
    mode = params.get("mode", "any")
    needles = params["substrings"]

    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", video_path,
             "-vf", f"fps={sample_fps}", f"{tmp}/f%05d.png"],
            capture_output=True, timeout=120,
        )
        frames = sorted(Path(tmp).glob("f*.png"))
        all_text = []
        for fp in frames:
            txt = pytesseract.image_to_string(Image.open(fp))
            if case_ins:
                txt = txt.lower()
            all_text.append(txt)
        joined = "\n".join(all_text)
        needles_n = [n.lower() if case_ins else n for n in needles]
        hits = [n for n in needles_n if n in joined]
        if mode == "all":
            ok = len(hits) == len(needles_n)
        else:
            ok = len(hits) > 0
        return (ok,
                f"sampled {len(frames)} frames, hit {len(hits)}/{len(needles_n)} substrings: {hits[:5]}")


def _kind_ffmpeg_decode_no_errors(video_path: str, params: dict) -> tuple[bool, str]:
    """No params. Pass iff `ffmpeg -v error -i FILE -f null -` produces no error lines."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", video_path, "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    err_text = out.stderr.strip()
    if err_text:
        return False, f"ffmpeg errors: {err_text[:200]}"
    return True, "decoded end-to-end with no errors"


def _kind_ffmpeg_loudness_lufs_range(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {min: float (LUFS, optional), max: float (LUFS, optional)}
    Uses ffmpeg loudnorm filter to extract integrated_LUFS.
    """
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", video_path,
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    # Last `{...}` block in stderr is the JSON
    blocks = re.findall(r"\{[^{}]*\}", out.stderr, re.DOTALL)
    if not blocks:
        return False, "could not parse loudnorm json"
    try:
        info = json.loads(blocks[-1])
        lufs = float(info.get("input_i", "nan"))
    except (json.JSONDecodeError, ValueError):
        return False, "could not parse loudnorm output"
    lo = float(params.get("min", -1e9))
    hi = float(params.get("max", 1e9))
    return (lo <= lufs <= hi, f"integrated_LUFS={lufs:.2f}, range=[{lo}, {hi}]")


def _kind_ffprobe_bitrate_min(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {min_bps: int}
    Pass iff video stream bit_rate (or format bit_rate as fallback) ≥ min_bps.
    """
    p = _ffprobe_full(video_path)
    v = _video_stream(p)
    br_str = v.get("bit_rate") or p.get("format", {}).get("bit_rate")
    if not br_str:
        return False, "no bit_rate field"
    try:
        br = int(br_str)
    except ValueError:
        return False, f"unparsable bit_rate: {br_str}"
    th = int(params["min_bps"])
    return (br >= th, f"bitrate={br} bps, min={th} bps")


def _kind_scenedetect_max_shot_duration(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {max_seconds: float}
    Pass iff no single shot exceeds max_seconds.
    """
    from scenedetect import SceneManager, open_video, ContentDetector

    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27))
    sm.detect_scenes(video=video)
    scenes = sm.get_scene_list()
    if not scenes:
        return False, "no scenes detected"
    durations = [(end - start).get_seconds() for start, end in scenes]
    max_dur = max(durations)
    th = float(params["max_seconds"])
    return (max_dur <= th,
            f"max_shot_duration={max_dur:.2f}s, threshold={th}s, n_scenes={len(scenes)}")


def _kind_librosa_silence_max_run(video_path: str, params: dict) -> tuple[bool, str]:
    """params: {max_seconds: float, threshold_db: float (default -40)}
    Pass iff no silent stretch (RMS below threshold) longer than max_seconds.
    """
    import librosa
    import numpy as np
    import tempfile

    th_db = float(params.get("threshold_db", -40))
    max_run = float(params["max_seconds"])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-i", video_path,
             "-vn", "-ar", "22050", "-ac", "1", tmp.name],
            capture_output=True, timeout=180,
        )
        y, sr = librosa.load(tmp.name, sr=22050, mono=True)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    silent = db < th_db
    # Find longest contiguous run of silent frames
    max_run_frames = 0
    cur = 0
    for s in silent:
        if s:
            cur += 1
            max_run_frames = max(max_run_frames, cur)
        else:
            cur = 0
    sec_per_frame = 512 / sr
    longest_silent_seconds = max_run_frames * sec_per_frame
    return (longest_silent_seconds <= max_run,
            f"longest_silent={longest_silent_seconds:.2f}s, max_allowed={max_run}s")


# ---- Dispatcher -------------------------------------------------------------

_KIND_REGISTRY = {
    "ffprobe_duration_range":       _kind_ffprobe_duration_range,
    "ffprobe_resolution_eq":        _kind_ffprobe_resolution_eq,
    "ffprobe_resolution_in":        _kind_ffprobe_resolution_in,
    "ffprobe_video_codec_in":       _kind_ffprobe_video_codec_in,
    "ffprobe_audio_codec_in":       _kind_ffprobe_audio_codec_in,
    "ffprobe_format_in":            _kind_ffprobe_format_in,
    "ffprobe_fps_range":            _kind_ffprobe_fps_range,
    "ffprobe_audio_streams":        _kind_ffprobe_audio_streams,
    "ffprobe_audio_channels_eq":    _kind_ffprobe_audio_channels_eq,
    "ffmpeg_audio_loudness_min":    _kind_ffmpeg_audio_loudness_min,
    "ffmpeg_blackdetect_max_run":   _kind_ffmpeg_blackdetect_max_run,
    "ffmpeg_freeze_max_run":        _kind_ffmpeg_freeze_max_run,
    "ffmpeg_brightness_at_times":   _kind_ffmpeg_brightness_at_times,
    "scenedetect_median_shot_duration": _kind_scenedetect_median_shot_duration,
    "pytesseract_substring_match":  _kind_pytesseract_substring_match,
    "librosa_silence_max_run":      _kind_librosa_silence_max_run,
    "ffmpeg_decode_no_errors":      _kind_ffmpeg_decode_no_errors,
    "ffmpeg_loudness_lufs_range":   _kind_ffmpeg_loudness_lufs_range,
    "ffprobe_bitrate_min":          _kind_ffprobe_bitrate_min,
    "scenedetect_max_shot_duration": _kind_scenedetect_max_shot_duration,
}


def list_kinds() -> list[str]:
    return sorted(_KIND_REGISTRY.keys())


def run_kind(kind: str, video_path: str, params: dict) -> tuple[bool, str]:
    """Dispatch a deterministic check by kind name.

    Returns (passed, evidence). On any exception, returns (False, "error: ...").
    """
    fn = _KIND_REGISTRY.get(kind)
    if fn is None:
        return False, f"unknown kind '{kind}'"
    try:
        return fn(video_path, params)
    except Exception as e:
        return False, f"error in {kind}: {type(e).__name__}: {e}"
