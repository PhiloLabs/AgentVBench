"""VLM-judged description scorer for the localization rubric.

What this is
------------
Calls Claude (haiku, cheap judge) with frames extracted from `broken.mp4`
around the injection window + the agent's `notes` text from `diagnosis.md`,
asks for a 4-criteria rubric judgment, caches the raw verdict to
`<pass1>/description_eval.json`, and returns the composed score.

Four criteria (each 0/1/2 → 0/1.25/2.5 pts; total 10 pts):

  1. DEFECT_CLASS       — does the text correctly name the defect type?
  2. PERCEPTUAL_CUE     — does it describe the observable symptom specifically?
  3. INTERNAL_CONSISTENCY — does the narrative match the window it claims?
  4. EVIDENCE           — does it cite tool output / measurements vs. guess?

Threat model: the VLM judge sees only `broken.mp4` (agent-visible) + the
agent's text. It never sees `sources/` (GT video) or `profile.json`. It's
told the GT defect TYPE so it can grade the DEFECT_CLASS criterion, but
not the GT window — the agent's own predicted window is the anchor for
INTERNAL_CONSISTENCY, so knowing truth would bias that judgment.

Env knobs
---------
    SCORE_VLM=0   disable this scorer (returns score=0, status=skipped)
    ANTHROPIC_API_KEY / ANTHROPIC_API_KEY_SECONDARY — required otherwise

The cache file keeps the full VLM response so re-running `score_session`
without `--vlm` (or with the API down) still picks up the prior verdict.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


CLAUDE_MODEL = "claude-haiku-4-5-20251001"   # cheap + fast judge
FRAMES_BEFORE_WINDOW = 2
FRAMES_IN_WINDOW = 4
FRAMES_AFTER_WINDOW = 2
PAD_SECONDS = 1.0
MAX_NOTES_CHARS = 3000

# Point-weights per criterion, scaled to max_points at call time.
# 4 criteria × 2 max raw-pts = 8 raw units → mapped to max_points linearly.
_CRITERIA = ("defect_class", "perceptual_cue", "internal_consistency", "evidence")


_VARIANT_BLURB = {
    "v01_frozen_scene": (
        "frozen_scene — within the defect window the video freezes on a single "
        "frame (motion stops while audio may continue)."
    ),
    "v02_downscale": (
        "downscale — within the defect window the video is lower-resolution "
        "than surrounding shots (noticeably softer; audio unaffected)."
    ),
    "v03_scene_swap": (
        "scene_swap — within the defect window two groups of consecutive "
        "scenes are in the wrong order (the original scene sequence was shuffled)."
    ),
    "v04_color_grade_shift": (
        "color_grade_shift — within the defect window the color grade differs "
        "from surrounding shots (hue / saturation / temperature shift); "
        "motion, focus, and framing are intact; audio unaffected."
    ),
    "v05_noise_floor_spike": (
        "noise_floor_spike — within the defect window an added noise layer "
        "(broadband hiss / pink rumble / brown rumble) raises the audio noise "
        "floor; dialogue timing, sync, and video are intact."
    ),
    "v06_duplicate_segment": (
        "duplicate_segment — at a scene boundary, a short stretch of the "
        "video plays twice in a row (the editor kept the overlap between "
        "clip A's tail and clip B's head). Both video and audio repeat. "
        "broken.mp4 is longer than the source by exactly the window length; "
        "the defect window brackets the SECOND copy."
    ),
}


def _load_api_key() -> str | None:
    """Prefer SECONDARY over PRIMARY, from env first, then a sibling .env.

    Looks for `.env` next to the kit root (parent of this file's `lib/`).
    On remote machines the user typically sets `ANTHROPIC_API_KEY` in the
    environment directly; the .env fallback is for local dev convenience.
    """
    found: dict[str, str] = {}
    for name in ("ANTHROPIC_API_KEY_SECONDARY", "ANTHROPIC_API_KEY"):
        if v := os.environ.get(name):
            found[name] = v
    # Search for .env next to the kit root (lib/rubrics/_description.py →
    # lib/rubrics → lib → kit_root). Falls back silently if absent — the
    # description sub-rubric returns a "skipped" status when no key is found.
    kit_root = Path(__file__).resolve().parents[2]
    for candidate in (kit_root / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                for name in ("ANTHROPIC_API_KEY_SECONDARY", "ANTHROPIC_API_KEY"):
                    if name in found:
                        continue  # env wins over .env
                    if line.strip().startswith(f"{name}="):
                        found[name] = line.strip().split("=", 1)[1].strip().strip("\"'")
            break
    return found.get("ANTHROPIC_API_KEY_SECONDARY") or found.get("ANTHROPIC_API_KEY")


def _extract_notes(diagnosis_md: Path) -> str:
    """Pull the free-text `notes:` field from diagnosis.md.

    The schema (see prompt_diagnose.md) puts it on a line like
    `- notes: ...` possibly followed by multiple indented lines. We grab
    everything from the `- notes:` marker to the end of file (or start of a
    new top-level list item).
    """
    try:
        text = diagnosis_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Case-insensitive, match `- notes:` and capture rest.
    m = re.search(r"^\s*-\s*notes\s*:\s*(.*)$", text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    # Take everything after — scoring rarely cares about terminating marker.
    return text[m.start(1):].strip()[:MAX_NOTES_CHARS]


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 0:
        return []
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _extract_frames(broken_mp4: Path, window: tuple[float, float],
                    tmpdir: Path) -> list[Path]:
    """Sample frames across the defect window with 1 s pad on each side."""
    w_start, w_end = float(window[0]), float(window[1])
    times_before = _linspace(max(0, w_start - PAD_SECONDS), w_start,
                             FRAMES_BEFORE_WINDOW + 1)[:-1]
    times_in = _linspace(w_start, w_end, FRAMES_IN_WINDOW + 2)[1:-1]
    times_after = _linspace(w_end, w_end + PAD_SECONDS,
                            FRAMES_AFTER_WINDOW + 1)[1:]
    all_times = sorted({round(t, 3) for t in times_before + times_in + times_after})

    frames: list[Path] = []
    for i, t in enumerate(all_times):
        out = tmpdir / f"frame_{i:02d}_{t:07.3f}.jpg"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(broken_mp4),
             "-vframes", "1", "-q:v", "3", str(out)],
            capture_output=True, timeout=30,
        )
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            frames.append(out)
    return frames


def _build_prompt(variant: str, predicted_window: tuple[float, float],
                  truth_window: tuple[float, float], notes: str, n_frames: int) -> str:
    variant_blurb = _VARIANT_BLURB.get(variant, variant)
    p_s, p_e = predicted_window
    t_s, t_e = truth_window
    return f"""You are judging an AI agent's diagnosis of a broken video edit.

CONTEXT
- Actual defect TYPE in this video: {variant_blurb}
- Actual defect WINDOW (ground truth): {t_s:.3f}s to {t_e:.3f}s
- The agent's predicted window: {p_s:.3f}s to {p_e:.3f}s
- {n_frames} frames from the broken video around the actual defect window
  (padded by {PAD_SECONDS:.0f}s) are attached below.

AGENT NOTES (free-text `notes:` field from diagnosis.md):
---
{notes}
---

Rate the agent's notes on these 4 criteria. Each scores 0, 1, or 2:
  0 = absent / wrong
  1 = partial / vague
  2 = present + specific

1. DEFECT_CLASS — does the text name the defect type correctly (e.g. "frozen scene", \
"downscale", "scene swap", or an unambiguous description like "frozen frame" / \
"soft-looking shot" / "shots out of order")?

2. PERCEPTUAL_CUE — does it describe the observable symptom specifically, grounded \
in what the frames show? Good: "motion stops for 2.7s, then resumes". \
Weak: "something looks off".

3. INTERNAL_CONSISTENCY — does the narrative match the window the agent claimed? \
The predicted start/end should be consistent with what the agent says they found.

4. EVIDENCE — does it cite concrete tool output or measurements (ffprobe values, \
freezedetect hits, per-frame diff, MD5s) rather than pure speculation?

Respond in STRICT JSON only, no prose outside the object:
{{"defect_class": N, "perceptual_cue": N, "internal_consistency": N, \
"evidence": N, "rationale": "one-line summary of what's right/wrong"}}
"""


def score_description(
    diagnosis_md: Path,
    broken_mp4: Path,
    truth_window: tuple[float, float],
    predicted_window: tuple[float, float] | None,
    variant: str,
    *,
    max_points: int = 9,
    cache_path: Path | None = None,
    model: str = CLAUDE_MODEL,
) -> dict[str, Any]:
    """Score the agent's `notes` against frames from `broken.mp4`.

    Never raises — all failure modes fold into `status` + `rationale`:
        "ok"         : VLM call succeeded
        "skipped"    : SCORE_VLM=0 or no API key
        "error"      : API call or parse failure
        "no_notes"   : diagnosis.md had no extractable `notes:` field
    """
    # Schema-safe zero return used on every error/skip path.
    zero = {
        "score": 0.0, "max": max_points, "status": "error",
        "model": model, "rationale": "",
        "criteria": [
            {"name": k, "raw": 0, "sub": 0.0, "max": max_points / (len(_CRITERIA) * 2) * 2}
            for k in _CRITERIA
        ],
        "n_frames": 0,
    }

    # Cache hit: return saved verdict.
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("max") == max_points and cached.get("status") == "ok":
                return cached
        except json.JSONDecodeError:
            pass

    if os.environ.get("SCORE_VLM") == "0":
        return {**zero, "status": "skipped", "rationale": "SCORE_VLM=0"}

    api_key = _load_api_key()
    if not api_key:
        return {**zero, "status": "skipped",
                "rationale": "no ANTHROPIC_API_KEY in env or .env"}

    if not broken_mp4.exists():
        return {**zero, "status": "error", "rationale": f"broken.mp4 not found: {broken_mp4}"}

    notes = _extract_notes(diagnosis_md)
    if not notes:
        return {**zero, "status": "no_notes", "rationale": "no `- notes:` field in diagnosis.md"}

    # Extract frames into a temp dir that cleans itself up.
    with tempfile.TemporaryDirectory(prefix="vlm_frames_") as td:
        tmpdir = Path(td)
        frames = _extract_frames(broken_mp4, truth_window, tmpdir)
        if not frames:
            return {**zero, "status": "error",
                    "rationale": "ffmpeg returned 0 frames — check broken.mp4 / window"}

        # Inline to avoid importing the SDK when SCORE_VLM=0 is common.
        try:
            import anthropic
        except ImportError:
            return {**zero, "status": "error", "rationale": "anthropic SDK not installed"}

        content: list[dict[str, Any]] = []
        for fp in frames:
            with open(fp, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
        pred_window = predicted_window if predicted_window else truth_window
        content.append({
            "type": "text",
            "text": _build_prompt(variant, pred_window, truth_window, notes, len(frames)),
        })

        client = anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model=model, max_tokens=512,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:
            return {**zero, "status": "error", "rationale": f"Anthropic API call failed: {e}"}

    # Outside the tempdir context — frames cleaned up.
    text = ""
    try:
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    except Exception:
        pass
    if not text.strip():
        return {**zero, "status": "error", "rationale": "empty VLM response"}

    # Extract JSON from the response (tolerates ```json fences or prose wrap).
    parsed: dict | None = None
    for pattern in (r"\{[^{}]*\}(?=\s*\Z|\s*$)", r"\{.*\}"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                break
            except json.JSONDecodeError:
                continue
    if parsed is None:
        return {**zero, "status": "error",
                "rationale": f"could not parse JSON from VLM response: {text[:200]!r}"}

    per_criterion_max = max_points / (len(_CRITERIA) * 2)  # 10 / 8 = 1.25 per raw-pt
    criteria_rows = []
    raw_total = 0
    for name in _CRITERIA:
        raw = parsed.get(name, 0)
        try:
            raw = int(raw)
        except (TypeError, ValueError):
            raw = 0
        raw = max(0, min(2, raw))
        criteria_rows.append({
            "name": name,
            "raw": raw,
            "sub": float(raw * per_criterion_max),
            "max": float(2 * per_criterion_max),
        })
        raw_total += raw

    result = {
        "score": float(raw_total * per_criterion_max),
        "max": max_points,
        "status": "ok",
        "model": model,
        "criteria": criteria_rows,
        "rationale": str(parsed.get("rationale") or "")[:400],
        "n_frames": len(frames),
    }

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, indent=2))
        except OSError:
            pass

    return result
