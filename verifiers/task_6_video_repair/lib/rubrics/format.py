"""Format scorer (universal, 10 pts).

Binary per-item ffprobe comparison of `fixed.mp4` against `source.mp4`.
Seven items; each passes or fails independently, score is linear in
number passed:

    score = max_points * passed / 7

All probes share one `ffprobe` call per file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


_CONFIG_PATH = Path(__file__).parent / "format_config.json"
_DEFAULT_TOLERANCES = {"duration_s": 0.05, "fps": 0.02}


def _load_tolerances(source_mp4: Path) -> dict[str, float]:
    """Merge `default` with any `per_source[<stem>]` override from format_config.json.

    Source-indexed so a new capture can pin its own tolerance without touching
    code — add a stem under `per_source` and the rubric picks it up.
    """
    tol = dict(_DEFAULT_TOLERANCES)
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        block = cfg.get("tolerances", {})
        tol.update(block.get("default", {}))
        tol.update(block.get("per_source", {}).get(source_mp4.stem, {}))
    except (OSError, ValueError):
        pass
    return tol


def _probe(path: Path) -> dict[str, Any]:
    res = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({path}): {res.stderr[-400:]}")
    return json.loads(res.stdout)


def _parse_fps(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


def _pick_stream(probe: dict[str, Any], codec_type: str) -> dict[str, Any] | None:
    for s in probe.get("streams", []):
        if s.get("codec_type") == codec_type:
            return s
    return None


def _summarize(probe: dict[str, Any]) -> dict[str, Any]:
    fmt = probe.get("format", {})
    video = _pick_stream(probe, "video") or {}
    audio = _pick_stream(probe, "audio") or {}
    fps = _parse_fps(video.get("avg_frame_rate")) or _parse_fps(video.get("r_frame_rate"))
    try:
        duration = float(fmt.get("duration")) if fmt.get("duration") else None
    except (TypeError, ValueError):
        duration = None
    try:
        asr = int(audio["sample_rate"]) if audio.get("sample_rate") else None
    except (TypeError, ValueError):
        asr = None
    w = video.get("width")
    h = video.get("height")
    return {
        "container": fmt.get("format_name"),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "fps": fps,
        "resolution": [w, h] if (w is not None and h is not None) else None,
        "duration_s": duration,
        "audio_sample_rate": asr,
    }


def _fail_all(max_points: int, error: str, tol: dict[str, float]) -> dict[str, Any]:
    none_item = {"pass": False, "expected": None, "actual": None}
    return {
        "score": 0.0,
        "max": max_points,
        "items": {
            "container": dict(none_item),
            "video_codec": dict(none_item),
            "audio_codec": dict(none_item),
            "fps": {**none_item, "tol": tol["fps"]},
            "resolution": dict(none_item),
            "duration_s": {**none_item, "tol": tol["duration_s"]},
            "audio_sample_rate": dict(none_item),
        },
        "error": error,
    }


def score_format(
    fixed_mp4: Path,
    source_mp4: Path,
    *,
    max_points: int = 5,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Binary per-item ffprobe comparison vs source.

    `tolerances` defaults to the per-source block in `format_config.json`;
    pass an explicit dict to override (tests, ad-hoc tuning).
    """
    fixed_mp4 = Path(fixed_mp4)
    source_mp4 = Path(source_mp4)
    tol = dict(_DEFAULT_TOLERANCES)
    tol.update(_load_tolerances(source_mp4))
    if tolerances:
        tol.update(tolerances)

    try:
        src = _summarize(_probe(source_mp4))
    except Exception as e:
        return _fail_all(max_points, f"source probe failed: {e}", tol)
    try:
        fix = _summarize(_probe(fixed_mp4))
    except Exception as e:
        return _fail_all(max_points, f"fixed probe failed: {e}", tol)

    def eq(a: Any, b: Any) -> bool:
        return a is not None and b is not None and a == b

    def fps_eq(a: float | None, b: float | None) -> bool:
        if a is None or b is None:
            return False
        return abs(a - b) <= tol["fps"]

    def dur_eq(a: float | None, b: float | None) -> bool:
        if a is None or b is None:
            return False
        return abs(a - b) <= tol["duration_s"]

    items = {
        "container": {
            "pass": eq(fix["container"], src["container"]),
            "expected": src["container"],
            "actual": fix["container"],
        },
        "video_codec": {
            "pass": eq(fix["video_codec"], src["video_codec"]),
            "expected": src["video_codec"],
            "actual": fix["video_codec"],
        },
        "audio_codec": {
            "pass": eq(fix["audio_codec"], src["audio_codec"]),
            "expected": src["audio_codec"],
            "actual": fix["audio_codec"],
        },
        "fps": {
            "pass": fps_eq(fix["fps"], src["fps"]),
            "expected": src["fps"],
            "actual": fix["fps"],
            "tol": tol["fps"],
        },
        "resolution": {
            "pass": eq(fix["resolution"], src["resolution"]),
            "expected": src["resolution"],
            "actual": fix["resolution"],
        },
        "duration_s": {
            "pass": dur_eq(fix["duration_s"], src["duration_s"]),
            "expected": src["duration_s"],
            "actual": fix["duration_s"],
            "tol": tol["duration_s"],
        },
        "audio_sample_rate": {
            "pass": eq(fix["audio_sample_rate"], src["audio_sample_rate"]),
            "expected": src["audio_sample_rate"],
            "actual": fix["audio_sample_rate"],
        },
    }

    passed = sum(1 for v in items.values() if v["pass"])
    score = max_points * passed / len(items)
    return {
        "score": float(score),
        "max": max_points,
        "items": items,
    }
