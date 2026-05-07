"""Localization scorer for combined-defect instances (Run #6+).

A combined-defect profile carries `injections: [...]` (plural list) instead
of the single-defect `injection: {...}`. Each injection has its own variant,
`start_s` / `end_s`, and `target_scene_index` (None for global defects like
v07_audio_video_desync). The agent emits one `## Diagnosis` block per defect.

Per-injection truth: `(start_s, end_s)` straight off the injection record.
Per-injection scene bounds: the named target scene if present, else the full
clip span (mirrors v07's "scene_hit always passes inside the video" rule).

Predicted blocks ↔ truth injections are matched best-IoU-per-truth by the
shared `score_with_scene_multi` helper, so the order in which the agent
reports defects doesn't matter and an agent that collapses two contiguous
defects into a single window still earns partial credit on both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rubrics._localize import (
    compose_localization_multi, DESCRIPTION_FRACTION,
    SCENE_WEIGHT, ENDPOINT_WEIGHT, IOU_WEIGHT,
)


VARIANT_ID = "_combined"


def score_localization(
    diagnosis_md: Path,
    profile: dict[str, Any],
    *,
    max_points: int = 35,
) -> dict[str, Any]:
    try:
        injections = profile["injections"]
        if not isinstance(injections, list) or not injections:
            raise KeyError("profile.injections must be a non-empty list")

        scenes = profile["narrative"]["scene_list"]
        video_dur = float(profile["format"]["duration_s"])

        truths_and_scenes: list[
            tuple[tuple[float, float], list[tuple[float, float]]]
        ] = []
        for inj in injections:
            truth = (float(inj["start_s"]), float(inj["end_s"]))
            tsi = inj.get("target_scene_index")
            if tsi is None:
                # Global defect (v07-style). Treat the whole clip as the
                # scene-hit target so any predicted window inside the video
                # passes the scene-hit floor.
                scene_bounds = [(0.0, video_dur)]
            else:
                sc = scenes[int(tsi)]
                scene_bounds = [(float(sc["start_s"]), float(sc["end_s"]))]
            truths_and_scenes.append((truth, scene_bounds))
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return _error_result(max_points, f"profile missing combined fields: {e}")

    return compose_localization_multi(
        diagnosis_md, truths_and_scenes, VARIANT_ID, max_points,
    )


def _error_result(max_points: int, rationale: str) -> dict[str, Any]:
    """Full-zero result matching the multi-window composite schema."""
    description_max = int(round(max_points * DESCRIPTION_FRACTION))
    window_max = max_points - description_max
    return {
        "score": 0.0, "max": max_points,
        "window_sub": 0.0, "window_max": window_max,
        "description_sub": 0.0, "description_max": description_max,
        "predicted_window_s": None, "truth_window_s": None,
        "iou": 0.0,
        "iou_sub": 0.0, "iou_max": window_max * IOU_WEIGHT,
        "scene_hit": False, "scene_hit_sub": 0.0, "scene_hit_max": window_max * SCENE_WEIGHT,
        "start_delta_s": None, "end_delta_s": None,
        "start_sub": 0.0, "start_max": window_max * ENDPOINT_WEIGHT,
        "end_sub": 0.0,   "end_max":   window_max * ENDPOINT_WEIGHT,
        "windows": [], "n_predicted": 0, "n_truth": 0,
        "description": {"score": 0.0, "max": description_max, "status": "skipped",
                        "rationale": "n/a (upstream error)", "criteria": []},
        "rationale": rationale,
    }
