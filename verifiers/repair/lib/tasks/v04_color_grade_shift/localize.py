"""Localization scorer for v04_color_grade_shift — window + VLM description."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rubrics._localize import (
    compose_localization, DESCRIPTION_FRACTION,
    SCENE_WEIGHT, ENDPOINT_WEIGHT, IOU_WEIGHT,
)


VARIANT_ID = "v04_color_grade_shift"


def score_localization(
    diagnosis_md: Path,
    profile: dict[str, Any],
    *,
    max_points: int = 35,
) -> dict[str, Any]:
    try:
        inj = profile["injection"]
        truth = (float(inj["start_s"]), float(inj["end_s"]))
        scenes = profile["narrative"]["scene_list"]
        sc = scenes[int(inj["target_scene_index"])]
        target_bounds = [(float(sc["start_s"]), float(sc["end_s"]))]
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return _error_result(max_points, f"profile missing injection/scene fields: {e}")
    return compose_localization(diagnosis_md, truth, target_bounds, VARIANT_ID, max_points)


def _error_result(max_points: int, rationale: str) -> dict[str, Any]:
    """Full-zero result that matches the composite schema."""
    description_max = int(round(max_points * DESCRIPTION_FRACTION))
    window_max = max_points - description_max
    return {
        "score": 0.0, "max": max_points,
        "window_sub": 0.0, "window_max": window_max,
        "description_sub": 0.0, "description_max": description_max,
        "predicted_window_s": None, "truth_window_s": None,
        "iou": 0.0, "iou_sub": 0.0, "iou_max": window_max * IOU_WEIGHT,
        "scene_hit": False, "scene_hit_sub": 0.0, "scene_hit_max": window_max * SCENE_WEIGHT,
        "start_delta_s": None, "end_delta_s": None,
        "start_sub": 0.0, "start_max": window_max * ENDPOINT_WEIGHT,
        "end_sub": 0.0, "end_max": window_max * ENDPOINT_WEIGHT,
        "description": {"score": 0.0, "max": description_max, "status": "skipped",
                        "rationale": "n/a (upstream error)", "criteria": []},
        "rationale": rationale,
    }
