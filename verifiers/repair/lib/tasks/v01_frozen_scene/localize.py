"""Localization scorer for v01_frozen_scene.

Single-window scoring across all three modes. For `frozen_av` and
`frozen_video_only` the spec produces one window and the truth is that
window's bounds. For `buffering` the spec produces multiple per-scene
windows; the truth here is the **outer span** (`min(start) → max(end)`
over all windows), exposed at the top level of the injection block as
`start_s` / `end_s`. The agent reports a single (start, end) covering
the affected stretch — buffering doesn't ask the agent to enumerate
each freeze separately.

Falls back to the legacy flat schema (no `windows` array) so any
pre-Run-#5 v01 instance is still scorable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rubrics._localize import (
    compose_localization, DESCRIPTION_FRACTION,
    SCENE_WEIGHT, ENDPOINT_WEIGHT, IOU_WEIGHT,
)


VARIANT_ID = "v01_frozen_scene"


def score_localization(
    diagnosis_md: Path,
    profile: dict[str, Any],
    *,
    max_points: int = 35,
) -> dict[str, Any]:
    try:
        inj = profile["injection"]
        scenes = profile["narrative"]["scene_list"]

        # Outer-span truth — works for both single- and multi-window
        # injections. The injector always sets flat start_s/end_s to
        # the outer span (single-window: the only window's bounds;
        # buffering: convex hull over all per-scene freezes).
        truth = (float(inj["start_s"]), float(inj["end_s"]))

        # Target-scene bounds. Single-window injections expose
        # `target_scene_index` at the top of the block; buffering
        # exposes a `windows` list, each with its own `target_scene_index`.
        # Score-with-scene treats target_bounds as a list, so we just
        # union them either way (currently SCENE_WEIGHT=0 so this is
        # cosmetic but kept correct).
        windows = inj.get("windows")
        if windows:
            target_bounds = [
                (float(scenes[int(w["target_scene_index"])]["start_s"]),
                 float(scenes[int(w["target_scene_index"])]["end_s"]))
                for w in windows
            ]
        else:
            sc = scenes[int(inj["target_scene_index"])]
            target_bounds = [(float(sc["start_s"]), float(sc["end_s"]))]
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return _error_result(max_points, f"profile missing v01 injection fields: {e}")

    return compose_localization(
        diagnosis_md, truth, target_bounds, VARIANT_ID, max_points,
    )


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
