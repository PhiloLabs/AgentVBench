"""Localization scorer for v03_scene_swap — per-group windows in broken coords.

A scene swap produces two distinct defect regions in the **broken** timeline:
group_b's new slot (where group_a used to be) and group_a's new slot (where
group_b used to be). Gapped swaps put untouched material between them, so a
single outer-span truth is too lenient. The agent reports each region
separately (`window_1_*`, `window_2_*`) and each region is graded against
its own group's broken-coord span.

Per-region truth: `[min(broken_start_s), max(broken_end_s)]` across the
scenes tagged with that group in the injection block.

Per-region scene bounds: the per-scene broken-coord spans, so a predicted
window landing inside the group's new location passes the scene-hit floor.

Predicted windows are assigned to regions best-match-first (see
`score_with_scene_multi`), so agents that collapse two contiguous regions
into a single window still earn partial credit on both regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rubrics._localize import (
    compose_localization_multi, DESCRIPTION_FRACTION,
    SCENE_WEIGHT, ENDPOINT_WEIGHT, IOU_WEIGHT,
)


VARIANT_ID = "v03_scene_swap"


def score_localization(
    diagnosis_md: Path,
    profile: dict[str, Any],
    *,
    max_points: int = 35,
) -> dict[str, Any]:
    try:
        inj = profile["injection"]
        groups: dict[str, list[tuple[float, float]]] = {"A": [], "B": []}
        i = 1
        while (entry := inj.get(f"scene_{i}")) is not None:
            g = entry.get("group")
            if g in groups:
                groups[g].append((
                    float(entry["broken_start_s"]),
                    float(entry["broken_end_s"]),
                ))
            i += 1
        if not groups["A"] or not groups["B"]:
            raise KeyError("injection block missing group-A/B scene entries")

        truths_and_scenes: list[
            tuple[tuple[float, float], list[tuple[float, float]]]
        ] = []
        for g in ("A", "B"):
            bounds = sorted(groups[g])
            truth = (min(s for s, _ in bounds), max(e for _, e in bounds))
            truths_and_scenes.append((truth, bounds))
    except (KeyError, TypeError, ValueError) as e:
        return _error_result(max_points, f"profile missing scene_swap fields: {e}")

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
