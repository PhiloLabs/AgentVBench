"""Shared helpers for per-variant localization scorers.

Parses `diagnosis.md` (schema pinned in `prompt_diagnose.md`) and exposes
timestamp + IoU primitives, plus the composite `compose_localization()`
that combines the window scorer with the VLM description scorer per
PROJECT.md §6.2 (30 + 5 of 35 pts).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


_TS_HMS = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")
_TS_MS = re.compile(r"^\s*(\d+):(\d{1,2}(?:\.\d+)?)\s*$")
_TS_SEC = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")
_FIELD = re.compile(r"^\s*-\s*(type|start|end)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_WINDOW_FIELD = re.compile(
    r"^\s*-\s*window[_\s-]?(\d+)[_\s-]?(start|end)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_DIAG_HEADING = re.compile(r"^\s*##\s+Diagnosis\b", re.IGNORECASE)


def parse_timestamp(raw: str) -> float | None:
    """Parse `HH:MM:SS.mmm`, `MM:SS.mmm`, or `SS.mmm`. Return seconds or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = _TS_HMS.match(s)
    if m:
        h, mi, se = m.groups()
        return float(h) * 3600.0 + float(mi) * 60.0 + float(se)
    m = _TS_MS.match(s)
    if m:
        mi, se = m.groups()
        return float(mi) * 60.0 + float(se)
    m = _TS_SEC.match(s)
    if m:
        return float(m.group(1))
    return None


def parse_diagnosis(diagnosis_md: Path) -> dict[str, str | None]:
    """Extract `type`, `start`, `end` lines from a diagnosis.md.

    Returns {"type": str|None, "start": str|None, "end": str|None}. Missing
    fields stay None so the caller can report a useful rationale.
    """
    try:
        text = Path(diagnosis_md).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"type": None, "start": None, "end": None}

    found: dict[str, str | None] = {"type": None, "start": None, "end": None}
    for line in text.splitlines():
        m = _FIELD.match(line)
        if not m:
            continue
        key = m.group(1).lower()
        val = m.group(2).strip()
        # Strip trailing inline-comment / parenthetical.
        if key in found and found[key] is None:
            found[key] = val
    return found


def parse_diagnosis_windows(
    diagnosis_md: Path, *, max_windows: int = 2
) -> list[tuple[float, float] | None]:
    """Extract ordered predicted windows from a multi-window diagnosis.md.

    Three parse paths, tried in order:

    1. `- window_N_start:` / `- window_N_end:` pairs (N = 1, 2, …) within a
       single `## Diagnosis` block. Returned in ascending-N order. Used by
       v03_scene_swap and any other variant that asks for numbered windows.
    2. Multiple `## Diagnosis` blocks, each with its own `- start:` / `- end:`.
       Used by combined-defect runs (Run #6+) where the agent emits one block
       per defect. Order = block order in the file.
    3. Single `- start:` / `- end:` (no heading multiplicity). Legacy
       single-window schema. Returns a 1-element list.

    Entries are `None` when a pair is missing, unparseable, or non-positive
    (end ≤ start). Caller decides whether to skip or zero-score those slots.
    """
    try:
        text = Path(diagnosis_md).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    # Path 1: numbered window_N_start / window_N_end pairs.
    starts: dict[int, str] = {}
    ends: dict[int, str] = {}
    for line in text.splitlines():
        m = _WINDOW_FIELD.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        kind = m.group(2).lower()
        val = m.group(3).strip()
        target = starts if kind == "start" else ends
        if idx not in target:  # first-wins, consistent with parse_diagnosis
            target[idx] = val

    if starts or ends:
        windows: list[tuple[float, float] | None] = []
        for i in sorted(set(starts) | set(ends)):
            s = parse_timestamp(starts.get(i)) if i in starts else None
            e = parse_timestamp(ends.get(i)) if i in ends else None
            if s is None or e is None or e <= s:
                windows.append(None)
            else:
                windows.append((s, e))
            if len(windows) >= max_windows:
                break
        return windows

    # Path 2: multi-block — one `## Diagnosis` heading per defect, each block
    # carrying its own `- start:` / `- end:` lines. Slice the text by heading
    # and parse each slice with the single-block primitives.
    blocks = _split_diagnosis_blocks(text)
    if len(blocks) > 1:
        out: list[tuple[float, float] | None] = []
        for block_text in blocks:
            fields = _parse_fields_from_text(block_text)
            s = parse_timestamp(fields.get("start"))
            e = parse_timestamp(fields.get("end"))
            if s is None or e is None or e <= s:
                out.append(None)
            else:
                out.append((s, e))
            if len(out) >= max_windows:
                break
        return out

    # Path 3: legacy single window.
    fields = parse_diagnosis(diagnosis_md)
    s = parse_timestamp(fields.get("start"))
    e = parse_timestamp(fields.get("end"))
    if s is None or e is None or e <= s:
        return []
    return [(s, e)]


def _split_diagnosis_blocks(text: str) -> list[str]:
    """Return the text of each `## Diagnosis` block. Pre-heading text is dropped.

    A block runs from one `## Diagnosis` heading to (but not including) the
    next one — a following `## Repair trajectory` heading or EOF terminates
    the last block. Returns [] if no Diagnosis heading is present.
    """
    lines = text.splitlines()
    starts: list[int] = [i for i, ln in enumerate(lines) if _DIAG_HEADING.match(ln)]
    if not starts:
        return []
    blocks: list[str] = []
    for i, start_idx in enumerate(starts):
        end_idx = starts[i + 1] if i + 1 < len(starts) else len(lines)
        # Trim at the next non-Diagnosis `## ` heading (e.g. `## Repair trajectory`).
        for j in range(start_idx + 1, end_idx):
            stripped = lines[j].lstrip()
            if stripped.startswith("## ") and not _DIAG_HEADING.match(lines[j]):
                end_idx = j
                break
        blocks.append("\n".join(lines[start_idx:end_idx]))
    return blocks


def _parse_fields_from_text(text: str) -> dict[str, str | None]:
    """First-wins parse of `- type:` / `- start:` / `- end:` lines from a block."""
    found: dict[str, str | None] = {"type": None, "start": None, "end": None}
    for line in text.splitlines():
        m = _FIELD.match(line)
        if not m:
            continue
        key = m.group(1).lower()
        val = m.group(2).strip()
        if key in found and found[key] is None:
            found[key] = val
    return found


def iou(pred: tuple[float, float], truth: tuple[float, float]) -> float:
    p0, p1 = float(pred[0]), float(pred[1])
    t0, t1 = float(truth[0]), float(truth[1])
    if p1 <= p0 or t1 <= t0:
        return 0.0
    inter = max(0.0, min(p1, t1) - max(p0, t0))
    union = max(p1, t1) - min(p0, t0)
    if union <= 0.0:
        return 0.0
    return float(inter / union)


# Localization points split into four deterministic, independently-scored
# components summing to the full 35-pt localization budget. The VLM
# description judge is REMOVED — every point of localization comes from
# scorer-computed quantities now.
#   scene_hit  : 5/35 — binary (predicted window overlaps any target scene)
#   start      : 10/35 — `exp(-3·|Δstart|)` exponential falloff
#   end        : 10/35 — `exp(-3·|Δend|)` exponential falloff
#   iou        : 10/35 — linear in interval IoU
# Multi-window cells (v03 scene_swap, v01 buffering): per-region budget is
# `max_points / N`; each region's start/end caps at half the global endpoint
# weight when N=2, etc. Scene_hit and IoU also split per-region. Sum across
# regions still totals max_points.
SCENE_WEIGHT = 5.0 / 35.0      # ≈ 0.1429
ENDPOINT_WEIGHT = 10.0 / 35.0  # ≈ 0.2857 each of (start, end); together ~0.5714
IOU_WEIGHT = 10.0 / 35.0       # ≈ 0.2857

# Each endpoint scores `endpoint_weight × max_points × exp(-ENDPOINT_DECAY × |Δ|)`.
# Calibrated at k=3 so Δ=0.05s → 86%, Δ=0.10s → 74%, Δ=1.0s → 5% — forgives
# sub-frame drift while still penalizing full-second misses. The IoU component
# covers the other failure mode: proportionally-small miss on a long defect
# window (e.g. v03 swap, 10s off a 96s window → exp=0 but IoU≈0.8).
ENDPOINT_DECAY = 3.0


def score_with_scene(
    diagnosis_md: Path,
    truth: tuple[float, float],
    target_scene_bounds: list[tuple[float, float]],
    max_points: int,
    *,
    scene_weight: float = SCENE_WEIGHT,
    endpoint_weight: float = ENDPOINT_WEIGHT,
    endpoint_decay: float = ENDPOINT_DECAY,
    iou_weight: float = IOU_WEIGHT,
) -> dict[str, Any]:
    """Scene-hit + per-endpoint exponential-falloff + IoU scorer.

    Four additive components:
      - Scene hit (binary, worth `scene_weight × max_points`): predicted
        window has non-empty intersection with any target-scene bound.
        Overlap test (not anchor-point test) so wide over-predictions that
        swallow a short target scene still get the narrative-region credit
        — IoU + endpoint exp-decay penalize the slop.
      - Start accuracy (worth `endpoint_weight × max_points`):
        `exp(-endpoint_decay × |Δstart|)`. Sub-second precision.
      - End accuracy (same weight + same formula, on Δend).
      - IoU (worth `iou_weight × max_points`): interval intersection over
        union. Scale-invariant — a 10% proportional miss on a 100s truth
        window still scores ~0.90, rescuing long-window defects (v03) that
        the absolute-Δ exponential would otherwise zero out.
    """
    scene_max = max_points * scene_weight
    endpoint_max_each = max_points * endpoint_weight
    iou_max = max_points * iou_weight
    truth_list = [float(truth[0]), float(truth[1])]

    def _result(
        predicted: list[float] | None,
        iou_val: float,
        scene_hit: bool,
        start_delta: float | None,
        end_delta: float | None,
        start_sub: float,
        end_sub: float,
        iou_sub: float,
        rationale: str,
    ) -> dict[str, Any]:
        scene_sub = scene_max if scene_hit else 0.0
        return {
            "score": float(scene_sub + start_sub + end_sub + iou_sub),
            "max": max_points,
            "predicted_window_s": predicted,
            "truth_window_s": truth_list,
            "iou": float(iou_val),
            "iou_sub": float(iou_sub),
            "iou_max": float(iou_max),
            "scene_hit": bool(scene_hit),
            "scene_hit_sub": float(scene_sub),
            "scene_hit_max": float(scene_max),
            "start_delta_s": float(start_delta) if start_delta is not None else None,
            "end_delta_s": float(end_delta) if end_delta is not None else None,
            "start_sub": float(start_sub),
            "start_max": float(endpoint_max_each),
            "end_sub": float(end_sub),
            "end_max": float(endpoint_max_each),
            "rationale": rationale,
        }

    fields = parse_diagnosis(diagnosis_md)
    start_raw = fields.get("start")
    end_raw = fields.get("end")
    if not start_raw or not end_raw:
        return _result(None, 0.0, False, None, None, 0.0, 0.0, 0.0,
                       f"diagnosis.md missing start/end fields (got {fields!r})")

    p_start = parse_timestamp(start_raw)
    p_end = parse_timestamp(end_raw)
    if p_start is None or p_end is None:
        return _result(None, 0.0, False, None, None, 0.0, 0.0, 0.0,
                       f"unparseable timestamps: start={start_raw!r}, end={end_raw!r}")

    if p_end <= p_start:
        return _result([p_start, p_end], 0.0, False, None, None, 0.0, 0.0, 0.0,
                       f"non-positive window: end ({p_end}) <= start ({p_start})")

    iou_val = iou((p_start, p_end), truth)
    # Scene hit = predicted window overlaps any target-scene bound (non-empty
    # intersection). Catches all five geometries — start/midpoint/end inside
    # scene, predicted ⊇ scene, scene ⊇ predicted — with one symmetric test.
    # An anchor-point check would miss "predicted swallows the scene" (wide
    # over-prediction across a short scene): the agent found the right region
    # but no anchor lands inside it. IoU + endpoint exp-decay still penalize
    # the slop, so a wide hit can't game its way past those components.
    scene_hit = any(
        p_end >= s and p_start <= e
        for s, e in target_scene_bounds
    )

    start_delta = abs(p_start - float(truth[0]))
    end_delta = abs(p_end - float(truth[1]))
    start_sub = endpoint_max_each * math.exp(-endpoint_decay * start_delta)
    end_sub = endpoint_max_each * math.exp(-endpoint_decay * end_delta)
    iou_sub = iou_max * iou_val

    rationale = (
        f"scene_hit={scene_hit}; "
        f"Δstart={start_delta:.3f}s (sub={start_sub:.2f}/{endpoint_max_each:.2f}), "
        f"Δend={end_delta:.3f}s (sub={end_sub:.2f}/{endpoint_max_each:.2f}); "
        f"IoU={iou_val:.4f} (sub={iou_sub:.2f}/{iou_max:.2f})"
    )
    return _result([p_start, p_end], iou_val, scene_hit,
                   start_delta, end_delta, start_sub, end_sub, iou_sub, rationale)


# ---------------------------------------------------------------------------
# Composite: 35-pt window scorer (no VLM description). The four components
# are scene_hit (5/35), start (10/35), end (10/35), IoU (10/35).
# ---------------------------------------------------------------------------

# Backward-compat shim: every variant's `tasks/v*/localize.py:_error_result`
# still imports `DESCRIPTION_FRACTION` to size the description sub-budget in
# its synthetic-error schema. We've removed the description scorer from the
# rubric, but keep the constant exported as 0.0 so those imports don't break.
# Future cleanup: the per-variant _error_result functions should be rewritten
# to drop the description block entirely.
DESCRIPTION_FRACTION = 0.0


def compose_localization(
    diagnosis_md: Path,
    truth: tuple[float, float],
    target_scene_bounds: list[tuple[float, float]],
    variant: str,
    max_points: int = 35,
) -> dict[str, Any]:
    """Window-only localization score. The VLM description judge has been
    removed; every point of the 35-pt localization budget comes from
    deterministic scorer-computed quantities (scene_hit + start + end + IoU).

    Keeps a `description` block in the return dict (zeroed, status="removed")
    so consumers that read it don't crash. New consumers should ignore it.
    """
    window = score_with_scene(diagnosis_md, truth, target_scene_bounds, max_points)

    return {
        "score": float(window.get("score", 0.0)),
        "max": max_points,
        # Window subtotals — `window_sub` equals the localization total now.
        "window_sub": float(window.get("score", 0.0)),
        "window_max": max_points,
        # Description block kept for backward-compat report code; always 0.
        "description_sub": 0.0,
        "description_max": 0,
        # Pass-through window detail.
        "predicted_window_s": window.get("predicted_window_s"),
        "truth_window_s":     window.get("truth_window_s"),
        "iou":                window.get("iou"),
        "iou_sub":            window.get("iou_sub"),
        "iou_max":            window.get("iou_max"),
        "scene_hit":          window.get("scene_hit"),
        "scene_hit_sub":      window.get("scene_hit_sub"),
        "scene_hit_max":      window.get("scene_hit_max"),
        "start_delta_s":      window.get("start_delta_s"),
        "end_delta_s":        window.get("end_delta_s"),
        "start_sub":          window.get("start_sub"),
        "start_max":          window.get("start_max"),
        "end_sub":            window.get("end_sub"),
        "end_max":            window.get("end_max"),
        # Vestigial description block (always zero now).
        "description": {
            "score": 0.0, "max": 0, "status": "removed",
            "rationale": "VLM description judge removed from the rubric",
            "criteria": [],
        },
        "rationale": window.get("rationale", ""),
    }


# ---------------------------------------------------------------------------
# Multi-window variant — defects that produce two (or more) distinct regions.
# v03 scene_swap is the current user: gapped swaps create two disturbed
# stretches (group_b's new slot + group_a's new slot) separated by untouched
# middle material. Grading one outer span is too lenient; grading per-region
# asks the agent to localize each disturbance.
# ---------------------------------------------------------------------------


def _score_single_region(
    pred: tuple[float, float],
    truth: tuple[float, float],
    scene_bounds: list[tuple[float, float]],
    max_points: float,
    *,
    scene_weight: float,
    endpoint_weight: float,
    endpoint_decay: float,
    iou_weight: float,
) -> dict[str, Any]:
    """Per-region sub-result. Same 4-component scoring as `score_with_scene`."""
    scene_max = max_points * scene_weight
    endpoint_max_each = max_points * endpoint_weight
    iou_max = max_points * iou_weight

    p_start, p_end = pred
    t_start, t_end = truth
    iou_val = iou(pred, truth)
    # See score_with_scene: overlap-based test, not anchor-point test.
    scene_hit = any(
        p_end >= s and p_start <= e
        for s, e in scene_bounds
    )
    scene_sub = scene_max if scene_hit else 0.0

    start_delta = abs(p_start - t_start)
    end_delta = abs(p_end - t_end)
    start_sub = endpoint_max_each * math.exp(-endpoint_decay * start_delta)
    end_sub = endpoint_max_each * math.exp(-endpoint_decay * end_delta)
    iou_sub = iou_max * iou_val

    return {
        "score": float(scene_sub + start_sub + end_sub + iou_sub),
        "max": float(max_points),
        "predicted_window_s": [float(p_start), float(p_end)],
        "truth_window_s": [float(t_start), float(t_end)],
        "iou": float(iou_val),
        "iou_sub": float(iou_sub),
        "iou_max": float(iou_max),
        "scene_hit": bool(scene_hit),
        "scene_hit_sub": float(scene_sub),
        "scene_hit_max": float(scene_max),
        "start_delta_s": float(start_delta),
        "end_delta_s": float(end_delta),
        "start_sub": float(start_sub),
        "start_max": float(endpoint_max_each),
        "end_sub": float(end_sub),
        "end_max": float(endpoint_max_each),
        "rationale": (
            f"scene_hit={scene_hit}; "
            f"Δstart={start_delta:.3f}s (sub={start_sub:.2f}/{endpoint_max_each:.2f}), "
            f"Δend={end_delta:.3f}s (sub={end_sub:.2f}/{endpoint_max_each:.2f}); "
            f"IoU={iou_val:.4f} (sub={iou_sub:.2f}/{iou_max:.2f})"
        ),
    }


def _empty_region_result(
    truth: tuple[float, float],
    max_points: float,
    *,
    scene_weight: float,
    endpoint_weight: float,
    iou_weight: float,
    rationale: str,
) -> dict[str, Any]:
    return {
        "score": 0.0, "max": float(max_points),
        "predicted_window_s": None,
        "truth_window_s": [float(truth[0]), float(truth[1])],
        "iou": 0.0,
        "iou_sub": 0.0, "iou_max": float(max_points * iou_weight),
        "scene_hit": False,
        "scene_hit_sub": 0.0, "scene_hit_max": float(max_points * scene_weight),
        "start_delta_s": None, "end_delta_s": None,
        "start_sub": 0.0, "start_max": float(max_points * endpoint_weight),
        "end_sub": 0.0, "end_max": float(max_points * endpoint_weight),
        "rationale": rationale,
    }


def score_with_scene_multi(
    diagnosis_md: Path,
    truths_and_scenes: list[tuple[tuple[float, float], list[tuple[float, float]]]],
    max_points: int,
    *,
    scene_weight: float = SCENE_WEIGHT,
    endpoint_weight: float = ENDPOINT_WEIGHT,
    endpoint_decay: float = ENDPOINT_DECAY,
    iou_weight: float = IOU_WEIGHT,
) -> dict[str, Any]:
    """Multi-region window scorer. Each region scored independently.

    `truths_and_scenes` = list of (truth_window, target_scene_bounds) per
    distinct defect region. Budget splits equally across regions; each region
    picks whichever predicted window scores highest against it (predictions
    are allowed to match more than one region — an agent that collapses two
    contiguous regions into one window still earns partial credit on both).

    Top-level `predicted_window_s` / `truth_window_s` / `iou` /
    `start_delta_s` / `end_delta_s` report the **outer-span** view across all
    regions, for backward compatibility with the single-window report UI.
    Per-region detail lives in `windows: [...]`.
    """
    if not truths_and_scenes:
        raise ValueError("truths_and_scenes must be non-empty")

    n_regions = len(truths_and_scenes)
    per_region_max = max_points / n_regions

    # Read more predictions than truths so over-predicting agents (common in
    # combined-defect runs where models emit one ## Diagnosis block per
    # suspected defect, including false positives) aren't truncated. Best-IoU
    # matching below naturally ignores extras — a hallucinated block is fine
    # as long as the correct one is still in the parsed list. v03's numbered
    # `window_N_*` schema is unaffected (path 1 in parse_diagnosis_windows
    # exits at the right N regardless of this cap).
    predicted = parse_diagnosis_windows(diagnosis_md, max_windows=max(2 * n_regions, 12))
    valid_preds = [p for p in predicted if p is not None]

    region_results: list[dict[str, Any]] = []
    for (truth, scene_bounds) in truths_and_scenes:
        if not valid_preds:
            region_results.append(_empty_region_result(
                truth, per_region_max,
                scene_weight=scene_weight,
                endpoint_weight=endpoint_weight,
                iou_weight=iou_weight,
                rationale="no parseable predicted window",
            ))
            continue
        best: dict[str, Any] | None = None
        for p in valid_preds:
            r = _score_single_region(
                p, truth, scene_bounds, per_region_max,
                scene_weight=scene_weight,
                endpoint_weight=endpoint_weight,
                endpoint_decay=endpoint_decay,
                iou_weight=iou_weight,
            )
            if best is None or r["score"] > best["score"]:
                best = r
        region_results.append(best)

    truth_outer = (
        min(t[0] for t, _ in truths_and_scenes),
        max(t[1] for t, _ in truths_and_scenes),
    )
    if valid_preds:
        pred_outer: tuple[float, float] | None = (
            min(p[0] for p in valid_preds),
            max(p[1] for p in valid_preds),
        )
        iou_outer = iou(pred_outer, truth_outer)
        start_delta_outer: float | None = abs(pred_outer[0] - truth_outer[0])
        end_delta_outer: float | None = abs(pred_outer[1] - truth_outer[1])
    else:
        pred_outer = None
        iou_outer = 0.0
        start_delta_outer = None
        end_delta_outer = None

    per_rationale = "; ".join(
        f"w{i+1}: {r['rationale']}" for i, r in enumerate(region_results)
    )
    return {
        "score":             float(sum(r["score"] for r in region_results)),
        "max":               int(max_points),
        "predicted_window_s": list(pred_outer) if pred_outer else None,
        "truth_window_s":    [float(truth_outer[0]), float(truth_outer[1])],
        "iou":               float(iou_outer),
        "iou_sub":           float(sum(r["iou_sub"] for r in region_results)),
        "iou_max":           float(sum(r["iou_max"] for r in region_results)),
        "scene_hit":         bool(any(r["scene_hit"] for r in region_results)),
        "scene_hit_sub":     float(sum(r["scene_hit_sub"] for r in region_results)),
        "scene_hit_max":     float(sum(r["scene_hit_max"] for r in region_results)),
        "start_delta_s":     float(start_delta_outer) if start_delta_outer is not None else None,
        "end_delta_s":       float(end_delta_outer) if end_delta_outer is not None else None,
        "start_sub":         float(sum(r["start_sub"] for r in region_results)),
        "start_max":         float(sum(r["start_max"] for r in region_results)),
        "end_sub":           float(sum(r["end_sub"] for r in region_results)),
        "end_max":           float(sum(r["end_max"] for r in region_results)),
        "windows":           region_results,
        "n_predicted":       len(valid_preds),
        "n_truth":           n_regions,
        "rationale":         f"n_predicted={len(valid_preds)} n_truth={n_regions} | {per_rationale}",
    }


def compose_localization_multi(
    diagnosis_md: Path,
    truths_and_scenes: list[tuple[tuple[float, float], list[tuple[float, float]]]],
    variant: str,
    max_points: int = 35,
) -> dict[str, Any]:
    """Multi-region version of `compose_localization`. No VLM description.

    Per-region budget is `max_points / N` (already in `score_with_scene_multi`).
    Each region's start/end weights cap at `endpoint_weight × per_region_max`,
    so when N=2 the per-window endpoint budget halves to 5/35; sum across
    regions still totals `endpoint_weight × max_points`.
    """
    window = score_with_scene_multi(diagnosis_md, truths_and_scenes, max_points)

    return {
        "score": float(window.get("score", 0.0)),
        "max": max_points,
        "window_sub": float(window.get("score", 0.0)),
        "window_max": max_points,
        "description_sub": 0.0,
        "description_max": 0,
        "predicted_window_s": window.get("predicted_window_s"),
        "truth_window_s":     window.get("truth_window_s"),
        "iou":                window.get("iou"),
        "iou_sub":            window.get("iou_sub"),
        "iou_max":            window.get("iou_max"),
        "scene_hit":          window.get("scene_hit"),
        "scene_hit_sub":      window.get("scene_hit_sub"),
        "scene_hit_max":      window.get("scene_hit_max"),
        "start_delta_s":      window.get("start_delta_s"),
        "end_delta_s":        window.get("end_delta_s"),
        "start_sub":          window.get("start_sub"),
        "start_max":          window.get("start_max"),
        "end_sub":            window.get("end_sub"),
        "end_max":            window.get("end_max"),
        "windows":            window.get("windows"),
        "n_predicted":        window.get("n_predicted"),
        "n_truth":            window.get("n_truth"),
        "description": {
            "score": 0.0, "max": 0, "status": "removed",
            "rationale": "VLM description judge removed from the rubric",
            "criteria": [],
        },
        "rationale":          window.get("rationale", ""),
    }
