"""Verifier for AgentVBench_100 / repair — broken-video repair.

Each cell of the benchmark gives the agent a `broken.mp4` containing one or
more defects (frozen scene, scene swap, color shift, audio noise, duplicate
segment, A/V desync; cell v7 stacks several). The agent must produce:

  - `fixed.mp4`  — the repaired video
  - `report.md`  — a per-defect localization with start/end timestamps

We score against the original `source.mp4` (verifier-only) and a per-cell
ground-truth profile (`ground_truth/<cell>/profile.json`). Three rubrics:

  format        ( 5 pt) — container/codec/fps/resolution/sample-rate match
  localization  (35 pt) — agent's reported window matches GT defect window
  edit          (60 pt) — SSIM (video) + xcorr (audio) on the affected region
                           plus optional spillover penalty outside the region

  final_score   = (format + localization + edit) / 100      in [0, 1]

The kit's three rubric implementations live verbatim under `lib/`:
  lib/rubrics/format.py     ffprobe + format_config.json
  lib/rubrics/_localize.py  per-cell description + window math
  lib/rubrics/edit.py       SSIM / dhash / xcorr signal-processing

Per-variant `localize.py` modules under `lib/tasks/<variant>/` adapt the
generic localization rubric to each defect type.

USAGE
-----
As a library:

    from verifiers.repair import score_task
    result = score_task(
        fixed_mp4=Path("agent_output/fixed.mp4"),
        report_md=Path("agent_output/report.md"),
        source_mp4=Path("path/to/v1_source.mp4"),
        task_id="bench-broken-cut-v1-s1",
    )
    print(result.final_score)

CLI:

    python -m verifiers.repair.score \
        --fixed-mp4 path/to/fixed.mp4 \
        --report-md path/to/report.md \
        --source-mp4 path/to/v1_source.mp4 \
        --task-id bench-broken-cut-v1-s1

REQUIREMENTS
------------
- ffmpeg / ffprobe on PATH (system dependency)
- numpy, opencv-python, scipy   (pip extras: agenticvbench[repair])
- the kit's `lib/` directory (already vendored alongside this file)

SOURCE VIDEO ACCESS
-------------------
The source.mp4 is hosted alongside the dataset at
``Anonymous47621123/AgentVBench_100`` under each repair cell (column
``verifier_reference_urls`` on the parquet — exactly one source per cell).
Reviewers can download it ahead of time; the scorer expects it as a local
path. To avoid leaking the answer to evaluated agents, the agent's runtime
sandbox should NOT mount the source — only `broken.mp4` is meant to be
visible at rollout time.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make the vendored kit's `lib/` directory importable so its
# `from rubrics import ...` and `import tasks.<variant>.localize`
# style imports work without packaging gymnastics.
_HERE = Path(__file__).resolve().parent
_LIB = _HERE / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

# Module-import-time check that the kit is present
if not (_LIB / "rubrics" / "__init__.py").exists():
    raise ImportError(
        f"repair kit missing — expected {_LIB / 'rubrics'} to exist. "
        "Did you copy lib/ into the repo?"
    )

_CELLS_JSON = _HERE / "cells.json"
_GROUND_TRUTH = _HERE / "ground_truth"


# ----------------------------------------------------------------------------- #
# Cell + profile resolution                                                     #
# ----------------------------------------------------------------------------- #


def _task_id_to_cell(task_id: str) -> str | None:
    """`bench-broken-cut-v1-s1` -> `v1/s1`. Returns None if unrecognized."""
    import re
    if task_id.startswith("verify-"):
        task_id = task_id[len("verify-"):]
    m = re.match(r"^bench-broken-cut-(v\d+)-(s\d+)$", task_id)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def _load_cell_meta(cell_id: str) -> dict | None:
    try:
        manifest = json.loads(_CELLS_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return manifest.get("cells", {}).get(cell_id)


def _edit_kwargs_for(variant: str) -> dict:
    """Per-variant scoring overrides from `lib/tasks/<variant>/task.json`."""
    task_json = _LIB / "tasks" / variant / "task.json"
    if not task_json.exists():
        return {}
    try:
        scoring = json.loads(task_json.read_text()).get("scoring") or {}
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(scoring, dict):
        return {}
    allowed = {
        "w_in", "w_out", "chroma_norm", "audio_max_lag_ms",
        "sample_fps", "chunk_duration_s", "hash_threshold",
        "lag_band_frames", "lag_penalty_per_frame", "lag_topk",
        "max_workers",
        "video_share", "audio_share",
        "video_w_in", "video_w_out", "audio_w_in", "audio_w_out",
    }
    return {k: v for k, v in scoring.items() if k in allowed}


def _score_localization(report_md: Path, profile: dict, variant: str) -> dict:
    """Dispatch to per-variant localize module."""
    try:
        mod = importlib.import_module(f"tasks.{variant}.localize")
        return mod.score_localization(report_md, profile)
    except (ImportError, AttributeError) as e:
        return {
            "score": 0.0, "max": 35,
            "error": f"import_failed: tasks.{variant}.localize: {e}",
        }


def _safe_float(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------- #
# Top-level scorer                                                              #
# ----------------------------------------------------------------------------- #


@dataclass
class RepairResult:
    """Per-cell scoring result. Subscores are normalized to [0, 1]; weights
    sum to 100. `final_score` is (format*5 + localization*35 + edit*60)/100."""

    task_id: str
    cell: str
    variant: str
    format_score: float
    format_max: float
    localization_score: float
    localization_max: float
    edit_score: float
    edit_max: float
    final_score: float
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def score_task(
    fixed_mp4: Path | str,
    report_md: Path | str,
    source_mp4: Path | str,
    task_id: str,
) -> RepairResult:
    """Score one broken-video repair instance.

    Parameters
    ----------
    fixed_mp4 : Path
        The agent's repaired video.
    report_md : Path
        The agent's localization report (markdown with `## Diagnosis` blocks).
    source_mp4 : Path
        The original (verifier-only) source video for this cell, fetched from
        the dataset's ``verifier_reference_urls``.
    task_id : str
        e.g. ``"bench-broken-cut-v1-s1"``.
    """
    fixed_p = Path(fixed_mp4)
    report_p = Path(report_md)
    source_p = Path(source_mp4)

    cell = _task_id_to_cell(task_id)
    if cell is None:
        return RepairResult(
            task_id=task_id, cell="", variant="",
            format_score=0.0, format_max=5.0,
            localization_score=0.0, localization_max=35.0,
            edit_score=0.0, edit_max=60.0,
            final_score=0.0,
            error=f"unrecognized task_id {task_id!r}",
        )
    cell_meta = _load_cell_meta(cell)
    if cell_meta is None:
        return RepairResult(
            task_id=task_id, cell=cell, variant="",
            format_score=0.0, format_max=5.0,
            localization_score=0.0, localization_max=35.0,
            edit_score=0.0, edit_max=60.0,
            final_score=0.0,
            error=f"no cell metadata for {cell}",
        )
    variant = cell_meta["variant"]
    profile_path = _GROUND_TRUTH / cell.replace("/", "/") / "profile.json"
    if not profile_path.exists():
        # cells.json carries the canonical relative path
        profile_path = _HERE / cell_meta["profile_path"]
    if not profile_path.exists():
        return RepairResult(
            task_id=task_id, cell=cell, variant=variant,
            format_score=0.0, format_max=5.0,
            localization_score=0.0, localization_max=35.0,
            edit_score=0.0, edit_max=60.0,
            final_score=0.0,
            error=f"missing GT profile: {profile_path}",
        )
    try:
        profile = json.loads(profile_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return RepairResult(
            task_id=task_id, cell=cell, variant=variant,
            format_score=0.0, format_max=5.0,
            localization_score=0.0, localization_max=35.0,
            edit_score=0.0, edit_max=60.0,
            final_score=0.0,
            error=f"bad GT profile: {e}",
        )

    # Lazy-import the kit (depends on numpy/cv2/scipy at import time)
    try:
        from rubrics import score_format, score_edit  # type: ignore[import-not-found]
    except ImportError as e:
        return RepairResult(
            task_id=task_id, cell=cell, variant=variant,
            format_score=0.0, format_max=5.0,
            localization_score=0.0, localization_max=35.0,
            edit_score=0.0, edit_max=60.0,
            final_score=0.0,
            error=f"kit import failed (try `pip install agenticvbench[repair]`): {e}",
        )

    detail: dict[str, Any] = {}

    # ---- format (5 pt) ---------------------------------------------------
    if fixed_p.exists():
        try:
            fmt = score_format(fixed_p, source_p)
        except Exception as e:
            fmt = {"score": 0.0, "max": 5, "error": f"{type(e).__name__}: {e}"}
    else:
        fmt = {"score": 0.0, "max": 5, "error": "missing fixed.mp4"}
    detail["format"] = fmt

    # ---- localization (35 pt) -------------------------------------------
    if report_p.exists():
        loc = _score_localization(report_p, profile, variant)
    else:
        loc = {"score": 0.0, "max": 35, "error": "missing report.md"}
    detail["localization"] = loc

    # ---- edit (60 pt) ---------------------------------------------------
    if fixed_p.exists() and source_p.exists():
        try:
            edit_kwargs = _edit_kwargs_for(variant)
            edit_kwargs.setdefault("max_workers", 2)
            edt = score_edit(fixed_p, source_p, profile, **edit_kwargs)
        except Exception as e:
            edt = {"score": 0.0, "max": 60, "error": f"{type(e).__name__}: {e}"}
    else:
        edt = {
            "score": 0.0, "max": 60,
            "error": (
                "missing fixed.mp4" if not fixed_p.exists() else "missing source.mp4"
            ),
        }
    detail["edit"] = edt

    fmt_max = _safe_float(fmt, "max", 5.0) or 5.0
    loc_max = _safe_float(loc, "max", 35.0) or 35.0
    edt_max = _safe_float(edt, "max", 60.0) or 60.0
    fmt_norm = _safe_float(fmt, "score") / fmt_max
    loc_norm = _safe_float(loc, "score") / loc_max
    edt_norm = _safe_float(edt, "score") / edt_max
    fmt_norm = max(0.0, min(1.0, fmt_norm))
    loc_norm = max(0.0, min(1.0, loc_norm))
    edt_norm = max(0.0, min(1.0, edt_norm))

    final = (fmt_norm * fmt_max + loc_norm * loc_max + edt_norm * edt_max) / (
        fmt_max + loc_max + edt_max
    )

    return RepairResult(
        task_id=task_id,
        cell=cell,
        variant=variant,
        format_score=fmt_norm,
        format_max=fmt_max,
        localization_score=loc_norm,
        localization_max=loc_max,
        edit_score=edt_norm,
        edit_max=edt_max,
        final_score=final,
        detail=detail,
    )


# ----------------------------------------------------------------------------- #
# CLI                                                                           #
# ----------------------------------------------------------------------------- #


def cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="avb-score-repair",
        description="Score one AgentVBench_100/repair broken-video repair solution.",
    )
    p.add_argument("--fixed-mp4", required=True, type=Path,
                   help="path to the agent's fixed.mp4")
    p.add_argument("--report-md", required=True, type=Path,
                   help="path to the agent's report.md (localization)")
    p.add_argument("--source-mp4", required=True, type=Path,
                   help="path to the verifier-only source.mp4 for this cell "
                        "(see verifier_reference_urls in the dataset)")
    p.add_argument("--task-id", required=True,
                   help="task_id, e.g. 'bench-broken-cut-v1-s1'")
    args = p.parse_args(argv)

    res = score_task(
        fixed_mp4=args.fixed_mp4,
        report_md=args.report_md,
        source_mp4=args.source_mp4,
        task_id=args.task_id,
    )
    print(res.to_json())
    return 0 if not res.error else 1


if __name__ == "__main__":
    sys.exit(cli())
