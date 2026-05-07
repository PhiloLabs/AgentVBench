"""Verifier for AgentVBench_100 / recap — creative recap.

The dataset row carries an inline ``rubric_items`` column: a list of dicts,
each item being one expert-authored check. There are three dispatch types:

  python    — a deterministic check on the agent's output video. Implemented
              in ``kinds.py`` (ffprobe duration / scenedetect cut count /
              audio loudness / OCR title-card / etc.). Returns (pass, evidence).
  llm       — a yes/no question answered by Gemini against the agent's
              output. Implemented in ``llm_judges.py`` — uploads the video
              to Gemini File API once per task, reuses the file URI across
              every llm item.
  compound  — ordered Python stages whose evidence is substituted into a
              final LLM stage's prompt placeholders. Short-circuits to a
              fail if any python stage fails.

For each item we record pass/fail × the item's signed weight (positive =
"earn N points if passed", negative = "lose N points if failed"; deduction
items model catastrophic violations like "the recap is wrong length"). The
final score is::

    final_score = sum(passed * weight) / sum(positive_weights)
                  clamped to [0, 1]

USAGE
-----
As a library:

    from verifiers.recap import score_task
    result = score_task(
        final_mp4=Path("agent_output/final.mp4"),
        rubric_items=row["rubric_items"],   # from the dataset
        task_id="cutbench-animated_out",
        gemini_api_key=os.environ["GEMINI_API_KEY"],
    )
    print(result.final_score)

CLI:

    python -m verifiers.recap.score \
        --final-mp4 path/to/final.mp4 \
        --task-id cutbench-animated_out \
        --dataset Anonymous47621123/AgentVBench_100

The CLI loads ``rubric_items`` from the dataset for the given ``task_id`` and
reads the Gemini API key from the ``GEMINI_API_KEY`` environment variable.

REQUIREMENTS
------------
- ffmpeg / ffprobe on PATH (system dependency)
- google-genai  (pip extras: agenticvbench[recap])
- a Google AI Studio API key in ``GEMINI_API_KEY``
- internet (for the Gemini File API)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .kinds import run_kind
from .llm_judges import (
    FileState,
    init_file_state,
    run_llm_items_async,
    _references_source,
    _call_gemini_sync,
)


# ----------------------------------------------------------------------------- #
# Item dispatchers                                                              #
# ----------------------------------------------------------------------------- #


def _clean_params(params: dict | None) -> dict:
    """Strip None-valued keys (the HF dataset's union schema fills every
    column for every row, leaving Python kinds with `eq=None` etc. on rows
    where they don't apply)."""
    if not params:
        return {}
    return {k: v for k, v in params.items() if v is not None}


def _run_python_item(item: dict, video_path: str) -> tuple[bool, str]:
    return run_kind(item.get("kind"), video_path, _clean_params(item.get("params")))


def _run_compound_item(
    item: dict, video_path: str, file_state: FileState, model: str,
) -> tuple[bool, str]:
    """Compound = ordered Python stages whose evidence is substituted into the
    LLM stage's `{var}` placeholders, then a single Gemini call decides
    pass/fail. If any Python stage fails, the compound short-circuits."""
    stages = item.get("stages", [])
    saved: dict[str, str] = {}
    py_evidence: list[str] = []

    for s in stages:
        if s.get("dispatch") == "python":
            kind = s.get("kind")
            ok, ev = run_kind(kind, video_path, _clean_params(s.get("params")))
            var = s.get("save_as") or kind
            saved[var] = ev
            py_evidence.append(f"{kind}: {ev}")
            if not ok:
                return False, f"python stage '{kind}' failed: {ev}"
        elif s.get("dispatch") == "llm":
            prompt = s.get("prompt", "")
            try:
                prompt = prompt.format(**saved)
            except (KeyError, IndexError):
                pass
            references_source = _references_source(prompt) or _references_source(
                item.get("criterion", "")
            )
            ok, reason = _call_gemini_sync(
                file_state, model, prompt, references_source,
            )
            return ok, f"{'; '.join(py_evidence)} → llm: {reason}"

    return True, "; ".join(py_evidence) or "no stages"


# ----------------------------------------------------------------------------- #
# Top-level scorer                                                              #
# ----------------------------------------------------------------------------- #


@dataclass
class ItemResult:
    item_id: str
    dispatch: str
    weight: float
    pillar: int
    passed: bool
    score: float
    reason: str


@dataclass
class CutbenchResult:
    task_id: str
    n_items: int
    items: list[ItemResult] = field(default_factory=list)
    pillar_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    final_score: float = 0.0
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


_PILLAR_NAMES = {0: "format", 1: "visual", 2: "narrative", 3: "polish"}


def score_task(
    final_mp4: Path | str,
    rubric_items: Sequence[dict],
    task_id: str = "",
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-3-flash-preview",
    max_concurrent_llm: int = 16,
) -> CutbenchResult:
    """Score one recap instance.

    Parameters
    ----------
    final_mp4 : Path
        Path to the agent's output video.
    rubric_items : Sequence[dict]
        The dataset row's ``rubric_items`` (28 items per task on average).
    task_id : str, optional
        Carried through for traceability.
    gemini_api_key : str, optional
        Google AI Studio API key. Defaults to ``$GEMINI_API_KEY``.
    gemini_model : str, optional
        Defaults to ``gemini-3-flash-preview``.
    max_concurrent_llm : int, optional
        Concurrency cap on the Gemini File API. Defaults to 16.
    """
    final_p = Path(final_mp4)
    items = list(rubric_items or [])

    if not final_p.exists():
        return CutbenchResult(
            task_id=task_id, n_items=len(items),
            error=f"final.mp4 not found at {final_p}",
        )
    if not items:
        return CutbenchResult(
            task_id=task_id, n_items=0,
            error="rubric_items is empty",
        )
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return CutbenchResult(
            task_id=task_id, n_items=len(items),
            error="GEMINI_API_KEY not provided (set --gemini-api-key or $GEMINI_API_KEY)",
        )

    # Strip openrouter/google/ prefix if a user passes it via --model
    model = gemini_model
    for prefix in ("openrouter/google/", "google/", "gemini/"):
        if model.startswith(prefix):
            model = model[len(prefix):]
            break

    # Upload the agent's video to Gemini File API once. Source video is
    # NOT uploaded — most "source-referencing" rubric items already encode
    # source content as text in the prompt (style anchors, motifs, fabrication
    # facts), and the LLM judges those against the output alone.
    file_state = init_file_state(
        output_video_path=str(final_p),
        source_video_path=None,
        api_key=api_key,
        needs_source=False,
    )

    # Bucketize by dispatch type
    py_items = [it for it in items if it.get("dispatch") == "python"]
    llm_items = [it for it in items if it.get("dispatch") == "llm"]
    compound_items = [it for it in items if it.get("dispatch") == "compound"]

    results: dict[str, tuple[bool, str]] = {}

    # Python items run synchronously in-process — fast, no LLM calls
    for it in py_items:
        results[it["id"]] = _run_python_item(it, str(final_p))

    # LLM items run in parallel via asyncio
    if llm_items:
        loop = asyncio.new_event_loop()
        try:
            llm_results = loop.run_until_complete(
                run_llm_items_async(
                    llm_items, file_state, model=model,
                    max_concurrent=max_concurrent_llm,
                )
            )
        finally:
            loop.close()
        results.update(llm_results)

    # Compound items: serialize (each calls Gemini at most once)
    for it in compound_items:
        results[it["id"]] = _run_compound_item(it, str(final_p), file_state, model)

    # Aggregate
    item_results: list[ItemResult] = []
    pillar_totals: dict[int, dict[str, float]] = {}

    earned = 0.0
    total_positive_weight = 0.0
    deduction = 0.0

    for it in items:
        item_id = it["id"]
        weight = float(it.get("weight", 1))
        pillar = int(it.get("pillar", 0))
        passed, reason = results.get(item_id, (False, "no result"))
        score = 1.0 if passed else 0.0

        item_results.append(ItemResult(
            item_id=item_id,
            dispatch=str(it.get("dispatch", "?")),
            weight=weight,
            pillar=pillar,
            passed=bool(passed),
            score=score,
            reason=reason[:300],
        ))

        if weight >= 0:
            total_positive_weight += weight
            if passed:
                earned += weight
            pt = pillar_totals.setdefault(pillar, {"weight": 0.0, "earned": 0.0})
            pt["weight"] += weight
            pt["earned"] += score * weight
        else:
            # Negative-weight = penalty: subtracts points iff the rubric fails.
            if not passed:
                deduction += -weight  # weight is negative, so this adds a positive deduction

    final = ((earned - deduction) / total_positive_weight) if total_positive_weight > 0 else 0.0
    final = max(0.0, min(1.0, final))

    pillar_breakdown = {}
    for pillar, t in sorted(pillar_totals.items()):
        if t["weight"] <= 0:
            continue
        pillar_breakdown[_PILLAR_NAMES.get(pillar, str(pillar))] = {
            "earned": t["earned"],
            "max": t["weight"],
            "score": t["earned"] / t["weight"],
        }

    return CutbenchResult(
        task_id=task_id,
        n_items=len(items),
        items=item_results,
        pillar_breakdown=pillar_breakdown,
        final_score=final,
    )


# ----------------------------------------------------------------------------- #
# CLI                                                                           #
# ----------------------------------------------------------------------------- #


def _load_rubric_from_dataset(dataset: str, task_id: str) -> list[dict]:
    """Look up `rubric_items` for `task_id` (recap family)."""
    p = Path(dataset)
    if p.exists() and p.suffix == ".parquet":
        import pyarrow.parquet as pq
        tbl = pq.read_table(p, columns=["task_family", "task_id", "rubric_items"])
    elif p.is_dir():
        candidate = p / "data" / "train-00000-of-00001.parquet"
        if not candidate.exists():
            raise FileNotFoundError(f"no parquet at {candidate}")
        import pyarrow.parquet as pq
        tbl = pq.read_table(candidate, columns=["task_family", "task_id", "rubric_items"])
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, split="train")
        for row in ds:
            if row.get("task_family") == "recap" and str(row["task_id"]) == str(task_id):
                return list(row["rubric_items"])
        raise KeyError(f"recap/task_id={task_id} not found in {dataset}")

    families = tbl.column("task_family").to_pylist()
    ids = [str(x) for x in tbl.column("task_id").to_pylist()]
    arr = tbl.column("rubric_items").to_pylist()
    for i, (fam, tid) in enumerate(zip(families, ids)):
        if fam == "recap" and tid == str(task_id):
            return list(arr[i])
    raise KeyError(f"recap/task_id={task_id} not in dataset")


def cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="avb-score-recap",
        description="Score one AgentVBench_100/recap solution.",
    )
    p.add_argument("--final-mp4", required=True, type=Path,
                   help="path to the agent's final.mp4")
    p.add_argument("--task-id", required=True,
                   help="task_id within recap, e.g. 'cutbench-animated_out' (the slug naming convention is preserved from the dataset)")
    p.add_argument("--dataset", default="Anonymous47621123/AgentVBench_100",
                   help="HF dataset id (default: Anonymous47621123/AgentVBench_100), "
                        "or a local parquet/dir path")
    p.add_argument("--gemini-model", default="gemini-3-flash-preview",
                   help="Gemini model to use for LLM-judge items")
    p.add_argument("--max-concurrent-llm", type=int, default=16,
                   help="parallel calls cap on Gemini File API")
    args = p.parse_args(argv)

    rubric_items = _load_rubric_from_dataset(args.dataset, args.task_id)

    res = score_task(
        final_mp4=args.final_mp4,
        rubric_items=rubric_items,
        task_id=args.task_id,
        gemini_model=args.gemini_model,
        max_concurrent_llm=args.max_concurrent_llm,
    )
    print(res.to_json())
    return 0 if not res.error else 1


if __name__ == "__main__":
    sys.exit(cli())
