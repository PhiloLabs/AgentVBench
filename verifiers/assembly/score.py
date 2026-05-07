"""Verifier for AgentVBench_100 / assembly — video assembly.

Given:
  - the agent's `solution.json` (a manifest with a `segments` list, each entry
    having a `source` field — the clip number the agent picked for that slot), and
  - the task's golden `correct_assembly_in_slot_order` (an inline column on the
    dataset),

we compute:

  assembly_score        = n_correct / n_slots                in [0, 1]
  strict_match          = 1.0 iff every slot picked correctly
  final_score           = assembly_score
  adjusted_final_score  = max(0, (final_score - 1/3) * 1.5)
                          (per SCORE_ADJUSTMENTS.md — chance-floor rescale,
                           putting random guessing at 0 and perfect at 1)

If the solution is missing or malformed, every score is 0.

USAGE
-----
As a library:

    from verifiers.assembly import score_task
    result = score_task(
        solution_json=Path("solution.json"),
        correct_assembly=["3.mp4", "7.mp4", "5.mp4", "11.mp4"],
    )
    print(result.adjusted_final_score)

CLI:

    python -m verifiers.assembly.score \
        --solution-json path/to/solution.json \
        --task-id 1 \
        --dataset Anonymous47621123/AgentVBench_100

The CLI reads `correct_assembly_in_slot_order` from the dataset for the given
`task_id`.

Pure stdlib + (optional) `datasets`/`pyarrow` — no LLM calls, no S3.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

CHANCE_FLOOR = 1.0 / 3.0


# ----------------------------------------------------------------------------- #
# Helpers                                                                       #
# ----------------------------------------------------------------------------- #


def _normalize_pick(src: object) -> str:
    """Normalize a manifest segment.source to '<N>.mp4' form.

    The agent may write ``"3"``, ``"3.mp4"``, or ``3``; the answer key uses
    the .mp4 suffix consistently.
    """
    s = str(src).strip()
    if not s.endswith(".mp4"):
        s = f"{s}.mp4"
    return s


def adjust_chance_floor(final_score: float, chance: float = CHANCE_FLOOR) -> float:
    """Rescale a score with a chance floor `chance` to [0, 1].

    Random picking gets 1/3 (since slots typically have 3+ candidates). We
    rescale so chance maps to 0 and perfect maps to 1, floored at 0:

        adjusted = max(0, (raw - chance) / (1 - chance))
    """
    span = 1.0 - chance
    if span <= 0:
        return 0.0
    return max(0.0, (final_score - chance) / span)


# ----------------------------------------------------------------------------- #
# Top-level scorer                                                              #
# ----------------------------------------------------------------------------- #


@dataclass
class AdjustedResult:
    task_id: int | str
    n_slots: int
    n_correct: int
    assembly_score: float
    strict_match: float
    final_score: float
    adjusted_final_score: float
    picks: list[str]
    correct: list[str]
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _zero_result(
    task_id, correct: list[str], picks: list[str], reason: str,
) -> AdjustedResult:
    return AdjustedResult(
        task_id=task_id,
        n_slots=len(correct),
        n_correct=0,
        assembly_score=0.0,
        strict_match=0.0,
        final_score=0.0,
        adjusted_final_score=0.0,
        picks=picks,
        correct=correct,
        error=reason,
    )


def score_task(
    solution_json: Path | dict,
    correct_assembly: Iterable[str],
    task_id: int | str = "",
) -> AdjustedResult:
    """Score one assembly instance.

    Parameters
    ----------
    solution_json : Path | dict
        Path to the agent's `solution.json`, or its already-parsed contents.
    correct_assembly : Iterable[str]
        The golden picks per slot (from the dataset's
        ``correct_assembly_in_slot_order``).
    task_id : int | str, optional
        Carried through for traceability.
    """
    correct = [_normalize_pick(x) for x in correct_assembly]
    n = len(correct)

    if isinstance(solution_json, (str, Path)):
        path = Path(solution_json)
        if not path.exists():
            return _zero_result(task_id, correct, [], f"solution.json not found at {path}")
        try:
            sol = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            return _zero_result(task_id, correct, [], f"solution.json invalid JSON: {e}")
    else:
        sol = solution_json

    segments = sol.get("segments")
    if not isinstance(segments, list):
        return _zero_result(task_id, correct, [],
                            "solution.json: segments is not a list")

    if len(segments) != n:
        picks = [_normalize_pick(seg.get("source", "")) for seg in segments]
        return _zero_result(
            task_id, correct, picks,
            f"malformed: expected {n} segments, got {len(segments)}",
        )

    picks = [_normalize_pick(seg.get("source", "")) for seg in segments]
    n_correct = sum(1 for p, c in zip(picks, correct) if p == c)
    score = n_correct / n if n else 0.0
    strict = 1.0 if picks == correct else 0.0
    adjusted = adjust_chance_floor(score, CHANCE_FLOOR)

    return AdjustedResult(
        task_id=task_id,
        n_slots=n,
        n_correct=n_correct,
        assembly_score=score,
        strict_match=strict,
        final_score=score,
        adjusted_final_score=adjusted,
        picks=picks,
        correct=correct,
    )


# ----------------------------------------------------------------------------- #
# CLI                                                                           #
# ----------------------------------------------------------------------------- #


def _load_correct_assembly_from_dataset(dataset: str, task_id: int | str) -> list[str]:
    """Look up `correct_assembly_in_slot_order` for `task_id` (assembly family)."""
    p = Path(dataset)
    if p.exists() and p.suffix == ".parquet":
        import pyarrow.parquet as pq
        tbl = pq.read_table(p, columns=["task_family", "task_id", "correct_assembly_in_slot_order"])
    elif p.is_dir():
        candidate = p / "data" / "train-00000-of-00001.parquet"
        if not candidate.exists():
            raise FileNotFoundError(f"no parquet at {candidate}")
        import pyarrow.parquet as pq
        tbl = pq.read_table(candidate, columns=["task_family", "task_id", "correct_assembly_in_slot_order"])
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, split="train")
        for row in ds:
            if row.get("task_family") == "assembly" and str(row["task_id"]) == str(task_id):
                return [str(x) for x in row["correct_assembly_in_slot_order"]]
        raise KeyError(f"assembly/task_id={task_id} not found in {dataset}")

    families = tbl.column("task_family").to_pylist()
    ids = [str(x) for x in tbl.column("task_id").to_pylist()]
    arr = tbl.column("correct_assembly_in_slot_order").to_pylist()
    for i, (fam, tid) in enumerate(zip(families, ids)):
        if fam == "assembly" and tid == str(task_id):
            return [str(x) for x in arr[i]]
    raise KeyError(f"assembly/task_id={task_id} not in dataset")


def cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="avb-score-assembly",
        description="Score one AgentVBench_100/assembly video-assembly solution.",
    )
    p.add_argument("--solution-json", required=True, type=Path,
                   help="path to the agent's solution.json")
    p.add_argument("--task-id", required=True,
                   help="task_id within assembly (one of 1, 2, 4, 5, 6, 7, 9, "
                        "10, 11, 12, 13, 14, 15, 18, 19, 20, 22, 24)")
    p.add_argument("--dataset", default="Anonymous47621123/AgentVBench_100",
                   help="HF dataset id (default: Anonymous47621123/AgentVBench_100), "
                        "or a local parquet/dir path")
    p.add_argument("--correct-assembly", default=None,
                   help="bypass dataset lookup; comma-separated golden picks, "
                        "e.g. '3.mp4,7.mp4,5.mp4,11.mp4'")
    args = p.parse_args(argv)

    if args.correct_assembly:
        correct = [s.strip() for s in args.correct_assembly.split(",")]
    else:
        correct = _load_correct_assembly_from_dataset(args.dataset, args.task_id)

    res = score_task(
        solution_json=args.solution_json,
        correct_assembly=correct,
        task_id=args.task_id,
    )
    print(res.to_json())
    return 0 if not res.error else 1


if __name__ == "__main__":
    sys.exit(cli())
