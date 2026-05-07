"""Verifier for AgentVBench_100 / sequencing — video ordering.

Given:
  - the agent's `solution.json` (a manifest with a `segments` list, each entry
    having a `source` field — the clip number the agent thinks belongs in that
    slot), and
  - the task's golden `correct_order` (an inline column on the dataset),

we compute three component scores and two final scores:

  nd_score   = 1 - normalized_footrule(predicted, correct)        in [0, 1]
  lis_score  = longest_correctly_ordered_subseq_ratio             in [0, 1]
  adj_score  = adjacent_transition_hit_rate                       in [0, 1]

  final_score          = 0.4 * nd_score + 0.3 * lis_score + 0.3 * adj_score
  adjusted_final_score = nd_score * lis_score * adj_score    (per
                         SCORE_ADJUSTMENTS.md — multiplicative composite,
                         which is harsher than the weighted sum and zeros out
                         when any component fails).

If the solution is missing or malformed (slot-set mismatch, missing manifest,
…), every score is 0.

USAGE
-----
As a library:

    from verifiers.sequencing import score_task
    result = score_task(
        solution_json=Path("solution.json"),    # the agent's manifest
        correct_order=["6","5","1","2","3"],    # from the dataset's correct_order
    )
    print(result.adjusted_final_score)

CLI:

    python -m verifiers.sequencing.score \
        --solution-json path/to/solution.json \
        --task-id 2 \
        --dataset Anonymous47621123/AgentVBench_100  # or a local parquet path

The CLI reads `correct_order` from the dataset for the given `task_id`.

Pure stdlib + (optional) `datasets`/`pyarrow` — no LLM calls, no S3.
"""
from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


# ----------------------------------------------------------------------------- #
# Metric implementations                                                        #
# ----------------------------------------------------------------------------- #


def metric_nd(pred: Sequence[str], correct: Sequence[str]) -> float:
    """Normalised footrule distance, in [0, 1]; lower is better.

    Sum of |position(c, pred) - position(c, correct)| over all clips, divided
    by the maximum possible such sum (n^2 // 2). Returns 0 for empty input.
    """
    pred_pos = {c: i for i, c in enumerate(pred)}
    correct_pos = {c: i for i, c in enumerate(correct)}
    n = len(correct)
    total = sum(abs(pred_pos[c] - correct_pos[c]) for c in correct_pos)
    max_total = (n * n) // 2
    return total / max_total if max_total else 0.0


def metric_lis(pred: Sequence[str], correct: Sequence[str]) -> float:
    """Longest correctly-ordered subsequence ratio, in [0, 1]; higher is better.

    Patience sorting on the rank array — `len(LIS) / len(pred)`.
    """
    rank = {c: i for i, c in enumerate(correct)}
    seq = [rank[c] for c in pred if c in rank]
    if not seq:
        return 0.0
    tails: list[int] = []
    for x in seq:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails) / len(pred)


def metric_adj(pred: Sequence[str], correct: Sequence[str]) -> float:
    """Fraction of true adjacent transitions caught, in [0, 1]; higher is better.

    For each adjacent pair (a, b) in `correct`, score 1 iff b immediately
    follows a in `pred`. With `n` correct, this is fraction of (n-1) pairs.
    """
    if len(correct) <= 1:
        return 1.0
    pred_pos = {c: i for i, c in enumerate(pred)}
    caught = 0
    for i in range(len(correct) - 1):
        a, b = correct[i], correct[i + 1]
        if pred_pos.get(b, -2) - pred_pos.get(a, -1) == 1:
            caught += 1
    return caught / (len(correct) - 1)


# ----------------------------------------------------------------------------- #
# Top-level scorer                                                              #
# ----------------------------------------------------------------------------- #


@dataclass
class AdjustedResult:
    """Per-task scoring result.

    `final_score` is the original weighted-sum composite kept around for
    backwards compatibility. `adjusted_final_score` is what the paper
    reports — the multiplicative composite per SCORE_ADJUSTMENTS.md.
    """

    task_id: int | str
    n_slots: int
    nd_score: float
    lis_score: float
    adj_score: float
    strict_match: float
    final_score: float
    adjusted_final_score: float
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _zero_result(task_id, n_slots: int, reason: str) -> AdjustedResult:
    return AdjustedResult(
        task_id=task_id,
        n_slots=n_slots,
        nd_score=0.0,
        lis_score=0.0,
        adj_score=0.0,
        strict_match=0.0,
        final_score=0.0,
        adjusted_final_score=0.0,
        error=reason,
    )


def score_task(
    solution_json: Path | dict,
    correct_order: Iterable[str],
    task_id: int | str = "",
) -> AdjustedResult:
    """Score one sequencing instance.

    Parameters
    ----------
    solution_json : Path | dict
        Either a path to the agent's `solution.json`, or its already-parsed
        contents. Expected shape: ``{"segments": [{"source": "<n>", ...}, ...]}``.
    correct_order : Iterable[str]
        The golden ordering for this task (from the dataset's `correct_order`).
        Each entry is a clip identifier, typically a string number like ``"6"``.
    task_id : int | str, optional
        Carried through to the result for traceability.

    Returns
    -------
    AdjustedResult
    """
    correct = [str(x) for x in correct_order]
    n = len(correct)

    if isinstance(solution_json, (str, Path)):
        path = Path(solution_json)
        if not path.exists():
            return _zero_result(task_id, n, f"solution.json not found at {path}")
        try:
            sol = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            return _zero_result(task_id, n, f"solution.json invalid JSON: {e}")
    else:
        sol = solution_json

    segments = sol.get("segments")
    if not isinstance(segments, list):
        return _zero_result(task_id, n, "solution.json: segments is not a list")

    pred = [str(seg.get("source", "")) for seg in segments]
    if sorted(pred) != sorted(correct):
        return _zero_result(
            task_id, n,
            f"slot set mismatch: expected {sorted(correct)}, got {sorted(pred)}",
        )

    nd = metric_nd(pred, correct)
    lis = metric_lis(pred, correct)
    adj = metric_adj(pred, correct)
    nd_score = 1.0 - nd
    strict = 1.0 if pred == correct else 0.0

    final = 0.4 * nd_score + 0.3 * lis + 0.3 * adj
    # SCORE_ADJUSTMENTS.md: multiplicative composite. If any component is 0
    # (worst-case slot set wrong, or a single inverted block), the product is 0.
    adjusted = nd_score * lis * adj

    return AdjustedResult(
        task_id=task_id,
        n_slots=n,
        nd_score=nd_score,
        lis_score=lis,
        adj_score=adj,
        strict_match=strict,
        final_score=final,
        adjusted_final_score=adjusted,
    )


# ----------------------------------------------------------------------------- #
# CLI                                                                           #
# ----------------------------------------------------------------------------- #


def _load_correct_order_from_dataset(dataset: str, task_id: int | str) -> list[str]:
    """Look up `correct_order` for a given task_id from a parquet path or HF id."""
    p = Path(dataset)
    if p.exists() and p.suffix == ".parquet":
        import pyarrow.parquet as pq
        tbl = pq.read_table(p, columns=["task_id", "correct_order"])
    elif p.is_dir():
        # repo-style: <dir>/data/train-00000-of-00001.parquet
        candidate = p / "data" / "train-00000-of-00001.parquet"
        if not candidate.exists():
            raise FileNotFoundError(f"no parquet at {candidate}")
        import pyarrow.parquet as pq
        tbl = pq.read_table(candidate, columns=["task_id", "correct_order"])
    else:
        # HF hub id — load via `datasets` and the relevant config
        from datasets import load_dataset
        ds = load_dataset(dataset, "sequencing", split="train")
        tbl = None
        for row in ds:
            if str(row["task_id"]) == str(task_id):
                return [str(x) for x in row["correct_order"]]
        raise KeyError(f"task_id={task_id} not found in {dataset}")

    ids = [str(x) for x in tbl.column("task_id").to_pylist()]
    correct = tbl.column("correct_order").to_pylist()
    try:
        i = ids.index(str(task_id))
    except ValueError as exc:
        raise KeyError(f"task_id={task_id} not in dataset") from exc
    return [str(x) for x in correct[i]]


def cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="avb-score-sequencing",
        description="Score one AgentVBench_100/sequencing video-ordering solution.",
    )
    p.add_argument("--solution-json", required=True, type=Path,
                   help="path to the agent's solution.json")
    p.add_argument("--task-id", required=True,
                   help="task_id within sequencing (1..28)")
    p.add_argument("--dataset", default="Anonymous47621123/AgentVBench_100",
                   help="HF dataset id (default: Anonymous47621123/AgentVBench_100), "
                        "or a local parquet/dir path")
    p.add_argument("--correct-order", default=None,
                   help="bypass dataset lookup; comma-separated golden order, "
                        "e.g. '6,5,1,2,3,9,4,8,7,10'")
    args = p.parse_args(argv)

    if args.correct_order:
        correct = [s.strip() for s in args.correct_order.split(",")]
    else:
        correct = _load_correct_order_from_dataset(args.dataset, args.task_id)

    res = score_task(
        solution_json=args.solution_json,
        correct_order=correct,
        task_id=args.task_id,
    )
    print(res.to_json())
    return 0 if not res.error else 1


if __name__ == "__main__":
    sys.exit(cli())
