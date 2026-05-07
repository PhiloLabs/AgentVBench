# Adjusted final score — task_5 and task_7

The paper reports **`adjusted_final_score`** as the headline number for each
of these two tasks. Both verifiers compute it alongside the raw
`final_score` (left in for backwards compatibility); the paper's tables are
the adjusted form.

## task_7_3 — chance-floor rescale

The original `final_score` for task_7_3 is per-slot accuracy
(`n_correct / n_slots`), which has a chance floor at `1/3` (slots typically
have 3 candidates; random picking expectation = 1/3). We rescale so chance
maps to 0 and perfect maps to 1, floored at 0:

```
adjusted_final_score = max(0, (final_score - 1/3) / (1 - 1/3))
                     = max(0, (final_score - 1/3) * 1.5)
```

| `final_score` | `adjusted_final_score` |
|---:|---:|
| ≤ 0.333 | 0.000 (floored) |
| 0.500 | 0.250 |
| 0.667 | 0.500 |
| 0.750 | 0.625 |
| 1.000 | 1.000 |

## task_5_4 — multiplicative composite

The original `final_score` is a weighted sum:
`0.4 * nd_score + 0.3 * lis_score + 0.3 * adj_score`. We replace it with
the product of the three components:

```
adjusted_final_score = nd_score * lis_score * adj_score
```

If any component is missing or zero, the product is `0`. This is harsher
than the weighted sum — a hard fail on any one metric drives the headline
to 0 even if the other two are fine. The intent is to ensure all three
ordering signals (footrule distance, longest correctly-ordered subsequence,
adjacent-transition hits) actually have to land for the system to score
non-zero.

If any component is missing because the verifier-evaluator errored
(observed in 18 cases in our runs, all with `final_score = 0`),
`adjusted_final_score = 0`.

Example: `nd_score=0.96, lis_score=0.90, adj_score=0.667`
→ `adjusted = 0.96 × 0.90 × 0.667 = 0.576`.

## Aggregating to a (harness, model) headline number

For a given (harness, model) on either task:

1. Pick the task IDs in scope (e.g. all 28 of task_5_4 or all 18 of task_7_3).
2. For each task ID, look up `adjusted_final_score` from each of the `N` runs
   (we used N = 3).
3. Treat missing runs as 0.
4. Final number = mean over all `(N × tasks)` cells.

Worked example for `codex_cli + gpt-5.5` on task_7_3, task IDs
`{1,2,4,5,6,7,9,10,11,12,13,14,15,18,19,20,22,24}`:

- run1 mean over 18 task IDs: 0.3958
- run2 mean over 18 task IDs: 0.3667
- run3 mean over 18 task IDs: 0.3806
- **Final: 0.3810** (mean of all 3 × 18 = 54 cells)
