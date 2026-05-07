# sequencing — video ordering

Each task in `sequencing` gives the agent N video clips (named `1.mp4`,
`2.mp4`, …, `N.mp4`) that are the shots of a single chapter from a short
film, renamed and shuffled. The agent reassembles them into the original
narrative order and writes a manifest (`solution.json`) listing the picked
order.

## Score

We compute three component scores against the dataset's `correct_order`:

| metric    | definition | range |
|---|---|---|
| `nd_score`  | `1 − normalized_footrule(pred, correct)` — lower footrule = better | [0, 1] |
| `lis_score` | longest correctly-ordered subsequence ratio (patience sorting) | [0, 1] |
| `adj_score` | fraction of true adjacent transitions preserved | [0, 1] |

```
final_score          = 0.4 × nd_score + 0.3 × lis_score + 0.3 × adj_score
adjusted_final_score = nd_score × lis_score × adj_score      ← headline
```

The headline `adjusted_final_score` is the multiplicative composite, which
zeros when any component fails. See [SCORE_ADJUSTMENTS.md](../../SCORE_ADJUSTMENTS.md)
for the rationale.

If the solution is missing, malformed, or doesn't match the correct
clip-set, every component is 0.

## Files

```
sequencing/
├── score.py    public entry point + CLI + metric implementations
└── README.md
```

Pure stdlib + (optional) `datasets`/`pyarrow` for dataset lookup. No LLM
calls, no system deps.

## Score one task

```bash
avb-score-sequencing \
    --solution-json path/to/agent_output/solution.json \
    --task-id 2
```

Or pass the correct order inline (no dataset access):

```bash
avb-score-sequencing \
    --solution-json path/to/agent_output/solution.json \
    --task-id 2 \
    --correct-order "6,5,1,2,3,9,4,8,7,10"
```

Programmatic:

```python
from verifiers.sequencing import score_task

result = score_task(
    solution_json="path/to/solution.json",
    correct_order=["6","5","1","2","3","9","4","8","7","10"],
    task_id=2,
)
print(result.adjusted_final_score)
```

## Manifest format

The agent's `solution.json`:

```json
{
  "segments": [
    {"output": [0.0, 2.5], "source": "6", "source_range": [0.0, 2.5]},
    {"output": [2.5, 4.1], "source": "5", "source_range": [0.0, 1.6]},
    ...
  ]
}
```

We extract the `source` field of each segment to recover the predicted
order. The `output` and `source_range` fields are not used by v0.1 of this
verifier (a future v2 will cross-check them against the actual
`solution.mp4`).
