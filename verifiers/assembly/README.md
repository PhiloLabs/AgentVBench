# assembly — video assembly

Each task in `assembly` gives the agent N candidate video clips and a
storyboard with M slots (each slot describes one shot in cinematic
language: `description`, `shot_size`, `camera_angle`, `lens_size`,
`camera_movement`). The agent picks one candidate per slot and concatenates
them in slot order, producing `solution.mp4` and a manifest
(`solution.json`).

## Score

Per-slot exact match against the dataset's golden picks
(`correct_assembly_in_slot_order`):

```
assembly_score       = n_correct / n_slots                          ∈ [0, 1]
strict_match         = 1.0 iff every slot picked correctly          ∈ {0, 1}
final_score          = assembly_score
adjusted_final_score = max(0, (final_score − 1/3) × 1.5)            ← headline
```

The headline `adjusted_final_score` rescales away the chance floor (random
picking gives ≈1/3 since slots have 3 candidates on average). See
[SCORE_ADJUSTMENTS.md](../../SCORE_ADJUSTMENTS.md).

## Files

```
assembly/
├── score.py    public entry point + CLI + adjustment math
└── README.md
```

Pure stdlib + (optional) `datasets`/`pyarrow` for dataset lookup. No LLM
calls, no system deps.

## Score one task

```bash
avb-score-assembly \
    --solution-json path/to/agent_output/solution.json \
    --task-id 1
```

Or pass the answer inline (no dataset access):

```bash
avb-score-assembly \
    --solution-json path/to/agent_output/solution.json \
    --task-id 1 \
    --correct-assembly "3.mp4,7.mp4,5.mp4,11.mp4"
```

Programmatic:

```python
from verifiers.assembly import score_task

result = score_task(
    solution_json="path/to/solution.json",
    correct_assembly=["3.mp4", "7.mp4", "5.mp4", "11.mp4"],
    task_id=1,
)
print(result.adjusted_final_score)
```

## Manifest format

The agent's `solution.json`:

```json
{
  "segments": [
    {"output": [0.0, 2.0], "source": "3", "source_range": [0.0, 2.0]},
    {"output": [2.0, 3.5], "source": "7", "source_range": [0.0, 1.5]},
    ...
  ]
}
```

We extract the `source` field of each segment, normalize to `<N>.mp4` form,
and compare slot-by-slot against the answer key. The `output` and
`source_range` fields are not used by v0.1 of this verifier (a future v2
will cross-check them against the actual `solution.mp4`).
