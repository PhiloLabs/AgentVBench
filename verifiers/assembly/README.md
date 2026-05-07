# assembly

Per-slot exact match against the dataset's `correct_assembly_in_slot_order`:

```
assembly_score       = n_correct / n_slots
final_score          = assembly_score
adjusted_final_score = max(0, (final_score − 1/3) × 1.5)
```

The chance-floor rescale (random picks score ≈1/3) maps random → 0 and perfect
→ 1, floored at 0. See [`../../SCORE_ADJUSTMENTS.md`](../../SCORE_ADJUSTMENTS.md).

Pure stdlib + (optional) `datasets`/`pyarrow` for dataset lookup. No system
deps.

## Score one task

```bash
avb-score-assembly --solution-json solution.json --task-id 1
```

## Manifest format

```json
{
  "segments": [
    {"output": [0.0, 2.0], "source": "3",  "source_range": [0.0, 2.0]},
    {"output": [2.0, 3.5], "source": "7",  "source_range": [0.0, 1.5]},
    ...
  ]
}
```

We extract `source` (normalized to `<N>.mp4`) and compare slot-by-slot. The
`output` and `source_range` fields are unused in v0.1.
