# sequencing

Three component scores against the dataset's `correct_order`:

| metric    | definition |
|---|---|
| `nd_score`  | `1 − normalized_footrule(pred, correct)` |
| `lis_score` | longest correctly-ordered subsequence ratio |
| `adj_score` | fraction of true adjacent transitions preserved |

```
final_score          = 0.4·nd_score + 0.3·lis_score + 0.3·adj_score
adjusted_final_score = nd_score · lis_score · adj_score
```

See [`../../SCORE_ADJUSTMENTS.md`](../../SCORE_ADJUSTMENTS.md) for the
adjusted-score derivation. If the manifest is missing, malformed, or doesn't
match the correct clip-set, every component is `0`.

Pure stdlib + (optional) `datasets`/`pyarrow` for dataset lookup. No system
deps.

## Score one task

```bash
avb-score-sequencing --solution-json solution.json --task-id 2
```

## Manifest format

```json
{
  "segments": [
    {"output": [0.0, 2.5], "source": "6", "source_range": [0.0, 2.5]},
    ...
  ]
}
```

We only consume `source` of each segment; `output` and `source_range` are
unused in v0.1.
