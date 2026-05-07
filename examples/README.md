# Examples

Sample solution files for the simple-scorer tasks (sequencing, assembly)
so you can run the CLIs end-to-end against the real dataset without
writing a manifest by hand.

## sequencing

`perfect_solution.json` is constructed from the dataset's
`correct_order` for `task_id=2` (The Snowflake Discovery, 13 clips). Score it:

```bash
avb-score-sequencing \
    --solution-json examples/sequencing/perfect_solution.json \
    --task-id 2
```

Expected:

```json
{
  ...,
  "nd_score": 1.0,
  "lis_score": 1.0,
  "adj_score": 1.0,
  "final_score": 1.0,
  "adjusted_final_score": 1.0
}
```

## assembly

`perfect_solution.json` is a 4-slot manifest. To score it as actually
correct, look up the golden picks for the task_id you care about and pass
them via `--correct-assembly`:

```bash
avb-score-assembly \
    --solution-json examples/assembly/perfect_solution.json \
    --task-id 1 \
    --correct-assembly "3.mp4,7.mp4,5.mp4,11.mp4"
```

(or omit `--correct-assembly` to look up from the dataset).

## repair

The repair verifier scores against a video file (`fixed.mp4` + `report.md`),
so we don't ship a sample solution in this repo (it would inflate the
tree). To score one, produce the agent's output yourself.
