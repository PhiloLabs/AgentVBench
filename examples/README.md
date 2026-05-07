# Examples

Sample solution files for the simple-scorer tasks (task_5, task_7) so you
can run the CLIs end-to-end against the real dataset without writing a
manifest by hand.

## task_5_video_order

`perfect_solution.json` is constructed from the dataset's
`correct_order` for `task_id=2` (The Snowflake Discovery, 13 clips). Score it:

```bash
avb-score-task-5 \
    --solution-json examples/task_5_video_order/perfect_solution.json \
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

## task_7_video_assembly

`perfect_solution.json` is a 4-slot manifest. To score it as actually
correct, look up the golden picks for the task_id you care about and pass
them via `--correct-assembly`:

```bash
avb-score-task-7 \
    --solution-json examples/task_7_video_assembly/perfect_solution.json \
    --task-id 1 \
    --correct-assembly "3.mp4,7.mp4,5.mp4,11.mp4"
```

(or omit `--correct-assembly` to look up from the dataset).

## task_4 and task_6

These tasks score against a video file (`final.mp4` / `fixed.mp4`), so we
don't ship sample solutions in this repo (would inflate the tree by
hundreds of MB). To score one, produce the agent's output yourself or
download a sample run from the dataset's companion artifact bundle (TBD).
