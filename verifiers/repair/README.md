# repair

The agent receives `broken.mp4` (one or more defects) and produces:

- **`fixed.mp4`** — the repaired video
- **`report.md`** — a localization log with one `## Diagnosis` block per defect

Three rubrics, weighted out of 100:

| rubric | weight | what it measures |
|---|---:|---|
| **format**       |  5 | container / codec / fps / resolution / sample-rate match (ffprobe) |
| **localization** | 35 | `## Diagnosis` blocks identify the defect window |
| **edit**         | 60 | SSIM (video) + cross-correlation (audio) on the affected region |

```
final_score = (format + localization + edit) / 100   ∈ [0, 1]
```

No separate adjusted form for repair.

## Layout

```
repair/
├── score.py
├── cells.json                   manifest mapping task_id → (variant, profile, source basename)
├── ground_truth/<vN>/<sM>/profile.json
└── lib/
    ├── rubrics/{format,_localize,edit}.py
    └── tasks/<variant>/{localize.py, task.json}
```

## Requires

- `ffmpeg` / `ffprobe` on `PATH`
- `pip install agenticvbench[repair]` (adds `numpy`, `opencv-python`, `scipy`)
- the per-cell **source** video, hosted with the dataset under
  `verifier_reference_urls`. Do not let an agent see the source during rollout —
  it trivializes the task.

## Score one task

```bash
avb-score-repair \
    --fixed-mp4 fixed.mp4 \
    --report-md report.md \
    --source-mp4 v1_source.mp4 \
    --task-id bench-broken-cut-v1-s1
```
