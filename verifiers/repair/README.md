# repair — broken-video repair

Each task in `repair` gives the agent a `broken.mp4` containing one or more
defects (frozen scene, scene swap, color shift, audio noise, duplicate
segment, A/V desync; the v7 cells stack several at once). The agent must:

1. Inspect the broken video, identify each defect's start/end timestamp.
2. Produce `fixed.mp4` — the repaired video.
3. Produce `report.md` — a localization log with one `## Diagnosis` block
   per defect, plus a brief `## Repair trajectory` block.

## Score

Three rubrics, weighted out of 100 points:

| rubric | weight | what it measures |
|---|---|---|
| **format**       |  5 pt | container / codec / fps / resolution / sample-rate match the input. Pure ffprobe. |
| **localization** | 35 pt | the agent's `## Diagnosis` blocks identify the defect window correctly (per-variant rubric — see `lib/tasks/<variant>/localize.py`). |
| **edit**         | 60 pt | SSIM (video) and cross-correlation (audio) on the affected region — does the repaired content actually match the source inside the defect window, without spilling damage outside it? |

```
final_score = (format + localization + edit) / 100   ∈ [0, 1]
```

There is **no separate adjusted form** for repair — `final_score` is what
the paper reports.

## What you need to score

- `fixed.mp4` — the agent's repaired output.
- `report.md` — the agent's localization log.
- `source.mp4` — the **original** uncorrupted video (verifier-only, fetched
  from the dataset's `verifier_reference_urls` column).
- `task_id` — e.g. `bench-broken-cut-v1-s1`.

## Layout

```
repair/
├── score.py                  public entry point + CLI
├── cells.json                manifest mapping task_id → (variant, GT profile, source basename)
├── ground_truth/
│   └── <vN>/<sM>/profile.json   per-cell defect ground truth (timestamps, etc.)
└── lib/                      vendored kit
    ├── rubrics/
    │   ├── format.py         ffprobe-driven format check (5 pt)
    │   ├── _localize.py      generic localization rubric base (35 pt)
    │   ├── _description.py   shared description-similarity helpers
    │   ├── edit.py           SSIM + xcorr signal-processing (60 pt)
    │   └── format_config.json
    └── tasks/                per-variant rubric specializations
        ├── v01_frozen_scene/{__init__.py, localize.py, task.json}
        ├── v03_scene_swap/...
        ├── v04_color_grade_shift/...
        ├── v05_noise_floor_spike/...
        ├── v06_duplicate_segment/...
        ├── v07_audio_video_desync/...
        └── _combined/        (v7 — multiple defects stacked)
```

## Requirements

- ffmpeg / ffprobe on PATH (system dep)
- `pip install agenticvbench[repair]` — adds numpy / opencv-python / scipy
  for SSIM, dhash, xcorr

## Score one task

```bash
avb-score-repair \
    --fixed-mp4 path/to/agent_output/fixed.mp4 \
    --report-md path/to/agent_output/report.md \
    --source-mp4 path/to/v1_source.mp4 \
    --task-id bench-broken-cut-v1-s1
```

The 6 source mp4s (`v1_source.mp4` through `v7_source.mp4`) live in the
dataset under `repair/sources/` and are referenced per-cell from
`verifier_reference_urls`.

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from verifiers.repair import score_task

ds = load_dataset("Anonymous47621123/AgentVBench_100", "repair", split="train")
row = next(r for r in ds if r["task_id"] == "bench-broken-cut-v1-s1")

# Download the per-cell source video from the dataset
src_url = row["verifier_reference_urls"][0]
# (resolve URL to local path — see examples/ for the helper)

result = score_task(
    fixed_mp4="path/to/fixed.mp4",
    report_md="path/to/report.md",
    source_mp4="path/to/v1_source.mp4",
    task_id=row["task_id"],
)
print(result.final_score)
```

## Privacy note

The source video must NOT be visible to the agent during rollout — it would
trivialize the task. The agent's runtime sandbox should mount only the
`broken.mp4` (from `reference_file_urls`); the verifier loads
`source.mp4` (from `verifier_reference_urls`) separately, after the agent
exits. The dataset hosts both publicly; honoring the privacy boundary is
the responsibility of whoever runs the agent.
