# AgenticVBench — verifiers

Per-task scoring code for the AgenticVBench benchmark. The dataset lives at
[`Anonymous47621123/AgentVBench_100`](https://huggingface.co/datasets/Anonymous47621123/AgentVBench_100)
on Hugging Face. This repo lets you score one candidate solution against the
dataset's ground truth.

```
verifiers/
├── recap/        # creative recap
├── sequencing/   # narrative sequencing
├── repair/       # broken-video repair
└── assembly/     # video assembly
```

## Install

```bash
git clone <this-repo>
cd agenticvbench-code
pip install -e .[all]
```

System deps: **ffmpeg / ffprobe** on `PATH`. **`GEMINI_API_KEY`** required for
the `recap` LLM-judge items.

## Score one task

```bash
avb-score-recap      --final-mp4 final.mp4       --task-id cutbench-animated_out
avb-score-sequencing --solution-json solution.json --task-id 2
avb-score-repair     --fixed-mp4 fixed.mp4 --report-md report.md \
                     --source-mp4 v1_source.mp4 --task-id bench-broken-cut-v1-s1
avb-score-assembly   --solution-json solution.json --task-id 1
```

Each CLI loads ground truth from `Anonymous47621123/AgentVBench_100` by default.
Pass `--dataset path/to/local/parquet` to score offline. See `examples/` for two
sample solutions.

## License

Apache-2.0.
