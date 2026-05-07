# AgenticVBench — verifiers

Per-task scoring code for three families of the AgenticVBench benchmark
(**sequencing**, **repair**, **assembly**). The dataset lives at
[`Anonymous47621123/AgenticVBench_100`](https://huggingface.co/datasets/Anonymous47621123/AgenticVBench_100)
on Hugging Face. This repo lets you score one candidate solution against
the dataset's ground truth.

The fourth family (**recap**) is graded by a learned rater trained on
human judgements; that grader is released separately and is not in this
repo.

```
verifiers/
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

System deps: **ffmpeg / ffprobe** on `PATH` (used by `repair`).

## Score one task

```bash
avb-score-sequencing --solution-json solution.json --task-id 2
avb-score-repair     --fixed-mp4 fixed.mp4 --report-md report.md \
                     --source-mp4 v1_source.mp4 --task-id bench-broken-cut-v1-s1
avb-score-assembly   --solution-json solution.json --task-id 1
```

Each CLI loads ground truth from `Anonymous47621123/AgenticVBench_100` by
default. Pass `--dataset path/to/local/parquet` to score offline. See
`examples/` for two sample solutions.

## License

Apache-2.0.
