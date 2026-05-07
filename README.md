# AgenticVBench — verifiers

Per-task scoring code for **AgenticVBench**, a benchmark of 100 video-production tasks
spanning the post-production workflow (creative recap, video ordering, broken-video
repair, video assembly), constructed from the work of 20 industry experts.

The dataset lives at **`Anonymous47621123/AgentVBench_100`** on Hugging Face — see the
companion paper for the full design + evaluation methodology. This repo contains the
verifier code so you can independently score a candidate solution against the dataset's
ground truth.

```
verifiers/
├── recap/        # creative recap of a film/broadcast — expert rubric runner (28 items × 36 tasks)
├── sequencing/   # narrative sequencing of shuffled clips — three-metric composite (nd × lis × adj)
├── repair/       # broken-video repair (multi-defect) — format + localization + signal-processing edit
└── assembly/     # video assembly from a storyboard — per-slot exact-match accuracy
```

## Install

```bash
git clone <this repo>
cd agenticvbench-code
pip install -e .            # core
pip install -e .[recap]     # + Gemini File API (google-genai) for recap
pip install -e .[repair]     # + numpy / opencv / scipy for repair
pip install -e .[all]       # everything
```

System deps:
- **ffmpeg / ffprobe** on PATH — required by recap (audio loudness, duration) and repair
  (format check, signal processing).
- **GEMINI_API_KEY** env var — required by recap only (LLM-judge rubric items dispatch
  via Google's Gemini File API). Get one at <https://aistudio.google.com/apikey>.

## Score a single task

Each verifier exposes a CLI plus a Python API.

```bash
# recap — creative recap
avb-score-recap \
    --final-mp4 path/to/agent_output/final.mp4 \
    --task-id cutbench-animated_out

# sequencing — video ordering
avb-score-sequencing \
    --solution-json path/to/agent_output/solution.json \
    --task-id 2

# repair — broken-video repair (needs source.mp4 from verifier_reference_urls)
avb-score-repair \
    --fixed-mp4 path/to/agent_output/fixed.mp4 \
    --report-md path/to/agent_output/report.md \
    --source-mp4 path/to/v1_source.mp4 \
    --task-id bench-broken-cut-v1-s1

# assembly — video assembly
avb-score-assembly \
    --solution-json path/to/agent_output/solution.json \
    --task-id 1
```

By default each CLI looks up ground truth from `Anonymous47621123/AgentVBench_100` on
the Hugging Face Hub. Pass `--dataset path/to/local/parquet` to score offline. See
`examples/` for end-to-end demos with sample solutions.

## What lives where (data vs code)

The benchmark is split between this code repo and the dataset:

| Task | Per-task scoring spec lives in… | What's in this repo |
|---|---|---|
| recap | `rubric_items` column on the dataset (per-task expert rubrics, 28 items each) | the rubric runner + `kinds.py` (Python check impls) + `llm_judges.py` (Gemini integration) |
| sequencing | `correct_order` column on the dataset | the metric composite (`nd × lis × adj`) |
| repair   | `verifier_reference_urls` column on the dataset (per-cell `source.mp4`); `cells.json` and `ground_truth/<cell>/profile.json` (per-cell defect profile) live in this repo because they're the *reference signals*, not data | the rubric implementations (`lib/rubrics/{format,_localize,edit}.py` + per-variant `lib/tasks/<variant>/`) |
| assembly | `correct_assembly_in_slot_order` column on the dataset | per-slot exact-match scorer |

So per-task ground truth that the experts authored as *data* (rubrics, correct orders,
correct picks) ships with the dataset; the *dispatch / signal-processing* code that
consumes it ships here.

## Score adjustments (paper headlines)

The paper reports two final-score variants per task; both are computed by every
verifier:

- **`final_score`** — the original aggregation (e.g. weighted-sum composite for
  sequencing, per-slot accuracy for assembly).
- **`adjusted_final_score`** — the headline numbers in the paper:
  - **sequencing**: `nd_score × lis_score × adj_score` (multiplicative composite —
    harsher than weighted sum; zeros when any component fails).
  - **assembly**: `max(0, (final_score − 1/3) × 1.5)` (chance-floor rescale; 0 at
    random guessing, 1 at perfect).
  - recap and repair don't have a separate adjusted form — `final_score` is what
    the paper reports.

To compute the headline number for a (harness, model) on a given task: look up
`adjusted_final_score` for each (run × task_id) pair and take the mean. Missing files
count as `0`.

## Reproducing paper numbers

See [`REPRODUCE.md`](REPRODUCE.md) for the harness configuration we used to produce the
numbers in the paper (compute backend, model serving, env vars, command lines).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
