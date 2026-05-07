# Reproducing AgenticVBench paper numbers

This document describes how to recreate the headline `adjusted_final_score`
numbers reported in the paper. The verifiers in this repo are the source of
truth for scoring; the steps below describe the agent-running side that
produces the `final.mp4` / `solution.json` / `fixed.mp4` files we feed into
those verifiers.

## What you need

- A solution per (model, harness, task, run) — the agent's output for one
  task instance under one configuration.
- The dataset at `Anonymous47621123/AgentVBench_100` (HF Hub).
- Per-task verifier from this repo to convert each solution into a score.

The paper aggregates scores as **mean(adjusted_final_score over runs × task_ids)**
within each (harness, model) cell, treating any missing solution as 0.

## Per-task expected outputs (what the agent produces)

| Task family   | Output files | Where the score comes from |
|---|---|---|
| task_4_2 cutbench       | `final.mp4`                                          | `avb-score-task-4 --final-mp4 …`  |
| task_5_4 video order    | `solution.mp4` + `solution.json` (manifest)          | `avb-score-task-5 --solution-json …` |
| task_6   video repair   | `fixed.mp4` + `report.md`                            | `avb-score-task-6 --fixed-mp4 … --report-md … --source-mp4 …`     |
| task_7_3 video assembly | `solution.mp4` + `solution.json` (manifest)          | `avb-score-task-7 --solution-json …` |

The `solution.mp4` files for task_5 and task_7 are not used by the verifier
in v0.1 (we score from the manifest), but the prompt asks the agent to
deliver them and a future v2 will cross-check the manifest against the
actual concatenation.

## Harness configurations evaluated in the paper

Each (model, harness) cell is run for `N` independent attempts on each task.
The headline table lists (harness, model) pairs across:

- **gemini_cli** — Google's Gemini CLI tool, with the agent-mode template.
- **claude_code** — Anthropic's Claude Code CLI.
- **codex_cli** — OpenAI's Codex CLI.
- **openhands_cli** — OpenHands CLI harness (`@openhands-cli`).
- **opencode** — opencode CLI (`opencode-ai/opencode`).
- **openclaw** — openclaw CLI agent.
- **openhands_sdk** — OpenHands SDK (Python, agent-loop dispatch).

We used `--max-iterations 200` and `--timeout 1800` (30 min wall-clock per
task) across all harnesses. The prompt for each task is the dataset's
`prompt` column verbatim. Materials referenced by `reference_file_urls` are
downloaded ahead of time and mounted at `/workspace/materials/`. Outputs are
expected at `/workspace/output/` per the prompt.

## Compute environment

We ran agent containers in a sandboxed cloud environment (compute backend
abstracted; the verifier code does not depend on the backend). Each task
container was given:

- 2 CPU cores, 8 GB RAM (16 GB for task_6 / task_4 due to ffmpeg + scenedetect
  memory peaks)
- 30-minute wall-clock cap
- Per-task scratch dir at `/workspace/`
- The required CLI binary (one per harness)
- Per-harness API keys via env (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, …)

After the container exits, we collect the output files and upload them to
object storage. Verification is a separate batch job that pulls each
solution and runs the matching verifier from this repo.

## Verifier compute

- task_4 cutbench: needs `GEMINI_API_KEY` (for the LLM-judge items). Each
  call uploads `final.mp4` to Gemini File API once, reuses the file URI for
  every `llm` and `compound` rubric item. ~16 concurrent calls/task. Median
  per-task verify wall-clock: ~90 s.
- task_5 video order: pure-stdlib Python. Median per-task verify
  wall-clock: <1 s.
- task_6 video repair: ffprobe + numpy + opencv-python + scipy. 2-worker
  ProcessPool. Median per-task verify wall-clock: 5–15 min (varies with
  source-video length and per-cell `chunk_duration_s`).
- task_7 video assembly: pure-stdlib. Median per-task verify
  wall-clock: <1 s.

## Suggested local repro flow

```bash
# 1) Pull a single task instance's input
huggingface-cli download Anonymous47621123/AgentVBench_100 \
    --repo-type dataset \
    --include "task_5_4/materials/2.zip" \
    --local-dir /tmp/avb_demo

unzip /tmp/avb_demo/task_5_4/materials/2.zip -d /tmp/avb_demo/work

# 2) Run an agent of your choice (manually, or via a CLI harness)
#    expected outputs: /tmp/avb_demo/work/output/solution.{mp4,json}

# 3) Score
avb-score-task-5 \
    --solution-json /tmp/avb_demo/work/output/solution.json \
    --task-id 2
```

For the full evaluation campaign that populates the paper's tables, run the
above for every (harness, model, task_id, run) cell and aggregate per the
formula above.

## Score-adjustment math (task_5 and task_7)

See [SCORE_ADJUSTMENTS.md](SCORE_ADJUSTMENTS.md) for the derivation. Both
adjustments are computed automatically by `score_task` and emitted as
`adjusted_final_score` on the result object.
