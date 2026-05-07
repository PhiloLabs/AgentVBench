# task_4 — cutbench creative-recap scoring

Each task in `task_4_2` asks an agent to cut a short creative recap of a film
or broadcast to a brief written by a domain expert. The agent's `final.mp4`
is graded against ~28 expert-authored rubric items, each one a yes/no check
authored alongside the brief.

The rubric items are part of the dataset (column `rubric_items`). This
verifier just consumes them, dispatches each item to the right backend, and
aggregates a weighted score.

## Rubric dispatch

Each item has a `dispatch` field that selects how it's checked:

| dispatch  | implementation | example check |
|---|---|---|
| `python`   | `kinds.run_kind` — deterministic, in-process. Backed by ffprobe / scenedetect / OCR / loudness analysis / etc. | "duration is exactly 60s ± 0.5", "≥ 4 distinct scene cuts within first 5 s" |
| `llm`      | `llm_judges.run_llm_items_async` — yes/no question to Gemini against the uploaded `final.mp4` | "Does the cut open on a hook (no slow ramp-up) within the first 1.5 s?" |
| `compound` | ordered Python stages → final LLM stage; Python evidence is substituted into the LLM prompt's `{var}` placeholders | "Voice-over present (python ffprobe loudness) AND it has the right tone (LLM)" |

Each item carries a signed `weight`:

- positive weight = "earn N points if passed"
- negative weight = "lose N points if failed" (catastrophic violations like
  "the recap is the wrong length")

Final score:

```
final_score = sum(passed × weight) / sum(positive_weights),  clamped to [0, 1]
```

There is **no separate adjusted form** for task_4 — `final_score` is what
the paper reports.

## Pillars

Items also carry an integer `pillar` in `{0: format, 1: visual, 2: narrative,
3: polish}`. The result object includes a per-pillar breakdown for analysis,
but the headline `final_score` is computed across all items.

## Files

```
task_4_cutbench/
├── score.py        public entry point + CLI
├── kinds.py        Python-dispatch checks (ffprobe, scenedetect, OCR, …)
├── llm_judges.py   Gemini File API integration for `llm` and `compound` dispatches
└── README.md
```

## Requirements

- ffmpeg / ffprobe on PATH (system dep — `brew install ffmpeg` or apt)
- `pip install agenticvbench[task4]` — adds `google-genai` for the Gemini
  File API
- `GEMINI_API_KEY` env var — get one at <https://aistudio.google.com/apikey>

## Score one task

```bash
avb-score-task-4 \
    --final-mp4 path/to/agent_output/final.mp4 \
    --task-id cutbench-animated_out
```

Programmatic:

```python
from verifiers.task_4_cutbench import score_task
from datasets import load_dataset

ds = load_dataset("Anonymous47621123/AgentVBench_100", "task_4_2", split="train")
row = next(r for r in ds if r["task_id"] == "cutbench-animated_out")
result = score_task(
    final_mp4="path/to/final.mp4",
    rubric_items=row["rubric_items"],
    task_id="cutbench-animated_out",
)
print(result.final_score, result.pillar_breakdown)
```

## Notes

- We do **not** upload the source film to Gemini. ~70% of source-referencing
  rubric items already encode the source content as text in the prompt
  (motifs, style anchors, fabrication facts), so the LLM can grade them
  using the agent's output alone. The remaining items take a small accuracy
  hit. This was a deliberate trade-off to (a) avoid Gemini File API
  size limits on multi-GB sports broadcasts, and (b) halve File API
  storage exposure per verifier run.
- The default model is `gemini-3-flash-preview`. Pass `--gemini-model
  <name>` to override.
