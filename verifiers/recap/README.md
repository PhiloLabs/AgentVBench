# recap

Each task carries an inline `rubric_items` column on the dataset (~28 expert-
authored checks per task). Items dispatch to one of three backends:

- **`python`** — deterministic checks (ffprobe / scenedetect / OCR / loudness),
  implemented in `kinds.py`.
- **`llm`** — yes/no questions answered by Gemini against the agent's output,
  implemented in `llm_judges.py` (uploads `final.mp4` to Gemini File API once
  per task, reuses the file URI).
- **`compound`** — ordered Python stages → final LLM stage, with Python
  evidence substituted into the LLM prompt's `{var}` placeholders.

Each item has a signed weight (positive = "earn N points"; negative = "lose N
points if failed"). Final score:

```
final_score = sum(passed × weight) / sum(positive_weights)        clamped to [0, 1]
```

There is no separate adjusted form for recap.

## Requires

- `ffmpeg` / `ffprobe` on `PATH`
- `pip install agenticvbench[recap]` (adds `google-genai`)
- `GEMINI_API_KEY` env var

## Score one task

```bash
avb-score-recap --final-mp4 final.mp4 --task-id cutbench-animated_out
```
