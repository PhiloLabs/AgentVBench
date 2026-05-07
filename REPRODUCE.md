# Reproducing scores

The verifiers in this repo score one solution at a time. To reproduce a
benchmark cell:

1. Run an agent on a task — the agent produces these files for each family:

   | family     | expected outputs                            |
   |---|---|
   | recap      | `final.mp4`                                 |
   | sequencing | `solution.mp4` + `solution.json` (manifest) |
   | repair     | `fixed.mp4` + `report.md`                   |
   | assembly   | `solution.mp4` + `solution.json` (manifest) |

2. Run the matching verifier (see top-level [`README.md`](README.md)).

3. Aggregate `adjusted_final_score` (or `final_score` for `recap`/`repair`)
   across runs and tasks. Treat missing files as `0`.

Verifier compute notes:

- **recap**: requires `GEMINI_API_KEY`; uploads `final.mp4` to Gemini File API
  once and reuses the file URI for every rubric item. Network-bound.
- **sequencing**, **assembly**: pure stdlib, sub-second per task.
- **repair**: ffprobe + numpy / opencv / scipy. CPU-bound; minutes per task.

Score-adjustment math: see [`SCORE_ADJUSTMENTS.md`](SCORE_ADJUSTMENTS.md).
