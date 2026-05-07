# Reproducing scores

The verifiers in this repo score one solution at a time. To reproduce a
benchmark cell:

1. Run an agent on a task — the agent produces these files for each family
   covered by this repo:

   | family     | expected outputs                            |
   |---|---|
   | sequencing | `solution.mp4` + `solution.json` (manifest) |
   | repair     | `fixed.mp4` + `report.md`                   |
   | assembly   | `solution.mp4` + `solution.json` (manifest) |

   The fourth family (**recap**) is scored by a learned rater released
   separately.

2. Run the matching verifier (see top-level [`README.md`](README.md)).

3. Aggregate `adjusted_final_score` (sequencing, assembly) or `final_score`
   (repair) across runs and tasks. Treat missing files as `0`.

Verifier compute notes:

- **sequencing**, **assembly**: pure stdlib, sub-second per task.
- **repair**: ffprobe + numpy / opencv / scipy. CPU-bound; minutes per task.

Score-adjustment math: see [`SCORE_ADJUSTMENTS.md`](SCORE_ADJUSTMENTS.md).
