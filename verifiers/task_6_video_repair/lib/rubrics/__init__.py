"""Universal scorers. Per-variant localization lives under `tasks/v*/localize.py`."""

from rubrics.edit import score_edit
from rubrics.format import score_format

__all__ = ["score_format", "score_edit"]
