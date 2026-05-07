"""Gemini-based LLM judge for cutbench rubric items.

Uploads each video file ONCE per task via Gemini File API and reuses the
file URI for every item's yes/no question. Falls back to inline base64 if
the File API path fails.

Public entry: ``run_llm_items_async(items, file_state, ...)``
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Gemini SDK is imported lazily so module import doesn't fail in environments
# without it (e.g. unit tests). The runtime container has it installed.
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GENAI_AVAILABLE = False


_DEFAULT_MODEL = "gemini-3-flash-preview"


# ---- File state -------------------------------------------------------------

@dataclass
class FileState:
    """Holds Gemini-uploaded file references for a single verify task."""
    output_video_path: str
    source_video_path: str | None = None
    output_uri: str | None = None
    source_uri: str | None = None
    output_mime: str = "video/mp4"
    source_mime: str = "video/mp4"
    client: object = None    # google.genai.Client


def _new_client(api_key: str):
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google.genai SDK not installed in this container")
    return genai.Client(api_key=api_key)


def _upload_and_wait(client, path: str, mime: str = "video/mp4"):
    """Upload a file via Gemini File API and wait until ACTIVE.

    Files start in PROCESSING state — Gemini extracts video features before
    they're usable. We poll up to ~5 min.
    """
    f = client.files.upload(file=path, config={"mime_type": mime})
    deadline = time.time() + 300
    while f.state.name == "PROCESSING" and time.time() < deadline:
        time.sleep(2)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file {f.name} did not become ACTIVE: {f.state.name}")
    return f


def init_file_state(
    output_video_path: str,
    source_video_path: str | None,
    api_key: str,
    needs_source: bool,
) -> FileState:
    """Open a Gemini client and upload the output video (and source if needed).

    Source is uploaded LAZILY — only if any item references it.
    """
    client = _new_client(api_key)
    out_file = _upload_and_wait(client, output_video_path)
    src_file = None
    if needs_source and source_video_path and Path(source_video_path).exists():
        src_file = _upload_and_wait(client, source_video_path)
    return FileState(
        output_video_path=output_video_path,
        source_video_path=source_video_path,
        output_uri=out_file.name if out_file else None,
        source_uri=src_file.name if src_file else None,
        client=client,
    )


# ---- Single item dispatch ---------------------------------------------------

_YES_NO_GUARD = (
    "\n\nRespond with exactly one line of strict JSON:\n"
    "{\"pass\": true|false, \"reason\": \"<≤30 word explanation>\"}\n"
)


def _build_user_content(prompt: str, file_state: FileState, references_source: bool):
    """Build the user-message content list with text + file URIs."""
    client = file_state.client
    out_file = client.files.get(name=file_state.output_uri)
    parts: list = [out_file]
    if references_source and file_state.source_uri:
        src_file = client.files.get(name=file_state.source_uri)
        parts.append(src_file)
        parts.append(
            "Above are TWO videos. The first is the agent's OUTPUT (the "
            "deliverable being scored). The second is the original SOURCE."
        )
    parts.append(prompt + _YES_NO_GUARD)
    return parts


def _references_source(prompt: str) -> bool:
    """Detect whether a prompt references the source video/transcript."""
    p = prompt.lower()
    needles = [
        "source transcript", "source video", "source's", "against the source",
        "the source's", "compared to source", "from the source",
        "in the source", "source audio", "source music", "source film",
    ]
    return any(n in p for n in needles)


def _parse_yes_no(text: str) -> tuple[bool, str]:
    """Parse `{pass: bool, reason: str}` from a Gemini response."""
    text = text.strip()
    # Try direct JSON parse
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return bool(obj.get("pass", False)), str(obj.get("reason", ""))[:200]
    except json.JSONDecodeError:
        pass
    # Fallback: look for "pass":true/false
    m = re.search(r'"pass"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true", text[:200]
    # Last-ditch: "yes"/"no"
    if re.search(r"\byes\b", text[:50], re.IGNORECASE):
        return True, text[:200]
    if re.search(r"\bno\b", text[:50], re.IGNORECASE):
        return False, text[:200]
    return False, f"unparseable: {text[:100]}"


async def _call_gemini_async(
    file_state: FileState,
    model: str,
    prompt: str,
    references_source: bool,
    sem: asyncio.Semaphore,
) -> tuple[bool, str]:
    """Make a single Gemini call, gated by `sem`."""
    async with sem:
        return await asyncio.to_thread(
            _call_gemini_sync, file_state, model, prompt, references_source,
        )


def _call_gemini_sync(
    file_state: FileState,
    model: str,
    prompt: str,
    references_source: bool,
) -> tuple[bool, str]:
    client = file_state.client
    contents = _build_user_content(prompt, file_state, references_source)
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=300,
                ),
            )
            text = resp.text or ""
            return _parse_yes_no(text)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return False, f"gemini error: {type(last_err).__name__}: {last_err}"


# ---- Public batch entry -----------------------------------------------------

async def run_llm_items_async(
    llm_items: list[dict],
    file_state: FileState,
    model: str = _DEFAULT_MODEL,
    max_concurrent: int = 16,
) -> dict[str, tuple[bool, str]]:
    """Dispatch every LLM item in parallel (bounded by max_concurrent).

    Returns: {item_id: (passed, reason)}
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(it: dict) -> tuple[str, tuple[bool, str]]:
        prompt = it.get("prompt") or it.get("criterion") or ""
        return it["id"], await _call_gemini_async(
            file_state, model, prompt,
            references_source=_references_source(prompt),
            sem=sem,
        )

    tasks = [_one(it) for it in llm_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out: dict[str, tuple[bool, str]] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"LLM item failed: {r}")
            continue
        item_id, passed_reason = r
        out[item_id] = passed_reason
    return out
