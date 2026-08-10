"""Repair UTF-8 mojibake in model output — e.g. ``universitÃ©`` → ``université``.

Sub-billion vision/OCR models trained on badly-encoded web text reproduce
double-encoded UTF-8 ("é" → "Ã©") when transcribing accented (especially French)
documents. The transport is UTF-8-clean end to end (httpx decodes the response
bytes as UTF-8, FastAPI re-encodes UTF-8), so the mojibake is the MODEL's own
output, not a charset bug in the pipeline.

ftfy's ENCODING fix repairs it surgically: only the byte-level mis-encoding, NOT
the quote / whitespace / HTML normalization ``fix_text`` also does — those would
alter OCR fidelity (curly quotes, ligatures, spacing an operator may rely on).
Idempotent and safe on already-correct text (a no-op).
"""

from __future__ import annotations

from typing import Any

import ftfy


def fix_mojibake(text: str | None) -> str | None:
    """Repair double-encoded UTF-8 in ``text``; pass through ``None``/empty.

    A no-op on correct text, so it is safe to apply unconditionally to any model
    output. Only the encoding is fixed (``ftfy.fix_encoding``), never spelling,
    punctuation, or whitespace.
    """
    if not text:
        return text
    return ftfy.fix_encoding(text)


def fix_completion_content(completion: Any) -> Any:
    """Repair mojibake in an OpenAI chat completion's assistant message content.

    Mutates the ``choices[].message.content`` in place — a plain string or the
    multimodal part list. Best-effort: any unexpected shape is left untouched
    (this must never break a passthrough response).
    """
    if not isinstance(completion, dict):
        return completion
    for choice in completion.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = fix_mojibake(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = fix_mojibake(part["text"])
    return completion
