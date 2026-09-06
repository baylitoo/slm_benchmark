"""Preserve the generation prompt of checkpoints trained to always think."""

from __future__ import annotations

import re
from typing import Any

from docie_bench.llm.model_profiles import ModelProfile


def uses_native_reasoning(profile: ModelProfile) -> bool:
    """Recognize LFM2.5-2.6B; allow explicit policy for opaque served aliases.

    This is checkpoint-specific: smaller LFM2 instruct models do not share its
    unconditional <think> generation prompt. Base checkpoints are excluded.
    """
    if "native_reasoning" in profile.options:
        return profile.options["native_reasoning"] is True
    for identifier in (profile.model, profile.name):
        leaf = identifier.rsplit("/", 1)[-1].lower()
        if re.match(r"^lfm2[._]?5[-_:]2[._]6b(?:$|[-_.:])", leaf) and not re.search(
            r"(?:^|[-_.:])base(?:$|[-_.:])", leaf
        ):
            return True
    return False


def apply_native_reasoning(body: dict[str, Any]) -> None:
    """Enable backend reasoning without inventing model-specific effort levels."""
    kwargs = dict(body.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = True
    kwargs.pop("reasoning_effort", None)
    body["chat_template_kwargs"] = kwargs
    body.pop("reasoning_effort", None)
