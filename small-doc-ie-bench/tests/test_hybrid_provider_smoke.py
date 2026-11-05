from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from docie_bench.llm.model_profiles import load_model_profiles
from docie_bench.llm.upstream_transport import post_chat_completion


@pytest.mark.asyncio
async def test_opt_in_real_frontier_provider_smoke() -> None:
    if os.environ.get("DOCIE_RUN_PROVIDER_SMOKE") != "1":
        pytest.skip("set DOCIE_RUN_PROVIDER_SMOKE=1 to call the configured provider")
    config_path = Path(os.environ.get("DOCIE_MODELS_CONFIG", "configs/models.yaml"))
    profiles = load_model_profiles(config_path)
    if "frontier" not in profiles:
        pytest.fail(f"profile 'frontier' is missing from {config_path}")
    async with httpx.AsyncClient() as client:
        completion = await post_chat_completion(
            profiles["frontier"],
            client,
            {
                "messages": [
                    {"role": "user", "content": "Reply with the single word ready."}
                ],
                "max_tokens": 8,
            },
        )
    content = completion["choices"][0]["message"].get("content")
    assert isinstance(content, str)
    assert content.strip()
