"""Tool-calling capability: the launch-args + family-flag contract.

The chat proxy always forwarded `tools` verbatim (pinned in test_chat_api),
but llama-server only RENDERS tools through its jinja engine -- a family
launched without --jinja answers a tools request with a plain completion,
silently. These tests pin the two halves of the fix: the flag never claims
what the launch args can't deliver, and the families API exposes it so a
caller can distinguish "tools unsupported here" from "model chose not to
call".
"""

from __future__ import annotations

from docie_bench.serving.model_store import FAMILIES


def test_every_tools_family_launches_with_jinja() -> None:
    # The invariant, checked for ALL families (present and future): claiming
    # tools without --jinja would resurrect the silent-ignore failure mode.
    for name, fam in FAMILIES.items():
        if fam.tools:
            assert "--jinja" in fam.llama_server_args, (
                f"family {name!r} claims tools but launches without --jinja"
            )


def test_chat_families_serve_tools_and_extraction_contracts_do_not() -> None:
    assert FAMILIES["openai_chat"].tools is True
    assert "--jinja" in FAMILIES["openai_chat"].llama_server_args
    assert FAMILIES["lfm2"].tools is True
    # nuextract3 launches with --jinja too -- for its extraction template, not
    # for chat. Tool calling is meaningless on an extraction contract.
    assert FAMILIES["nuextract3"].tools is False
    # Non-chat surfaces never claim tools.
    for name in ("embedding", "reranker", "multi_vector", "encoder_gliner", "transformers"):
        assert FAMILIES[name].tools is False, name


def test_families_api_exposes_the_tools_flag() -> None:
    import asyncio

    from docie_bench.inngest.serving_api import list_families

    payload = asyncio.run(list_families())
    by_name = {fam["name"]: fam for fam in payload}
    assert by_name["openai_chat"]["tools"] is True
    assert by_name["nuextract3"]["tools"] is False
