import pytest

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.reasoning import uses_native_reasoning


@pytest.mark.parametrize(("model", "expected"), [
    ("lfm2.5-2.6b", True),
    ("LiquidAI/LFM2.5-2.6B-GGUF", True),
    ("LFM2.5-2.6B-Q4_K_M.gguf", True),
    ("lfm2.5:2.6b", True),
    ("lfm25_2_6b", True),
    ("LiquidAI/LFM2.5-2.6B-Base", False),
    ("lfm25_350m", False),
    ("lfm2.5-1.2b-instruct", False),
    ("lfm2.5-vl-1.6b", False),
    ("qwen3", False),
])
def test_checkpoint_detection(model: str, expected: bool) -> None:
    profile = ModelProfile(name="alias", model=model, base_url="http://test", api_key="k")
    assert uses_native_reasoning(profile) is expected


def test_native_reasoning_policy_overrides_detection() -> None:
    for model, enabled in (("opaque-alias", True), ("lfm2.5-2.6b", False)):
        profile = ModelProfile(
            name="alias", model=model, base_url="http://test", api_key="k",
            options={"native_reasoning": enabled},
        )
        assert uses_native_reasoning(profile) is enabled
