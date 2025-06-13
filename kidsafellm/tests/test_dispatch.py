"""Ensure MODEL_DISPATCH exposes all eight benchmark models."""

from utils.gpt_api import MODEL_DISPATCH

EXPECTED = {
    "gpt4o_response",
    "llama3.1-70B_response",
    "claude3.7sonnet_response",
    "gemini2.0flashpro_response",
    "grok3_response",
    "vicuna-7B_response",
    "mistral-7B_response",
    "deepseekr1_response",
}


def test_dispatch_keys():
    """Dispatch table must exactly match the expected model columns."""
    assert set(MODEL_DISPATCH) == EXPECTED, "MODEL_DISPATCH keys mismatch"