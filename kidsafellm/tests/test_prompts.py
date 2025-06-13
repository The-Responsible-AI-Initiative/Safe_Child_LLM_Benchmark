"""Quick sanity checks for the PROMPTS list."""

from kidsafellm.prompts import PROMPTS


def test_prompt_count():
    """Exactly 100 prompts are required for the red-team benchmark."""
    assert len(PROMPTS) == 100, "PROMPTS must contain exactly 100 items"


def test_prompt_uniqueness():
    """Prompts should be unique to avoid double-counting."""
    assert len(set(PROMPTS)) == 100, "PROMPTS contains duplicates"


def test_prompt_non_empty():
    """No prompt string should be blank or whitespace-only."""
    assert all(p.strip() for p in PROMPTS), "PROMPTS contains empty strings"