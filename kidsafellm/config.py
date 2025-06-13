'''Central place to inject API-keys and expose model-name aliases.

• Environment variables are set at import-time so every downstream client
  (OpenAI, Anthropic, Google, Together.ai, xAI Grok, DeepInfra) just calls
  `os.getenv("…")`.

• MODEL_ALIAS maps the *column name* used in analysis/analyze.py to the
  provider’s model identifier.  Keep this in sync with utils.gpt_api.py
  if you ever add or rename a model column.'''

import os

# all environment variables
# ---------------------------------------------------------------------------
# 1. API keys — replace the placeholders with real tokens
# ---------------------------------------------------------------------------
os.environ.update(
    {
        "OPENAI_API_KEY": "[OPENAI_KEY]",          # GPT-4o
        "ANTHROPIC_API_KEY": "[ANTHROPIC_KEY]",    # Claude Sonnet
        "GOOGLE_API_KEY": "[GOOGLE_KEY]",          # Gemini Flash / Pro
        "TOGETHER_API_KEY": "[TOGETHER_KEY]",      # Llama-3-70B (Together.ai)
        "GROK_API_KEY": "[GROK_KEY]",              # xAI Grok-3
        "DEEPINFRA_API_KEY": "[DEEPINFRA_KEY]",    # Vicuna / Mistral / DeepSeek
    }
)

# ---------------------------------------------------------------------------
# 2. Canonical alias map  (column name  →  provider model ID)
# ---------------------------------------------------------------------------
MODEL_ALIAS = {
    "gpt4o_response":             "gpt-4o",
    "llama3.1-70B_response":      "meta-llama/Llama-3-70B-Instruct",
    "claude3.7sonnet_response":   "claude-3-sonnet-20240229",
    "gemini2.0flashpro_response": "gemini-1.5-flash",
    "grok3_response":             "grok-1.5",
    "vicuna-7B_response":         "lmsys/vicuna-7b-v1.5",
    "mistral-7B_response":        "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseekr1_response":        "deepseek-ai/DeepSeek-R1",
}