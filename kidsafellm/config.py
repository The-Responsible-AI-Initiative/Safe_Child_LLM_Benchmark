# Centralized management for API keys
import os

# all environment variables
os.environ.update({
    "OPENAI_API_KEY":      "[OPENAI_KEY]",      # GPT-4o
    "ANTHROPIC_API_KEY":   "[ANTHROPIC_KEY]",   # Claude Sonnet
    "GOOGLE_API_KEY":      "[GOOGLE_KEY]",      # Gemini Flash / Pro
    "TOGETHER_API_KEY":    "[TOGETHER_KEY]",    # Llama-3-70B (Together.ai host)
    "GROK_API_KEY":        "[GROK_KEY]",        # xAI Grok-3
    "DEEPINFRA_API_KEY":   "[DEEPINFRA_KEY]",   # Vicuna / Mistral / DeepSeek
})

MODEL_ALIAS = {
    "gpt4o_response":            "gpt-4o",
    "llama3.1-70B_response":     "meta-llama/Llama-3-70B-Instruct",   # together host
    "claude3.7sonnet_response":  "claude-3-sonnet-20240229",
    "gemini2.0flashpro_response":"gemini-1.5-flash",                  # Google name
    "grok3_response":            "grok-1.5",                          # xAI
    "vicuna-7B_response":        "lmsys/vicuna-7b-v1.5",
    "mistral-7B_response":       "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseekr1_response":       "deepseek-ai/DeepSeek-R1",
}

# proprietary async models and local processing models
async_models = ['gpt-4o-mini', 'gpt-4o-2024-08-06', 'gemini-1.5-flash', 'gemini-1.5-pro', 'claude3-haiku', 'claude3.5-sonnet', 'o3-mini', 'gemini-2.0-flash', 'deepseek-r1', 'llama3.3-70b', 'mistral-8x7b']
local_models = ['mistral', 'llama3-instruct', 'vicuna', 'vicuna-13b']