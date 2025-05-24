# Centralized management of all API keys
import os

# One-time setup of all environment variables
os.environ.update({
    "OPENAI_API_KEY": "[API_KEY_REMOVED]",
    "GOOGLE_API_KEY": "[API_KEY_REMOVED]",
    "ANTHROPIC_API_KEY": "[API_KEY_REMOVED]",
})

# Remove original DEEPSEEK_CONFIG configuration
# Add DeepInfra configuration
os.environ["DEEPINFRA_API_KEY"] = "[API_KEY_REMOVED]"

model_path_dicts = {
    "vicuna": "lmsys/vicuna-7b-v1.5",
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
    'llama3-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    "vicuna-13b": "lmsys/vicuna-13b-v1.5",
    'galactica': 'facebook/galactica-6.7b',
    'darwin': 'darwin-7b',
    'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
    'claude3.5-sonnet': 'claude-3-5-sonnet-20240620',
    'claude3-haiku': 'claude-3-haiku-20240307',
    'o1-mini': 'o1-mini',
    'deepseek-r1': "deepseek-ai/DeepSeek-R1",
    'llama3.3-70b': "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    'mistral-8x7b': "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

# Model type classification: proprietary async models and local batch processing models
async_models = ['gpt-4o-mini', 'gpt-4o-2024-08-06', 'gemini-1.5-flash', 'gemini-1.5-pro', 'claude3-haiku', 'claude3.5-sonnet', 'o3-mini', 'gemini-2.0-flash', 'deepseek-r1', 'llama3.3-70b', 'mistral-8x7b']
local_models = ['mistral', 'llama3-instruct', 'vicuna', 'vicuna-13b']