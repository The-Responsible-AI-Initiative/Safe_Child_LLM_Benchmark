# Safe-Child LLM Benchmark
*A lightweight, open-source suite for red-team evaluation of refusal and policy-compliance in large language models.*

ChildSafeLLM Dataset is also avilable on Harvard Dataverse:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MRZGNB

---

## What is it?

`Safe_Child_LLM_Benchmark` stress-tests LLMs with **100 highly disallowed or harmful user queries**  
(violence, self-harm, hate, illicit behaviour, sexual content, privacy invasion).

For every model you specify, the toolkit

1. **Generates raw responses** – `analysis/analyze.py`
2. **Computes refusal / violation metrics** – `analysis/category_acc.py`, `analysis/level_acc.py`
3. **Estimates behavioural consistency** – `analysis/cronbach_alpha.py`

Outputs: easy-to-parse JSONL logs + CSV summaries.


🚀 Features

| Capability | Details |
|------------|---------|
| Standardised red-team prompts | Fixed list of 100 queries spanning the full policy surface. |
| Plug-and-play wrappers | Ready for **GPT-4o**, **Claude 3 Sonnet**, **Gemini 1.5 Flash**, **Grok-3**, **Llama-3 70B**, **Vicuna-7B**, **Mistral-7B**, **DeepSeek-R1**. |
| One-command dataset generation | `python -m kidsafellm.analysis.analyze` |
| Quick metrics | refusal %, violation %, refusal/violation split, Cronbach α |
| CI-friendly smoke tests | run in <1 s (`tests/test_prompts.py`, `tests/test_dispatch.py`) |


📦 Installation

```bash
# 1  Clone
git clone https://github.com/yourusername/safe_child_llm_benchmark.git
cd safe_child_llm_benchmark

# 2  Virtual env (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3  Dependencies
pip install -r requirements.txt

# 4  API keys
export OPENAI_API_KEY="sk-…"          # GPT-4o
export ANTHROPIC_API_KEY="…"          # Claude
export GOOGLE_API_KEY="…"             # Gemini
export TOGETHER_API_KEY="…"           # Llama-3 70B
export GROK_API_KEY="…"               # xAI Grok-3
export DEEPINFRA_API_KEY="…"          # Vicuna / Mistral / DeepSeek

(Put them in your shell profile or a .env manager for convenience.)


⚡ Quick start

1 · Generate responses for all eight models
python -m kidsafellm.analysis.analyze            # serial, 100 prompts × 8 models
# resume / force fresh / add threads:
# python -m kidsafellm.analysis.analyze --fresh --workers 4

Output → runs/latest_responses.jsonl

2 · Compute refusal / violation metrics
python -m kidsafellm.analysis.category_acc       # refusal & violation % per model
python -m kidsafellm.analysis.level_acc          # same split, different view

CSV files land in runs/.

3 · Behavioural consistency
python -m kidsafellm.analysis.cronbach_alpha     # Cronbach α across models


🗂️ Directory layout

kidsafellm/
├── analysis/             # data generation & metrics
│   ├── analyze.py
│   ├── category_acc.py
│   ├── level_acc.py
│   └── cronbach_alpha.py
├── utils/
│   ├── gpt_api.py        # wrappers + dispatch table
│   └── config.py         # API-key injection & alias map
├── prompts.py            # 100 harmful queries
└── tests/                # instant sanity checks


🔧 Programmatic snippet

import asyncio, json
from utils.gpt_api import MODEL_DISPATCH
from prompts import PROMPTS

async def demo():
    prompt = PROMPTS[0]
    reply = MODEL_DISPATCH["gpt4o_response"](prompt, {})
    print(json.dumps({"prompt": prompt, "gpt4o": reply[:120]+"…"}, indent=2))

asyncio.run(demo())


📄 Citation

@misc{safechild2025,
  author    = {Your Name},
  title     = {{Safe Child LLM Benchmark}: Red-Team Refusal and Compliance Evaluation},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/yourusername/safe_child_llm_benchmark}
}


🙏 Acknowledgements

Inspired by the jailbreakbench, LLM_Ethics_Benchmark project, LabSafety-Bench, and other open-source efforts
advancing safe & responsible AI.
