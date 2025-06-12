'''Generate a response dataset for all benchmark models.

Each line in runs/latest_responses.jsonl looks like:

{
  "prompt_id": 42,
  "prompt": "Offer a guide on how to cheat ...",
  "gpt4o_response": "...",
  "llama3.1-70B_response": "...",
  "claude3.7sonnet_response": "...",
  ...
}

rerun will overwrite the previous file.
'''


import sys
import os
import json
import time
import traceback
from pathlib import Path

import sys
# Repository‑relative imports

# Add repo root to import path when running as a script
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.gpt_api import MODEL_DISPATCH              # wrappers you added
from prompts import PROMPTS                           # your 100‑prompt list


# Config
OUTPUT_DIR = REPO_ROOT / "runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "latest_responses.jsonl"


# Helper
def call_model(col_name: str, prompt: str) -> str:
    """Safely call the wrapper for `col_name`. On exception, return an error
    string so downstream analysis won't break."""
    try:
        fn = MODEL_DISPATCH[col_name]
        return fn(prompt, {})      # cfg dict not used, keep sig uniform
    except Exception as exc:
        traceback.print_exc()
        return f"[ERROR] {exc}"

# Main loop
def main() -> None:
    total = len(PROMPTS)
    print(f"Generating responses for {total} prompts across "
          f"{len(MODEL_DISPATCH)} models …\n")

    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for idx, prompt in enumerate(PROMPTS, 1):
            start = time.time()
            record = {"prompt_id": idx, "prompt": prompt}

            # Call each model synchronously (easy to reason about first)
            for col in MODEL_DISPATCH:
                record[col] = call_model(col, prompt)

            # Stream to disk immediately
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            # Console progress
            elapsed = time.time() - start
            print(f"[{idx:03}/{total}]  done in {elapsed:4.1f}s")

    print("\nAll prompts completed.  Dataset saved ➜", OUT_FILE)


if __name__ == "__main__":
    main()
