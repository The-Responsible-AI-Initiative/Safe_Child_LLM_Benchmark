'''Generate a response dataset for all benchmark models.

EEach JSON-line in the output looks like:
{
  "prompt_id": 42,
  "prompt": "Offer a guide on how to cheat ...",
  "gpt4o_response": "...",
  "llama3.1-70B_response": "...",
  ...
}

• Default output  : runs/latest_responses.jsonl
• Safe to resume  : adds only missing prompts unless --fresh is used
• Concurrency     : set --workers >1 to parallelise across prompts

----------------------------------------------------------------------------
USAGE
----------------------------------------------------------------------------

# serial run
python -m kidsafellm.analysis.analyze

# resume if interrupted
python -m kidsafellm.analysis.analyze

# force a fresh rebuild
python -m kidsafellm.analysis.analyze --fresh

# run with 4 worker threads
python -m kidsafellm.analysis.analyze --workers 4

'''


import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository-relative imports
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.gpt_api import MODEL_DISPATCH
from prompts import PROMPTS

# ---------------------------------------------------------------------------
# Helper: safe model call
# ---------------------------------------------------------------------------
def call_model(col_name: str, prompt: str) -> str:
    """Call the wrapper associated with *col_name* and return the raw text.
    Any Exception is caught and turned into an '[ERROR] …' string so that
    downstream analysis never breaks."""
    try:
        fn = MODEL_DISPATCH[col_name]
        return fn(prompt, {})  # cfg dict unused for now
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return f"[ERROR] {exc}"


# ---------------------------------------------------------------------------
# Core per-prompt job
# ---------------------------------------------------------------------------
def process_prompt(idx: int, prompt: str) -> dict:
    """Return a dict containing the prompt and every model's response."""
    record = {"prompt_id": idx, "prompt": prompt}
    for col in MODEL_DISPATCH:
        record[col] = call_model(col, prompt)
    return record


# ---------------------------------------------------------------------------
# CLI / main entry
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "runs" / "latest_responses.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing file and start over",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent worker threads (1 = serial)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Resume logic
    # ---------------------------------------------------------------------
    done_ids: set[int] = set()
    if out_path.exists() and not args.fresh:
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["prompt_id"])
                except Exception:
                    pass
        print(f"⚠️  Found {len(done_ids)} completed prompts – will resume.")

    # ---------------------------------------------------------------------
    # Dispatch jobs (serial or threaded)
    # ---------------------------------------------------------------------
    start_run = time.time()
    total = len(PROMPTS)
    to_run = [
        (idx, p)
        for idx, p in enumerate(PROMPTS, 1)
        if idx not in done_ids
    ]

    if args.workers == 1:
        # ---------- serial ----------
        with out_path.open("a", encoding="utf-8") as fout:
            for idx, prompt in to_run:
                t0 = time.time()
                rec = process_prompt(idx, prompt)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(
                    f"[{idx:03}/{total}] done in {time.time() - t0:4.1f}s"
                )
    else:
        # ---------- threaded ----------
        with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open(
            "a", encoding="utf-8"
        ) as fout:
            future_to_idx = {
                pool.submit(process_prompt, idx, prompt): idx
                for idx, prompt in to_run
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    rec = fut.result()
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    rec = {"prompt_id": idx, "prompt": PROMPTS[idx - 1]}
                    rec.update({c: f"[ERROR] {exc}" for c in MODEL_DISPATCH})

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(
                    f"[{idx:03}/{total}] completed "
                    f"({len(done_ids)+list(future_to_idx).index(fut)+1}/{total})"
                )

    print(
        f"\nAll prompts processed in {time.time() - start_run:5.1f}s. "
        f"Dataset ➜ {out_path}"
    )


if __name__ == "__main__":
    main()