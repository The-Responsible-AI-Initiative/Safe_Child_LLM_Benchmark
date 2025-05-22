import json
import os
from collections.abc import Callable, Iterable
from datetime import date, datetime
from typing import Any, Literal

import torch

import kidsafetybench as ksb
from kidsafetybench.artifact import KidSafetyArtifact, KidSafetyParameters, KidSafetyInfo
from kidsafetybench.classifier import KidSafetyJailbreakJudge
from kidsafetybench.config import (
    API_KEYS,
    LOGS_PATH,
    NUM_KIDSAFETY_DEFAULT,
    Model
)
from kidsafetybench.defenses import DEFENSES
from kidsafetybench.llm.litellm import LLMLiteLLM
from kidsafetybench.llm.llm_output import LLMOutput
from kidsafetybench.llm.llm_wrapper import LLM

SUBMISSIONS_PATH = "submissions"


def check_eval_logs_empty(submitted_models: Iterable[str]):
    nonempty_logs = []
    for model in submitted_models:
        log_evals_path = os.path.join(LOGS_PATH, "eval", model)
        if os.path.exists(log_evals_path) and os.listdir(log_evals_path):
            nonempty_logs.append(log_evals_path)
        else:
            os.makedirs(log_evals_path, exist_ok=True)
    if nonempty_logs:
        raise ValueError(
            f"Warning, these evaluation log directories are not empty: {nonempty_logs}."
            "Please delete or rename them before running a fresh evaluation."
        )


def evaluate_prompts(
    prompts_dict: dict[str, dict[str, str]],
    llm_provider: Literal["vllm", "litellm"] = "litellm",
    defense: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate the given dict of model_name -> {behavior -> prompt} pairs.
    Returns a nested dictionary with the raw prompts, responses, and classification results.
    """
    submitted_models = set(prompts_dict.keys())
    # If you always expect 100 prompts (the default dataset size):
    for model_name, behaviors_dict in prompts_dict.items():
        if len(behaviors_dict) != NUM_KIDSAFETY_DEFAULT:
            raise ValueError(
                f"Model {model_name} has {len(behaviors_dict)} prompts, but expected {NUM_KIDSAFETY_DEFAULT}."
            )

    check_eval_logs_empty(submitted_models)

    # Decide which LLM wrapper to use
    if llm_provider == "vllm":
        try:
            from kidsafetybench.llm.vllm import LLMvLLM
        except ImportError:
            raise ImportError("vllm not installed. Please install vllm or use llm_provider='litellm'.")
        llm_constructor = lambda model_name: LLMvLLM(model_name=model_name)
    elif llm_provider == "litellm":
        llm_constructor = lambda model_name: LLMLiteLLM(
            model_name=model_name, api_key=os.environ[API_KEYS[Model(model_name)]]
        )
    else:
        raise ValueError(f"Invalid llm_provider: {llm_provider}.")

    if defense is not None and defense not in DEFENSES:
        raise ValueError(f"Invalid defense: {defense}. Must be one of {DEFENSES}.")

    evaluation = {}
    for model_name, behaviors_dict in prompts_dict.items():
        llm = llm_constructor(model_name)
        model_dict = _generate_full_model_dict(
            model_name=model_name, llm=llm, behavior_prompt_dict=behaviors_dict, defense=defense
        )
        evaluation[model_name] = model_dict
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    evaluation["details"] = {
        "llm_provider": llm_provider,
        "evaluation_date": datetime.now().date().isoformat(),
        "defense": defense
    }

    os.makedirs(SUBMISSIONS_PATH, exist_ok=True)
    evaluation_path = os.path.join(SUBMISSIONS_PATH, "evaluation.json")
    with open(evaluation_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=4)
        print(f"Evaluation results saved at {evaluation_path}.")
    return evaluation


def _generate_full_model_dict(
    model_name: str,
    llm: LLM,
    behavior_prompt_dict: dict[str, str],
    defense: str | None
) -> dict:
    valid_behaviors = []
    valid_prompts = []
    for beh, pr in behavior_prompt_dict.items():
        # skip if missing
        if pr is not None:
            valid_behaviors.append(beh)
            valid_prompts.append(pr)

    print(f"Querying responses for {model_name} with {len(valid_behaviors)} prompts.")
    outputs = llm._get_responses(valid_prompts, max_new_tokens=None, defense=defense)
    responses = outputs.responses

    # Log to “eval”
    for i, beh in enumerate(valid_behaviors):
        cur_output = LLMOutput(
            responses=[responses[i]],
            prompt_tokens=[outputs.prompt_tokens[i]],
            completion_tokens=[outputs.completion_tokens[i]],
        )
        llm.log_responses(
            behavior=beh,
            prompts=[valid_prompts[i]],
            llm_output=cur_output,
            phase="eval",
        )

    print(f"Classifying responses for {model_name}...")
    # use whichever classifier you want
    classifier = KidSafetyJailbreakJudge(os.environ.get("TOGETHER_API_KEY", ""))
    classifications = classifier(valid_prompts, responses)

    out_dict = {}
    for i, beh in enumerate(valid_behaviors):
        out_dict[beh] = {
            "prompt": valid_prompts[i],
            "response": responses[i],
            "violates_safety": classifications[i],
        }

    # For behaviors that had None prompts
    for beh, pr in behavior_prompt_dict.items():
        if pr is None:
            out_dict[beh] = {"prompt": None, "response": None, "violates_safety": False}

    total_violations = sum(classifications)
    print(
        f"Total safety violations for {model_name}: {total_violations}/{len(valid_behaviors)}"
    )
    return out_dict


def create_submission(
    method_name: str,
    safety_type: str,
    method_params: dict[str, Any] | None = None,
    default_values: dict[str, dict[str, int]] = {},
) -> dict:
    """
    Summarizes logs + evaluation.json into a final “artifact” per model.
    """
    if not os.path.exists(SUBMISSIONS_PATH):
        os.makedirs(SUBMISSIONS_PATH)
    evaluation_file = os.path.join(SUBMISSIONS_PATH, "evaluation.json")
    if not os.path.exists(evaluation_file):
        raise ValueError("No evaluation.json found. Please run evaluate_prompts first.")

    with open(evaluation_file, "r", encoding="utf-8") as f:
        evaluation = json.load(f)

    # Separate out details
    eval_details = evaluation["details"]
    submitted_models = [m for m in evaluation.keys() if m != "details"]

    # Aggregate logs
    test_logs = _aggregate_logs(os.path.join(LOGS_PATH, "test"), submitted_models)
    eval_logs = _aggregate_logs(os.path.join(LOGS_PATH, "eval"), submitted_models)

    model_summaries = _compute_summary_stats(evaluation, test_logs, default_values)

    submission = {
        "summaries": model_summaries,
        "test_logs": test_logs,
        "eval_logs": eval_logs
    }

    submission_file = os.path.join(SUBMISSIONS_PATH, "submission.json")
    with open(submission_file, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=4)
        print(f"Submission file created at {submission_file}.")

    _create_artifacts(model_summaries, method_name, safety_type, method_params, eval_details)

    return submission


def _create_artifacts(model_summaries, method_name, safety_type, method_params, eval_details):
    for model_name, data in model_summaries.items():
        # Build parameters
        params = KidSafetyParameters(
            method=method_name,
            model=model_name,
            defense=eval_details["defense"],
            safety_type=safety_type,
            safety_violation_rate=data["model_summary"]["attack_success_rate"],  # rename as needed
            total_number_of_violations=data["model_summary"]["total_number_of_violations"],
            number_of_submitted_prompts=data["model_summary"]["number_of_submitted_prompts"],
            total_number_of_queries=data["model_summary"]["total_number_of_queries"],
            total_prompt_tokens=data["model_summary"]["total_prompt_tokens"],
            total_response_tokens=data["model_summary"]["total_response_tokens"],
            evaluation_date=date.fromisoformat(eval_details["evaluation_date"]),
            evaluation_llm_provider=eval_details["llm_provider"],
            method_parameters=method_params or {},
        )

        # Wrap each row in a KidSafetyInfo
        results = data["violations_list"]

        artifact = KidSafetyArtifact(parameters=params, results=results)

        file_path = os.path.join(SUBMISSIONS_PATH, f"{model_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=4, default=_artifact_encoder)
            print(f"KidSafety artifact for {model_name} created at {file_path}.")


def _compute_summary_stats(evaluation: dict, test_logs: dict, default_values: dict[str, dict[str, int]]):
    summary_dict = {}
    # skip the “details” key
    for model_name in [m for m in evaluation if m != "details"]:
        # gather data for that model
        model_eval = evaluation[model_name]
        # example building a summary
        summary_dict[model_name] = _compute_model_summary(model_name, model_eval, test_logs.get(model_name, {}), default_values.get(model_name, {}))
    return summary_dict


def _compute_model_summary(
    model_name: str,
    model_eval: dict,   # behavior => {prompt, response, violates_safety}
    log_dict: dict,
    default_vals: dict[str, int]
):
    # For example, load your dataset:
    ds = ksb.read_dataset().as_dataframe()
    total_violations = 0
    submitted_prompts = 0

    total_prompt_tokens = 0
    total_response_tokens = 0
    total_queries_used = 0

    violations_list = []

    for idx, row in ds.iterrows():
        goal = row["goal"]
        category = row["category"]
        behavior = row["behavior"]

        if behavior not in model_eval:
            raise ValueError(f"Behavior {behavior} not in model evaluation for {model_name}.")
        record = model_eval[behavior]
        prompt = record["prompt"]
        response = record["response"]
        violates = record["violates_safety"]

        if prompt is not None:
            submitted_prompts += 1
        if violates:
            total_violations += 1

        # optionally parse logs if you track queries/prompt_tokens/response_tokens
        queries = 0
        prompt_tokens = 0
        response_tokens = 0
        # ... fill from logs or your defaults

        # For demonstration, we just fill from defaults if provided:
        if "number_of_queries" in default_vals:
            queries = default_vals["number_of_queries"]
        if "prompt_tokens" in default_vals:
            prompt_tokens = default_vals["prompt_tokens"]
        if "response_tokens" in default_vals:
            response_tokens = default_vals["response_tokens"]

        total_queries_used += queries
        total_prompt_tokens += prompt_tokens
        total_response_tokens += response_tokens

        # build the KidSafetyInfo
        info = KidSafetyInfo(
            index=idx,
            goal=goal,
            behavior=behavior,
            category=category,
            prompt=prompt,
            response=response,
            number_of_queries=queries if queries > 0 else None,
            queries_to_jailbreak=None,  # or set if you track the exact query that triggered violation
            prompt_tokens=prompt_tokens if prompt_tokens > 0 else None,
            response_tokens=response_tokens if response_tokens > 0 else None,
            violates_safety=violates
        )
        violations_list.append(info)

    model_summary = {
        "total_number_of_violations": total_violations,
        "attack_success_rate": total_violations / len(ds),
        "number_of_submitted_prompts": submitted_prompts,
        "total_prompt_tokens": total_prompt_tokens or None,
        "total_response_tokens": total_response_tokens or None,
        "total_number_of_queries": total_queries_used or None
    }
    return {"model_summary": model_summary, "violations_list": violations_list}


def _artifact_encoder(obj):
    """JSON encoder helper for artifact objects."""
    if isinstance(obj, (KidSafetyInfo, KidSafetyParameters, KidSafetyArtifact)):
        return obj.model_dump()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def _aggregate_logs(path: str, submitted_models: list[str]) -> dict:
    """Aggregate logs from the given path, if any exist."""
    out = {}
    if os.path.isdir(path):
        for model_name in os.listdir(path):
            if model_name in submitted_models:
                model_logs = {}
                model_dir = os.path.join(path, model_name)
                for file in os.listdir(model_dir):
                    if file.endswith(".json"):
                        behavior_name = file.replace(".json", "")
                        with open(os.path.join(model_dir, file), "r", encoding="utf-8") as f:
                            model_logs[behavior_name] = json.load(f)
                out[model_name] = model_logs
    return out