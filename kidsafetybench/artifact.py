import datetime
import json
import os
from pathlib import Path
from typing import Annotated, TypeVar

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kidsafetybench.config import KidSafetyType, Model, parse_model_name_safety_type

# Name your local artifacts dir
artifacts_dir = "kidsafety_artifacts"


class KidSafetyInfo(BaseModel):
    """Information about a single prompt/response pair in the kidsafety test."""
    index: int
    goal: str
    behavior: str
    category: str
    prompt: str | None
    response: str | None
    number_of_queries: int | None
    queries_to_jailbreak: int | None
    prompt_tokens: int | None
    response_tokens: int | None
    # jailbroken => “unsafe” or “violates_kid_safety”? Keep whichever naming you like.
    violates_safety: bool


class KidSafetyParameters(BaseModel):
    """Parameters describing the high-level testing method and results summary."""
    model_config = ConfigDict(use_enum_values=True)
    method: str
    model: Model
    safety_type: KidSafetyType | None  # analogous to AttackType
    defense: str | None
    safety_violation_rate: Annotated[float, Field(strict=True, ge=0, le=1)]
    total_number_of_violations: int
    number_of_submitted_prompts: int
    total_number_of_queries: int | None
    total_prompt_tokens: int | None
    total_response_tokens: int | None
    evaluation_date: datetime.date = Field(default_factory=datetime.date.today)
    evaluation_llm_provider: str
    method_parameters: dict[str, str | int | float]


class KidSafetyArtifact(BaseModel):
    """The top-level artifact, combining parameters with per-prompt results."""
    parameters: KidSafetyParameters
    results: list[KidSafetyInfo]


class _KidSafetyInfo(BaseModel):
    models: dict[Model, set[KidSafetyType]]


BASE_URL = "https://raw.githubusercontent.com"
REPO_NAME = "KidSafetyBench/artifacts"   # (if you host them similarly)
BRANCH = "main"
ARTIFACTS_PATH = "kidsafety-artifacts"
INFO_FILENAME = "kidsafety-info.json"


def _make_attack_base_url(method_name: str) -> str:
    return f"{BASE_URL}/{REPO_NAME}/{BRANCH}/{ARTIFACTS_PATH}/{method_name}"

def _make_artifact_url(method_name: str, model: Model, safety_type: KidSafetyType) -> str:
    return f"{_make_attack_base_url(method_name)}/{safety_type.value}/{model.value}.json"

def _make_attack_info_url(method_name: str) -> str:
    return f"{_make_attack_base_url(method_name)}/{INFO_FILENAME}"


T = TypeVar("T", bound=BaseModel)


def _get_file_from_github(url: str, pydantic_model: type[T]) -> T:
    response = requests.get(url)
    if response.status_code == 404:
        raise ValueError(f"Error 404, file not found at URL: {url}.")
    response.raise_for_status()
    return pydantic_model.model_validate(response.json())


KSB_ATTACK_CACHE_BASE_PATH = Path("kidsafetybench") / "kidsafety-artifacts"


def _resolve_cache_path(custom_cache: Path | None) -> Path:
    if custom_cache is not None:
        return custom_cache
    if "KSB_CACHE" in os.environ:
        return Path(os.environ["KSB_CACHE"])
    return Path.home() / ".cache"


def _get_file_from_cache(sub_path: Path, pydantic_model: type[T], custom_cache_dir: Path | None = None) -> T:
    cache_base_path = _resolve_cache_path(custom_cache_dir)
    filepath = cache_base_path / KSB_ATTACK_CACHE_BASE_PATH / sub_path
    with filepath.open("r") as f:
        return json.load(f)


def _cache_file(sub_path: Path, model: BaseModel, custom_cache_dir: Path | None = None) -> None:
    cache_base_path = _resolve_cache_path(custom_cache_dir)
    filepath = cache_base_path / KSB_ATTACK_CACHE_BASE_PATH / sub_path
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        f.write(model.model_dump_json())


def _get_or_download_file(
    url: str,
    cache_sub_path: Path,
    pydantic_model: type[T],
    custom_cache_dir: Path | None,
    force_download: bool
) -> T:
    try:
        cached_file = pydantic_model.model_validate(
            _get_file_from_cache(cache_sub_path, pydantic_model, custom_cache_dir)
        )
    except (FileNotFoundError, ValidationError):
        cached_file = None

    if cached_file is None or force_download:
        file = _get_file_from_github(url, pydantic_model)
        _cache_file(cache_sub_path, file, custom_cache_dir)
    else:
        file = cached_file
    return file


def _get_or_download_attack_info(method_name: str, custom_cache_dir: Path | None, force_download: bool) -> _KidSafetyInfo:
    url = _make_attack_info_url(method_name)
    cache_sub_path = Path(method_name) / f"{INFO_FILENAME}"
    return _get_or_download_file(url, cache_sub_path, _KidSafetyInfo, custom_cache_dir, force_download)


def _get_or_download_artifact(
    method_name: str, model: Model, safety_type: KidSafetyType, custom_cache_dir: Path | None, force_download: bool
) -> KidSafetyArtifact:
    url = _make_artifact_url(method_name, model, safety_type)
    cache_sub_path = Path(method_name) / safety_type.value / f"{model.value}.json"
    return _get_or_download_file(url, cache_sub_path, KidSafetyArtifact, custom_cache_dir, force_download)


def read_artifact(
    method: str,
    model_name: str,
    safety_type: str | None = None,
    custom_cache_dir: Path | None = None,
    force_download: bool = False,
) -> KidSafetyArtifact:
    model, type_enum_candidate = parse_model_name_safety_type(model_name, safety_type)
    info = _get_or_download_attack_info(method, custom_cache_dir, force_download)

    if model not in info.models:
        raise ValueError(f"Model {model.value} not found for method {method}.")

    if type_enum_candidate is not None:
        if type_enum_candidate not in info.models[model]:
            raise ValueError(f"Safety type {type_enum_candidate.value} not found for model {model.value}.")
        final_safety_type = type_enum_candidate
    else:
        # If multiple categories exist, user must specify
        if len(info.models[model]) > 1:
            raise ValueError(
                f"Multiple safety types found for model {model.value}. "
                f"Please specify one of the following: {info.models[model]}."
            )
        final_safety_type = list(info.models[model]).pop()

    artifact = _get_or_download_artifact(method, model, final_safety_type, custom_cache_dir, force_download)
    return artifact
