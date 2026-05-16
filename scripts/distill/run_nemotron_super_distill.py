from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from scripts.distill.common import (
    DEFAULT_DISTILL_DIR,
    build_parallel_thinking_prompt,
    build_parallel_thinking_student_prompt,
    build_teacher_prompt,
    extract_json_object,
    has_answer_value,
    infer_answer_format,
    infer_symbol_subtype,
    load_with_family,
    normalize_answer,
    render_parallel_thinking_response,
    validate_parallel_thinking_response,
)


DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_FREE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_TITLE = "Nemotron Super Distillation Runner"

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
STOP_STATUS_CODES = {401, 402, 403}

PROVIDER_PROFILES: dict[str, dict[str, object]] = {
    "nvidia": {
        "base_url": DEFAULT_NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "model": DEFAULT_NVIDIA_MODEL,
        "rpm_limit": 36.0,
        "daily_request_budget": 0,
    },
    "openrouter-free": {
        "base_url": DEFAULT_OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
        "model": DEFAULT_OPENROUTER_FREE_MODEL,
        "rpm_limit": 18.0,
        "daily_request_budget": 900,
    },
}


@dataclass(frozen=True)
class RuntimeConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    output_format: str
    temperature: float
    top_p: float | None
    max_tokens: int | None
    timeout_seconds: int
    json_mode: bool
    condition_on_gold: bool
    assistant_format: str
    branches: int
    max_retries: int
    rpm_limit: float
    workers: int
    daily_request_budget: int
    title: str | None
    http_referer: str | None


@dataclass(frozen=True)
class WorkItem:
    sample_id: str
    family: str
    prompt: str
    gold_answer: str | None
    candidate_index: int


class RateLimiter:
    def __init__(self, rpm_limit: float) -> None:
        if rpm_limit <= 0:
            raise ValueError("--rpm-limit must be positive.")
        self.interval_seconds = 60.0 / rpm_limit
        self.lock = threading.Lock()
        self.next_allowed_time = time.monotonic()

    def wait(self, stop_event: threading.Event) -> bool:
        with self.lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self.next_allowed_time - now)
            self.next_allowed_time = max(now, self.next_allowed_time) + self.interval_seconds

        if wait_seconds <= 0:
            return True
        return not stop_event.wait(wait_seconds)


class AttemptBudget:
    def __init__(self, daily_request_budget: int) -> None:
        self.limit = daily_request_budget
        self.used = 0
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            self.used += 1
            if self.limit > 0 and self.used > self.limit:
                return False
            return True

    def snapshot(self) -> int:
        with self.lock:
            return self.used


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-call Nemotron 3 Super through NVIDIA hosted NIM or OpenRouter "
            "for RFT/DPO/ORPO distillation data."
        )
    )
    parser.add_argument(
        "--provider",
        choices=["nvidia", "openrouter-free", "custom"],
        default="nvidia",
        help="Provider profile. Use custom with explicit --base-url/--model/--api-key-env.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Source CSV with prompt and optional answer columns.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Append-only output JSONL. Defaults to data/distill/<provider>_<format>_v1.jsonl.",
    )
    parser.add_argument(
        "--resume-against-jsonl",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional JSONL files to use for resume keys (status=ok only). "
            "This lets you write outputs to a new file while skipping already "
            "successful keys recorded elsewhere."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["teacher", "parallel"],
        default="teacher",
        help="teacher emits short supervision; parallel emits branch/consensus supervision.",
    )
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Repeatable family filter. Example: --family bit --family symbol",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens. Use 0 to omit max_tokens from the request.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument(
        "--condition-on-gold",
        action="store_true",
        help=(
            "Include the gold answer in the teacher prompt. Use only for "
            "answer-conditioned SFT bootstrap data, not unbiased rollouts."
        ),
    )
    parser.add_argument(
        "--branches",
        type=int,
        choices=range(2, 9),
        default=3,
        metavar="{2..8}",
        help="Number of branches for --output-format parallel.",
    )
    parser.add_argument(
        "--assistant-format",
        choices=["json", "nemotron-cot"],
        default="json",
        help="Assistant target format for --output-format parallel.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--rpm-limit",
        type=float,
        default=None,
        help="Request starts per minute. Defaults to provider profile.",
    )
    parser.add_argument(
        "--daily-request-budget",
        type=int,
        default=None,
        help="Maximum API attempts for this run. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=0.0,
        help="Stop dispatching new work after this many hours. Use 0 for unlimited.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip successful records already present in the output JSONL.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not skip successful records already present in the output JSONL.",
    )
    parser.add_argument("--http-referer", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print row/request estimates and exit without calling the API.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> RuntimeConfig:
    profile = PROVIDER_PROFILES.get(args.provider, {})
    base_url = args.base_url or profile.get("base_url")
    api_key_env = args.api_key_env or profile.get("api_key_env")
    model = args.model or profile.get("model")
    rpm_limit = args.rpm_limit if args.rpm_limit is not None else profile.get("rpm_limit")
    daily_budget = (
        args.daily_request_budget
        if args.daily_request_budget is not None
        else profile.get("daily_request_budget", 0)
    )
    title = args.title
    if title is None and args.provider == "openrouter-free":
        title = DEFAULT_TITLE

    missing = [
        name
        for name, value in (
            ("--base-url", base_url),
            ("--api-key-env", api_key_env),
            ("--model", model),
            ("--rpm-limit", rpm_limit),
        )
        if value in (None, "")
    ]
    if missing:
        raise ValueError(
            f"{args.provider} provider is missing required settings: {', '.join(missing)}"
        )
    if args.samples_per_row < 1:
        raise ValueError("--samples-per-row must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative.")

    return RuntimeConfig(
        provider=args.provider,
        base_url=str(base_url),
        api_key_env=str(api_key_env),
        model=str(model),
        output_format=args.output_format,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        timeout_seconds=args.timeout_seconds,
        json_mode=args.json_mode,
        condition_on_gold=args.condition_on_gold,
        assistant_format=args.assistant_format,
        branches=args.branches,
        max_retries=args.max_retries,
        rpm_limit=float(rpm_limit),
        workers=args.workers,
        daily_request_budget=int(daily_budget or 0),
        title=title,
        http_referer=args.http_referer,
    )


def load_api_key(env_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ValueError(f"Invalid API key environment variable name: {env_name}")

    api_key = os.environ.get(env_name, "").strip()
    if api_key:
        return api_key

    try:
        import subprocess

        api_key = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f'[System.Environment]::GetEnvironmentVariable("{env_name}", "User")',
            ],
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        api_key = ""

    return api_key


def output_path_for(args: argparse.Namespace, config: RuntimeConfig) -> Path:
    if args.output_jsonl is not None:
        return args.output_jsonl.resolve()
    provider_slug = config.provider.replace("-", "_")
    return (
        DEFAULT_DISTILL_DIR / f"{provider_slug}_{config.output_format}_v1.jsonl"
    ).resolve()


def existing_success_keys(output_jsonl: Path) -> set[tuple[str, str, str, str, int]]:
    if not output_jsonl.exists():
        return set()

    keys: set[tuple[str, str, str, str, int]] = set()
    for line in output_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            continue
        sample_id = payload.get("id")
        if sample_id is None:
            continue
        output_format = payload.get("output_format")
        if not isinstance(output_format, str):
            output_format = (
                "parallel" if isinstance(payload.get("parallel_thinking"), dict) else "teacher"
            )
        try:
            candidate_index = int(payload.get("candidate_index", 0))
        except (TypeError, ValueError):
            candidate_index = 0
        keys.add(
            (
                str(payload.get("provider", "")),
                str(payload.get("teacher_model", "")),
                output_format,
                str(sample_id),
                candidate_index,
            )
        )
    return keys


def work_key(config: RuntimeConfig, item: WorkItem) -> tuple[str, str, str, str, int]:
    return (
        config.provider,
        config.model,
        config.output_format,
        item.sample_id,
        item.candidate_index,
    )


def load_work_items(
    args: argparse.Namespace,
    config: RuntimeConfig,
    output_jsonl: Path,
) -> tuple[list[WorkItem], int]:
    input_csv = args.input_csv.resolve()
    dataframe = load_with_family(input_csv)
    if "prompt" not in dataframe.columns:
        raise RuntimeError(f"{input_csv} must contain a prompt column.")
    if args.condition_on_gold and "answer" not in dataframe.columns:
        raise RuntimeError("--condition-on-gold requires an input CSV with an answer column.")
    if "id" not in dataframe.columns:
        dataframe["id"] = [str(index) for index in range(len(dataframe))]

    if args.family:
        dataframe = dataframe[dataframe["family"].isin(set(args.family))].copy()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")

    dataframe = dataframe.sort_values("id").reset_index(drop=True)
    if args.num_shards > 1:
        shard_positions = [
            index for index in range(len(dataframe)) if index % args.num_shards == args.shard_index
        ]
        dataframe = dataframe.iloc[shard_positions].copy()
    if args.offset:
        dataframe = dataframe.iloc[args.offset :].copy()
    if args.limit:
        dataframe = dataframe.iloc[: args.limit].copy()

    existing_keys: set[tuple[str, str, str, str, int]] = set()
    if args.resume:
        existing_keys |= existing_success_keys(output_jsonl)
        for extra in args.resume_against_jsonl:
            try:
                existing_keys |= existing_success_keys(extra.resolve())
            except Exception:  # noqa: BLE001
                # Best-effort: resume should not fail if an auxiliary file is unreadable.
                continue
    work_items: list[WorkItem] = []
    for row in dataframe.itertuples(index=False):
        raw_gold = getattr(row, "answer", None)
        gold_answer = normalize_answer(raw_gold) if has_answer_value(raw_gold) else None
        for candidate_index in range(args.samples_per_row):
            item = WorkItem(
                sample_id=str(row.id),
                family=str(row.family),
                prompt=str(row.prompt),
                gold_answer=gold_answer,
                candidate_index=candidate_index,
            )
            if work_key(config, item) not in existing_keys:
                work_items.append(item)

    return work_items, len(dataframe)


def default_headers_for(config: RuntimeConfig) -> dict[str, str]:
    headers: dict[str, str] = {}
    if config.http_referer:
        headers["HTTP-Referer"] = config.http_referer
    if config.title:
        headers["X-OpenRouter-Title"] = config.title
    return headers


def build_prompt(
    item: WorkItem,
    config: RuntimeConfig,
) -> tuple[str, str | None, str | None, str | None]:
    if config.output_format == "teacher":
        prompt = build_teacher_prompt(item.prompt, item.family)
        if config.condition_on_gold and item.gold_answer is not None:
            prompt += (
                "\nGold answer for answer-conditioned rationale construction: "
                f"{item.gold_answer}\n"
                "- Use the gold answer only to verify and select a concise, supported path.\n"
                "- Do not invent unsupported rules just to fit the gold answer.\n"
            )
        return prompt, None, None, None

    answer_format = infer_answer_format(
        family=item.family,
        prompt=item.prompt,
        answer=item.gold_answer,
    )
    symbol_subtype = (
        infer_symbol_subtype(item.prompt, item.gold_answer)
        if item.family == "symbol"
        else None
    )
    prompt = build_parallel_thinking_prompt(
        prompt=item.prompt,
        family=item.family,
        branches=config.branches,
        answer_format=answer_format,
        symbol_subtype=symbol_subtype,
        gold_answer=item.gold_answer if config.condition_on_gold else None,
    )
    student_prompt = build_parallel_thinking_student_prompt(
        prompt=item.prompt,
        family=item.family,
        branches=config.branches,
        answer_format=answer_format,
        symbol_subtype=symbol_subtype,
    )
    return prompt, student_prompt, answer_format, symbol_subtype


def call_api_once(
    config: RuntimeConfig,
    item: WorkItem,
    prompt: str,
    api_key: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    system_content = (
        "You are a precise reasoning-data generator. Return only valid JSON "
        "that follows the requested schema."
        if config.output_format == "parallel"
        else "You are a precise reasoning teacher. Return only valid JSON."
    )
    payload: dict[str, object] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
        "timeout": config.timeout_seconds,
    }
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        payload["top_p"] = config.top_p
    if config.json_mode:
        payload["response_format"] = {"type": "json_object"}

    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        default_headers=default_headers_for(config) or None,
    )
    response = client.chat.completions.create(**payload)
    body = response.model_dump()
    raw_body = json.dumps(body, ensure_ascii=False)
    try:
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content")
        usage = body.get("usage", {})
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected response for id={item.sample_id}: {raw_body[:1200]}"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            f"API response had no parseable content for id={item.sample_id}: "
            f"{raw_body[:1200]}"
        )

    return extract_json_object(content), usage, content


def validate_teacher_response(response: dict[str, object]) -> None:
    # Only final_answer is hard-required for downstream scoring.
    # rule_summary/reasoning_brief/confidence are best-effort and may be repaired.
    if "final_answer" not in response:
        raise ValueError("Teacher response missing keys: ['final_answer']")
    if not normalize_answer(response["final_answer"]):
        raise ValueError("Teacher response has an empty final_answer.")


def coerce_teacher_response(response: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    missing: list[str] = []
    for key in ("rule_summary", "reasoning_brief", "confidence"):
        if key not in response:
            missing.append(key)

    rule_summary = response.get("rule_summary")
    if not isinstance(rule_summary, str):
        rule_summary = ""
        if "rule_summary" not in missing:
            missing.append("rule_summary")
    rule_summary = rule_summary.strip()

    reasoning_brief = response.get("reasoning_brief")
    if not isinstance(reasoning_brief, str):
        reasoning_brief = ""
        if "reasoning_brief" not in missing:
            missing.append("reasoning_brief")
    reasoning_brief = reasoning_brief.strip()

    if not reasoning_brief and rule_summary:
        reasoning_brief = rule_summary

    confidence_value = response.get("confidence")
    try:
        confidence = float(confidence_value)
        if math.isnan(confidence):
            raise ValueError("nan")
    except Exception:  # noqa: BLE001
        confidence = 0.5
        if "confidence" not in missing:
            missing.append("confidence")
    confidence = max(0.0, min(1.0, confidence))

    response = dict(response)
    response["rule_summary"] = rule_summary
    response["reasoning_brief"] = reasoning_brief
    response["confidence"] = confidence
    return response, sorted(set(missing))


def status_code_from_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if retry_after is None:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


def is_retryable_error(exc: Exception) -> bool:
    status_code = status_code_from_exception(exc)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    text = str(exc).lower()
    return any(token in text for token in ("timeout", "temporarily", "rate limit"))


def sleep_before_retry(
    exc: Exception,
    attempt: int,
    stop_event: threading.Event,
) -> bool:
    retry_after = retry_after_seconds(exc)
    if retry_after is None:
        retry_after = min(90.0, (2 ** min(attempt, 6)) + random.uniform(0.0, 1.5))
    return not stop_event.wait(retry_after)


def reward_fields(
    verified_answer_exact: bool | None,
    condition_on_gold: bool,
) -> tuple[float | None, str, str]:
    if verified_answer_exact is True:
        return (
            1.0,
            "sft_bootstrap_only" if condition_on_gold else "positive",
            "sft_bootstrap_only" if condition_on_gold else "sft_and_dapo_chosen",
        )
    if verified_answer_exact is False:
        return -1.0, "negative_answer", "dapo_rejected_only"
    return None, "unverified", "manual_review"


def success_record(
    item: WorkItem,
    config: RuntimeConfig,
    response: dict[str, object],
    usage: dict[str, object],
    raw_content: str,
    student_prompt: str | None,
    answer_format: str | None,
    symbol_subtype: str | None,
) -> dict[str, object]:
    teacher_answer = normalize_answer(response["final_answer"])
    verified_answer_exact = (
        teacher_answer == item.gold_answer if item.gold_answer is not None else None
    )
    reward, reward_label, use_for = reward_fields(
        verified_answer_exact=verified_answer_exact,
        condition_on_gold=config.condition_on_gold,
    )
    base_record: dict[str, object] = {
        "id": item.sample_id,
        "family": item.family,
        "prompt": item.prompt,
        "gold_answer": item.gold_answer,
        "provider": config.provider,
        "teacher_model": config.model,
        "output_format": config.output_format,
        "candidate_index": item.candidate_index,
        "teacher_answer": teacher_answer,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "json_mode": config.json_mode,
        "conditioning": (
            "gold_answer_provided" if config.condition_on_gold else "answer_hidden"
        ),
        "verified_answer_exact": verified_answer_exact,
        "reward": reward,
        "reward_label": reward_label,
        "use_for": use_for,
        "usage": usage,
        "raw_response_text": raw_content,
        "status": "ok",
        "generated_at": int(time.time()),
    }

    if config.output_format == "teacher":
        repaired = response.get("_repaired_missing_keys")
        if isinstance(repaired, list) and repaired:
            base_record["teacher_repaired"] = True
            base_record["teacher_missing_keys"] = repaired
        base_record.update(
            {
                "rule_summary": str(response["rule_summary"]).strip(),
                "reasoning_brief": str(response["reasoning_brief"]).strip(),
                "confidence": float(response["confidence"]),
            }
        )
        return base_record

    assistant_response = render_parallel_thinking_response(
        response,
        response_format=config.assistant_format,
    )
    branch_answers = [
        normalize_answer(branch["candidate_answer"])
        for branch in response["branches"]
        if isinstance(branch, dict)
    ]
    consensus = response["consensus"]
    agreement = consensus.get("agreement") if isinstance(consensus, dict) else None
    base_record.update(
        {
            "answer_format": answer_format,
            "symbol_subtype": symbol_subtype,
            "student_prompt": student_prompt,
            "parallel_thinking": response,
            "assistant_response": assistant_response,
            "messages": [
                {"role": "user", "content": student_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "branch_answers": branch_answers,
            "agreement": agreement,
            "confidence": float(response["confidence"]),
            "branches": config.branches,
            "assistant_format": config.assistant_format,
        }
    )
    return base_record


def failure_record(
    item: WorkItem,
    config: RuntimeConfig,
    error: str,
    status_code: int | None,
    attempts: int,
) -> dict[str, object]:
    return {
        "id": item.sample_id,
        "family": item.family,
        "prompt": item.prompt,
        "gold_answer": item.gold_answer,
        "provider": config.provider,
        "teacher_model": config.model,
        "output_format": config.output_format,
        "candidate_index": item.candidate_index,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "json_mode": config.json_mode,
        "conditioning": (
            "gold_answer_provided" if config.condition_on_gold else "answer_hidden"
        ),
        "attempts": attempts,
        "status_code": status_code,
        "reward": -1.0,
        "reward_label": "negative_format_or_api_error",
        "use_for": "debug_or_format_reward_only",
        "status": "error",
        "error": error,
        "generated_at": int(time.time()),
    }


def run_one_item(
    item: WorkItem,
    config: RuntimeConfig,
    api_key: str,
    limiter: RateLimiter,
    budget: AttemptBudget,
    stop_event: threading.Event,
) -> dict[str, object]:
    prompt, student_prompt, answer_format, symbol_subtype = build_prompt(item, config)
    last_error = ""
    last_status_code: int | None = None
    attempts = 0

    for attempt in range(1, config.max_retries + 2):
        if stop_event.is_set():
            return {"_skip_write": True, "stop_reason": "stopped"}
        if not budget.acquire():
            stop_event.set()
            return {"_skip_write": True, "stop_reason": "daily_request_budget_exhausted"}
        if not limiter.wait(stop_event):
            return {"_skip_write": True, "stop_reason": "stopped"}

        attempts = attempt
        try:
            response, usage, raw_content = call_api_once(
                config=config,
                item=item,
                prompt=prompt,
                api_key=api_key,
            )
            if config.output_format == "teacher":
                response, missing = coerce_teacher_response(response)
                if missing:
                    response["_repaired_missing_keys"] = missing
                validate_teacher_response(response)
            else:
                validate_parallel_thinking_response(
                    response=response,
                    expected_branches=config.branches,
                )
            return success_record(
                item=item,
                config=config,
                response=response,
                usage=usage,
                raw_content=raw_content,
                student_prompt=student_prompt,
                answer_format=answer_format,
                symbol_subtype=symbol_subtype,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            last_status_code = status_code_from_exception(exc)
            if last_status_code in STOP_STATUS_CODES:
                stop_event.set()
                break
            if attempt > config.max_retries or not is_retryable_error(exc):
                break
            if not sleep_before_retry(exc, attempt, stop_event):
                return {"_skip_write": True, "stop_reason": "stopped"}

    return failure_record(
        item=item,
        config=config,
        error=last_error,
        status_code=last_status_code,
        attempts=attempts,
    )


def print_first_prompt(work_items: list[WorkItem], config: RuntimeConfig) -> None:
    if not work_items:
        print("No pending work items.")
        return
    prompt, student_prompt, answer_format, symbol_subtype = build_prompt(
        work_items[0],
        config,
    )
    print("--- resolved config ---")
    print(f"provider:        {config.provider}")
    print(f"base_url:        {config.base_url}")
    print(f"api_key_env:     {config.api_key_env}")
    print(f"model:           {config.model}")
    print(f"output_format:   {config.output_format}")
    print(f"rpm_limit:       {config.rpm_limit}")
    print(f"workers:         {config.workers}")
    print(f"daily_budget:    {config.daily_request_budget or 'unlimited'}")
    print(f"json_mode:       {config.json_mode}")
    print(f"answer_format:   {answer_format}")
    print(f"symbol_subtype:  {symbol_subtype}")
    print("\n--- teacher prompt ---\n")
    print(prompt)
    if student_prompt:
        print("\n--- student prompt ---\n")
        print(student_prompt)


def print_estimate(
    selected_rows: int,
    work_items: list[WorkItem],
    config: RuntimeConfig,
    max_runtime_hours: float,
) -> None:
    total_pending = len(work_items)
    request_cap = (
        config.daily_request_budget if config.daily_request_budget > 0 else total_pending
    )
    runtime_cap = total_pending
    if max_runtime_hours > 0:
        runtime_cap = math.floor(max_runtime_hours * 60.0 * config.rpm_limit)
    dispatchable = min(total_pending, request_cap, runtime_cap)
    theoretical_minutes = dispatchable / config.rpm_limit if config.rpm_limit else 0.0
    print(f"Selected source rows:         {selected_rows}")
    print(f"Pending candidate requests:   {total_pending}")
    print(f"Provider:                     {config.provider}")
    print(f"Model:                        {config.model}")
    print(f"RPM limit:                    {config.rpm_limit}")
    print(f"Workers:                      {config.workers}")
    print(f"Daily request budget:         {config.daily_request_budget or 'unlimited'}")
    print(f"Max runtime hours:            {max_runtime_hours or 'unlimited'}")
    print(f"Dispatchable this run:        {dispatchable}")
    print(f"Theoretical minimum runtime:  {theoretical_minutes:.1f} minutes")


def run_batch(
    work_items: list[WorkItem],
    output_jsonl: Path,
    config: RuntimeConfig,
    api_key: str,
    max_runtime_hours: float,
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(config.rpm_limit)
    budget = AttemptBudget(config.daily_request_budget)
    stop_event = threading.Event()
    start_time = time.monotonic()
    max_runtime_seconds = max_runtime_hours * 3600.0 if max_runtime_hours > 0 else 0.0
    iterator = iter(work_items)
    in_flight: set[concurrent.futures.Future[dict[str, object]]] = set()
    submitted = 0
    written = 0
    errors = 0
    stop_reason = ""

    def runtime_exceeded() -> bool:
        return bool(
            max_runtime_seconds and time.monotonic() - start_time >= max_runtime_seconds
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
        with output_jsonl.open("a", encoding="utf-8") as handle:
            while True:
                while (
                    len(in_flight) < config.workers
                    and not stop_event.is_set()
                    and not runtime_exceeded()
                ):
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    future = executor.submit(
                        run_one_item,
                        item,
                        config,
                        api_key,
                        limiter,
                        budget,
                        stop_event,
                    )
                    in_flight.add(future)
                    submitted += 1

                if not in_flight:
                    if runtime_exceeded():
                        stop_reason = "max_runtime_exceeded"
                    elif stop_event.is_set() and not stop_reason:
                        stop_reason = "stopped"
                    break

                done, in_flight = concurrent.futures.wait(
                    in_flight,
                    timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    record = future.result()
                    if record.get("_skip_write"):
                        stop_reason = str(record.get("stop_reason") or "stopped")
                        continue
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    written += 1
                    if record.get("status") == "error":
                        errors += 1
                    print(
                        f"[written={written} submitted={submitted} attempts={budget.snapshot()}] "
                        f"id={record.get('id')} cand={record.get('candidate_index')} "
                        f"family={record.get('family')} status={record.get('status')} "
                        f"verified={record.get('verified_answer_exact')}"
                    )
                    if record.get("status_code") in STOP_STATUS_CODES:
                        stop_reason = f"stop_status_{record.get('status_code')}"

                if runtime_exceeded() and not stop_reason:
                    stop_reason = "max_runtime_exceeded"
                if stop_reason:
                    stop_event.set()

    print("--- run summary ---")
    print(f"Submitted work items: {submitted}")
    print(f"API attempts:         {budget.snapshot()}")
    print(f"Records written:      {written}")
    print(f"Errors written:       {errors}")
    print(f"Stop reason:          {stop_reason or 'completed'}")
    print(f"Output JSONL:         {output_jsonl}")


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    output_jsonl = output_path_for(args, config)
    work_items, selected_rows = load_work_items(args, config, output_jsonl)

    print_estimate(
        selected_rows=selected_rows,
        work_items=work_items,
        config=config,
        max_runtime_hours=args.max_runtime_hours,
    )
    print(f"Output JSONL:                  {output_jsonl}")

    if args.dry_run:
        print_first_prompt(work_items, config)
        return
    if args.estimate_only:
        return
    if not work_items:
        print("No pending work items.")
        return

    api_key = load_api_key(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{config.api_key_env} is not set. Export it before running this script."
        )

    run_batch(
        work_items=work_items,
        output_jsonl=output_jsonl,
        config=config,
        api_key=api_key,
        max_runtime_hours=args.max_runtime_hours,
    )


if __name__ == "__main__":
    main()
