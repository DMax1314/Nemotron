from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from scripts.distill.common import (
    DEFAULT_DISTILL_DIR,
    build_parallel_thinking_prompt,
    build_parallel_thinking_student_prompt,
    extract_json_object,
    infer_answer_format,
    infer_symbol_subtype,
    load_with_family,
    normalize_answer,
    render_parallel_thinking_response,
    validate_parallel_thinking_response,
)


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_TITLE = "Nemotron Parallel Thinking Distillation Runner"


def load_existing_ids(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()

    existing_ids: set[str] = set()
    for line in output_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("status") == "ok":
            existing_ids.add(str(payload["id"]))
    return existing_ids


def load_api_key(env_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ValueError(f"Invalid API key environment variable name: {env_name}")

    api_key = os.environ.get(env_name, "").strip()
    if api_key:
        return api_key

    try:
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


def call_api_once(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    timeout_seconds: int,
    json_mode: bool,
) -> tuple[dict[str, object], dict[str, object], str]:
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise reasoning-data generator. Return only valid "
                    "JSON that follows the requested schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "timeout": timeout_seconds,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**payload)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"API request failed: {exc}") from exc

    body = response.model_dump()
    raw_body = json.dumps(body, ensure_ascii=False)
    try:
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content")
        usage = body.get("usage", {})
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {raw_body[:1200]}") from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"API response had no parseable content: {raw_body[:1200]}")

    parsed = extract_json_object(content)
    return parsed, usage, content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call an OpenAI-compatible API to generate parallel-thinking "
            "distillation data."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Source CSV with prompt and optional answer columns.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_DISTILL_DIR / "parallel_thinking_teacher_v1.jsonl",
        help="Append-only output JSONL file.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TEACHER_MODEL", DEFAULT_MODEL),
        help="Teacher model name or provider-specific model slug.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument(
        "--branches",
        type=int,
        choices=range(2, 9),
        default=3,
        metavar="{2..8}",
        help="Number of parallel-thinking branches to request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature. Recommended default for distillation is 0.2.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling parameter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Optional max_tokens override. Use 0 to omit it from the API payload.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Request provider JSON mode via response_format=json_object.",
    )
    parser.add_argument(
        "--condition-on-gold",
        action="store_true",
        help=(
            "Include the gold answer in the teacher prompt. Use this only for "
            "answer-conditioned rationale generation / SFT bootstrap data, not "
            "for unbiased rollout reward estimates."
        ),
    )
    parser.add_argument(
        "--assistant-format",
        choices=["json", "nemotron-cot"],
        default="json",
        help=(
            "Stored assistant target format. Use json for DAPO data plumbing, "
            "or nemotron-cot for Nemotron-style SFT targets ending in boxed answer."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Repeatable family filter. Example: --family symbol --family bit",
    )
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip ids that already exist in the output JSONL.",
    )
    parser.add_argument(
        "--http-referer",
        default=None,
        help="Optional OpenRouter attribution header.",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Optional OpenRouter title header.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first teacher prompt and student prompt, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_jsonl = args.output_jsonl.resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    dataframe = load_with_family(input_csv)
    if args.family:
        dataframe = dataframe[dataframe["family"].isin(set(args.family))].copy()

    dataframe = dataframe.sort_values("id").reset_index(drop=True)
    if args.offset:
        dataframe = dataframe.iloc[args.offset :].copy()
    if args.limit:
        dataframe = dataframe.iloc[: args.limit].copy()

    existing_ids = load_existing_ids(output_jsonl) if args.resume else set()
    pending_rows = dataframe[~dataframe["id"].astype(str).isin(existing_ids)].copy()
    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    if args.condition_on_gold and "answer" not in pending_rows.columns:
        raise RuntimeError("--condition-on-gold requires an input CSV with an answer column.")

    print(f"Loaded rows after filtering: {len(dataframe)}")
    print(f"Pending rows after resume:   {len(pending_rows)}")
    print(f"Output JSONL:                {output_jsonl}")
    print(f"Teacher model:               {args.model}")
    print(f"Base URL:                    {args.base_url}")
    print(f"Branches:                    {args.branches}")
    print(f"Temperature:                 {args.temperature}")
    print(f"Top-p:                       {args.top_p if args.top_p is not None else 'default'}")
    print(f"Max tokens:                  {max_tokens if max_tokens is not None else 'default'}")
    print(f"JSON mode:                   {args.json_mode}")
    print(f"Condition on gold:           {args.condition_on_gold}")
    print(f"Assistant format:            {args.assistant_format}")

    if pending_rows.empty:
        print("No pending rows left.")
        return

    if args.dry_run:
        first_row = pending_rows.iloc[0]
        gold_answer = str(first_row["answer"]) if "answer" in pending_rows.columns else None
        answer_format = infer_answer_format(
            family=first_row["family"],
            prompt=first_row["prompt"],
            answer=gold_answer,
        )
        symbol_subtype = (
            infer_symbol_subtype(first_row["prompt"], gold_answer)
            if first_row["family"] == "symbol"
            else None
        )
        teacher_prompt = build_parallel_thinking_prompt(
            prompt=first_row["prompt"],
            family=first_row["family"],
            branches=args.branches,
            answer_format=answer_format,
            symbol_subtype=symbol_subtype,
            gold_answer=gold_answer if args.condition_on_gold else None,
        )
        student_prompt = build_parallel_thinking_student_prompt(
            prompt=first_row["prompt"],
            family=first_row["family"],
            branches=args.branches,
            answer_format=answer_format,
            symbol_subtype=symbol_subtype,
        )
        print("\n--- teacher prompt ---\n")
        print(teacher_prompt)
        print("\n--- student prompt ---\n")
        print(student_prompt)
        return

    api_key = load_api_key(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{args.api_key_env} is not set. Export it before running this script."
        )

    default_headers: dict[str, str] = {}
    if args.http_referer:
        default_headers["HTTP-Referer"] = args.http_referer
    if args.title:
        default_headers["X-OpenRouter-Title"] = args.title

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        default_headers=default_headers or None,
    )

    with output_jsonl.open("a", encoding="utf-8") as handle:
        for row_index, row in enumerate(
            pending_rows.itertuples(index=False), start=1
        ):
            gold_answer = str(row.answer) if hasattr(row, "answer") else None
            answer_format = infer_answer_format(
                family=row.family,
                prompt=row.prompt,
                answer=gold_answer,
            )
            symbol_subtype = (
                infer_symbol_subtype(row.prompt, gold_answer)
                if row.family == "symbol"
                else None
            )
            teacher_prompt = build_parallel_thinking_prompt(
                prompt=row.prompt,
                family=row.family,
                branches=args.branches,
                answer_format=answer_format,
                symbol_subtype=symbol_subtype,
                gold_answer=gold_answer if args.condition_on_gold else None,
            )
            student_prompt = build_parallel_thinking_student_prompt(
                prompt=row.prompt,
                family=row.family,
                branches=args.branches,
                answer_format=answer_format,
                symbol_subtype=symbol_subtype,
            )
            last_error: str | None = None

            for attempt in range(1, args.max_retries + 2):
                try:
                    response, usage, raw_content = call_api_once(
                        client=client,
                        model=args.model,
                        prompt=teacher_prompt,
                        temperature=args.temperature,
                        max_tokens=max_tokens,
                        top_p=args.top_p,
                        timeout_seconds=args.timeout_seconds,
                        json_mode=args.json_mode,
                    )
                    validate_parallel_thinking_response(
                        response=response,
                        expected_branches=args.branches,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    time.sleep(min(attempt, 3))
            else:
                failure_record = {
                    "id": row.id,
                    "family": row.family,
                    "answer_format": answer_format,
                    "symbol_subtype": symbol_subtype,
                    "prompt": row.prompt,
                    "gold_answer": gold_answer,
                    "teacher_model": args.model,
                    "branches": args.branches,
                    "conditioning": (
                        "gold_answer_provided"
                        if args.condition_on_gold
                        else "answer_hidden"
                    ),
                    "assistant_format": args.assistant_format,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "json_mode": args.json_mode,
                    "reward": -1.0,
                    "reward_label": "negative_format_or_api_error",
                    "use_for": "debug_or_format_reward_only",
                    "status": "error",
                    "error": last_error,
                }
                handle.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{row_index}/{len(pending_rows)}] "
                    f"id={row.id} family={row.family} error"
                )
                continue

            teacher_answer = normalize_answer(response["final_answer"])
            gold_answer = normalize_answer(gold_answer) if gold_answer is not None else None
            assistant_response = render_parallel_thinking_response(
                response,
                response_format=args.assistant_format,
            )
            branch_answers = [
                normalize_answer(branch["candidate_answer"])
                for branch in response["branches"]
                if isinstance(branch, dict)
            ]
            consensus = response["consensus"]
            agreement = (
                consensus.get("agreement")
                if isinstance(consensus, dict)
                else None
            )
            verified_answer_exact = (
                teacher_answer == gold_answer if gold_answer is not None else None
            )
            if verified_answer_exact is True:
                reward = 1.0
                reward_label = "positive"
                use_for = (
                    "sft_bootstrap_only"
                    if args.condition_on_gold
                    else "sft_and_dapo_chosen"
                )
            elif verified_answer_exact is False:
                reward = -1.0
                reward_label = "negative_answer"
                use_for = "dapo_rejected_only"
            else:
                reward = None
                reward_label = "unverified"
                use_for = "manual_review"

            record = {
                "id": row.id,
                "family": row.family,
                "answer_format": answer_format,
                "symbol_subtype": symbol_subtype,
                "prompt": row.prompt,
                "student_prompt": student_prompt,
                "gold_answer": gold_answer,
                "teacher_model": args.model,
                "teacher_answer": teacher_answer,
                "parallel_thinking": response,
                "assistant_response": assistant_response,
                "messages": [
                    {"role": "user", "content": student_prompt},
                    {"role": "assistant", "content": assistant_response},
                ],
                "branch_answers": branch_answers,
                "agreement": agreement,
                "confidence": float(response["confidence"]),
                "branches": args.branches,
                "conditioning": (
                    "gold_answer_provided"
                    if args.condition_on_gold
                    else "answer_hidden"
                ),
                "assistant_format": args.assistant_format,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "json_mode": args.json_mode,
                "verified_answer_exact": verified_answer_exact,
                "reward": reward,
                "reward_label": reward_label,
                "use_for": use_for,
                "usage": usage,
                "raw_response_text": raw_content,
                "status": "ok",
                "generated_at": int(time.time()),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{row_index}/{len(pending_rows)}] id={row.id} "
                f"family={row.family} agreement={agreement} "
                f"verified={verified_answer_exact} reward_label={reward_label}"
            )


if __name__ == "__main__":
    main()
