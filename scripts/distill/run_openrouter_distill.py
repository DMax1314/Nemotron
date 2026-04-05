from __future__ import annotations

import argparse
import json
import os
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
    build_teacher_prompt,
    extract_json_object,
    load_with_family,
    normalize_answer,
)


OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_DEFAULT_TITLE = "Nemotron Distillation Runner"
DEFAULT_FAMILY_REASONING_EFFORT = {
    "symbol": "xhigh",
    "bit": "high",
    "cipher": "high",
    "gravity": "medium",
    "unit": "medium",
    "roman": "low",
}


def build_reasoning_config(
    family: str,
    reasoning_enabled: bool,
    reasoning_effort: str | None,
    family_effort_map: dict[str, str],
    reasoning_max_tokens: int | None,
    exclude_reasoning: bool,
) -> dict[str, object] | None:
    if not reasoning_enabled and reasoning_effort is None and reasoning_max_tokens is None:
        return None

    config: dict[str, object] = {}
    selected_effort = family_effort_map.get(family, reasoning_effort)

    if reasoning_max_tokens is not None:
        config["max_tokens"] = reasoning_max_tokens
    elif selected_effort:
        config["effort"] = selected_effort
    else:
        config["enabled"] = True

    if exclude_reasoning:
        config["exclude"] = True
    elif reasoning_enabled and "enabled" not in config and "effort" not in config and "max_tokens" not in config:
        config["enabled"] = True

    return config or None


def call_openrouter_once(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    timeout_seconds: int,
    stream_mode: bool,
    json_mode: bool,
    reasoning_config: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object], str, object]:
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise reasoning teacher. Return only valid JSON.",
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
    if reasoning_config is not None:
        payload["extra_body"] = {"reasoning": reasoning_config}

    if stream_mode:
        try:
            stream = client.chat.completions.create(**payload, stream=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenRouter stream request failed: {exc}") from exc

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_details_parts: list[object] = []
        usage: dict[str, object] = {}
        chunk_count = 0
        print("stream_connected", flush=True)
        for chunk in stream:
            chunk_count += 1
            body = chunk.model_dump()
            choices = body.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    content_parts.append(content)
                reasoning = delta.get("reasoning")
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
                reasoning_details = delta.get("reasoning_details")
                if reasoning_details:
                    reasoning_details_parts.append(reasoning_details)
            if body.get("usage"):
                usage = body["usage"]
            if chunk_count == 1 or chunk_count % 10 == 0:
                print(
                    f"stream_progress chunks={chunk_count} "
                    f"reasoning_chars={sum(len(part) for part in reasoning_parts)} "
                    f"content_chars={sum(len(part) for part in content_parts)}",
                    flush=True,
                )

        content_text = "".join(content_parts)
        reasoning_text = "".join(reasoning_parts)
        reasoning_blob: object = reasoning_details_parts if reasoning_details_parts else reasoning_text
        raw_body = json.dumps(
            {
                "content": content_text,
                "reasoning": reasoning_text,
                "reasoning_details": reasoning_details_parts,
                "usage": usage,
                "stream_mode": True,
                "chunks": chunk_count,
            },
            ensure_ascii=False,
        )
    else:
        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        body = response.model_dump()
        raw_body = json.dumps(body, ensure_ascii=False)
        try:
            choice = body["choices"][0]
            message = choice["message"]
            content_text = message.get("content")
            usage = body.get("usage", {})
            reasoning_blob = message.get("reasoning_details")
            if not reasoning_blob:
                reasoning_blob = message.get("reasoning")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected OpenRouter response: {raw_body[:1200]}"
            ) from exc

    if isinstance(content_text, str) and content_text.strip():
        parsed = extract_json_object(content_text)
        return parsed, usage, content_text, reasoning_blob

    raise RuntimeError(
        "OpenRouter response had no parseable content. "
        f"Raw response: {raw_body[:1200]}"
    )


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


def load_openrouter_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        api_key = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[System.Environment]::GetEnvironmentVariable(\"OPENROUTER_API_KEY\", \"User\")",
            ],
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        api_key = ""

    return api_key


def parse_family_effort_args(entries: list[str]) -> dict[str, str]:
    effort_map: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --family-reasoning-effort entry '{entry}'. Use family=effort."
            )
        family, effort = entry.split("=", 1)
        family = family.strip()
        effort = effort.strip()
        if not family or not effort:
            raise ValueError(
                f"Invalid --family-reasoning-effort entry '{entry}'. Use family=effort."
            )
        effort_map[family] = effort
    return effort_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call OpenRouter directly as a teacher model for distillation."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Source CSV with prompt/answer pairs.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_DISTILL_DIR / "openrouter_teacher_v1.jsonl",
        help="Append-only output JSONL file.",
    )
    parser.add_argument(
        "--model",
        default=OPENROUTER_DEFAULT_MODEL,
        help="OpenRouter model slug.",
    )
    parser.add_argument(
        "--base-url",
        default=OPENROUTER_DEFAULT_BASE_URL,
        help="OpenRouter OpenAI-compatible base URL.",
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
        default=None,
        help="Optional max_tokens override.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode to observe long generations and aggregate deltas locally.",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Request OpenRouter JSON mode via response_format=json_object.",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable OpenRouter reasoning using the unified reasoning config.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Global OpenRouter reasoning effort override.",
    )
    parser.add_argument(
        "--family-reasoning-effort",
        action="append",
        default=[],
        help="Repeatable override like symbol=xhigh or bit=high.",
    )
    parser.add_argument(
        "--auto-family-effort",
        action="store_true",
        help="Use built-in family-specific reasoning effort defaults.",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        help="Optional reasoning token budget override.",
    )
    parser.add_argument(
        "--exclude-reasoning",
        action="store_true",
        help="Let the model reason internally but omit reasoning from the response body.",
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
        default=OPENROUTER_DEFAULT_TITLE,
        help="Optional OpenRouter title header.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt payload and exit without calling OpenRouter.",
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

    print(f"Loaded rows after filtering: {len(dataframe)}")
    print(f"Pending rows after resume:   {len(pending_rows)}")
    print(f"Output JSONL:                {output_jsonl}")
    print(f"Teacher model:               {args.model}")
    print(f"Temperature:                 {args.temperature}")
    print(f"Top-p:                       {args.top_p if args.top_p is not None else 'default'}")
    print(f"Streaming:                   {args.stream}")
    print(f"JSON mode:                   {args.json_mode}")
    print(f"Reasoning enabled:           {args.reasoning}")
    print(f"Reasoning effort:            {args.reasoning_effort or 'auto/default'}")
    print(f"Reasoning max tokens:        {args.reasoning_max_tokens if args.reasoning_max_tokens is not None else 'default'}")
    print(f"Exclude reasoning:           {args.exclude_reasoning}")

    if pending_rows.empty:
        print("No pending rows left.")
        return

    family_effort_map = (
        dict(DEFAULT_FAMILY_REASONING_EFFORT) if args.auto_family_effort else {}
    )
    family_effort_map.update(parse_family_effort_args(args.family_reasoning_effort))

    if args.dry_run:
        first_row = pending_rows.iloc[0]
        reasoning_config = build_reasoning_config(
            family=str(first_row["family"]),
            reasoning_enabled=args.reasoning,
            reasoning_effort=args.reasoning_effort,
            family_effort_map=family_effort_map,
            reasoning_max_tokens=args.reasoning_max_tokens,
            exclude_reasoning=args.exclude_reasoning,
        )
        print(f"Resolved family:              {first_row['family']}")
        print(f"Resolved reasoning config:    {reasoning_config}")
        print(build_teacher_prompt(first_row["prompt"], first_row["family"]))
        return

    api_key = load_openrouter_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it in the shell before running this script."
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
            prompt = build_teacher_prompt(row.prompt, row.family)
            reasoning_config = build_reasoning_config(
                family=row.family,
                reasoning_enabled=args.reasoning,
                reasoning_effort=args.reasoning_effort,
                family_effort_map=family_effort_map,
                reasoning_max_tokens=args.reasoning_max_tokens,
                exclude_reasoning=args.exclude_reasoning,
            )
            last_error: str | None = None

            for attempt in range(1, args.max_retries + 2):
                try:
                    response, usage, raw_content, raw_reasoning = call_openrouter_once(
                        client=client,
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        timeout_seconds=args.timeout_seconds,
                        stream_mode=args.stream,
                        json_mode=args.json_mode,
                        reasoning_config=reasoning_config,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    time.sleep(min(attempt, 3))
            else:
                failure_record = {
                    "id": row.id,
                    "family": row.family,
                    "prompt": row.prompt,
                    "gold_answer": str(row.answer) if hasattr(row, "answer") else None,
                    "teacher_model": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "stream": args.stream,
                    "json_mode": args.json_mode,
                    "reasoning": reasoning_config,
                    "status": "error",
                    "error": last_error,
                }
                handle.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{row_index}/{len(pending_rows)}] id={row.id} family={row.family} error"
                )
                continue

            teacher_answer = normalize_answer(response["final_answer"])
            gold_answer = normalize_answer(row.answer) if hasattr(row, "answer") else None
            record = {
                "id": row.id,
                "family": row.family,
                "prompt": row.prompt,
                "gold_answer": gold_answer,
                "teacher_model": args.model,
                "teacher_answer": teacher_answer,
                "rule_summary": str(response["rule_summary"]).strip(),
                "reasoning_brief": str(response["reasoning_brief"]).strip(),
                "confidence": float(response["confidence"]),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "stream": args.stream,
                "json_mode": args.json_mode,
                "reasoning": reasoning_config,
                "verified_answer_exact": (
                    teacher_answer == gold_answer if gold_answer is not None else None
                ),
                "usage": usage,
                "raw_reasoning": raw_reasoning,
                "raw_response_text": raw_content,
                "status": "ok",
                "generated_at": int(time.time()),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{row_index}/{len(pending_rows)}] id={row.id} family={row.family} "
                f"verified={record['verified_answer_exact']}"
            )


if __name__ == "__main__":
    main()
