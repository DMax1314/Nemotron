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


LONGCAT_DEFAULT_URL = "https://api.longcat.chat/openai/v1/chat/completions"
LONGCAT_DEFAULT_BASE_URL = "https://api.longcat.chat/openai/v1"


def call_longcat_once(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    timeout_seconds: int,
    stream_mode: bool,
) -> tuple[dict[str, object], dict[str, object], str]:
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

    if stream_mode:
        try:
            stream = client.chat.completions.create(**payload, stream=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LongCat SDK stream request failed: {exc}") from exc

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, object] = {}
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            body = chunk.model_dump()
            choices = body.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning_parts.append(str(delta["reasoning_content"]))
                if delta.get("content"):
                    content_parts.append(str(delta["content"]))
            if body.get("usage"):
                usage = body["usage"]
            if chunk_count % 50 == 0:
                print(
                    f"stream_progress chunks={chunk_count} "
                    f"reasoning_chars={sum(len(part) for part in reasoning_parts)} "
                    f"content_chars={sum(len(part) for part in content_parts)}"
                )

        content = "".join(content_parts)
        reasoning_content = "".join(reasoning_parts)
        raw_body = json.dumps(
            {
                "content": content,
                "reasoning_content": reasoning_content,
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
            raise RuntimeError(f"LongCat SDK request failed: {exc}") from exc

        body = response.model_dump()
        raw_body = json.dumps(body, ensure_ascii=False)
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LongCat response: {raw_body[:1200]}") from exc

        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        usage = body.get("usage", {})

    if isinstance(content, str) and content.strip():
        parsed = extract_json_object(content)
        return parsed, usage, content

    if isinstance(reasoning_content, str) and reasoning_content.strip():
        try:
            parsed = extract_json_object(reasoning_content)
            return parsed, usage, reasoning_content
        except Exception:  # noqa: BLE001
            completion_tokens = usage.get("completion_tokens")
            raise RuntimeError(
                "LongCat returned reasoning_content but no final content. "
                f"completion_tokens={completion_tokens}, max_tokens={max_tokens}. "
                "The model likely exhausted its budget in hidden reasoning before producing the final JSON answer. "
                "Try increasing --max-tokens, reducing prompt burden, or switching away from the thinking model for this slice. "
                f"Raw response: {raw_body[:1200]}"
            )

    usage = body.get("usage", {})
    raise RuntimeError(
        "LongCat response had neither parseable content nor parseable reasoning_content. "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call LongCat directly as a teacher model for distillation."
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
        default=DEFAULT_DISTILL_DIR / "longcat_teacher_v1.jsonl",
        help="Append-only output JSONL file.",
    )
    parser.add_argument(
        "--model",
        default="LongCat-Flash-Thinking-2601",
        help="LongCat model name.",
    )
    parser.add_argument(
        "--base-url",
        default=LONGCAT_DEFAULT_BASE_URL,
        help="Direct LongCat OpenAI-compatible base URL.",
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
        help="Optional nucleus sampling parameter. Leave unset unless testing it explicitly.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override. Leave unset to use LongCat's model default.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode to observe long reasoning runs and aggregate deltas locally.",
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
        "--dry-run",
        action="store_true",
        help="Print the first prompt payload and exit without calling LongCat.",
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

    if pending_rows.empty:
        print("No pending rows left.")
        return

    if args.dry_run:
        first_row = pending_rows.iloc[0]
        print(build_teacher_prompt(first_row["prompt"], first_row["family"]))
        return

    api_key = os.environ.get("LONGCAT_API_KEY")
    if not api_key:
        api_key = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[System.Environment]::GetEnvironmentVariable(\"LONGCAT_API_KEY\", \"User\")",
            ],
            text=True,
        ).strip()
    if not api_key:
        raise RuntimeError(
            "LONGCAT_API_KEY is not set. Export it in the shell before running this script."
        )
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    with output_jsonl.open("a", encoding="utf-8") as handle:
        for row_index, row in enumerate(
            pending_rows.itertuples(index=False), start=1
        ):
            prompt = build_teacher_prompt(row.prompt, row.family)
            last_error: str | None = None

            for attempt in range(1, args.max_retries + 2):
                try:
                    response, usage, raw_content = call_longcat_once(
                        client=client,
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        timeout_seconds=args.timeout_seconds,
                        stream_mode=args.stream,
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
                "verified_answer_exact": (
                    teacher_answer == gold_answer if gold_answer is not None else None
                ),
                "usage": usage,
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
