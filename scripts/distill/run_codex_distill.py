from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill.common import (
    DEFAULT_DISTILL_DIR,
    REPO_ROOT,
    build_teacher_prompt,
    load_with_family,
    normalize_answer,
)


def run_codex_once(
    codex_bin: str,
    model: str,
    reasoning_effort: str | None,
    schema_path: Path,
    prompt: str,
    workdir: Path,
) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        output_path = Path(handle.name)

    command = [
        codex_bin,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "-C",
        str(workdir),
        "-s",
        "read-only",
        "-m",
        model,
    ]
    if reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    command.extend(
        [
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "-",
        ]
    )

    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with exit code {completed.returncode}\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )

    try:
        response = json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        output_path.unlink(missing_ok=True)

    return response


def resolve_codex_binary(codex_bin: str) -> str:
    if codex_bin != "codex":
        return codex_bin

    for candidate in ("codex.cmd", "codex.exe", "codex"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return codex_bin


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
        description="Use local Codex CLI as a teacher to generate distillation data."
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
        default=DEFAULT_DISTILL_DIR / "codex_teacher_v1.jsonl",
        help="Append-only output JSONL file.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path("scripts/distill/teacher_response.schema.json"),
        help="JSON schema file for structured Codex output.",
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional Codex reasoning effort override.",
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Repeatable family filter. Example: --family symbol --family bit",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--codex-workdir",
        type=Path,
        default=REPO_ROOT / "tmp",
        help="Neutral working directory for codex exec.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip ids that already exist in the output JSONL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt payload and exit without calling Codex.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_jsonl = args.output_jsonl.resolve()
    schema_path = args.schema_path.resolve()
    codex_workdir = args.codex_workdir.resolve()
    codex_bin = resolve_codex_binary(args.codex_bin)
    codex_workdir.mkdir(parents=True, exist_ok=True)
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
    print(f"Codex binary:                {codex_bin}")
    print(f"Teacher model:               {args.model}")
    print(f"Reasoning effort:            {args.reasoning_effort or 'default'}")

    if pending_rows.empty:
        print("No pending rows left.")
        return

    if args.dry_run:
        first_row = pending_rows.iloc[0]
        print(build_teacher_prompt(first_row["prompt"], first_row["family"]))
        return

    with output_jsonl.open("a", encoding="utf-8") as handle:
        for row_index, row in enumerate(
            pending_rows.itertuples(index=False), start=1
        ):
            prompt = build_teacher_prompt(row.prompt, row.family)
            last_error: str | None = None

            for attempt in range(1, args.max_retries + 2):
                try:
                    response = run_codex_once(
                        codex_bin=codex_bin,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        schema_path=schema_path,
                        prompt=prompt,
                        workdir=codex_workdir,
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
                    "reasoning_effort": args.reasoning_effort,
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
                "reasoning_effort": args.reasoning_effort,
                "teacher_answer": teacher_answer,
                "rule_summary": str(response["rule_summary"]).strip(),
                "reasoning_brief": str(response["reasoning_brief"]).strip(),
                "confidence": float(response["confidence"]),
                "verified_answer_exact": (
                    teacher_answer == gold_answer if gold_answer is not None else None
                ),
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
