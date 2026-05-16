from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill.common import (  # noqa: E402
    DEFAULT_DISTILL_DIR,
    normalize_answer,
    render_parallel_thinking_response,
)


DEFAULT_PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
DEFAULT_THINKING_STUB = (
    "Infer the rule from the examples, apply it to the target, and keep the "
    "answer format exact."
)
POSITIVE_LABELS = {"positive", "sft_and_dapo_chosen", "sft_bootstrap_only"}
NEGATIVE_LABELS = {"negative_answer", "dapo_rejected_only"}


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    family: str | None
    user_content: str
    assistant_content: str
    answer: str
    gold_answer: str | None
    reward_label: str
    reward: float | None
    confidence: float | None
    source_file: str
    source_line: int
    synthetic: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build RFT and DPO/ORPO preference datasets from scored teacher "
            "or rollout JSONL files."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        type=Path,
        required=True,
        help="Input JSONL file. Repeat this argument to merge multiple rollouts.",
    )
    parser.add_argument(
        "--output-rft-jsonl",
        type=Path,
        default=DEFAULT_DISTILL_DIR / "rft_messages_v1.jsonl",
        help="Output JSONL for SFT/RFT. Uses messages plus prompt/completion fields.",
    )
    parser.add_argument(
        "--output-preference-jsonl",
        type=Path,
        default=DEFAULT_DISTILL_DIR / "preference_pairs_v1.jsonl",
        help="Output JSONL for DPO/ORPO. Uses conversational prompt/chosen/rejected.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_DISTILL_DIR / "rft_preference_report_v1.json",
        help="Summary report path.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["base", "raw", "student"],
        default="base",
        help=(
            "base: prompt plus suffix; raw: original prompt only; "
            "student: use student_prompt when present."
        ),
    )
    parser.add_argument(
        "--prompt-suffix",
        default=DEFAULT_PROMPT_SUFFIX,
        help="Suffix appended when --prompt-mode=base.",
    )
    parser.add_argument(
        "--assistant-format",
        choices=["nemotron-cot", "existing", "answer-only"],
        default="nemotron-cot",
        help=(
            "How to render candidate assistant completions. nemotron-cot is "
            "recommended for the current adapter training path."
        ),
    )
    parser.add_argument(
        "--max-rft-per-id",
        type=int,
        default=2,
        help="Maximum positive completions kept per sample id. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--max-pairs-per-id",
        type=int,
        default=4,
        help="Maximum chosen/rejected pairs emitted per sample id. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--synthesize-gold-chosen",
        action="store_true",
        help=(
            "If an id has rejected candidates but no accepted candidate, synthesize "
            "a minimal chosen answer from gold_answer for preference training."
        ),
    )
    parser.add_argument(
        "--include-synthetic-gold-rft",
        action="store_true",
        help="Also include synthesized gold-chosen records in the RFT output.",
    )
    parser.add_argument(
        "--strict-jsonl",
        action="store_true",
        help="Fail on malformed JSONL lines instead of skipping them.",
    )
    return parser.parse_args()


def iter_jsonl(paths: Iterable[Path], strict: bool) -> tuple[list[dict[str, object]], Counter]:
    records: list[dict[str, object]] = []
    stats: Counter = Counter()

    for path in paths:
        resolved = path.resolve()
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats["malformed_jsonl_lines"] += 1
                    if strict:
                        raise ValueError(
                            f"Malformed JSONL at {resolved}:{line_number}: {exc}"
                        ) from exc
                    continue

                if not isinstance(record, dict):
                    stats["non_object_jsonl_records"] += 1
                    continue
                record["_source_file"] = str(resolved)
                record["_source_line"] = line_number
                records.append(record)
                stats["jsonl_records"] += 1

    return records, stats


def append_prompt_suffix(prompt: str, suffix: str) -> str:
    prompt = prompt.strip()
    suffix = suffix.strip()
    if not suffix:
        return prompt
    if suffix in prompt:
        return prompt
    return prompt + "\n" + suffix


def user_content_from_record(record: dict[str, object], args: argparse.Namespace) -> str | None:
    if args.prompt_mode == "student":
        student_prompt = record.get("student_prompt")
        if isinstance(student_prompt, str) and student_prompt.strip():
            return student_prompt.strip()

    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if args.prompt_mode == "raw":
        return prompt.strip()
    return append_prompt_suffix(prompt, args.prompt_suffix)


def as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def boolish(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def candidate_polarity(record: dict[str, object]) -> str | None:
    verified = boolish(record.get("verified_answer_exact"))
    if verified is True:
        return "positive"
    if verified is False:
        return "negative"

    reward_label = str(record.get("reward_label", "")).strip()
    use_for = str(record.get("use_for", "")).strip()
    if reward_label in POSITIVE_LABELS or use_for in POSITIVE_LABELS:
        return "positive"
    if reward_label in NEGATIVE_LABELS or use_for in NEGATIVE_LABELS:
        return "negative"

    reward = as_float(record.get("reward"))
    if reward is not None:
        if reward > 0:
            return "positive"
        if reward < 0:
            return "negative"
    return None


def answer_from_record(record: dict[str, object]) -> str | None:
    for key in ("teacher_answer", "final_answer", "candidate_answer"):
        value = record.get(key)
        if isinstance(value, str) and normalize_answer(value):
            return normalize_answer(value)

    parallel_thinking = record.get("parallel_thinking")
    if isinstance(parallel_thinking, dict):
        value = parallel_thinking.get("final_answer")
        if isinstance(value, str) and normalize_answer(value):
            return normalize_answer(value)

    return None


def render_nemotron_cot(record: dict[str, object], answer: str) -> str:
    assistant_response = record.get("assistant_response")
    if (
        isinstance(assistant_response, str)
        and record.get("assistant_format") == "nemotron-cot"
        and "\\boxed{" in assistant_response
    ):
        return assistant_response.strip()

    parallel_thinking = record.get("parallel_thinking")
    if isinstance(parallel_thinking, dict):
        try:
            return render_parallel_thinking_response(
                parallel_thinking,
                response_format="nemotron-cot",
            ).strip()
        except Exception:
            pass

    reasoning = record.get("reasoning_brief")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = record.get("rule_summary")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = DEFAULT_THINKING_STUB

    return reasoning.strip() + f"\n</think>\n\\boxed{{{answer}}}"


def assistant_content_from_record(
    record: dict[str, object],
    answer: str,
    assistant_format: str,
) -> str | None:
    assistant_response = record.get("assistant_response")
    if assistant_format == "existing" and isinstance(assistant_response, str):
        return assistant_response.strip() or None
    if assistant_format == "answer-only":
        return f"</think>\n\\boxed{{{answer}}}"
    return render_nemotron_cot(record, answer)


def candidate_from_record(
    record: dict[str, object],
    args: argparse.Namespace,
) -> Candidate | None:
    sample_id = str(record.get("id", "")).strip()
    if not sample_id:
        return None

    polarity = candidate_polarity(record)
    if polarity not in {"positive", "negative"}:
        return None

    answer = answer_from_record(record)
    if not answer:
        return None

    user_content = user_content_from_record(record, args)
    if not user_content:
        return None

    assistant_content = assistant_content_from_record(
        record=record,
        answer=answer,
        assistant_format=args.assistant_format,
    )
    if not assistant_content:
        return None

    gold_answer = record.get("gold_answer")
    gold_answer = normalize_answer(gold_answer) if gold_answer is not None else None
    confidence = as_float(record.get("confidence"))

    return Candidate(
        sample_id=sample_id,
        family=str(record.get("family", "")).strip() or None,
        user_content=user_content,
        assistant_content=assistant_content,
        answer=answer,
        gold_answer=gold_answer,
        reward_label=polarity,
        reward=as_float(record.get("reward")),
        confidence=confidence,
        source_file=str(record.get("_source_file", "")),
        source_line=int(record.get("_source_line", 0)),
    )


def synthesize_gold_candidate(candidate: Candidate) -> Candidate | None:
    if not candidate.gold_answer:
        return None
    return Candidate(
        sample_id=candidate.sample_id,
        family=candidate.family,
        user_content=candidate.user_content,
        assistant_content=DEFAULT_THINKING_STUB
        + f"\n</think>\n\\boxed{{{candidate.gold_answer}}}",
        answer=candidate.gold_answer,
        gold_answer=candidate.gold_answer,
        reward_label="positive",
        reward=1.0,
        confidence=None,
        source_file=candidate.source_file,
        source_line=candidate.source_line,
        synthetic=True,
    )


def candidate_sort_key(candidate: Candidate) -> tuple[float, str, int]:
    confidence = candidate.confidence if candidate.confidence is not None else -1.0
    reward = candidate.reward if candidate.reward is not None else 0.0
    return (confidence, str(reward), -candidate.source_line)


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[Candidate] = []
    for candidate in candidates:
        key = (
            candidate.sample_id,
            candidate.user_content,
            candidate.assistant_content,
            candidate.answer,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def rft_record(candidate: Candidate) -> dict[str, object]:
    prompt = [{"role": "user", "content": candidate.user_content}]
    completion = [{"role": "assistant", "content": candidate.assistant_content}]
    return {
        "id": candidate.sample_id,
        "family": candidate.family,
        "prompt": prompt,
        "completion": completion,
        "messages": prompt + completion,
        "answer": candidate.answer,
        "gold_answer": candidate.gold_answer,
        "source_file": candidate.source_file,
        "source_line": candidate.source_line,
        "synthetic": candidate.synthetic,
    }


def preference_record(
    chosen: Candidate,
    rejected: Candidate,
    pair_index: int,
) -> dict[str, object]:
    return {
        "id": f"{chosen.sample_id}::pair{pair_index}",
        "sample_id": chosen.sample_id,
        "family": chosen.family or rejected.family,
        "prompt": [{"role": "user", "content": chosen.user_content}],
        "chosen": [{"role": "assistant", "content": chosen.assistant_content}],
        "rejected": [{"role": "assistant", "content": rejected.assistant_content}],
        "chosen_answer": chosen.answer,
        "rejected_answer": rejected.answer,
        "gold_answer": chosen.gold_answer or rejected.gold_answer,
        "chosen_source_file": chosen.source_file,
        "chosen_source_line": chosen.source_line,
        "rejected_source_file": rejected.source_file,
        "rejected_source_line": rejected.source_line,
        "chosen_synthetic": chosen.synthetic,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    raw_records, stats = iter_jsonl(args.input_jsonl, strict=args.strict_jsonl)

    candidates_by_id: dict[str, list[Candidate]] = defaultdict(list)
    for record in raw_records:
        candidate = candidate_from_record(record, args)
        if candidate is None:
            stats["skipped_records"] += 1
            continue
        candidates_by_id[candidate.sample_id].append(candidate)
        stats[f"{candidate.reward_label}_candidates"] += 1

    rft_records: list[dict[str, object]] = []
    preference_records: list[dict[str, object]] = []

    for sample_id in sorted(candidates_by_id):
        candidates = dedupe_candidates(candidates_by_id[sample_id])
        positives = [
            candidate for candidate in candidates if candidate.reward_label == "positive"
        ]
        negatives = [
            candidate for candidate in candidates if candidate.reward_label == "negative"
        ]
        positives.sort(key=candidate_sort_key, reverse=True)
        negatives.sort(key=candidate_sort_key, reverse=True)

        if not positives and args.synthesize_gold_chosen and negatives:
            synthesized = synthesize_gold_candidate(negatives[0])
            if synthesized is not None:
                positives.append(synthesized)
                stats["synthetic_gold_chosen"] += 1

        rft_limit = args.max_rft_per_id or len(positives)
        for positive in positives[:rft_limit]:
            if positive.synthetic and not args.include_synthetic_gold_rft:
                continue
            rft_records.append(rft_record(positive))

        if not positives or not negatives:
            continue

        pair_limit = args.max_pairs_per_id or (len(positives) * len(negatives))
        pair_count = 0
        for positive in positives:
            for negative in negatives:
                if positive.user_content != negative.user_content:
                    continue
                if positive.answer == negative.answer:
                    continue
                pair_count += 1
                preference_records.append(
                    preference_record(
                        chosen=positive,
                        rejected=negative,
                        pair_index=pair_count,
                    )
                )
                if pair_count >= pair_limit:
                    break
            if pair_count >= pair_limit:
                break

    rft_count = write_jsonl(args.output_rft_jsonl, rft_records)
    preference_count = write_jsonl(args.output_preference_jsonl, preference_records)

    report = {
        "input_jsonl": [str(path.resolve()) for path in args.input_jsonl],
        "output_rft_jsonl": str(args.output_rft_jsonl.resolve()),
        "output_preference_jsonl": str(args.output_preference_jsonl.resolve()),
        "prompt_mode": args.prompt_mode,
        "assistant_format": args.assistant_format,
        "sample_ids": len(candidates_by_id),
        "rft_records": rft_count,
        "preference_pairs": preference_count,
        "stats": dict(stats),
    }
    args.report_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_json.resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
