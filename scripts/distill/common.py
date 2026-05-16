from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISTILL_DIR = REPO_ROOT / "data" / "distill"


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.solver_baseline import classify_prompt  # noqa: E402


def normalize_answer(value: object) -> str:
    return " ".join(str(value).strip().split())


FAMILY_FORMAT_HINTS = {
    "bit": "Return exactly 8 binary digits such as 01001011. No spaces, no quotes, no prefixes.",
    "gravity": "Return a decimal number only. No units. Usually this benchmark uses 2 decimal places.",
    "unit": "Return a decimal number only, with 2 digits after the decimal point. No units.",
    "roman": "Return uppercase Roman numerals only, such as LXVII. No Arabic numerals, no spaces.",
    "cipher": "Return lowercase English words separated by single spaces. No punctuation.",
    "symbol": "Return the exact transformed expression only. It may be a short integer or a compact symbol string, usually 1 to 4 characters, with no spaces.",
}

FAMILY_REASONING_HINTS = {
    "bit": "Infer the bitwise transformation from multiple input-output examples before applying it to the target binary string.",
    "gravity": "Infer the modified gravitational constant from the examples, then apply the same physics formula to the target case.",
    "unit": "Infer the hidden conversion factor from the examples, then apply it to the target measurement.",
    "roman": "Infer the target numeral system and convert the final number into the same representation.",
    "cipher": "Infer the text transformation or substitution rule from the examples, then apply it consistently to the target phrase.",
    "symbol": "Infer the hidden symbol-level transformation from several example equations. Pay close attention to character substitutions, ordering, and dropped or inserted symbols.",
}

FAMILY_PARALLEL_SCAFFOLDS = {
    "bit": """- Branches should include per-output-bit or boolean-function checks when useful.
- Compare the target bits against the example rows before choosing each output bit.
- Treat 8-bit formatting as part of the answer, not as a post-processing detail.""",
    "gravity": """- Infer the hidden gravitational constant from d = 0.5*g*t^2.
- Account for rounding in the observed distances before applying the target time.
- The selected answer should be the rounded decimal distance only.""",
    "unit": """- Infer the hidden conversion factor by checking ratios across all examples.
- Account for two-decimal rounding before applying the target measurement.
- The selected answer should be the rounded decimal value only.""",
    "roman": """- Verify that all examples match standard Roman numerals.
- Convert the target number by place value.
- The selected answer should use uppercase Roman numeral symbols only.""",
    "cipher": """- Build a consistent one-to-one substitution map from the examples.
- Verify target words against the observed Wonderland vocabulary and letter constraints.
- The selected answer should be lowercase words separated by single spaces.""",
    "symbol": """- Compare character substitutions, ordering, dropped symbols, inserted symbols, and whole-expression rewrites.
- Do not solve symbol tasks by nearest-neighbor copying unless the branch explicitly verifies why the copied rule transfers.
- If the target appears numeric-like, prefer arithmetic/operator hypotheses; if symbolic-like, prefer character-level rewrite hypotheses.""",
}


def has_answer_value(value: object) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def classify_answer_format(answer: object) -> str:
    text = normalize_answer(answer)
    if re.fullmatch(r"[01]{8}", text):
        return "binary_8bit"
    if re.fullmatch(r"-?\d+\.\d+", text):
        return "decimal"
    if re.fullmatch(r"-?\d+", text):
        return "integer"
    if re.fullmatch(r"[IVXLCDM]+", text):
        return "roman"
    if re.fullmatch(r"[a-z]+(?: [a-z]+)*", text):
        return "lowercase_text"
    if re.fullmatch(r"[A-Za-z ]+", text):
        return "text"
    return "symbolic"


def infer_answer_format(
    family: str,
    prompt: str = "",
    answer: object | None = None,
) -> str:
    if has_answer_value(answer):
        return classify_answer_format(answer)
    if family == "bit":
        return "binary_8bit"
    if family in {"gravity", "unit"}:
        return "decimal"
    if family == "roman":
        return "roman"
    if family == "cipher":
        return "lowercase_text"
    if family == "symbol":
        example_outputs = [
            output.strip()
            for output in re.findall(r"^[^=\n]+? = ([^\n]+)$", prompt, flags=re.M)
        ]
        if example_outputs:
            formats = [classify_answer_format(output) for output in example_outputs]
            numeric_count = sum(answer_format == "integer" for answer_format in formats)
            if numeric_count > len(formats) / 2:
                return "integer"
        return "symbolic"
    return "unknown"


def infer_symbol_subtype(prompt: str, answer: object | None = None) -> str | None:
    if has_answer_value(answer):
        answer_format = classify_answer_format(answer)
        if answer_format in {"integer", "decimal"}:
            return "symbol_numeric_like"
        return "symbol_symbolic_like"

    example_outputs = [
        output.strip()
        for output in re.findall(r"^[^=\n]+? = ([^\n]+)$", prompt, flags=re.M)
    ]
    if not example_outputs:
        return "symbol_unknown"

    formats = [classify_answer_format(output) for output in example_outputs]
    numeric_count = sum(answer_format in {"integer", "decimal"} for answer_format in formats)
    return (
        "symbol_numeric_like"
        if numeric_count > len(formats) / 2
        else "symbol_symbolic_like"
    )


def build_family_scaffold(
    family: str,
    answer_format: str | None = None,
    symbol_subtype: str | None = None,
) -> str:
    scaffold = FAMILY_PARALLEL_SCAFFOLDS.get(
        family,
        "- Compare multiple examples before deciding the rule.",
    )
    metadata_lines = []
    if answer_format:
        metadata_lines.append(f"- Expected answer_format label: {answer_format}.")
    if symbol_subtype:
        metadata_lines.append(f"- Symbol subtype label: {symbol_subtype}.")
    if metadata_lines:
        return scaffold + "\n" + "\n".join(metadata_lines)
    return scaffold


def build_teacher_prompt(prompt: str, family: str) -> str:
    format_hint = FAMILY_FORMAT_HINTS.get(
        family,
        "Return only the exact final answer string for this puzzle family.",
    )
    reasoning_hint = FAMILY_REASONING_HINTS.get(
        family,
        "Infer the hidden transformation from the examples and apply it to the target input.",
    )
    return f"""You are generating high-quality teacher supervision for a reasoning benchmark.

Task:
- Infer the hidden transformation rule from the worked examples in the prompt.
- Apply the same rule to the final query.
- Produce supervision that is maximally useful for training another model on this benchmark.

Return a JSON object that exactly matches this schema:
{{
  "final_answer": "string",
  "rule_summary": "string",
  "reasoning_brief": "string",
  "confidence": 0.0
}}

Rules:
- final_answer: provide only the exact final answer, with no extra text.
- final_answer must match the expected surface form for this family.
- Do not wrap the answer in \\boxed{{}}.
- Do not include units, labels, equations, or explanation inside final_answer.
- rule_summary: one short sentence, ideally under 30 words.
- reasoning_brief: 2 to 4 short sentences, concrete and concise, no markdown bullets.
- confidence: a number between 0 and 1.
- Compare multiple examples before deciding the rule.
- Prefer exact answer formatting over stylistic variation.
- Keep the hidden reasoning short and move quickly to the final JSON answer.
- Do not spend the full budget on internal analysis.
- Return only raw JSON.
- Do not use markdown.
- Do not use code fences.

Family hint: {family}
Answer format expectation: {format_hint}
Solving focus: {reasoning_hint}

Puzzle:
{prompt}
"""


def build_parallel_thinking_prompt(
    prompt: str,
    family: str,
    branches: int,
    answer_format: str | None = None,
    symbol_subtype: str | None = None,
    gold_answer: str | None = None,
) -> str:
    format_hint = FAMILY_FORMAT_HINTS.get(
        family,
        "Return only the exact final answer string for this puzzle family.",
    )
    reasoning_hint = FAMILY_REASONING_HINTS.get(
        family,
        "Infer the hidden transformation from the examples and apply it to the target input.",
    )
    family_scaffold = build_family_scaffold(
        family=family,
        answer_format=answer_format,
        symbol_subtype=symbol_subtype,
    )
    gold_answer_block = ""
    if has_answer_value(gold_answer):
        gold_answer_block = f"""
Gold answer for answer-conditioned rationale construction: {normalize_answer(gold_answer)}
- Use the gold answer only to verify and select a concise, supported path.
- Do not invent unsupported rules just to fit the gold answer.
- If no branch can support the gold answer from the examples, set consensus.agreement to "uncertain" and say that the rationale is weak.
"""
    return f"""You are generating synthetic teacher supervision for a reasoning benchmark.

Task:
- Infer the hidden transformation rule from the worked examples in the prompt.
- Solve the final query using parallel thinking.
- Produce {branches} independent branches. Each branch should use a meaningfully different strategy, such as direct rule induction, inverse checking, example consistency, boundary/format checking, or alternative hypothesis elimination.
- Then aggregate the branches into a single final answer.

Return a JSON object that exactly matches this shape:
{{
  "final_answer": "string",
  "branches": [
    {{
      "branch_id": "A",
      "strategy": "string",
      "rule_hypothesis": "string",
      "evidence": "string",
      "candidate_answer": "string",
      "confidence": 0.0
    }}
  ],
  "consensus": {{
    "selected_answer": "string",
    "agreement": "unanimous | majority | tie_break | uncertain",
    "selected_branch_ids": ["A"],
    "resolution": "string"
  }},
  "confidence": 0.0
}}

Rules:
- Return only raw JSON.
- Do not use markdown or code fences.
- Use exactly {branches} branch objects.
- final_answer and consensus.selected_answer must be identical.
- final_answer must match the expected surface form for this family.
- candidate_answer must also follow the family answer format.
- Do not wrap answers in \\boxed{{}}.
- Do not include units, labels, equations, or explanation inside final_answer.
- Keep each branch concise: one short sentence for strategy, one for rule_hypothesis, and one to three short sentences for evidence.
- Do not expose long private chain-of-thought. The fields should be compact audit notes, not exhaustive hidden reasoning.
- Compare multiple examples before deciding the rule.
- If branches disagree, use consensus.resolution to briefly explain why the selected answer wins.

Family hint: {family}
Answer format expectation: {format_hint}
Solving focus: {reasoning_hint}
Family-specific scaffold:
{family_scaffold}
{gold_answer_block}

Puzzle:
{prompt}
"""


def build_parallel_thinking_student_prompt(
    prompt: str,
    family: str,
    branches: int,
    answer_format: str | None = None,
    symbol_subtype: str | None = None,
) -> str:
    format_hint = FAMILY_FORMAT_HINTS.get(
        family,
        "Return only the exact final answer string for this puzzle family.",
    )
    answer_format = answer_format or infer_answer_format(family=family, prompt=prompt)
    subtype_line = (
        f"\nSymbol subtype: {symbol_subtype}."
        if symbol_subtype
        else ""
    )
    return f"""{prompt.strip()}

Use parallel thinking with exactly {branches} concise branches. Return valid JSON with keys:
- parallel_thinking
- consensus
- answer

Answer format label: {answer_format}.{subtype_line}
The final answer must follow this format: {format_hint}
"""


def validate_parallel_thinking_response(
    response: dict[str, object],
    expected_branches: int,
) -> None:
    required_keys = {"final_answer", "branches", "consensus", "confidence"}
    missing_keys = required_keys - response.keys()
    if missing_keys:
        raise ValueError(f"Parallel-thinking response missing keys: {sorted(missing_keys)}")

    final_answer = normalize_answer(response["final_answer"])
    if not final_answer:
        raise ValueError("Parallel-thinking response has an empty final_answer.")

    branches = response["branches"]
    if not isinstance(branches, list):
        raise ValueError("Parallel-thinking response field 'branches' must be a list.")
    if len(branches) != expected_branches:
        raise ValueError(
            f"Expected {expected_branches} branches, got {len(branches)}."
        )

    branch_ids: set[str] = set()
    for index, branch in enumerate(branches):
        if not isinstance(branch, dict):
            raise ValueError(f"Branch {index} must be an object.")
        for key in (
            "branch_id",
            "strategy",
            "rule_hypothesis",
            "evidence",
            "candidate_answer",
            "confidence",
        ):
            if key not in branch:
                raise ValueError(f"Branch {index} missing key: {key}")
        branch_id = str(branch["branch_id"]).strip()
        if not branch_id:
            raise ValueError(f"Branch {index} has an empty branch_id.")
        if branch_id in branch_ids:
            raise ValueError(f"Duplicate branch_id: {branch_id}")
        branch_ids.add(branch_id)
        if not normalize_answer(branch["candidate_answer"]):
            raise ValueError(f"Branch {branch_id} has an empty candidate_answer.")

    consensus = response["consensus"]
    if not isinstance(consensus, dict):
        raise ValueError("Parallel-thinking response field 'consensus' must be an object.")
    selected_answer = normalize_answer(consensus.get("selected_answer", ""))
    if selected_answer != final_answer:
        raise ValueError(
            "final_answer and consensus.selected_answer must match exactly after normalization."
        )
    if consensus.get("agreement") not in {
        "unanimous",
        "majority",
        "tie_break",
        "uncertain",
    }:
        raise ValueError("consensus.agreement has an unsupported value.")


def render_parallel_thinking_response(
    response: dict[str, object],
    response_format: str = "json",
) -> str:
    answer = normalize_answer(response["final_answer"])
    if response_format == "nemotron-cot":
        lines: list[str] = []
        for branch in response["branches"]:
            if not isinstance(branch, dict):
                continue
            branch_id = str(branch["branch_id"]).strip()
            strategy = str(branch["strategy"]).strip()
            hypothesis = str(branch["rule_hypothesis"]).strip()
            evidence = str(branch["evidence"]).strip()
            candidate = normalize_answer(branch["candidate_answer"])
            lines.append(
                f"Branch {branch_id} ({strategy}): {hypothesis} "
                f"Evidence: {evidence} Candidate: {candidate}."
            )

        consensus = response["consensus"]
        if isinstance(consensus, dict):
            agreement = str(consensus.get("agreement", "")).strip()
            resolution = str(consensus.get("resolution", "")).strip()
            lines.append(
                f"Consensus ({agreement}): {resolution} Selected answer: {answer}."
            )
        return "\n".join(lines) + f"\n</think>\n\\boxed{{{answer}}}"

    rendered = {
        "parallel_thinking": response["branches"],
        "consensus": response["consensus"],
        "answer": answer,
    }
    return json.dumps(rendered, ensure_ascii=False)


def extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model returned empty text.")

    candidates = [stripped]
    fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    candidates.extend(fenced_matches)

    brace_matches = re.findall(r"(\{.*\})", stripped, flags=re.S)
    candidates.extend(brace_matches)

    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse a JSON object from model text: {stripped[:400]}")


def load_with_family(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    dataframe["family"] = dataframe["prompt"].map(classify_prompt)
    return dataframe
