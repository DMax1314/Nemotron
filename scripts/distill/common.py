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
