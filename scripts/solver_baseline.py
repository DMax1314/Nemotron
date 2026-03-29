import argparse
import difflib
import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

MASK_8BIT = 0xFF


def round_half_up(value: float, digits: int = 2) -> str:
    quant = "0." + ("0" * (digits - 1)) + "1"
    return str(Decimal(str(value)).quantize(Decimal(quant), rounding=ROUND_HALF_UP))


def classify_prompt(prompt: str) -> str:
    first_line = prompt.split("\n", 1)[0]
    if "bit manipulation rule transforms 8-bit binary numbers" in first_line:
        return "bit"
    if "gravitational constant has been secretly changed" in first_line:
        return "gravity"
    if "secret unit conversion is applied to measurements" in first_line:
        return "unit"
    if "numbers are secretly converted into a different numeral system" in first_line:
        return "roman"
    if "secret encryption rules are used on text" in first_line:
        return "cipher"
    if "secret set of transformation rules is applied to equations" in first_line:
        return "symbol"
    return "other"


def int_to_roman(number: int) -> str:
    numerals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    parts: list[str] = []
    remainder = number
    for value, symbol in numerals:
        while remainder >= value:
            parts.append(symbol)
            remainder -= value
    return "".join(parts)


def rotate_left_8bit(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (8 - shift))) & MASK_8BIT


def rotate_right_8bit(value: int, shift: int) -> int:
    return ((value >> shift) | (value << (8 - shift))) & MASK_8BIT


def shift_left_8bit(value: int, shift: int) -> int:
    return (value << shift) & MASK_8BIT


def shift_right_8bit(value: int, shift: int) -> int:
    return (value >> shift) & MASK_8BIT


def majority(a: int, b: int, c: int) -> int:
    return ((a & b) | (a & c) | (b & c)) & MASK_8BIT


def choose(a: int, b: int, c: int) -> int:
    return ((a & b) | ((~a) & c)) & MASK_8BIT


def build_bit_candidates() -> list[tuple[str, Callable[[int], int]]]:
    unaries: list[tuple[str, Callable[[int], int]]] = [
        ("id", lambda x: x),
        ("not", lambda x: (~x) & MASK_8BIT),
    ]
    for shift in range(1, 8):
        unaries.extend(
            [
                (f"rol{shift}", lambda x, shift=shift: rotate_left_8bit(x, shift)),
                (f"ror{shift}", lambda x, shift=shift: rotate_right_8bit(x, shift)),
                (f"shl{shift}", lambda x, shift=shift: shift_left_8bit(x, shift)),
                (f"shr{shift}", lambda x, shift=shift: shift_right_8bit(x, shift)),
            ]
        )

    binary_ops: list[tuple[str, Callable[[int, int], int]]] = [
        ("xor", lambda a, b: a ^ b),
        ("and", lambda a, b: a & b),
        ("or", lambda a, b: a | b),
    ]
    ternary_ops: list[tuple[str, Callable[[int, int, int], int]]] = [
        ("maj", majority),
        ("ch", choose),
    ]

    candidates: list[tuple[str, Callable[[int], int]]] = []
    for name, transform in unaries:
        candidates.append((name, lambda x, transform=transform: transform(x)))

    for (name_a, fn_a), (name_b, fn_b) in product(unaries, repeat=2):
        for op_name, op_fn in binary_ops:
            candidates.append(
                (
                    f"{op_name}({name_a},{name_b})",
                    lambda x, fn_a=fn_a, fn_b=fn_b, op_fn=op_fn: op_fn(
                        fn_a(x), fn_b(x)
                    )
                    & MASK_8BIT,
                )
            )
            candidates.append(
                (
                    f"not({op_name}({name_a},{name_b}))",
                    lambda x, fn_a=fn_a, fn_b=fn_b, op_fn=op_fn: (~op_fn(
                        fn_a(x), fn_b(x)
                    ))
                    & MASK_8BIT,
                )
            )

    for (name_a, fn_a), (name_b, fn_b), (name_c, fn_c) in product(
        unaries, repeat=3
    ):
        for op_name, op_fn in ternary_ops:
            candidates.append(
                (
                    f"{op_name}({name_a},{name_b},{name_c})",
                    lambda x, fn_a=fn_a, fn_b=fn_b, fn_c=fn_c, op_fn=op_fn: op_fn(
                        fn_a(x), fn_b(x), fn_c(x)
                    )
                    & MASK_8BIT,
                )
            )
            candidates.append(
                (
                    f"not({op_name}({name_a},{name_b},{name_c}))",
                    lambda x, fn_a=fn_a, fn_b=fn_b, fn_c=fn_c, op_fn=op_fn: (~op_fn(
                        fn_a(x), fn_b(x), fn_c(x)
                    ))
                    & MASK_8BIT,
                )
            )
    return candidates


BIT_CANDIDATES = build_bit_candidates()


def extract_plaintext_vocabulary(train_df: pd.DataFrame) -> set[str]:
    vocab: set[str] = set()
    for _, row in train_df.iterrows():
        prompt = row["prompt"]
        if classify_prompt(prompt) != "cipher":
            continue
        for cipher_text, plain_text in re.findall(r"([^\n]+?) -> ([^\n]+)", prompt):
            if "Now, decrypt the following text:" in cipher_text:
                continue
            vocab.update(plain_text.split())
        vocab.update(str(row["answer"]).split())
    return vocab


@dataclass
class SolverContext:
    id_to_answer: dict[str, str]
    prompt_to_answer: dict[str, str]
    cipher_vocabulary: set[str]


def load_context(train_df: pd.DataFrame) -> SolverContext:
    return SolverContext(
        id_to_answer=dict(zip(train_df["id"], train_df["answer"])),
        prompt_to_answer=dict(zip(train_df["prompt"], train_df["answer"])),
        cipher_vocabulary=extract_plaintext_vocabulary(train_df),
    )


def solve_gravity(prompt: str) -> Optional[str]:
    pairs = re.findall(r"For t = ([0-9.]+)s, distance = ([0-9.]+) m", prompt)
    target = re.search(
        r"Now, determine the falling distance for t = ([0-9.]+)s", prompt
    )
    if not pairs or target is None:
        return None

    lower_bound = 0.0
    upper_bound = float("inf")
    for time_value, distance_value in pairs:
        time_float = float(time_value)
        distance_float = float(distance_value)
        lower_bound = max(
            lower_bound, 2 * (distance_float - 0.005) / (time_float * time_float)
        )
        upper_bound = min(
            upper_bound, 2 * (distance_float + 0.005) / (time_float * time_float)
        )

    if lower_bound > upper_bound:
        return None

    gravity = (lower_bound + upper_bound) / 2
    target_time = float(target.group(1))
    return round_half_up(0.5 * gravity * target_time * target_time)


def solve_unit_conversion(prompt: str) -> Optional[str]:
    pairs = re.findall(r"([0-9.]+) m becomes ([0-9.]+)", prompt)
    target = re.search(
        r"Now, convert the following measurement: ([0-9.]+) m", prompt
    )
    if not pairs or target is None:
        return None

    lower_bound = 0.0
    upper_bound = float("inf")
    for input_value, output_value in pairs:
        input_float = float(input_value)
        output_float = float(output_value)
        lower_bound = max(lower_bound, (output_float - 0.005) / input_float)
        upper_bound = min(upper_bound, (output_float + 0.005) / input_float)

    if lower_bound > upper_bound:
        return None

    factor = (lower_bound + upper_bound) / 2
    target_value = float(target.group(1))
    return round_half_up(factor * target_value)


def solve_roman(prompt: str) -> Optional[str]:
    match = re.search(
        r"Now, write the number ([0-9]+) in the Wonderland numeral system\.", prompt
    )
    if match is None:
        return None
    return int_to_roman(int(match.group(1)))


def consistent_substitution(
    cipher_text: str,
    plain_text: str,
    cipher_to_plain: dict[str, str],
    plain_to_cipher: dict[str, str],
) -> Optional[tuple[dict[str, str], dict[str, str]]]:
    if len(cipher_text) != len(plain_text):
        return None

    next_cipher_to_plain = dict(cipher_to_plain)
    next_plain_to_cipher = dict(plain_to_cipher)
    for cipher_char, plain_char in zip(cipher_text, plain_text):
        if cipher_char == " " and plain_char == " ":
            continue
        if cipher_char in next_cipher_to_plain and next_cipher_to_plain[cipher_char] != plain_char:
            return None
        if plain_char in next_plain_to_cipher and next_plain_to_cipher[plain_char] != cipher_char:
            return None
        next_cipher_to_plain[cipher_char] = plain_char
        next_plain_to_cipher[plain_char] = cipher_char
    return next_cipher_to_plain, next_plain_to_cipher


def solve_cipher(prompt: str, vocabulary: set[str]) -> Optional[str]:
    lines = [line.strip() for line in prompt.strip().splitlines() if line.strip()]
    cipher_to_plain: dict[str, str] = {}
    plain_to_cipher: dict[str, str] = {}
    example_plain_words: set[str] = set()
    target_text: Optional[str] = None

    for line in lines:
        if " -> " in line:
            cipher_text, plain_text = line.split(" -> ", 1)
            if "Now, decrypt the following text:" in cipher_text:
                continue
            result = consistent_substitution(
                cipher_text, plain_text, cipher_to_plain, plain_to_cipher
            )
            if result is None:
                return None
            cipher_to_plain, plain_to_cipher = result
            example_plain_words.update(plain_text.split())
        elif line.startswith("Now, decrypt the following text: "):
            target_text = line.split(": ", 1)[1]

    if target_text is None:
        return None

    target_words = target_text.split()
    candidate_lists: list[list[str]] = []
    for cipher_word in target_words:
        partial = "".join(cipher_to_plain.get(char, "?") for char in cipher_word)
        candidates = []
        for word in vocabulary:
            if len(word) != len(cipher_word):
                continue
            if any(partial_char != "?" and partial_char != word_char for partial_char, word_char in zip(partial, word)):
                continue
            candidates.append(word)
        candidate_lists.append(
            sorted(set(candidates), key=lambda word: (word not in example_plain_words, word))
        )

    order = sorted(range(len(target_words)), key=lambda idx: len(candidate_lists[idx]))

    def backtrack(
        position: int,
        current_cipher_to_plain: dict[str, str],
        current_plain_to_cipher: dict[str, str],
        chosen_words: dict[int, str],
    ) -> Optional[dict[int, str]]:
        if position == len(order):
            return chosen_words

        word_index = order[position]
        cipher_word = target_words[word_index]
        for candidate_word in candidate_lists[word_index]:
            result = consistent_substitution(
                cipher_word,
                candidate_word,
                current_cipher_to_plain,
                current_plain_to_cipher,
            )
            if result is None:
                continue
            next_chosen_words = dict(chosen_words)
            next_chosen_words[word_index] = candidate_word
            solved = backtrack(
                position + 1, result[0], result[1], next_chosen_words
            )
            if solved is not None:
                return solved
        return None

    solved_words = backtrack(0, cipher_to_plain, plain_to_cipher, {})
    if solved_words is None:
        return None
    return " ".join(solved_words[index] for index in range(len(target_words)))


def parse_bit_prompt(prompt: str) -> tuple[tuple[tuple[int, int], ...], int]:
    pairs = tuple(
        (int(input_bits, 2), int(output_bits, 2))
        for input_bits, output_bits in re.findall(r"([01]{8}) -> ([01]{8})", prompt)
    )
    target_match = re.search(r"Now, determine the output for: ([01]{8})", prompt)
    if target_match is None:
        raise ValueError("Bit prompt target not found.")
    return pairs, int(target_match.group(1), 2)


def solve_bit_prompt(prompt: str) -> Optional[str]:
    examples, target_value = parse_bit_prompt(prompt)

    cached_examples = tuple(sorted(examples))
    best_score = -1
    best_prediction: Optional[str] = None
    for expression_name, expression_fn in BIT_CANDIDATES:
        score = 0
        for input_value, output_value in cached_examples:
            if expression_fn(input_value) == output_value:
                score += 1
        if score > best_score:
            best_score = score
            best_prediction = format(expression_fn(target_value), "08b")
        if score == len(cached_examples):
            return best_prediction
    return best_prediction


def solve_symbol(prompt: str) -> Optional[str]:
    pairs = re.findall(r"([^\n]+?) = ([^\n]+)", prompt)
    target = re.search(r"Now, determine the result for: (.+)$", prompt, re.MULTILINE)
    if not pairs or target is None:
        return None

    target_text = target.group(1).strip()
    best_output = None
    best_score = -1.0
    for input_text, output_text in pairs:
        score = difflib.SequenceMatcher(None, input_text.strip(), target_text).ratio()
        if score > best_score:
            best_score = score
            best_output = output_text.strip()
    return best_output


def solve_prompt(prompt: str, context: SolverContext) -> Optional[str]:
    family = classify_prompt(prompt)
    if family == "bit":
        return solve_bit_prompt(prompt)
    if family == "gravity":
        return solve_gravity(prompt)
    if family == "unit":
        return solve_unit_conversion(prompt)
    if family == "roman":
        return solve_roman(prompt)
    if family == "cipher":
        return solve_cipher(prompt, context.cipher_vocabulary)
    if family == "symbol":
        return solve_symbol(prompt)
    return None


def exact_match(prediction: Optional[str], answer: str) -> bool:
    return prediction is not None and str(prediction).strip() == str(answer).strip()


def evaluate_train(train_df: pd.DataFrame, context: SolverContext) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    totals = Counter()
    solved = Counter()
    correct = Counter()

    for _, row in train_df.iterrows():
        prompt = row["prompt"]
        family = classify_prompt(prompt)
        prediction = solve_prompt(prompt, context)
        is_correct = exact_match(prediction, row["answer"])

        totals[family] += 1
        if prediction is not None:
            solved[family] += 1
        if is_correct:
            correct[family] += 1

        records.append(
            {
                "id": row["id"],
                "family": family,
                "prediction": prediction,
                "answer": row["answer"],
                "is_correct": is_correct,
            }
        )

    train_predictions = pd.DataFrame(records)
    family_report = {}
    for family in sorted(totals):
        family_report[family] = {
            "total": totals[family],
            "solved": solved[family],
            "correct": correct[family],
            "accuracy": correct[family] / totals[family],
        }

    report = {
        "overall_accuracy": train_predictions["is_correct"].mean(),
        "family_report": family_report,
    }
    return train_predictions, report


def build_submission(
    test_df: pd.DataFrame,
    context: SolverContext,
    lookup_by_id: bool,
    lookup_by_prompt: bool,
) -> pd.DataFrame:
    predictions: list[dict] = []
    for _, row in test_df.iterrows():
        prediction = None
        if lookup_by_id and row["id"] in context.id_to_answer:
            prediction = context.id_to_answer[row["id"]]
        elif lookup_by_prompt and row["prompt"] in context.prompt_to_answer:
            prediction = context.prompt_to_answer[row["prompt"]]
        else:
            prediction = solve_prompt(row["prompt"], context)

        predictions.append(
            {
                "id": row["id"],
                "prediction": "" if prediction is None else prediction,
                "family": classify_prompt(row["prompt"]),
                "used_lookup": bool(
                    (lookup_by_id and row["id"] in context.id_to_answer)
                    or (lookup_by_prompt and row["prompt"] in context.prompt_to_answer)
                ),
            }
        )
    return pd.DataFrame(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based baseline and submission generator for the Nemotron competition."
    )
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument(
        "--submission-out",
        default="submission_rule_solver.csv",
        help="Output path for the Kaggle submission CSV",
    )
    parser.add_argument(
        "--train-predictions-out",
        default="train_rule_solver_predictions.csv",
        help="Output path for train-set predictions and diagnostics",
    )
    parser.add_argument(
        "--report-out",
        default="rule_solver_report.json",
        help="Output path for the evaluation report",
    )
    parser.add_argument(
        "--lookup-test-by-id",
        action="store_true",
        help="Use exact id lookup from train to test before running the solver.",
    )
    parser.add_argument(
        "--lookup-test-by-prompt",
        action="store_true",
        help="Use exact prompt lookup from train to test before running the solver.",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    context = load_context(train_df)

    train_predictions, report = evaluate_train(train_df, context)
    train_predictions.to_csv(args.train_predictions_out, index=False)

    submission_with_diagnostics = build_submission(
        test_df=test_df,
        context=context,
        lookup_by_id=args.lookup_test_by_id,
        lookup_by_prompt=args.lookup_test_by_prompt,
    )
    submission_with_diagnostics[["id", "prediction"]].to_csv(
        args.submission_out, index=False
    )

    report.update(
        {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "test_exact_id_overlap": int(test_df["id"].isin(train_df["id"]).sum()),
            "test_exact_prompt_overlap": int(
                test_df["prompt"].isin(train_df["prompt"]).sum()
            ),
            "test_predictions": submission_with_diagnostics.to_dict(orient="records"),
        }
    )

    Path(args.report_out).write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"Train accuracy: {report['overall_accuracy']:.4f}")
    print(json.dumps(report["family_report"], indent=2))
    print(f"Submission written to: {args.submission_out}")
    print(f"Train predictions written to: {args.train_predictions_out}")
    print(f"Report written to: {args.report_out}")


if __name__ == "__main__":
    main()
