# %%
import difflib
from glob import glob
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
import zipfile

import pandas as pd

# %%
DATA_DIR = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge"
OUTPUT_DIR = "/kaggle/working"
MASK_8BIT = 0xFF


def find_input_file(filename):
    direct_path = f"{DATA_DIR}/{filename}"
    matches = [direct_path] if glob(direct_path) else []
    if not matches:
        matches = sorted(glob(f"/kaggle/input/**/{filename}", recursive=True))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under /kaggle/input")
    return matches[0]


def round_half_up(value, digits=2):
    quant = "0." + ("0" * (digits - 1)) + "1"
    return str(Decimal(str(value)).quantize(Decimal(quant), rounding=ROUND_HALF_UP))


def classify_prompt(prompt):
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


def int_to_roman(number):
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
    parts = []
    remainder = number
    for value, symbol in numerals:
        while remainder >= value:
            parts.append(symbol)
            remainder -= value
    return "".join(parts)


def rotate_left_8bit(value, shift):
    return ((value << shift) | (value >> (8 - shift))) & MASK_8BIT


def rotate_right_8bit(value, shift):
    return ((value >> shift) | (value << (8 - shift))) & MASK_8BIT


def shift_left_8bit(value, shift):
    return (value << shift) & MASK_8BIT


def shift_right_8bit(value, shift):
    return value >> shift


def majority(a, b, c):
    return ((a & b) | (a & c) | (b & c)) & MASK_8BIT


def choose(a, b, c):
    return ((a & b) | ((~a) & c)) & MASK_8BIT


def build_bit_candidates():
    unaries = [
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

    binary_ops = [
        ("xor", lambda a, b: a ^ b),
        ("and", lambda a, b: a & b),
        ("or", lambda a, b: a | b),
    ]
    ternary_ops = [
        ("maj", majority),
        ("ch", choose),
    ]

    candidates = []
    for name, transform in unaries:
        candidates.append((name, lambda x, transform=transform: transform(x)))

    for (name_a, fn_a), (name_b, fn_b) in product(unaries, repeat=2):
        for op_name, op_fn in binary_ops:
            candidates.append(
                (
                    f"{op_name}({name_a},{name_b})",
                    lambda x, fn_a=fn_a, fn_b=fn_b, op_fn=op_fn: op_fn(fn_a(x), fn_b(x))
                    & MASK_8BIT,
                )
            )
            candidates.append(
                (
                    f"not({op_name}({name_a},{name_b}))",
                    lambda x, fn_a=fn_a, fn_b=fn_b, op_fn=op_fn: (~op_fn(fn_a(x), fn_b(x)))
                    & MASK_8BIT,
                )
            )

    for (name_a, fn_a), (name_b, fn_b), (name_c, fn_c) in product(unaries, repeat=3):
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


def extract_plaintext_vocabulary(train_df):
    vocab = set()
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


def consistent_substitution(cipher_text, plain_text, cipher_to_plain, plain_to_cipher):
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


def solve_cipher(prompt, vocabulary):
    lines = [line.strip() for line in prompt.strip().splitlines() if line.strip()]
    cipher_to_plain = {}
    plain_to_cipher = {}
    example_plain_words = set()
    target_text = None

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
    candidate_lists = []
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

    def backtrack(position, current_cipher_to_plain, current_plain_to_cipher, chosen_words):
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
            solved = backtrack(position + 1, result[0], result[1], next_chosen_words)
            if solved is not None:
                return solved
        return None

    solved_words = backtrack(0, cipher_to_plain, plain_to_cipher, {})
    if solved_words is None:
        return None
    return " ".join(solved_words[index] for index in range(len(target_words)))


def solve_gravity(prompt):
    pairs = re.findall(r"For t = ([0-9.]+)s, distance = ([0-9.]+) m", prompt)
    target = re.search(r"Now, determine the falling distance for t = ([0-9.]+)s", prompt)
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


def solve_unit_conversion(prompt):
    pairs = re.findall(r"([0-9.]+) m becomes ([0-9.]+)", prompt)
    target = re.search(r"Now, convert the following measurement: ([0-9.]+) m", prompt)
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


def solve_roman(prompt):
    match = re.search(
        r"Now, write the number ([0-9]+) in the Wonderland numeral system\.", prompt
    )
    if match is None:
        return None
    return int_to_roman(int(match.group(1)))


def solve_bit_prompt(prompt, bit_candidates):
    examples = [
        (int(input_bits, 2), int(output_bits, 2))
        for input_bits, output_bits in re.findall(r"([01]{8}) -> ([01]{8})", prompt)
    ]
    target_match = re.search(r"Now, determine the output for: ([01]{8})", prompt)
    if target_match is None:
        return None
    target_value = int(target_match.group(1), 2)

    best_score = -1
    best_prediction = None
    for _, candidate_fn in bit_candidates:
        score = 0
        for input_value, output_value in examples:
            if candidate_fn(input_value) == output_value:
                score += 1
        if score > best_score:
            best_score = score
            best_prediction = format(candidate_fn(target_value), "08b")
        if score == len(examples):
            return best_prediction
    return best_prediction


def solve_symbol(prompt):
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


def solve_prompt(prompt, context, bit_candidates):
    family = classify_prompt(prompt)
    if family == "bit":
        return solve_bit_prompt(prompt, bit_candidates)
    if family == "gravity":
        return solve_gravity(prompt)
    if family == "unit":
        return solve_unit_conversion(prompt)
    if family == "roman":
        return solve_roman(prompt)
    if family == "cipher":
        return solve_cipher(prompt, context["cipher_vocabulary"])
    if family == "symbol":
        return solve_symbol(prompt)
    return None


# %%
train_path = find_input_file("train.csv")
test_path = find_input_file("test.csv")

print(f"train_path = {train_path}")
print(f"test_path = {test_path}")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

context = {
    "id_to_answer": dict(zip(train_df["id"], train_df["answer"])),
    "prompt_to_answer": dict(zip(train_df["prompt"], train_df["answer"])),
    "cipher_vocabulary": extract_plaintext_vocabulary(train_df),
}
bit_candidates = build_bit_candidates()

print(f"Train rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")
print("Test family counts:")
print(test_df["prompt"].apply(classify_prompt).value_counts())

# %%
submission_rows = []
for _, row in test_df.iterrows():
    used_lookup = False
    if row["id"] in context["id_to_answer"]:
        prediction = context["id_to_answer"][row["id"]]
        used_lookup = True
    elif row["prompt"] in context["prompt_to_answer"]:
        prediction = context["prompt_to_answer"][row["prompt"]]
        used_lookup = True
    else:
        prediction = solve_prompt(row["prompt"], context, bit_candidates)

    submission_rows.append(
        {
            "id": row["id"],
            "prediction": "" if prediction is None else prediction,
            "family": classify_prompt(row["prompt"]),
            "used_lookup": used_lookup,
        }
    )

submission_df = pd.DataFrame(submission_rows)
submission_df[["id", "prediction"]].to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
with zipfile.ZipFile(f"{OUTPUT_DIR}/submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(f"{OUTPUT_DIR}/submission.csv", arcname="submission.csv")

report = {
    "train_rows": int(len(train_df)),
    "test_exact_id_overlap": int(test_df["id"].isin(train_df["id"]).sum()),
    "test_exact_prompt_overlap": int(test_df["prompt"].isin(train_df["prompt"]).sum()),
    "test_predictions": submission_df.to_dict(orient="records"),
}

with open(f"{OUTPUT_DIR}/rule_solver_report.json", "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, ensure_ascii=True)

print(submission_df)
print("submission.csv written to /kaggle/working/submission.csv")
print("submission.zip written to /kaggle/working/submission.zip")
