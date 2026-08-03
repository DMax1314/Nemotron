import math
import re


def extract_final_answer(text: str) -> str | None:
    """Extracts the final answer from a text string using a cascade of
    strategies.

    This function returns the last non-empty ``\\boxed{...}`` content. Handles
    truncated boxes (missing closing brace) via ``(?:\\}|$)``.

    Example: ``\\boxed{42}`` → ``"42"``

    Args:
        text: The text to extract the final answer from.

    Returns:
        The extracted answer as a stripped string, or ``None`` if the text
        contains no non-empty lines.
    """

    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    return None


def verify(
    correct_answer: str,
    predicted_answer: str,
    *,
    compare_float_closeness: bool = True,
) -> bool:
    """Verifies if the predicted answer is the same as the correct answer.

    This function first strips whitespaces from two given answers, and convert
    all letters in both answers into lowercase.

    Then, it compares the two strings, and return True if they are the
    identical. If the two strings are not identical, it converts both answers
    into floating-point numbers and check if they are close enough. Two numbers
    are considered to be close enough if

        1. the relative difference is less than 1%, or
        2. the absolute difference is at most 0.00001.

    It returns True if the two numbers are close enough.

    Args:
        correct_answer: The correct answer to compare with.
        predicted_answer: The predicted answer to compare with.
        compare_float_closeness: Whether to compare float closeness when the two
        answers are not identical.

    Returns:
        True if the predicted answer is considered correct, and False otherwise.
    """

    correct_answer = correct_answer.strip().lower()
    predicted_answer = predicted_answer.strip().lower()

    if correct_answer == predicted_answer:
        return True

    if not compare_float_closeness:
        return False

    try:
        return math.isclose(
            float(correct_answer),
            float(predicted_answer),
            rel_tol=1e-2,
            abs_tol=1e-5,
        )
    except Exception:
        return False


def should_compare_closeness(puzzle_type: str) -> bool:
    """Determines whether to compare float closeness based on the puzzle type.

    For certain puzzle types such as 'gravity' and 'unit', the final answer is a
    real number, and it is reasonable to consider a predicted answer correct if
    it is close enough to the correct answer.

    Args:
        puzzle_type: The type of the puzzle, as inferred by infer_puzzle_type().
    """

    return puzzle_type == 'gravity' or puzzle_type == 'unit'
