from typing import Literal

# Define constants for the puzzle types. Note that in the public training set
# and test set, no puzzles have the type "other".
PUZZLE_BIT = 'bit'
PUZZLE_GRAVITY = 'gravity'
PUZZLE_UNIT = 'unit'
PUZZLE_ROMAN = 'roman'
PUZZLE_CIPHER = 'cipher'
PUZZLE_SYMBOL = 'symbol'
PUZZLE_OTHER = 'other'

# Define a type for the puzzle types.
PuzzleType = Literal[
    'bit',
    'gravity',
    'unit',
    'roman',
    'cipher',
    'symbol',
    'other',
]


def infer_puzzle_type(prompt: str) -> PuzzleType:
    """Infers the puzzle type based on the first line of the prompt.

    This function examines the first line of the input prompt and checks for
    specific keywords to determine the puzzle type.

    Args:
        prompt: The input prompt to classify.

    Returns:
        A string representing the puzzle type.
    """

    first_line = prompt.split('\n', 1)[0]

    if 'bit' in first_line:
        return PUZZLE_BIT
    if 'gravitational' in first_line:
        return PUZZLE_GRAVITY
    if 'unit' in first_line:
        return PUZZLE_UNIT
    if 'numeral' in first_line:
        return PUZZLE_ROMAN
    if 'encryption' in first_line:
        return PUZZLE_CIPHER
    if 'equations' in first_line:
        return PUZZLE_SYMBOL

    return PUZZLE_OTHER
