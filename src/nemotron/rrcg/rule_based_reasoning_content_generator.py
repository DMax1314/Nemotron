from nemotron.puzzle.puzzle_type import PuzzleType, infer_puzzle_type
from nemotron.rrcg.generator import Generator
from nemotron.rrcg.generators import get_generator_class_by_puzzle_type


class RuleBasedReasoningContentGenerator:
    def __init__(self, puzzle: str):
        self.puzzle: str = puzzle
        self.puzzle_type: PuzzleType = infer_puzzle_type(puzzle)
        self.generator: Generator = get_generator_class_by_puzzle_type(
            self.puzzle_type
        )(puzzle)

    def generate_reasoning_content(self) -> tuple[str, str]:
        """Solve the puzzle and return its answer and reasoning content.

        Returns:
            A pair ``(answer, reasoning_content)`` produced by the
            type-specific generator.
        """

        return self.generator.generate_reasoning_content()
