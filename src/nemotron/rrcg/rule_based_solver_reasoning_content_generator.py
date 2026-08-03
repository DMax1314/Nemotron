from puzzle.puzzle_type import PuzzleType, infer_puzzle_type
from rrcg.generator import Generator
from rrcg.generators import get_generator_class_by_puzzle_type


class RuleBasedSolverReasoningContentGenerator:
    def __init__(self, puzzle: str):
        self.puzzle: str = puzzle
        self.puzzle_type: PuzzleType = infer_puzzle_type(puzzle)
        self.generator: Generator = get_generator_class_by_puzzle_type(
            self.puzzle_type
        )(puzzle)

    def generate_reasoning_content(self) -> str:
        """Generates reasoning content for the given puzzle.

        Returns:
            A string containing the reasoning content generated for the puzzle.
        """

        return self.generator.generate_reasoning_content()
