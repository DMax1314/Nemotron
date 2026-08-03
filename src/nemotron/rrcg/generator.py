from abc import ABC, abstractmethod


class Generator(ABC):
    """Represents a rule-based reasoning content generator (RRCG) for a given
    puzzle. For simplicity, the class name is just "Generator".

    Given a puzzle, the generator first solves the puzzle and then generates
    reasoning content based on the solution.

    Every puzzle in this challenge is solvable. Some puzzles may be solved
    within several steps, others may require brute-force search.

    Attributes:
        puzzle: The puzzle for which the reasoning content is generated.
    """

    def __init__(self, puzzle: str):
        """Initializes a Generator instance with the given puzzle.

        Args:
            puzzle: The puzzle for which the reasoning content is generated.
        """

        self.puzzle: str = puzzle

    @abstractmethod
    def generate_reasoning_content(self) -> str:
        """Generates reasoning content for the given puzzle.

        Implementation of this method should solve the puzzle and generate
        a complete reasoning content in natural language.

        Returns:
            A string containing the reasoning content generated for the puzzle.
        """

        pass
