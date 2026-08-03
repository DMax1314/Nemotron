from abc import ABC, abstractmethod


class Generator(ABC):
    """Represents a rule-based reasoning content generator (RRCG) for a given
    puzzle. For simplicity, the class name is just "Generator".

    Given a puzzle, the generator first solves it and then produces a
    self-contained explanation of that solution. The answer is returned
    separately so callers can use it without extracting it from the prose.

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
    def generate_reasoning_content(self) -> tuple[str, str]:
        """Solve the puzzle and generate its reasoning content.

        Implementations must derive an answer and a complete natural-language
        explanation that is consistent with that answer.

        Returns:
            A pair ``(answer, reasoning_content)``. ``answer`` is the exact
            answer string, while ``reasoning_content`` contains the complete
            explanation and its final boxed answer marker.
        """
