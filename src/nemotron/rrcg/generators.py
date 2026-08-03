from typing import Type, override

from puzzle.puzzle_type import PuzzleType
from rrcg.generator import Generator


class BitGenerator(Generator):
    @override
    def generate_reasoning_content(self) -> str:
        return ''


class GravityGenerator(Generator):
    @override
    def generate_reasoning_content(self) -> str:
        return ''


class UnitGenerator(Generator):
    @override
    def generate_reasoning_content(self) -> str:
        return ''


class RomanGenerator(Generator):
    @override
    def generate_reasoning_content(self) -> str:
        return ''


class CipherGenerator(Generator):
    @override
    def generate_reasoning_content(self) -> str:
        return ''


class SymbolGenerator(Generator):
    @override
    def generate_reasoning_content(self) -> str:
        return ''


def get_generator_class_by_puzzle_type(
    puzzle_type: PuzzleType,
) -> Type[Generator]:
    """Returns the appropriate generator based on the puzzle type.

    Args:
        puzzle_type: The type of the puzzle.

    Returns:
        An instance of a generator corresponding to the puzzle type.
    """
    if puzzle_type == 'bit':
        return BitGenerator
    elif puzzle_type == 'gravity':
        return GravityGenerator
    elif puzzle_type == 'unit':
        return UnitGenerator
    elif puzzle_type == 'roman':
        return RomanGenerator
    elif puzzle_type == 'cipher':
        return CipherGenerator
    elif puzzle_type == 'symbol':
        return SymbolGenerator
    else:
        raise ValueError(f'Unsupported puzzle type: {puzzle_type}')
