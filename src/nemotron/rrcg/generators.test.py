"""Regression test for rule-based reasoning-content generation.

Run directly with:
``source env.sh && python src/nemotron/rrcg/generators.test.py``.

The training data is intentionally used as a complete regression corpus: each
prompt must produce an answer that passes the same comparison rule used by the
challenge evaluator.
"""

import unittest
from pathlib import Path
from typing import Any, cast

from nemotron.common.io import load_csv
from nemotron.configs import RAW_DATA_DIR
from nemotron.puzzle.answer import should_compare_closeness, verify
from nemotron.puzzle.puzzle_type import infer_puzzle_type
from nemotron.rrcg.rule_based_reasoning_content_generator import (
    RuleBasedReasoningContentGenerator,
)


class TestRuleBasedReasoningContentGenerator(unittest.TestCase):
    """Check every public training puzzle against its known answer."""

    def test_training_answers(self) -> None:
        """Generate and verify an answer for every training prompt."""
        training_data = load_csv(Path(RAW_DATA_DIR) / 'train.csv')
        failures: list[str] = []

        num_rows: int = len(training_data)
        i: int = 0
        print()
        for row in training_data.itertuples(index=False):
            row = cast(Any, row)

            i += 1
            print(f'[{i}/{num_rows}] Checking {row.id}...')

            puzzle_type = infer_puzzle_type(row.prompt)
            try:
                answer, _ = RuleBasedReasoningContentGenerator(
                    row.prompt
                ).generate_reasoning_content()
            except Exception as error:
                failures.append(f'{row.id}: {type(error).__name__}: {error}')
                continue

            if not verify(
                str(row.answer),
                answer,
                compare_float_closeness=should_compare_closeness(puzzle_type),
            ):
                failures.append(
                    f'{row.id}: expected {row.answer!r}, got {answer!r}'
                )

        self.assertFalse(
            failures,
            'Rule-based generator failed training examples:\n'
            + '\n'.join(failures),
        )


if __name__ == '__main__':
    unittest.main()
