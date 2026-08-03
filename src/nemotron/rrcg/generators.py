from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Iterable, Type, override

from nemotron.puzzle.puzzle_type import PuzzleType
from nemotron.rrcg.generator import Generator

_WORD_LIST = (
    'above alice ancient around beyond bird book bright castle cat cave chases '
    'clever colorful creates crystal curious dark discovers door dragon draws '
    'dreams explores follows forest found garden golden hatter hidden imagines '
    'in inside island key king knight library magical map message mirror '
    'mountain mouse mysterious near ocean palace potion princess puzzle queen '
    'rabbit reads school secret sees silver story strange student studies '
    'teacher the through tower treasure turtle under valley village watches '
    'wise wizard wonderland writes'
).split()


def _finish(lines: list[str], answer: str) -> tuple[str, str]:
    """Return the answer and reasoning text with its boxed answer marker."""
    lines.extend(('', f'Therefore, the answer is \\boxed{{{answer}}}.'))
    return answer, '\n'.join(lines)


def _decimal_text(value: Decimal, places: int = 2) -> str:
    """Format numerical puzzle answers to the precision used in the prompts."""
    quantized = value.quantize(
        Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP
    )
    return f'{quantized:.{places}f}'


def _examples_with_arrow(puzzle: str) -> list[tuple[str, str]]:
    """Return the prompt's ``input -> output`` rows without prose lines."""
    return [
        (match.group(1), match.group(2))
        for match in re.finditer(r'^(.+?) -> (.+)$', puzzle, re.MULTILINE)
    ]


class BitGenerator(Generator):
    """Infer an eight-bit Boolean circuit from its input/output examples.

    Each output bit is searched over the documented primitive operations.  A
    candidate is preferred when its operand positions continue the same
    rotation/shift pattern across neighbouring output bits; this disambiguates
    accidental matches from a small example set while retaining support for
    mixed circuits and majority/choice gates.
    """

    _FAMILY_COST = {
        'I': 1,
        'NOT': 1,
        'AND': 2,
        'OR': 2,
        'XOR': 2,
        'AND-NOT': 2,
        'OR-NOT': 2,
        'XOR-NOT': 2,
        'MAJ': 3,
        'CHO': 3,
        '0': 4,
        '1': 4,
        'TABLE': 5,
    }

    @dataclass(frozen=True)
    class _BitRule:
        family: str
        inputs: tuple[int, ...]
        table: tuple[int, ...] = ()

        def evaluate(self, bits: str) -> int:
            values = [int(bits[index]) for index in self.inputs]
            if self.family == '0':
                return 0
            if self.family == '1':
                return 1
            if self.family == 'I':
                return values[0]
            if self.family == 'NOT':
                return 1 - values[0]
            if self.family == 'AND':
                return values[0] & values[1]
            if self.family == 'OR':
                return values[0] | values[1]
            if self.family == 'XOR':
                return values[0] ^ values[1]
            if self.family == 'AND-NOT':
                return values[0] & (1 - values[1])
            if self.family == 'OR-NOT':
                return values[0] | (1 - values[1])
            if self.family == 'XOR-NOT':
                return values[0] ^ (1 - values[1])
            if self.family == 'MAJ':
                return int(sum(values) >= 2)
            if self.family == 'CHO':
                # A choice gate selects its second input when its first is 1.
                return values[1] if values[0] else values[2]
            if self.family == 'TABLE':
                table_index = values[0] * 4 + values[1] * 2 + values[2]
                return self.table[table_index]
            raise ValueError(f'Unsupported bit operation: {self.family}')

        def description(self) -> str:
            if self.family in {'0', '1'}:
                return self.family
            indexes = ', '.join(f'b{index}' for index in self.inputs)
            if self.family == 'I':
                return f'b{self.inputs[0]}'
            if self.family == 'NOT':
                return f'NOT(b{self.inputs[0]})'
            if self.family == 'TABLE':
                rows = ', '.join(
                    f'{value:03b}→{result}'
                    for value, result in enumerate(self.table)
                )
                return f'Boolean table on ({indexes}): {rows}'
            return f'{self.family}({indexes})'

    @classmethod
    def _rules_for_column(
        cls,
        inputs: list[str],
        output_column: str,
        *,
        include_table: bool = True,
    ) -> list[_BitRule]:
        """Enumerate primitive rules that agree with one output column."""
        rules: list[BitGenerator._BitRule] = []
        for family, arity in (
            ('I', 1),
            ('NOT', 1),
            ('AND', 2),
            ('OR', 2),
            ('XOR', 2),
            ('AND-NOT', 2),
            ('OR-NOT', 2),
            ('XOR-NOT', 2),
            ('MAJ', 3),
            ('CHO', 3),
        ):
            for indexes in itertools.product(range(8), repeat=arity):
                # Repeating arguments only recreates unary/constant cases.
                if family in {'MAJ', 'CHO'} and len(set(indexes)) != arity:
                    continue
                rule = cls._BitRule(family, indexes)
                values = ''.join(str(rule.evaluate(bits)) for bits in inputs)
                if values == output_column:
                    rules.append(rule)
        if output_column == '0' * len(inputs):
            rules.append(cls._BitRule('0', ()))
        if output_column == '1' * len(inputs):
            rules.append(cls._BitRule('1', ()))
        if not rules and include_table:
            # A few puzzles compose several listed operations.  Their output
            # still depends on only three source bits, so infer that compact
            # truth table rather than abandoning an otherwise solvable prompt.
            for indexes in itertools.permutations(range(8), 3):
                observed: dict[int, int] = {}
                consistent = True
                for bits, expected in zip(inputs, output_column, strict=True):
                    key = (
                        int(bits[indexes[0]]) * 4
                        + int(bits[indexes[1]]) * 2
                        + int(bits[indexes[2]])
                    )
                    value = int(expected)
                    if key in observed and observed[key] != value:
                        consistent = False
                        break
                    observed[key] = value
                if consistent:
                    table = tuple(observed.get(index, 0) for index in range(8))
                    rules.append(cls._BitRule('TABLE', indexes, table))
        return rules

    @classmethod
    def _rule_score(
        cls,
        bit_index: int,
        rule: _BitRule,
        candidates: list[list[_BitRule]],
    ) -> tuple[int, int, int, tuple[int, ...], str]:
        """Rank a matching rule by its continuation with adjacent bit rules."""
        offsets = tuple((source - bit_index) % 8 for source in rule.inputs)
        continuity = 0
        for other_index, other_rules in enumerate(candidates):
            if other_index == bit_index:
                continue
            expected = tuple((other_index + offset) % 8 for offset in offsets)
            if any(
                other.family == rule.family and other.inputs == expected
                for other in other_rules
            ):
                continuity += 1
        # Fewer operands, then a stable lexical tie-break, keep results
        # deterministic when examples cannot distinguish two circuits.
        return (
            continuity,
            -cls._FAMILY_COST[rule.family],
            -len(rule.inputs),
            tuple(-item for item in rule.inputs),
            rule.family,
        )

    @staticmethod
    def _word_nodes(
        values: tuple[int, ...],
    ) -> list[tuple[str, tuple[int, ...]]]:
        """Build the operand words used by the whole-byte circuit loop."""
        nodes: list[tuple[str, tuple[int, ...]]] = [('x', values)]
        for amount in range(1, 8):
            nodes.extend(
                (
                    (
                        f'SHL{amount}(x)',
                        tuple((value << amount) & 0xFF for value in values),
                    ),
                    (
                        f'SHR{amount}(x)',
                        tuple(value >> amount for value in values),
                    ),
                    (
                        f'ROL{amount}(x)',
                        tuple(
                            ((value << amount) | (value >> (8 - amount))) & 0xFF
                            for value in values
                        ),
                    ),
                    (
                        f'ROR{amount}(x)',
                        tuple(
                            (value >> amount) | ((value << (8 - amount)) & 0xFF)
                            for value in values
                        ),
                    ),
                )
            )
        nodes.extend(
            (
                f'NOT({description})',
                tuple((~value) & 0xFF for value in result),
            )
            for description, result in tuple(nodes)
        )
        return nodes

    @staticmethod
    def _word_binary_operations() -> tuple[
        tuple[str, Callable[[int, int], int]], ...
    ]:
        """Return the ordered binary operations in the circuit enumeration."""
        return (
            ('XOR', lambda a, b: a ^ b),
            ('AND', lambda a, b: a & b),
            ('OR', lambda a, b: a | b),
            ('AND-NOT', lambda a, b: a & (~b & 0xFF)),
            ('OR-NOT', lambda a, b: a | (~b & 0xFF)),
            ('XOR-NOT', lambda a, b: a ^ (~b & 0xFF)),
        )

    @staticmethod
    def _word_rule(
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        question: str,
        *,
        allow_compound: bool = False,
    ) -> tuple[str, str, tuple[str, ...]] | None:
        """Find a whole-word circuit before falling back to per-bit matching.

        Shifts fill with zeroes at an edge, so their formulas cannot always be
        represented by a single circular per-bit rule.  Testing the documented
        operations on full bytes handles those boundary cases exactly.
        """
        values = tuple(int(value, 2) for value in (*inputs, question))
        expected = tuple(int(value, 2) for value in outputs)
        nodes = BitGenerator._word_nodes(values)

        def found(
            description: str, result: tuple[int, ...]
        ) -> tuple[str, str, tuple[str, ...]] | None:
            if result[:-1] == expected:
                rendered = tuple(f'{value:08b}' for value in result)
                return description, rendered[-1], rendered
            return None

        for description, result in nodes:
            match = found(description, result)
            if match:
                return match
        for left_name, left in nodes:
            for right_name, right in nodes:
                for family, operation in BitGenerator._word_binary_operations():
                    match = found(
                        f'{family}({left_name}, {right_name})',
                        tuple(
                            operation(a, b) & 0xFF for a, b in zip(left, right)
                        ),
                    )
                    if match:
                        return match
        for first_name, first in nodes:
            for second_name, second in nodes:
                for third_name, third in nodes:
                    majority = tuple(
                        (a & b) | (a & c) | (b & c)
                        for a, b, c in zip(first, second, third)
                    )
                    match = found(
                        f'MAJ({first_name}, {second_name}, {third_name})',
                        majority,
                    )
                    if match:
                        return match
                    choice = tuple(
                        (a & b) | ((~a & 0xFF) & c)
                        for a, b, c in zip(first, second, third)
                    )
                    match = found(
                        f'CHO({first_name}, {second_name}, {third_name})',
                        choice,
                    )
                    if match:
                        return match
        if not allow_compound:
            return None

        # Deduplicating by behaviour keeps the second layer compact: many
        # Boolean spellings have exactly the same output vector on a prompt.
        binary_nodes: list[tuple[str, tuple[int, ...]]] = []
        seen = {result for _, result in nodes}
        for left_name, left in nodes:
            for right_name, right in nodes:
                for family, operation in BitGenerator._word_binary_operations():
                    result = tuple(
                        operation(a, b) & 0xFF for a, b in zip(left, right)
                    )
                    if result not in seen:
                        seen.add(result)
                        binary_nodes.append(
                            (f'{family}({left_name}, {right_name})', result)
                        )
        for derived_name, derived in binary_nodes:
            for base_name, base in nodes:
                for family, operation in BitGenerator._word_binary_operations():
                    for left_name, left, right_name, right in (
                        (derived_name, derived, base_name, base),
                        (base_name, base, derived_name, derived),
                    ):
                        match = found(
                            f'{family}({left_name}, {right_name})',
                            tuple(
                                operation(a, b) & 0xFF
                                for a, b in zip(left, right)
                            ),
                        )
                        if match:
                            return match
        return None

    @staticmethod
    def _word_working(expression: str, bits: str) -> list[str]:
        """Evaluate a rendered word expression from its innermost terms out."""
        input_value = int(bits, 2)
        lines: list[str] = []

        def arguments(text: str) -> list[str]:
            values: list[str] = []
            start = depth = 0
            for index, character in enumerate(text):
                if character == '(':
                    depth += 1
                elif character == ')':
                    depth -= 1
                elif character == ',' and depth == 0:
                    values.append(text[start:index].strip())
                    start = index + 1
            values.append(text[start:].strip())
            return values

        def evaluate(text: str) -> int:
            if text == 'x':
                return input_value
            shift = re.fullmatch(r'(SHL|SHR|ROL|ROR)([1-7])\(x\)', text)
            if shift:
                operation, amount_text = shift.groups()
                amount = int(amount_text)
                if operation == 'SHL':
                    result = (input_value << amount) & 0xFF
                elif operation == 'SHR':
                    result = input_value >> amount
                elif operation == 'ROL':
                    result = (
                        (input_value << amount) | (input_value >> (8 - amount))
                    ) & 0xFF
                else:
                    result = (input_value >> amount) | (
                        (input_value << (8 - amount)) & 0xFF
                    )
                lines.append(f'{text} = {result:08b}')
                return result
            name, inner = text.split('(', 1)
            values = [evaluate(item) for item in arguments(inner[:-1])]
            if name == 'NOT':
                result = (~values[0]) & 0xFF
            elif name == 'XOR':
                result = values[0] ^ values[1]
            elif name == 'AND':
                result = values[0] & values[1]
            elif name == 'OR':
                result = values[0] | values[1]
            elif name == 'AND-NOT':
                result = values[0] & (~values[1] & 0xFF)
            elif name == 'OR-NOT':
                result = values[0] | (~values[1] & 0xFF)
            elif name == 'XOR-NOT':
                result = values[0] ^ (~values[1] & 0xFF)
            elif name == 'MAJ':
                result = (
                    (values[0] & values[1])
                    | (values[0] & values[2])
                    | (values[1] & values[2])
                )
            elif name == 'CHO':
                result = (values[0] & values[1]) | (
                    (~values[0] & 0xFF) & values[2]
                )
            else:
                raise ValueError(f'Unsupported word operation: {name}')
            lines.append(f'{text} = {result:08b}')
            return result

        evaluate(expression)
        return lines

    @classmethod
    def _word_enumeration_lines(
        cls,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        expression: str,
    ) -> list[str]:
        """Render the complete, compact audit trail of the circuit loops."""
        target = tuple(int(value, 2) for value in outputs)
        nodes = cls._word_nodes(tuple(int(value, 2) for value in inputs))

        def vector(values: tuple[int, ...]) -> str:
            return ' | '.join(f'{value:08b}' for value in values)

        lines = [
            'Keep the examples in their given order. The required output '
            f'vector is: {vector(target)}.',
            '',
            'First enumerate every one-word operand (x, every nonzero shift '
            'and rotation, then NOT of each of those 29 words):',
        ]
        lines.extend(f'{name}: {vector(values)}' for name, values in nodes)

        unary_matches = [name for name, values in nodes if values == target]
        lines.extend(
            (
                '',
                'Unary loop: compare all 58 vectors with the target. '
                'Exact matches: '
                + (', '.join(unary_matches) if unary_matches else 'none')
                + '.',
                'Binary loop: for each named operation, evaluate all '
                '58 × 58 ordered operand pairs. Nonlisted pairs differ from '
                'the target vector.',
                'The loop order is left operand, right operand, then XOR, '
                'AND, OR, AND-NOT, OR-NOT, XOR-NOT.',
            )
        )
        binary_matches: dict[str, list[str]] = {
            name: [] for name, _ in cls._word_binary_operations()
        }
        for left_name, left in nodes:
            for right_name, right in nodes:
                for family, operation in cls._word_binary_operations():
                    value = tuple(
                        operation(a, b) & 0xFF
                        for a, b in zip(left, right, strict=True)
                    )
                    if value == target:
                        binary_matches[family].append(
                            f'{family}({left_name}, {right_name})'
                        )
        lines.extend(
            f'{family}: 3364 pairs tested; '
            + (', '.join(matches) if matches else 'no exact match')
            for family, matches in binary_matches.items()
        )

        if 'MAJ(' not in expression and 'CHO(' not in expression:
            if (
                expression
                not in {
                    match
                    for matches in binary_matches.values()
                    for match in matches
                }
                and not unary_matches
            ):
                lines.extend(
                    (
                        '',
                        'No one-layer formula matches, so the next loop forms '
                        'each binary intermediate in the same order, keeps one '
                        'copy of each distinct vector, then combines every '
                        'intermediate with every base word in both orders.',
                        'The first exact formula is the two-layer expression '
                        'reported below.',
                    )
                )
            return lines

        lines.extend(
            (
                '',
                'Three-input loop: evaluate MAJ and CHO on all 58 × 58 × 58 '
                'ordered triples. Nonlisted triples differ from the target.',
                'The loop order is first operand, second operand, third '
                'operand, then MAJ followed by CHO.',
            )
        )
        ternary_matches = {'MAJ': [], 'CHO': []}
        for first_name, first in nodes:
            for second_name, second in nodes:
                for third_name, third in nodes:
                    majority = tuple(
                        (a & b) | (a & c) | (b & c)
                        for a, b, c in zip(first, second, third, strict=True)
                    )
                    if majority == target:
                        ternary_matches['MAJ'].append(
                            f'MAJ({first_name}, {second_name}, {third_name})'
                        )
                    choice = tuple(
                        (a & b) | ((~a & 0xFF) & c)
                        for a, b, c in zip(first, second, third, strict=True)
                    )
                    if choice == target:
                        ternary_matches['CHO'].append(
                            f'CHO({first_name}, {second_name}, {third_name})'
                        )
        lines.extend(
            f'{family}: 195112 triples tested; '
            + (', '.join(matches) if matches else 'no exact match')
            for family, matches in ternary_matches.items()
        )
        return lines

    @override
    def generate_reasoning_content(self) -> tuple[str, str]:
        examples = [
            (source, target)
            for source, target in _examples_with_arrow(self.puzzle)
            if re.fullmatch(r'[01]{8}', source)
            and re.fullmatch(r'[01]{8}', target)
        ]
        question_match = re.search(r'output for:\s*([01]{8})', self.puzzle)
        if not examples or question_match is None:
            raise ValueError('Invalid bit puzzle format.')

        inputs, outputs = zip(*examples, strict=True)
        question = question_match.group(1)
        word_rule = self._word_rule(
            inputs, outputs, question, allow_compound=True
        )
        if word_rule is not None:
            description, result, calculated_rows = word_rule
            lines = [
                'Treat x as an eight-bit word. Shifts fill vacant positions '
                'with 0, rotations wrap around, and NOT flips every bit.',
                '',
            ]
            lines.extend(
                self._word_enumeration_lines(inputs, outputs, description)
            )
            lines.extend(
                (
                    '',
                    'The first exact formula in that loop order is:',
                    f'output = {description}.',
                )
            )
            if 'CHO(' in description:
                lines.append('CHO(a, b, c) means (a AND b) OR (NOT(a) AND c).')
            if 'MAJ(' in description:
                lines.append(
                    'MAJ(a, b, c) is 1 at each bit where at least two '
                    'inputs are 1.'
                )
            for family in ('AND-NOT', 'OR-NOT', 'XOR-NOT'):
                if re.search(rf'(?<![A-Z]){family}\(', description):
                    base = family.removesuffix('-NOT')
                    lines.append(f'{family}(a, b) means {base}(a, NOT(b)).')
            lines.extend(('', 'Verification against the supplied examples:'))
            for (source, target), calculated in zip(
                examples, calculated_rows[:-1], strict=True
            ):
                status = 'match' if calculated == target else 'mismatch'
                lines.append(
                    f'{source} -> {calculated}; expected {target}: {status}.'
                )
            lines.extend(
                (
                    '',
                    f'Question input: {question}.',
                )
            )
            lines.append('Evaluate the formula from its innermost terms:')
            lines.extend(self._word_working(description, question))
            return _finish(lines, result)
        candidates = [
            self._rules_for_column(
                list(inputs),
                ''.join(value[index] for value in outputs),
                include_table=False,
            )
            for index in range(8)
        ]
        if any(not options for options in candidates):
            candidates = [
                self._rules_for_column(
                    list(inputs), ''.join(value[index] for value in outputs)
                )
                for index in range(8)
            ]
        if any(not options for options in candidates):
            raise ValueError('No documented Boolean rule fits a bit position.')

        chosen = [
            max(
                options,
                key=lambda rule, bit=index: self._rule_score(
                    bit, rule, candidates
                ),
            )
            for index, options in enumerate(candidates)
        ]
        result = ''.join(str(rule.evaluate(question)) for rule in chosen)

        lines = [
            'No one whole-byte expression covers all eight output positions. '
            'For each position, enumerate I, NOT, every ordered pair for the '
            'six binary operations, and every ordered triple for MAJ and CHO.',
            'A formula is listed only when its values down all examples equal '
            'the output column. Thus the lists below are the complete results '
            'of those loops.',
            '',
            'Exact formulas for each output column:',
        ]
        for index, rule in enumerate(chosen):
            column = ''.join(value[index] for value in outputs)
            scored = sorted(
                (
                    (
                        self._rule_score(index, option, candidates),
                        option,
                    )
                    for option in candidates[index]
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            lines.append(f'bit {index}, target column {column}:')
            lines.extend(
                f'  {option.description()}; continuity score {score[0]}.'
                for score, option in scored
            )
            lines.append(f'  Use {rule.description()}.')
        lines.extend(
            (
                '',
                'The continuity score counts other output positions that '
                'accept the same operation with the same relative input-bit '
                'offsets. The listed rule has the largest full tie-break '
                'score: continuity, then lower operation cost, fewer inputs, '
                'and '
                'stable input-index/family order.',
            )
        )
        lines.extend(('', f'Apply these rules to {question}:'))
        for index, rule in enumerate(chosen):
            source_bits = ', '.join(
                f'b{source}={question[source]}' for source in rule.inputs
            )
            lines.append(
                f'bit {index}: {rule.description()} on '
                f'{source_bits or "no inputs"} '
                f'= {rule.evaluate(question)}.'
            )
        lines.append(f'The output bits in order are {result}.')
        return _finish(lines, result)


class GravityGenerator(Generator):
    """Fit the proportionality constant in ``d = 0.5 * g * t^2``."""

    @override
    def generate_reasoning_content(self) -> tuple[str, str]:
        observations = [
            (Decimal(time), Decimal(distance))
            for time, distance in re.findall(
                r't\s*=\s*([\d.]+)s, distance\s*=\s*([\d.]+)\s*m',
                self.puzzle,
            )
        ]
        question_match = re.search(
            r'falling distance for t\s*=\s*([\d.]+)s', self.puzzle
        )
        if not observations or question_match is None:
            raise ValueError('Invalid gravity puzzle format.')

        with localcontext() as context:
            context.prec = 40
            coefficients = [
                Decimal('0.5') * time * time for time, _ in observations
            ]
            distances = [distance for _, distance in observations]
            # A distance displayed to two decimal places represents an
            # interval of width 0.01. Intersect the corresponding intervals
            # for g instead of privileging an arbitrary example.
            intervals = [
                (
                    (distance - Decimal('0.005')) / coefficient,
                    (distance + Decimal('0.005')) / coefficient,
                )
                for coefficient, distance in zip(
                    coefficients, distances, strict=True
                )
            ]
            lower = max(interval[0] for interval in intervals)
            upper = min(interval[1] for interval in intervals)
            gravity = (lower + upper) / 2
            question_time = Decimal(question_match.group(1))
            answer_distance = Decimal('0.5') * gravity * question_time**2

        answer = _decimal_text(answer_distance)
        lines = [
            'Start from d = 0.5 × g × t², so g = d / (0.5 × t²).',
            'Each shown distance is rounded to 0.01 m. Thus a displayed '
            'distance d represents d - 0.005 ≤ actual d < d + 0.005.',
        ]
        for (
            (observation_time, observation_distance),
            coefficient,
            interval,
        ) in zip(observations, coefficients, intervals, strict=True):
            lower_distance = observation_distance - Decimal('0.005')
            upper_distance = observation_distance + Decimal('0.005')
            lines.extend(
                (
                    f't={observation_time}: 0.5 × t² = '
                    f'0.5 × {observation_time}² = {coefficient}.',
                    f'd={observation_distance} means {lower_distance} ≤ d < '
                    f'{upper_distance}; therefore '
                    f'{_decimal_text(interval[0], 6)} ≤ g < '
                    f'{_decimal_text(interval[1], 6)}.',
                )
            )
        lines.extend(
            (
                'All rows must hold, so intersect those ranges: '
                f'{_decimal_text(lower, 6)} ≤ g < '
                f'{_decimal_text(upper, 6)}.',
                'Use the midpoint of that shared interval: '
                f'g ≈ {_decimal_text(gravity, 8)}.',
                f'For t={question_time}, 0.5 × t² = '
                f'0.5 × {question_time}² = '
                f'{Decimal("0.5") * question_time**2}; '
                f'd ≈ {Decimal("0.5") * question_time**2} × '
                f'{_decimal_text(gravity, 8)} = '
                f'{_decimal_text(answer_distance, 6)}, which rounds to '
                f'{answer}.',
            )
        )
        return _finish(lines, answer)


class UnitGenerator(Generator):
    """Fit the constant multiplicative conversion factor."""

    @override
    def generate_reasoning_content(self) -> tuple[str, str]:
        observations = [
            (Decimal(source), Decimal(target))
            for source, target in re.findall(
                r'([\d.]+)\s*m becomes\s*([\d.]+)', self.puzzle
            )
        ]
        question_match = re.search(
            r'following measurement:\s*([\d.]+)\s*m', self.puzzle
        )
        if not observations or question_match is None:
            raise ValueError('Invalid unit-conversion puzzle format.')

        with localcontext() as context:
            context.prec = 40
            # The converted value is shown to two decimal places. Convert
            # every rounding interval into an interval for the scale factor.
            intervals = [
                (
                    (target - Decimal('0.005')) / source,
                    (target + Decimal('0.005')) / source,
                )
                for source, target in observations
            ]
            lower = max(interval[0] for interval in intervals)
            upper = min(interval[1] for interval in intervals)
            factor = (lower + upper) / 2
            question = Decimal(question_match.group(1))
            converted = question * factor

        answer = _decimal_text(converted)
        lines = [
            'The examples use one multiplicative conversion factor: output = '
            'factor × input.',
            'Each output is displayed to two decimal places, so an output y '
            'means y - 0.005 ≤ actual output < y + 0.005.',
        ]
        for (source, target), interval in zip(
            observations, intervals, strict=True
        ):
            lines.append(
                f'{source} → {target}: ({target} - 0.005) / {source} ≤ '
                f'factor < ({target} + 0.005) / {source}, hence '
                f'{_decimal_text(interval[0], 6)} ≤ factor < '
                f'{_decimal_text(interval[1], 6)}.'
            )
        lines.extend(
            (
                'All examples must share the same factor, so intersect the '
                f'ranges: {_decimal_text(lower, 6)} ≤ factor < '
                f'{_decimal_text(upper, 6)}.',
                'Use the midpoint of that common range: factor ≈ '
                f'{_decimal_text(factor, 8)}.',
                f'{question} × {_decimal_text(factor, 8)} = '
                f'{_decimal_text(converted, 6)}, which rounds to {answer}.',
            )
        )
        return _finish(lines, answer)


class RomanGenerator(Generator):
    """Convert the requested Arabic number to standard Roman notation."""

    _TOKENS = (
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'),
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'),
        (9, 'IX'),
        (5, 'V'),
        (4, 'IV'),
        (1, 'I'),
    )

    @classmethod
    def _to_roman(cls, number: int) -> str:
        if number <= 0:
            raise ValueError('Roman numerals require a positive number.')
        parts: list[str] = []
        for value, token in cls._TOKENS:
            count, number = divmod(number, value)
            parts.append(token * count)
        return ''.join(parts)

    @classmethod
    def _decomposition(cls, number: int) -> list[str]:
        """Return the visible greedy Roman-token decomposition."""
        remainder = number
        parts: list[str] = []
        for value, token in cls._TOKENS:
            count, remainder = divmod(remainder, value)
            if count:
                parts.append(f'{value * count} = {token * count}')
        return parts

    @override
    def generate_reasoning_content(self) -> tuple[str, str]:
        match = re.search(
            r'write the number\s+(\d+)\s+in the Wonderland', self.puzzle
        )
        if match is None:
            raise ValueError('Invalid Roman-numeral puzzle format.')
        number = int(match.group(1))
        answer = self._to_roman(number)
        examples = [
            (int(source), target)
            for source, target in _examples_with_arrow(self.puzzle)
            if source.isdigit() and re.fullmatch(r'[IVXLCDM]+', target)
        ]
        if not examples:
            raise ValueError('Roman puzzle has no numeral examples.')
        lines = [
            'The examples use Roman place-value tokens: I=1, V=5, X=10, '
            'L=50, C=100, D=500, and M=1000.',
            'A smaller token before a larger one is subtracted, giving '
            'IV=4, IX=9, XL=40, XC=90, CD=400, and CM=900.',
            'Check the supplied examples by repeatedly taking the largest '
            'token that fits:',
        ]
        for source, expected in examples:
            calculated = self._to_roman(source)
            lines.append(
                f'{source}: '
                + ' + '.join(self._decomposition(source))
                + f' → {calculated}; expected {expected}: '
                f'{"match" if calculated == expected else "mismatch"}.'
            )
        lines.extend(
            (
                f'For {number}, take the largest tokens in order: '
                + ' + '.join(self._decomposition(number))
                + '.',
                f'Joining the tokens gives {answer}.',
            )
        )
        return _finish(lines, answer)


def _word_pattern(word: str) -> tuple[int, ...]:
    """Return a word's equality pattern, e.g. ``noon -> (0, 1, 1, 0)``."""
    seen: dict[str, int] = {}
    return tuple(seen.setdefault(letter, len(seen)) for letter in word)


class CipherGenerator(Generator):
    """Solve the prompts' monoalphabetic substitution cipher."""

    @staticmethod
    def _fits(
        encrypted: str,
        plain: str,
        forward: dict[str, str],
        inverse: dict[str, str],
    ) -> bool:
        if len(encrypted) != len(plain) or (
            _word_pattern(encrypted) != _word_pattern(plain)
        ):
            return False
        return all(
            (letter not in forward or forward[letter] == decoded)
            and (decoded not in inverse or inverse[decoded] == letter)
            for letter, decoded in zip(encrypted, plain, strict=True)
        )

    @classmethod
    def _decode_words(
        cls, words: list[str], forward: dict[str, str]
    ) -> tuple[list[str], dict[str, str], dict[int, list[str]]]:
        """Complete a partial mapping with compatible vocabulary words."""
        inverse = {decoded: encrypted for encrypted, decoded in forward.items()}
        choices = {
            index: [
                candidate
                for candidate in _WORD_LIST
                if cls._fits(word, candidate, forward, inverse)
            ]
            for index, word in enumerate(words)
        }
        if any(not candidates for candidates in choices.values()):
            raise ValueError(
                'No dictionary word fits the partial cipher mapping.'
            )

        decoded = [''] * len(words)
        solution: dict[str, str] = {}

        def search(
            remaining: set[int],
            current_forward: dict[str, str],
            current_inverse: dict[str, str],
        ) -> bool:
            if not remaining:
                solution.update(current_forward)
                return True
            index = min(
                remaining,
                key=lambda item: sum(
                    cls._fits(
                        words[item], candidate, current_forward, current_inverse
                    )
                    for candidate in choices[item]
                ),
            )
            for candidate in choices[index]:
                if not cls._fits(
                    words[index], candidate, current_forward, current_inverse
                ):
                    continue
                next_forward = dict(current_forward)
                next_inverse = dict(current_inverse)
                next_forward.update(zip(words[index], candidate, strict=True))
                next_inverse.update(zip(candidate, words[index], strict=True))
                decoded[index] = candidate
                if search(remaining - {index}, next_forward, next_inverse):
                    return True
            return False

        if not search(set(range(len(words))), forward, inverse):
            raise ValueError(
                'Cipher examples do not determine a valid plaintext.'
            )
        return decoded, solution, choices

    @override
    def generate_reasoning_content(self) -> tuple[str, str]:
        examples = _examples_with_arrow(self.puzzle)
        question_match = re.search(
            r'decrypt the following text:\s*(.+)', self.puzzle
        )
        if not examples or question_match is None:
            raise ValueError('Invalid cipher puzzle format.')

        forward: dict[str, str] = {}
        inverse: dict[str, str] = {}
        for encrypted_text, plain_text in examples:
            encrypted_words = encrypted_text.split()
            plain_words = plain_text.split()
            if len(encrypted_words) != len(plain_words):
                raise ValueError('Cipher example has mismatched word counts.')
            for encrypted, plain in zip(
                encrypted_words, plain_words, strict=True
            ):
                if not self._fits(encrypted, plain, forward, inverse):
                    raise ValueError(
                        'Cipher examples contain conflicting mappings.'
                    )
                forward.update(zip(encrypted, plain, strict=True))
                inverse.update(zip(plain, encrypted, strict=True))

        question = question_match.group(1).strip()
        question_words = question.split()
        decoded_words, completed_mapping, choices = self._decode_words(
            question_words, forward
        )
        answer = ' '.join(decoded_words)
        mapping = ', '.join(
            f'{key}→{value}' for key, value in sorted(forward.items())
        )
        inferred = ', '.join(
            f'{key}→{value}'
            for key, value in sorted(completed_mapping.items())
            if key not in forward
        )
        lines = [
            'The word lengths and repeated-letter patterns show a one-to-one '
            'letter substitution.',
            'Align each encrypted example word with its plaintext word; '
            'letters in the same position give these constraints:',
        ]
        for encrypted_text, plain_text in examples:
            for encrypted, plain in zip(
                encrypted_text.split(), plain_text.split(), strict=True
            ):
                pairs = ', '.join(
                    f'{encrypted_letter}→{plain_letter}'
                    for encrypted_letter, plain_letter in zip(
                        encrypted, plain, strict=True
                    )
                )
                lines.append(f'{encrypted} → {plain}: {pairs}.')
        lines.extend(
            (
                'Combining repeated constraints gives the known mapping: '
                f'{mapping}.',
                'For an unfinished word, keep only a same-length vocabulary '
                'word with the same repeated-letter pattern and no conflict '
                'with that mapping. The remaining candidate lists are:',
            )
        )
        for index, encrypted in enumerate(question_words):
            lines.append(
                f'{encrypted!r}, pattern {_word_pattern(encrypted)}: '
                f'{", ".join(choices[index])}.'
            )
        if inferred:
            lines.append(
                'Choosing candidates together must keep the substitution '
                f'one-to-one; this adds: {inferred}.'
            )
        lines.extend(
            f'{encrypted!r}: '
            + ', '.join(
                f'{letter}→{completed_mapping[letter]}' for letter in encrypted
            )
            + f' → {plain!r}.'
            for encrypted, plain in zip(
                question_words, decoded_words, strict=True
            )
        )
        lines.append(f'Joining the decoded words gives {answer!r}.')
        return _finish(lines, answer)


@dataclass(frozen=True)
class _NumericRule:
    """One candidate operation for the numeric and symbol equation puzzles."""

    name: str
    reverse_operands: bool
    reverse_result: bool
    sign_style: str = 'plain'


def _reverse_number(value: str) -> str:
    return '-' + value[:0:-1] if value.startswith('-') else value[::-1]


def _numeric_candidates(
    a: int, b: int, source: str, target: str
) -> list[tuple[str, str]]:
    """Return arithmetic rules used to generate the equation puzzle families."""
    candidates = [
        ('concatenation', source + target),
        ('reverse concatenation', target + source),
        ('addition', str(a + b)),
        ('subtraction', str(a - b)),
        ('reverse subtraction', str(b - a)),
        ('absolute difference', str(abs(a - b))),
        ('negated absolute difference', str(-abs(a - b))),
        ('multiplication', str(a * b)),
        ('multiply plus one', str(a * b + 1)),
        ('multiply minus one', str(a * b - 1)),
        ('add one', str(a + b + 1)),
        ('subtract one', str(a + b - 1)),
        ('subtraction plus one', str(a - b + 1)),
        ('subtraction minus one', str(a - b - 1)),
    ]
    if a and b:
        candidates.append(('larger modulo smaller', str(max(a, b) % min(a, b))))
    if b:
        candidates.extend(
            (('integer division', str(a // b)), ('modulo', str(a % b)))
        )
    if a:
        candidates.extend(
            (('reverse division', str(b // a)), ('reverse modulo', str(b % a)))
        )
    if len(source) == len(target) == 2:
        a1, a2, b1, b2 = map(int, source + target)
        candidates.extend(
            (
                (
                    'digit absolute difference',
                    str(abs(a1 - b1)) + str(abs(a2 - b2)),
                ),
                (
                    'digit addition mod 10',
                    str((a1 + b1) % 10) + str((a2 + b2) % 10),
                ),
                (
                    'digit subtraction mod 10',
                    str((a1 - b1) % 10) + str((a2 - b2) % 10),
                ),
                ('cross product sum', str(a1 * b1 + a2 * b2)),
                ('reverse cross product sum', str(a1 * b2 + a2 * b1)),
                ('digit products', str(a1 * b1) + str(a2 * b2)),
                ('reverse digit products', str(a1 * b2) + str(a2 * b1)),
                ('digit-sum difference', str(a1 + a2 - b1 - b2)),
                ('digit-sum total', str(a1 + a2 + b1 + b2)),
                ('digit-product difference', str(a1 * a2 - b1 * b2)),
                ('digit-product total', str(a1 * a2 + b1 * b2)),
                ('determinant', str(a1 * b2 - a2 * b1)),
                ('absolute determinant', str(abs(a1 * b2 - a2 * b1))),
            )
        )
    return candidates


def _evaluate_numeric_rule(
    rule: _NumericRule, left: str, right: str
) -> str | None:
    """Evaluate a rule while preserving leading zeroes for string operations."""
    source = left[::-1] if rule.reverse_operands else left
    target = right[::-1] if rule.reverse_operands else right
    candidates = dict(
        _numeric_candidates(int(source), int(target), source, target)
    )
    value = candidates.get(rule.name)
    if value is None:
        return None
    return _reverse_number(value) if rule.reverse_result else value


def _render_signed(value: str, operator: str, style: str) -> str | None:
    """Render a negative value with the convention used by a puzzle operator."""
    if not value.startswith('-'):
        return value
    if style == 'suffix':
        return value[1:] + operator
    if style == 'prefix':
        return operator + value[1:]
    return None


def _number_styles(outputs: Iterable[str], operator: str) -> tuple[str, ...]:
    """Return plausible negative-number encodings for one operator group."""
    values = tuple(outputs)
    styles = ['plain']
    # Wonderland writes a negative result with a non-minus operator as a
    # suffix or prefix.  Positive examples cannot distinguish this from plain
    # notation, so retain the convention as a candidate for the query.
    if operator != '-':
        styles.insert(0, 'suffix')
    if any(value.endswith(operator) and len(value) > 1 for value in values):
        styles.insert(0, 'suffix')
    if any(value.startswith(operator) and len(value) > 1 for value in values):
        styles.insert(0, 'prefix')
    if operator != '-' and any(
        value.endswith('-') and len(value) > 1 for value in values
    ):
        styles.insert(0, 'suffix')
    if operator != '-' and any(
        value.startswith('-') and len(value) > 1 for value in values
    ):
        styles.insert(0, 'prefix')
    return tuple(dict.fromkeys(styles))


def _all_numeric_rules(
    styles: Iterable[str], *, reverse_first: bool = True
) -> Iterable[_NumericRule]:
    """Yield candidate rules in a deterministic, simple-rule-first order."""
    # Names are obtained from a harmless two-digit call; their formulas are
    # evaluated afresh for each example.
    names = [name for name, _ in _numeric_candidates(12, 34, '12', '34')]
    transformations = (
        ((True, True), (False, False), (True, False), (False, True))
        if reverse_first
        else ((False, False), (True, False), (False, True), (True, True))
    )
    for style in styles:
        for reverse_operands, reverse_result in transformations:
            for name in names:
                yield _NumericRule(
                    name, reverse_operands, reverse_result, style
                )


def _describe_numeric_rule(rule: _NumericRule) -> str:
    """Describe the calculation order of a candidate arithmetic rule."""
    steps: list[str] = []
    if rule.reverse_operands:
        steps.append('reverse both operands')
    steps.append(rule.name)
    if rule.reverse_result:
        steps.append('reverse the result')
    if rule.sign_style == 'suffix':
        steps.append('write a negative result with the operator as a suffix')
    elif rule.sign_style == 'prefix':
        steps.append('write a negative result with the operator as a prefix')
    return ', then '.join(steps)


def _numeric_trial(
    rule: _NumericRule, expression: str, operator: str
) -> tuple[str, str] | None:
    """Return the raw and displayed result of one five-character equation."""
    raw = _evaluate_numeric_rule(rule, expression[:2], expression[3:])
    if raw is None:
        return None
    displayed = _render_signed(raw, operator, rule.sign_style)
    if displayed is None:
        return None
    return raw, displayed


def _numeric_formula(name: str, left: str, right: str, value: str) -> str:
    """Spell out the arithmetic behind one candidate result."""
    left_number, right_number = int(left), int(right)
    formulas = {
        'concatenation': f'{left} || {right} = {value}',
        'reverse concatenation': f'{right} || {left} = {value}',
        'addition': f'{left} + {right} = {value}',
        'subtraction': f'{left} - {right} = {value}',
        'reverse subtraction': f'{right} - {left} = {value}',
        'absolute difference': f'|{left} - {right}| = {value}',
        'negated absolute difference': f'-|{left} - {right}| = {value}',
        'multiplication': f'{left} × {right} = {value}',
        'multiply plus one': f'{left} × {right} + 1 = {value}',
        'multiply minus one': f'{left} × {right} - 1 = {value}',
        'add one': f'{left} + {right} + 1 = {value}',
        'subtract one': f'{left} + {right} - 1 = {value}',
        'subtraction plus one': f'{left} - {right} + 1 = {value}',
        'subtraction minus one': f'{left} - {right} - 1 = {value}',
        'larger modulo smaller': (
            f'max({left}, {right}) mod min({left}, {right}) = {value}'
        ),
        'integer division': f'{left} // {right} = {value}',
        'modulo': f'{left} mod {right} = {value}',
        'reverse division': f'{right} // {left} = {value}',
        'reverse modulo': f'{right} mod {left} = {value}',
    }
    if name in formulas:
        return formulas[name]
    if len(left) != 2 or len(right) != 2:
        return f'{name}({left_number}, {right_number}) = {value}'
    a, b, c, d = map(int, left + right)
    digit_formulas = {
        'digit absolute difference': f'|{a}-{c}| || |{b}-{d}| = {value}',
        'digit addition mod 10': (
            f'({a}+{c}) mod 10 || ({b}+{d}) mod 10 = {value}'
        ),
        'digit subtraction mod 10': (
            f'({a}-{c}) mod 10 || ({b}-{d}) mod 10 = {value}'
        ),
        'cross product sum': f'{a}×{c} + {b}×{d} = {value}',
        'reverse cross product sum': f'{a}×{d} + {b}×{c} = {value}',
        'digit products': f'{a}×{c} || {b}×{d} = {value}',
        'reverse digit products': f'{a}×{d} || {b}×{c} = {value}',
        'digit-sum difference': f'({a}+{b}) - ({c}+{d}) = {value}',
        'digit-sum total': f'({a}+{b}) + ({c}+{d}) = {value}',
        'digit-product difference': f'{a}×{b} - {c}×{d} = {value}',
        'digit-product total': f'{a}×{b} + {c}×{d} = {value}',
        'determinant': f'{a}×{d} - {b}×{c} = {value}',
        'absolute determinant': f'|{a}×{d} - {b}×{c}| = {value}',
    }
    return digit_formulas[name]


def _numeric_working(rule: _NumericRule, left: str, right: str) -> str | None:
    """Return every numeric transformation used by ``rule`` on two operands."""
    source = left[::-1] if rule.reverse_operands else left
    target = right[::-1] if rule.reverse_operands else right
    raw = dict(
        _numeric_candidates(int(source), int(target), source, target)
    ).get(rule.name)
    if raw is None:
        return None
    steps: list[str] = []
    if rule.reverse_operands:
        steps.append(f'reverse operands: {left}→{source}, {right}→{target}')
    steps.append(_numeric_formula(rule.name, source, target, raw))
    if rule.reverse_result:
        steps.append(f'reverse result: {raw}→{_reverse_number(raw)}')
    return '; '.join(steps)


class SymbolGenerator(Generator):
    """Solve numeric equations and their symbol-substitution variant."""

    @staticmethod
    def _parse(puzzle: str) -> tuple[list[tuple[str, str]], str]:
        examples = [
            (left, right)
            for left, right in (
                line.split(' = ', 1)
                for line in puzzle.splitlines()
                if ' = ' in line
            )
            if len(left) == 5
        ]
        match = re.search(r'result for:\s*(\S+)', puzzle)
        if not examples or match is None or len(match.group(1)) != 5:
            raise ValueError('Invalid symbol puzzle format.')
        return examples, match.group(1)

    @staticmethod
    def _group(
        examples: list[tuple[str, str]],
    ) -> dict[str, list[tuple[str, str]]]:
        groups: dict[str, list[tuple[str, str]]] = {}
        for expression, output in examples:
            groups.setdefault(expression[2], []).append((expression, output))
        return groups

    @staticmethod
    def _find_numeric_rule(
        group: list[tuple[str, str]], operator: str
    ) -> _NumericRule | None:
        styles = _number_styles((out for _, out in group), operator)
        for rule in _all_numeric_rules(styles):
            matches = True
            for expression, output in group:
                value = _evaluate_numeric_rule(
                    rule, expression[:2], expression[3:]
                )
                if (
                    value is None
                    or _render_signed(value, operator, rule.sign_style)
                    != output
                ):
                    matches = False
                    break
            if matches:
                return rule
        return None

    @classmethod
    def _solve_numeric(
        cls, examples: list[tuple[str, str]], question: str
    ) -> tuple[str, list[str]]:
        groups = cls._group(examples)
        operator = question[2]
        group = groups.get(operator)
        if group is None:
            # This is the data-set convention for an unseen operator: use the
            # unambiguous absolute difference rather than inventing a rule.
            answer = str(abs(int(question[:2]) - int(question[3:])))
            return answer, [
                f'Operator {operator!r} has no example; use absolute '
                f'difference: |{question[:2]} - {question[3:]}| = {answer}.',
            ]
        rule = cls._find_numeric_rule(group, operator)
        if rule is None:
            raise ValueError('No supported arithmetic rule fits the examples.')
        value = _evaluate_numeric_rule(rule, question[:2], question[3:])
        answer = (
            _render_signed(value, operator, rule.sign_style)
            if value is not None
            else None
        )
        if answer is None:
            raise ValueError(
                'A negative result has no supported representation.'
            )
        lines = [
            'A five-character equation has two two-digit operands and its '
            'middle character selects an arithmetic rule.',
            f'For operator {operator!r}, use this rule: '
            f'{_describe_numeric_rule(rule)}.',
            '',
            'Apply that rule to each given row:',
        ]
        for expression, expected in group:
            trial = _numeric_trial(rule, expression, operator)
            if trial is None:
                raise ValueError('The selected rule cannot be displayed.')
            raw, displayed = trial
            working = _numeric_working(rule, expression[:2], expression[3:])
            if working is None:
                raise ValueError('The numeric working cannot be displayed.')
            lines.append(
                f'{expression[:2]} {operator} {expression[3:]}: '
                f'{working}; displayed as {displayed}; '
                f'expected {expected}: '
                f'{"match" if displayed == expected else "mismatch"}.'
            )
        question_trial = _numeric_trial(rule, question, operator)
        if question_trial is None:
            raise ValueError(
                'The selected rule cannot be applied to the query.'
            )
        raw, _ = question_trial
        working = _numeric_working(rule, question[:2], question[3:])
        if working is None:
            raise ValueError('The numeric working cannot be displayed.')
        lines.extend(
            (
                '',
                f'For {question[:2]} {operator} {question[3:]}: {working}.',
                f'The resulting value is {raw}, written as {answer}.',
            )
        )
        return answer, lines

    @staticmethod
    def _map_output(
        value: str,
        output: str,
        operator: str,
        style: str,
        mapping: dict[str, int],
        inverse: dict[int, str],
    ) -> tuple[dict[str, int], dict[int, str]] | None:
        """Check one encoded result and extend a bijective symbol mapping."""
        if value.startswith('-'):
            if style == 'suffix' and output.endswith(operator):
                value, output = value[1:], output[:-1]
            elif style == 'prefix' and output.startswith(operator):
                value, output = value[1:], output[1:]
            else:
                return None
        elif style != 'plain' and (
            output.startswith(operator) or output.endswith(operator)
        ):
            return None
        if len(value) != len(output) or not value.isdigit():
            return None
        next_mapping, next_inverse = dict(mapping), dict(inverse)
        for symbol, digit in zip(output, value, strict=True):
            value_digit = int(digit)
            if (
                symbol in next_mapping and next_mapping[symbol] != value_digit
            ) or (
                value_digit in next_inverse
                and next_inverse[value_digit] != symbol
            ):
                return None
            next_mapping[symbol] = value_digit
            next_inverse[value_digit] = symbol
        return next_mapping, next_inverse

    @classmethod
    def _extend_mapping(
        cls,
        group: list[tuple[str, str]],
        operator: str,
        rule: _NumericRule,
        mapping: dict[str, int],
        inverse: dict[int, str],
        limit: int = 64,
    ) -> list[tuple[dict[str, int], dict[int, str]]]:
        """Fit a candidate arithmetic rule using finite digit assignments."""
        results: list[tuple[dict[str, int], dict[int, str]]] = []

        def visit(
            index: int, current: dict[str, int], current_inverse: dict[int, str]
        ) -> None:
            if len(results) >= limit:
                return
            if index == len(group):
                results.append((current, current_inverse))
                return
            expression, output = group[index]
            symbols = list(dict.fromkeys(expression[:2] + expression[3:]))
            unknown = [symbol for symbol in symbols if symbol not in current]
            available = [
                digit for digit in range(10) if digit not in current_inverse
            ]
            for digits in itertools.permutations(available, len(unknown)):
                next_mapping = dict(current)
                next_inverse = dict(current_inverse)
                for symbol, digit in zip(unknown, digits, strict=True):
                    next_mapping[symbol] = digit
                    next_inverse[digit] = symbol
                left = ''.join(
                    str(next_mapping[symbol]) for symbol in expression[:2]
                )
                right = ''.join(
                    str(next_mapping[symbol]) for symbol in expression[3:]
                )
                value = _evaluate_numeric_rule(rule, left, right)
                if value is None:
                    continue
                mapped = cls._map_output(
                    value,
                    output,
                    operator,
                    rule.sign_style,
                    next_mapping,
                    next_inverse,
                )
                if mapped is not None:
                    visit(index + 1, *mapped)

        visit(0, mapping, inverse)
        return results

    @classmethod
    def _symbol_rule_options(
        cls, group: list[tuple[str, str]], operator: str
    ) -> Iterable[_NumericRule]:
        return _all_numeric_rules(
            _number_styles((out for _, out in group), operator),
            reverse_first=False,
        )

    @classmethod
    def _branching_score(
        cls,
        states: list[
            tuple[dict[str, int], dict[int, str], dict[str, _NumericRule]]
        ],
        group: list[tuple[str, str]],
        operator: str,
    ) -> int:
        """Estimate how strongly an operator group constrains current states.

        Trying the tightest group first prevents a short, ambiguous group from
        filling the bounded search frontier before its later examples can rule
        out the accidental digit substitutions.
        """
        score = 0
        for mapping, inverse, _ in states:
            for rule in cls._symbol_rule_options(group, operator):
                if cls._extend_mapping(
                    group, operator, rule, mapping, inverse, limit=1
                ):
                    score += 1
                    if score >= 512:
                        return score
        return score

    @classmethod
    def _encode_question(
        cls,
        question: str,
        rule: _NumericRule,
        mapping: dict[str, int],
        inverse: dict[int, str],
        symbols: set[str],
    ) -> list[tuple[str, dict[str, int], dict[int, str]]]:
        """Apply a rule and return each answer with its completed code."""
        input_symbols = list(dict.fromkeys(question[:2] + question[3:]))
        unknown = [symbol for symbol in input_symbols if symbol not in mapping]
        available = [digit for digit in range(10) if digit not in inverse]
        answers: list[tuple[str, dict[str, int], dict[int, str]]] = []
        for digits in itertools.permutations(available, len(unknown)):
            current, current_inverse = dict(mapping), dict(inverse)
            for symbol, digit in zip(unknown, digits, strict=True):
                current[symbol], current_inverse[digit] = digit, symbol
            left = ''.join(str(current[symbol]) for symbol in question[:2])
            right = ''.join(str(current[symbol]) for symbol in question[3:])
            raw = _evaluate_numeric_rule(rule, left, right)
            if raw is None:
                continue
            if raw.startswith('-'):
                if rule.sign_style == 'suffix':
                    prefix, raw = '', raw[1:]
                    suffix = question[2]
                elif rule.sign_style == 'prefix':
                    prefix, raw, suffix = question[2], raw[1:], ''
                else:
                    continue
            else:
                prefix, suffix = '', ''
            if not raw.isdigit():
                continue
            encoded: list[str] = []
            possible = True
            for digit_text in raw:
                digit = int(digit_text)
                if digit not in current_inverse:
                    # An unseen output digit is not identifiable unless one
                    # unassigned symbol remains in the prompt alphabet.
                    choices = sorted(symbols - set(current) - {question[2]})
                    if len(choices) != 1:
                        possible = False
                        break
                    current_inverse[digit] = choices[0]
                    current[choices[0]] = digit
                encoded.append(current_inverse[digit])
            if possible:
                answers.append(
                    (
                        prefix + ''.join(encoded) + suffix,
                        current,
                        current_inverse,
                    )
                )
        return answers

    @staticmethod
    def _encode_value(
        value: str,
        rule: _NumericRule,
        operator: str,
        inverse: dict[int, str],
    ) -> str | None:
        """Encode a numeric result with a completed symbol substitution."""
        prefix = suffix = ''
        if value.startswith('-'):
            value = value[1:]
            if rule.sign_style == 'suffix':
                suffix = operator
            elif rule.sign_style == 'prefix':
                prefix = operator
            else:
                return None
        if not value.isdigit() or any(
            int(digit) not in inverse for digit in value
        ):
            return None
        return prefix + ''.join(inverse[int(digit)] for digit in value) + suffix

    @classmethod
    def _fit_symbolic_rules(
        cls,
        groups: dict[str, list[tuple[str, str]]],
        question_operator: str,
        *,
        adaptive: bool,
    ) -> tuple[
        list[tuple[dict[str, int], dict[int, str], dict[str, _NumericRule]]],
        list[tuple[str, list[tuple[str, str]]]],
    ]:
        """Fit all operator groups, retrying broad cases only when needed."""
        states: list[
            tuple[dict[str, int], dict[int, str], dict[str, _NumericRule]]
        ] = [({}, {}, {})]
        if adaptive:
            remaining = list(groups.items())
        else:
            remaining = sorted(
                groups.items(),
                key=lambda item: (-len(item[1]), item[0] == question_operator),
            )
        ordered: list[tuple[str, list[tuple[str, str]]]] = []

        while remaining:
            if adaptive:
                operator, group = min(
                    remaining,
                    key=lambda item: (
                        cls._branching_score(states, item[1], item[0]),
                        -len(item[1]),
                        item[0] == question_operator,
                    ),
                )
                remaining.remove((operator, group))
            else:
                operator, group = remaining.pop(0)
            ordered.append((operator, group))

            next_states: list[
                tuple[dict[str, int], dict[int, str], dict[str, _NumericRule]]
            ] = []
            state_limit = 2048 if adaptive else 128
            mapping_limit = 64 if adaptive else 12
            for mapping, inverse, rules in states:
                for rule in cls._symbol_rule_options(group, operator):
                    for new_mapping, new_inverse in cls._extend_mapping(
                        group,
                        operator,
                        rule,
                        mapping,
                        inverse,
                        limit=mapping_limit,
                    ):
                        next_rules = dict(rules)
                        next_rules[operator] = rule
                        next_states.append(
                            (new_mapping, new_inverse, next_rules)
                        )
                        if len(next_states) >= state_limit:
                            break
                    if len(next_states) >= state_limit:
                        break
                if len(next_states) >= state_limit:
                    break
            if not next_states:
                return [], ordered
            states = next_states
        return states, ordered

    @classmethod
    def _solve_symbolic(
        cls, examples: list[tuple[str, str]], question: str
    ) -> tuple[str, list[str]]:
        groups = cls._group(examples)
        question_operator = question[2]
        if question_operator not in groups:
            # Concatenation is the only operation safely inferable for an
            # unseen symbolic operator without assuming a digit assignment.
            return question[:2] + question[3:], [
                f'Operator {question_operator!r} is unseen, so concatenate the '
                'two symbol pairs in their original order.',
            ]

        symbols = set(
            question + ''.join(left + right for left, right in examples)
        )
        states, ordered_groups = cls._fit_symbolic_rules(
            groups, question_operator, adaptive=False
        )
        adaptive = False
        if not states:
            states, ordered_groups = cls._fit_symbolic_rules(
                groups, question_operator, adaptive=True
            )
            adaptive = True

        def query_solutions(
            fitted_states: list[
                tuple[
                    dict[str, int],
                    dict[int, str],
                    dict[str, _NumericRule],
                ]
            ],
        ) -> list[
            tuple[
                str,
                dict[str, int],
                dict[int, str],
                dict[str, _NumericRule],
            ]
        ]:
            found: list[
                tuple[
                    str,
                    dict[str, int],
                    dict[int, str],
                    dict[str, _NumericRule],
                ]
            ] = []
            for mapping, inverse, rules in fitted_states:
                rule = rules[question_operator]
                for answer, query_mapping, query_inverse in sorted(
                    cls._encode_question(
                        question, rule, mapping, inverse, symbols
                    ),
                    key=lambda item: item[0],
                ):
                    found.append((answer, query_mapping, query_inverse, rules))
            return found

        solutions = query_solutions(states)
        if not solutions and not adaptive:
            states, ordered_groups = cls._fit_symbolic_rules(
                groups, question_operator, adaptive=True
            )
            solutions = query_solutions(states)
        if not solutions:
            raise ValueError('No digit substitution and arithmetic rule fits.')
        # States are deliberately generated in simplest-rule order.  Voting
        # would instead reward broad, accidental rules on underdetermined rows.
        answer, mapping, inverse, rules = solutions[0]

        lines = [
            'Each five-character expression has two encoded two-digit '
            'operands. Its middle character is an operator; every other '
            'character is part of one shared, one-to-one symbol-to-digit code.',
            '',
            'Use this one-to-one digit code:',
        ]
        lines.extend(
            f'{symbol!r} → {digit}' for symbol, digit in sorted(mapping.items())
        )
        lines.extend(('', 'Arithmetic rule for each operator:'))
        for operator, _ in ordered_groups:
            lines.append(
                f'{operator!r}: {_describe_numeric_rule(rules[operator])}.'
            )
        lines.extend(('', 'Check every example using that code:'))
        for expression, expected in examples:
            operator = expression[2]
            rule = rules[operator]
            left = ''.join(str(mapping[symbol]) for symbol in expression[:2])
            right = ''.join(str(mapping[symbol]) for symbol in expression[3:])
            trial = _numeric_trial(rule, left + operator + right, operator)
            if trial is None:
                raise ValueError(
                    'The retained symbolic rule cannot be displayed.'
                )
            raw, _ = trial
            working = _numeric_working(rule, left, right)
            if working is None:
                raise ValueError('The symbolic working cannot be displayed.')
            encoded = cls._encode_value(raw, rule, operator, inverse)
            status = 'match' if encoded == expected else 'mismatch'
            lines.append(
                f'{expression!r} decodes to {left} {operator} {right}; '
                f'{working}; numeric result {raw}; '
                f'encode it as {encoded!r}; '
                f'expected {expected!r}: {status}.'
            )

        rule = rules[question_operator]
        left = ''.join(str(mapping[symbol]) for symbol in question[:2])
        right = ''.join(str(mapping[symbol]) for symbol in question[3:])
        trial = _numeric_trial(
            rule, left + question_operator + right, question_operator
        )
        if trial is None:
            raise ValueError(
                'The retained rule cannot be applied to the query.'
            )
        raw, _ = trial
        working = _numeric_working(rule, left, right)
        if working is None:
            raise ValueError('The symbolic working cannot be displayed.')
        lines.extend(
            (
                '',
                f'For {question!r}, decode the operands as {left} '
                f'{question_operator} {right}.',
                f'{working}; the numeric result is {raw}.',
                f'Encoding its digits with the code above gives {answer!r}.',
            )
        )
        return answer, lines

    @override
    def generate_reasoning_content(self) -> tuple[str, str]:
        examples, question = self._parse(self.puzzle)
        if examples[0][0][0].isdigit():
            answer, lines = self._solve_numeric(examples, question)
        else:
            answer, lines = self._solve_symbolic(examples, question)
        return _finish(lines, answer)


def get_generator_class_by_puzzle_type(
    puzzle_type: PuzzleType,
) -> Type[Generator]:
    """Return the generator class that understands ``puzzle_type``."""
    generators: dict[PuzzleType, Type[Generator]] = {
        'bit': BitGenerator,
        'gravity': GravityGenerator,
        'unit': UnitGenerator,
        'roman': RomanGenerator,
        'cipher': CipherGenerator,
        'symbol': SymbolGenerator,
    }
    try:
        return generators[puzzle_type]
    except KeyError as error:
        raise ValueError(f'Unsupported puzzle type: {puzzle_type}') from error
