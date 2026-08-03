from pathlib import Path
from typing import Annotated

from typer import Argument, echo

from nemotron.cli.common import make_typer
from nemotron.common.io import file_exists, load_csv
from nemotron.rrcg.rule_based_reasoning_content_generator import (
    RuleBasedReasoningContentGenerator,
)

app = make_typer()


@app.command()
def generate(
    prompt_id: Annotated[
        str,
        Argument(
            help='The ID of the prompt (puzzle) to generate reasoning'
            'context for.'
        ),
    ],
) -> None:
    train_path = Path('data/raw/train.csv')
    if not file_exists(train_path):
        return echo(
            'Data files not found. Please run `nemotron data fetch` to '
            'download the dataset.'
        )

    train_df = load_csv(train_path)
    matches = train_df.loc[train_df['id'] == prompt_id, 'prompt']
    prompt: str | None = str(matches.iloc[0]) if not matches.empty else None

    if prompt is None:
        echo(f'Prompt with ID {prompt_id} does not exist.')
        exit(1)

    generator = RuleBasedReasoningContentGenerator(prompt)
    _, reasoning_content = generator.generate_reasoning_content()
    echo(reasoning_content)
