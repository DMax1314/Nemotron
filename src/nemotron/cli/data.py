import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated, cast

import pandas as pd
from termcolor import colored
from typer import Argument, Option, echo

from nemotron.cli.common import make_typer
from nemotron.common.io import file_exists, load_csv, save_csv
from nemotron.configs import RAW_DATA_DIR
from nemotron.puzzle.puzzle_type import infer_puzzle_type

app = make_typer()


@app.command()
def download(
    filename: Annotated[
        str,
        Argument(help='Competition file to download.'),
    ],
) -> None:
    """Downloads a file from the Kaggle competition.

    This command uses the `kagglehub` package to download files from the Kaggle
    competition. It requires the user to have a Kaggle API token set up in their
    environment.

    On Linux/macOS, export the Kaggle API token in the shell configuration file
    as follows:

        export KAGGLE_API_TOKEN=KGAT_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    If you don't have a Kaggle API token, you can obtain one from Kaggle account
    settings. Open the following URL in your browser after logging in to Kaggle:

        https://www.kaggle.com/settings

    Go to the "API Tokens" tab and click on "Generate New Token". Then,  Enter a
    token name and click "Generate". Copy the API token and export if in the
    shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`) as mentioned
    above.

    The downloaded files will be saved to the raw data directory.

    In this challenge, there are only two files for the time being:

        - `train.csv`
        - `test.csv`

    Example of downloading `train.csv`:

        nemotron data download train.csv
    """

    import kagglehub

    from nemotron.configs import RAW_DATA_DIR

    kagglehub.competition_download(
        'nvidia-nemotron-model-reasoning-challenge',
        path=filename,
        output_dir=RAW_DATA_DIR,
        force_download=True,
    )


@app.command()
def fetch() -> None:
    """Downloads all necessary files for the competition.

    This command downloads both the `train.csv` and `test.csv` files from the
    Kaggle competition using the `download` command.
    """

    download('train.csv')
    download('test.csv')


@app.command()
def stats(
    show_details: bool = Option(
        False,
        '--details',
        '-d',
        help='Show basic statistics about the dataset.',
    ),
) -> None:
    """Displays basic statistics about the dataset.

    This command reads the `train.csv` and `test.csv` files from the raw data
    directory and displays the number of samples in each set.

    If the `--details` flag is provided, it also infers the puzzle type from the
    prompts and shows the count of each puzzle type in both the training and
    test sets.
    """

    train_path = Path(f'{RAW_DATA_DIR}/train.csv')
    test_path = Path(f'{RAW_DATA_DIR}/test.csv')

    if not file_exists(train_path) or not file_exists(test_path):
        return echo(
            'Data files not found. Please run `nemotron data fetch` to '
            'download the datasets.'
        )

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    def display_stats(df: pd.DataFrame, name: str):
        echo(f'{name} set:')
        echo(f'Number of samples: {len(df)}')

        if not show_details:
            return

        puzzle_type_map = defaultdict(int)
        for _, row in df.iterrows():
            prompt = str(row['prompt'])
            puzzle_type = infer_puzzle_type(prompt)
            puzzle_type_map[puzzle_type] += 1

        for puzzle_type, count in puzzle_type_map.items():
            echo(f'  - [{puzzle_type}]: {count}')

    display_stats(train_df, 'Training')
    echo()
    display_stats(test_df, 'Test')


@app.command()
def show(
    num_total: Annotated[
        int,
        Argument(help='Number of examples to display.'),
    ] = 1,
    puzzle_type=Option(
        None, '--type', '-t', help='Filter examples by puzzle type.'
    ),
) -> None:
    """Displays examples of the dataset.

    This command reads the `train.csv` file from the `data/raw` directory and
    displays a specified number of examples from the training set. Each example
    consists of a prompt and its corresponding answer.

    If the `--type` flag is provided, it filters the examples by the specified
    puzzle type before displaying them. The puzzle type is inferred from the
    prompt using the `infer_puzzle_type` function. Only examples that match the
    specified puzzle type will be displayed. The puzzle types include:

        - bit
        - gravity
        - unit
        - roman
        - cipher
        - symbol

    """

    train_path = Path('data/raw/train.csv')

    if not file_exists(train_path):
        return echo(
            'Data files not found. Please run `nemotron data fetch` to '
            'download the dataset.'
        )

    train_df = load_csv(train_path)
    terminal_size = shutil.get_terminal_size()

    num_total = min(num_total, len(train_df))
    num_displayed = 0
    print()
    for i in range(len(train_df)):
        if num_displayed >= num_total:
            break

        prompt = str(train_df.loc[i, 'prompt'])
        answer = str(train_df.loc[i, 'answer'])

        if puzzle_type is not None:
            prompt_puzzle_type = infer_puzzle_type(prompt)
            if prompt_puzzle_type != puzzle_type:
                continue

        print(colored(prompt, 'yellow'))
        print()
        print(colored(answer, 'magenta'))
        print()

        if num_displayed < num_total - 1:
            print('=' * terminal_size.columns + '\n')

        num_displayed += 1


@app.command()
def extract(
    puzzle_types_str=Option(
        None,
        '--types',
        '-t',
        help='Filter examples by puzzle types; separated with comma.',
    ),
    limit=Option(
        -1, '--limit', '-l', help='Limit the number of examples to extract.'
    ),
    columns=Option(
        None,
        '--columns',
        '-c',
        help='Columns to include in the output CSV file; separated with comma.',
    ),
    output_file=Option(
        'data/extracted.csv',
        '--output',
        '-o',
        help='Output file path for the extracted CSV data.',
    ),
) -> None:
    """Extracts a subset of the training data.

    This command reads the `train.csv` file from the `data/raw` directory and
    extracts a subset of the data based on the specified puzzle types. The
    extracted data is saved to a new CSV file.
    """

    train_path = Path(f'{RAW_DATA_DIR}/train.csv')
    if not file_exists(train_path):
        return echo(
            'Data files not found. Please run `nemotron data fetch` to '
            'download the datasets.'
        )

    train_df = load_csv(train_path)
    train_df['type'] = train_df['prompt'].apply(infer_puzzle_type)

    # Filter by puzzle types if specified.
    if puzzle_types_str:
        puzzle_types: list[str] = puzzle_types_str.split(',')
        train_df = train_df[train_df['type'].isin(puzzle_types)]

    # Select specified columns if provided.
    if columns:
        column_list = columns.split(',')
        train_df = train_df[column_list]

    # Limit the number of examples if specified.
    limit = int(limit)
    if limit > 0:
        train_df = cast(pd.DataFrame, train_df).head(limit)

    save_csv(cast(pd.DataFrame, train_df), output_file)
