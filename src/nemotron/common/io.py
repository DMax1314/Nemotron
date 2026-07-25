import json
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd


def file_exists(path: PathLike) -> bool:
    """Checks if a file exists at the given path.

    This function returns True if the file exists and is a regular file, and
    False otherwise.

    Args:
        path: The path to the file.
    """

    path = Path(path)
    return path.exists() and path.is_file()


def load_csv(path: PathLike, **kwargs) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame.

    Args:
        path: The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to `pd.read_csv`.

    Returns:
        A pandas DataFrame containing the data from the CSV file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """

    path = Path(path)
    if not file_exists(path):
        raise FileNotFoundError(f'File not found: {path}')

    return pd.read_csv(path, **kwargs)


def save_csv(
    frame: pd.DataFrame, path: PathLike, *, index: bool = False, **kwargs
) -> None:
    """Writes a pandas DataFrame to a CSV file.

    This function ensures the parent directory of the output path exists before
    writing the CSV file.

    Args:
        frame: The DataFrame to write.
        path: The path to the output CSV file.
        index: Whether to write the DataFrame index to the CSV file. Defaults
            to False.
        **kwargs: Additional keyword arguments to pass to `pd.DataFrame.to_csv`.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=index, **kwargs)


def save_json(
    payload: Any,
    path: PathLike,
    indent: int = 2,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> None:
    """Writes a Python object to a JSON file.

    Args:
        payload: The Python object to write to the JSON file.
        path: The path to the output JSON file.
        indent: The number of spaces to use for indentation in the JSON file.
            Defaults to 2.
        ensure_ascii: Whether to escape non-ASCII characters in the JSON file.
            Defaults to True.
        sort_keys: Whether to sort the keys in the JSON file. Defaults to False.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            default=str,
        )
        + '\n',
        encoding='utf-8',
    )
