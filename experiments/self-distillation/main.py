# Copyright (c) 2026 James Chen, Zhendong Li, Rong Pan
# SPDX-License-Identifier: MIT

# This script implements self-distillation for the following models:
#
#     Nemotron 3 Nano 30B A3B BF16
#
# =============================================================================
# Because vLLM is not installabl on macOS, we use an inline comment
# `pyright: ignore[reportMissingImports]` to suppress the import errors given by
# pyright in this script.

import argparse
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from statistics import mean
from typing import Any, cast

import pandas as pd
import torch

# The Kaggle dataset reference for the 0.86 LoRA adapter. The script will
# download this adapter from this Kaggle dataset if the `adapter_path`
# configuration value is set.
LORA_ADAPTER_KAGGLE_DATASET_REF = 'leegongman/0-86-adapter'

# The extra prompt instruction appended to the original prompt (puzzle) to guide
# the model.
PROMPT_INSTRUCTION = (
    '\nPlease put your final answer inside `\\boxed{}`. '
    + 'For example: `\\boxed{your answer}`'
)

# Configure vLLM's FlashInfer sampler. During generation, the model produces
# logits for the next token. A sampler turns those logits into an actual next
# token. FlashInfer provides CUDA/GPU kernels that do this more efficiently,
# especially for batched LLM serving. Its sampling APIs include operations like
# `top_k_sampling` and `top_p_sampling`.
#
# Setting this environment variable to '1' enables the FlashInfer sampler; '0'
# to disable it.
os.environ['VLLM_USE_FLASH_INFER'] = '0'

# Configure vLLM's worker multiprocessing method. vLLM can use different method
# for multiprocessing when it spawns worker processes to handle inference
# requests. Options include `fork` and `spawn`.
#
# Set it to 'spawn' to enforce safe CUSA multiprocessing behavior.
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for loading the model and generating rollouts.

    The following entries are set the same as the evaluation metric for this
    challenge:

        - max_lora_rank = 32
        - max_tokens = 7680
        - max_model_len = 8192

    We encourage the model not to generate overlong rollouts, so that the final
    answer is not truncated.
    """

    # Maximum LoRA adapter rank vLLM will support.
    max_lora_rank: int = 32

    # Maximum number of new tokens the model may generate per rollout. If
    # exceeded, the model will stop and return what it has generated so far,
    # and hence the final answer may be truncated.
    max_tokens: int = 7680

    # Maximum total context length vLLM will support for each request: prompt
    # tokens plus generated tokens. This should be at least large enough for
    # your prompt plus max_tokens.
    max_model_len: int = 8192

    # The settings for rollouts. Each tuple is (num_rollouts, temperature).
    # For example, (3, 1.2) means to generate 3 rollouts with temperature 1.2
    # for each prompt.
    #
    # Lower temperature makes outputs more deterministic; higher temperature
    # makes them more diverse. For reasoning task self-distillation, temperature
    # is usually set between 1.0 and 2.0.
    rollout_settings: list[tuple[int, float]] = field(
        default_factory=lambda: [
            (3, 1.2),
            (3, 1.5),
            (3, 1.8),
        ]
    )

    # Nucleus sampling cutoff. The model samples only from tokens whose
    # cumulative probability reaches this value.
    #
    # For self-distillation task without a verifier (teacher model), we want to
    # encourage more diversity in the generated rollouts, so this value is often
    # set between 0.8 and 0.9.
    top_p: float = 0.85

    # The number of GPUs to use for tensor parallelism.
    tensor_parallel_size: int = 1

    # The number of data-parallel vLLM replicas to run.
    data_parallel_size: int = 1

    # The fraction of GPU memory to utilize for vLLM. 0.85 means vLLM can use
    # about 85% of available VRAM.
    gpu_memory_utilization: float = 0.85

    # Model weight/compute dtype. 'auto' lets vLLM choose based on the
    # checkpoint, commonly bfloat16 or float16.
    dtype: str = 'auto'

    # Whether to enable vLLM prefix caching.
    enable_prefix_caching: bool = False


@dataclass(frozen=True)
class Config:
    """Configuration for rollout generation.

    For the following paths:

        - output_dir
        - model_path
        - train_csv_path
        - adapter_path

    If they are not absolute paths, they are interpreted as relative to
    `data_root`. For example, if `data_root` is `/nemotron` and `train_csv_path`
    is `train.csv`, then the full path to the training CSV file is
    `/nemotron/train.csv`.
    """

    # Data root directory path. It contains datasets, models, adapters, and
    # other assets.
    data_root: Path

    # Output directory path.
    output_dir: Path

    # Model directory path.
    model_path: Path

    # Training CSV file path.
    train_csv_path: Path

    # Adapter directory path. If set, the script will load the adapter and apply
    # it to the base model.
    adapter_path: Path | None = None

    # Optional Kaggle model reference. If set, the script will download the
    # model from Kaggle and use it instead of the local model path.
    kagglehub_model_ref: str | None = None

    # The number of examples to skip from the beginning of the training set.
    offset: int = 0

    # The number of examples to process in the training set. Keep this small for
    # a smoke test, then remove it for the full run. If `-1`, the script
    # processes the entire training set.
    limit: int = -1

    # The list of puzzle types to consider. If set, the script will only process
    # examples whose puzzle type is in this list. In CLI, this can be passed as
    # a comma-separated string, e.g. `--puzzle-types=symbol,unit`.
    puzzle_types: list[str] | None = field(default_factory=lambda: None)

    # Random seed for repeatable preprocessing and rollout generation.
    seed: int = 42

    # Model configuration.
    model_config: ModelConfig = field(default_factory=ModelConfig)


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""

    parser = argparse.ArgumentParser(description='Generate rollouts.')

    parser.add_argument('--data-root', type=Path, default=Path(os.getcwd()))
    parser.add_argument('--output-dir', type=Path, default=Path('output'))
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('models/nemotron-3-nano-30b-a3b-bf16'),
    )
    parser.add_argument(
        '--train-csv-path', type=Path, default=Path('train.csv')
    )
    parser.add_argument('--adapter-path', type=Path, default=None)
    parser.add_argument('--kagglehub-model-ref', default=None)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--puzzle-types', type=str, default=None)
    parser.add_argument('--seed', type=int, default=471)

    # Model configuration arguments
    parser.add_argument('--max-lora-rank', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=7680)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--top-p', type=float, default=0.85)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--data-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--dtype', default='auto')
    parser.add_argument(
        '--enable-prefix-caching',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f'Ignoring unknown arguments, likely from Jupyter: {unknown}')
    return args


def build_config(args: argparse.Namespace) -> Config:
    """Merges CLI arguments into one explicit config object."""

    data_root = Path(args.data_root.expanduser())

    def get_abs_path(path: Path) -> Path:
        if isinstance(path, Path) and not path.is_absolute():
            path = data_root / path
        return path

    adapter_path = args.adapter_path
    return Config(
        data_root=data_root,
        output_dir=get_abs_path(args.output_dir),
        model_path=get_abs_path(args.model_path),
        train_csv_path=get_abs_path(args.train_csv_path),
        adapter_path=get_abs_path(adapter_path) if adapter_path else None,
        kagglehub_model_ref=args.kagglehub_model_ref
        if args.kagglehub_model_ref
        else None,
        offset=args.offset,
        puzzle_types=[t.strip() for t in args.puzzle_types.split(',')]
        if args.puzzle_types
        else None,
        limit=args.limit,
        seed=args.seed,
        model_config=ModelConfig(
            max_lora_rank=args.max_lora_rank,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            enable_prefix_caching=args.enable_prefix_caching,
        ),
    )


def print_config(config: Config) -> None:
    """Prints the configuration in a readable format."""

    print('\n[[ Config ]]')
    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        print(f'{field}: {value}')


def path_has_content(path: Path) -> bool:
    """Returns true when a file exists or a directory is non-empty."""

    if path.is_file():
        return True
    return path.is_dir() and any(path.iterdir())


def import_kagglehub():
    """Imports kagglehub only when a Kaggle download is required."""

    try:
        return import_module('kagglehub')
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'kagglehub is required to download Kaggle assets. Install it with '
            '`pip install kagglehub`, or omit the Kaggle download arguments '
            'and point --model-path/--adapter-path at existing local files.'
        ) from exc


def download_model_from_kaggle(config: Config) -> None:
    """Downloads the model from Kaggle if `kagglehub_model_ref` is set and the
    local model directory does not already exist or is empty. If the local model
    directory exists and is not empty, it skips the download to avoid
    overwriting existing files. If the download proceeds, it uses kagglehub to
    download the model from Kaggle and saves it to the specified local path.
    """

    if not config.kagglehub_model_ref:
        return

    print('\n[[ Model Download ]]')
    model_dir = config.model_path
    if path_has_content(model_dir):
        print(f'Model directory already exists and is not empty: {model_dir}')
        print(
            'Skipping download. If you want to re-download, please remove '
            'the existing directory.'
        )
    else:
        print('Downloading model from Kaggle...')
        kagglehub = import_kagglehub()
        kagglehub.model_download(
            config.kagglehub_model_ref,
            output_dir=str(config.model_path),
        )
        print(f'Downloaded model to {model_dir}')


def download_adapter_from_kaggle(config: Config) -> None:
    """Downloads the LoRA adapter from Kaggle if `adapter_path` is set and the
    local adapter directory does not already exist or is empty. If the local
    adapter directory exists and is not empty, it skips the download to avoid
    overwriting existing files. If the download proceeds, it uses kagglehub to
    download the adapter from Kaggle and saves it to the specified local path.
    """

    if not config.adapter_path:
        return

    print('\n[[ LoRA Adapter Download ]]')
    adapter_dir = config.adapter_path
    if path_has_content(adapter_dir):
        print(f'Adapter path already exists and is not empty: {adapter_dir}')
        print(
            'Skipping download. If you want to re-download, please remove the '
            'existing directory.'
        )
    else:
        print('Downloading LoRA adapter from Kaggle...')
        kagglehub = import_kagglehub()
        kagglehub.dataset_download(
            LORA_ADAPTER_KAGGLE_DATASET_REF,
            output_dir=str(adapter_dir),
        )
        print(f'Downloaded LoRA adapter to {adapter_dir}')


def ensure_paths(config: Config) -> None:
    """Ensures that the specified paths exist and are valid. For the model path
    and training CSV path, it checks that they exist and raises an error if not.
    For the output directory and the rollout results directory, it checks if
    they exist and create them if not.
    """

    print('\n[[ Path Ensurance ]]')
    if not config.model_path.exists():
        raise FileNotFoundError(
            f'Model path does not exist: {config.model_path}'
        )
    if not config.train_csv_path.exists():
        raise FileNotFoundError(
            f'Training CSV path does not exist: {config.train_csv_path}'
        )
    if not config.output_dir.exists():
        print(f'Output directory does not exist, creating: {config.output_dir}')
        config.output_dir.mkdir(parents=True, exist_ok=True)

    rollout_results_dir = config.output_dir / 'rollout_results'
    if not rollout_results_dir.exists():
        print(
            'Rollout results directory does not exist, '
            f'creating: {config.output_dir}'
        )
        rollout_results_dir.mkdir(parents=True, exist_ok=True)


def print_runtime_environment():
    """Prints information about the runtime environment, including Python
    version, PyTorch version, CUDA availability, and GPU details if available.
    This can help with debugging and ensuring that the environment is set up
    correctly for running the self-distillation script, which relies on PyTorch
    and CUDA for efficient model inference and rollout generation.
    """

    print('\n[[ Runtime Environment ]]')
    print(f'Python: {sys.version}')
    print(f'Torch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    if not torch.cuda.is_available():
        print(
            'Warning: CUDA GPU is not available. This script is designed to '
            'run with a CUDA GPU.'
        )

    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=name,driver_version,memory.total,memory.used,'
                'memory.free,temperature.gpu,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        for idx, line in enumerate(result.stdout.strip().splitlines()):
            name, driver, mem_total, mem_used, mem_free, temp, util = [
                x.strip() for x in line.split(',')
            ]

            print(f'GPU {idx}: {name}')
            print(f'  Driver version: {driver}')
            print(f'  Memory total:   {mem_total} MiB')
            print(f'  Memory used:    {mem_used} MiB')
            print(f'  Memory free:    {mem_free} MiB')
            print(f'  Temperature:    {temp} °C')
            print(f'  Utilization:    {util}%')
    except FileNotFoundError:
        print(
            'nvidia-smi not found. This may not be an NVIDIA GPU environment.'
        )
    except subprocess.CalledProcessError as e:
        print('Failed to run nvidia-smi.')
        print(e.stderr)


def seed_everything(seed: int) -> None:
    """seeds all relevant random number generators for reproducibility. this
    includes seeding python's built-in `random` module and pytorch's random
    number generator.
    """

    random.seed(seed)
    torch.manual_seed(seed)


def infer_puzzle_type(prompt: str) -> str:
    """Infers the puzzle type based on the first line of the prompt.

    This function examines the first line of the input prompt and checks for
    specific keywords to determine the puzzle type.

    Args:
        prompt: The input prompt to classify.

    Returns:
        A string representing the puzzle type.
    """

    first_line = prompt.split('\n', 1)[0]

    if 'bit' in first_line:
        return 'bit'
    if 'gravitational' in first_line:
        return 'gravity'
    if 'unit' in first_line:
        return 'unit'
    if 'numeral' in first_line:
        return 'roman'
    if 'encryption' in first_line:
        return 'cipher'
    if 'equations' in first_line:
        return 'symbol'

    return 'other'


def load_training_data(config: Config) -> pd.DataFrame:
    """Loads the training data from the specified CSV file, applies the offset
    and limit for selecting a subset of the data, and infers the puzzle type for
    each example based on the prompt. It also filters the examples based on the
    specified puzzle types in the config, if any.

    It then prints the number of loaded examples and the distribution of puzzle
    types in the training data.

    Returns:
        A pandas DataFrame containing the loaded training data with an
        additional 'type' column for the inferred puzzle type.
    """

    print('\n[[ Load Training Data ]]')
    print('Loading training data from CSV...')
    train_df = pd.read_csv(config.train_csv_path)
    if config.offset > 0:
        train_df = train_df.iloc[config.offset :]
    if config.limit > 0:
        train_df = train_df.iloc[: config.limit]

    print(
        f'Loaded {len(train_df)} training examples from {config.train_csv_path}'
    )
    train_df['type'] = train_df['prompt'].apply(infer_puzzle_type)
    print('Puzzle distribution over types:')
    for _puzzle_type, _count in train_df['type'].value_counts().items():
        print(f'  - {_puzzle_type}: {_count} examples')

    if config.puzzle_types:
        train_df = train_df[train_df['type'].isin(config.puzzle_types)]
        print(
            f'After filtering by puzzle types {config.puzzle_types}, '
            f'{len(train_df)} examples remain.'
        )

    return cast(pd.DataFrame, train_df)


def load_model_and_tokenizer(config: Config) -> tuple[Any, Any]:
    """Loads vLLM and tokenizer.

    vLLM imports are intentionally inside this function so spawned child
    processes can import this module without constructing another engine.

    Returns:
        A tuple of (llm, tokenizer) where `llm` is the loaded vLLM model and
        `tokenizer` is the corresponding tokenizer.
    """

    from transformers import AutoTokenizer
    from vllm import LLM  # pyright: ignore[reportMissingImports]

    print('\n[[ Model and Tokenizer Loading ]]')

    llm = LLM(
        model=str(config.model_path),
        enable_lora=True,
        max_lora_rank=config.model_config.max_lora_rank,
        max_model_len=config.model_config.max_model_len,
        tensor_parallel_size=config.model_config.tensor_parallel_size,
        data_parallel_size=config.model_config.data_parallel_size,
        gpu_memory_utilization=config.model_config.gpu_memory_utilization,
        dtype=config.model_config.dtype,
        enable_prefix_caching=config.model_config.enable_prefix_caching,
        trust_remote_code=True,
        seed=config.seed,
    )
    print(f'Loaded model from: {config.model_path}')

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )
    print(f'Loaded tokenizer: {tokenizer.__class__.__name__}')

    return llm, tokenizer


def convert_prompt_into_model_input(prompt: str, tokenizer: Any) -> str:
    """Converts a prompt into an exact chat format expected by the model and
    tokenizer.

    This function first appends the `PROMPT_INSTRUCTION` to each prompt, and
    converts the resulting string into a formatted chat message that can be
    understood by the model and tokenizer.

    More specifically, `apply_chat_template` takes the message list and renders
    it through the Jinja template stored in the tokenizer's config (the value o
    `chat_template`). That template inserts the model-specific special tokens,
    such as role markers, turn delimiters, system prefix, and etc.

    In this script, the formatted prompts are called "model inputs".
    """

    prompt_with_instruction = prompt + PROMPT_INSTRUCTION
    return tokenizer.apply_chat_template(
        # Creates a one-message chat conversation from the user.
        [{'role': 'user', 'content': prompt_with_instruction}],
        # Returns a string prompt instead of token IDs.
        tokenize=False,
        # appends the model’s assistant-start marker so generation can begin.
        add_generation_prompt=True,
        # enables the model’s reasoning/thinking mode if the tokenizer template
        # supports it.
        enable_thinking=True,
    )


def extract_final_answer(text: str) -> str | None:
    """Extracts the final answer from a text string using a cascade of
    strategies.

    This function returns the last non-empty ``\\boxed{...}`` content. Handles
    truncated boxes (missing closing brace) via ``(?:\\}|$)``.

    Example: ``\\boxed{42}`` → ``"42"``

    Args:
        text: The text to extract the final answer from.

    Returns:
        The extracted answer as a stripped string, or ``None`` if the text
        contains no non-empty lines.
    """

    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    return None


def verify(
    correct_answer: str,
    predicted_answer: str,
    *,
    compare_float_closeness: bool = True,
) -> bool:
    """Verifies if the predicted answer is the same as the correct answer.

    This function first strips whitespaces from two given answers, and convert
    all letters in both answers into lowercase.

    Then, it compares the two strings, and return True if they are the
    identical. If the two strings are not identical, it converts both answers
    into floating-point numbers and check if they are close enough. Two numbers
    are considered to be close enough if

        1. the relative difference is less than 1%, or
        2. the absolute difference is at most 0.00001.

    It returns True if the two numbers are close enough.

    Args:
        correct_answer: The correct answer to compare with.
        predicted_answer: The predicted answer to compare with.
        compare_float_closeness: Whether to compare float closeness when the two
        answers are not identical.

    Returns:
        True if the predicted answer is considered correct, and False otherwise.
    """

    correct_answer = correct_answer.strip().lower()
    predicted_answer = predicted_answer.strip().lower()

    if correct_answer == predicted_answer:
        return True

    if not compare_float_closeness:
        return False

    try:
        return math.isclose(
            float(correct_answer),
            float(predicted_answer),
            rel_tol=1e-2,
            abs_tol=1e-5,
        )
    except Exception:
        return False


def should_compare_closeness(puzzle_type: str) -> bool:
    """Determines whether to compare float closeness based on the puzzle type.

    For certain puzzle types such as 'gravity' and 'unit', the final answer is a
    real number, and it is reasonable to consider a predicted answer correct if
    it is close enough to the correct answer.

    Args:
        puzzle_type: The type of the puzzle, as inferred by infer_puzzle_type().
    """

    return puzzle_type == 'gravity' or puzzle_type == 'unit'


@dataclass(frozen=True)
class RolloutResult:
    # Rollout group; 2 represents the second group. For each prompt, there are
    # N many groups, where N is the summation of the first element of the tuples
    # in `config.model_config.rollout_settings`.
    group: int

    # Prompt ID in the original train CSV.
    prompt_id: str

    # The type of the puzzle inferred by infer_puzzle_type().
    puzzle_type: str

    # The correct answer of the prompt. This is given in the train CSV.
    correct_answer: str

    # The inference result from the vLLM. This includes both the CoT part and
    # final the final answer part.
    output: str

    # The predicted answer. This is extracted from the final answer part.
    predicted_answer: str | None

    # Whether the predicted answer is correct.
    is_correct: bool


def generate_rollouts_for_one_prompt(
    config: Config,
    llm: Any,
    tokenizer: Any,
    prompt_id: str,
    prompt: str,
    correct_answer: str,
) -> None:
    """Generates rollouts for one prompt, and saves the rollout results to a CSV
    file.

    This function first converts the prompt into a model input. Then, it
    iterates over the rollout settings specified in the config, and for each
    rollout setting entry, it generates the specified number of rollouts with
    the specified temperature using the vLLM model.

    For each generated rollout, it extracts the predicted answer using
    `extract_final_answer()`, verifies if the predicted answer is correct using
    `verify()`, and creates a `RolloutResult` instance to store the results.

    Finally, it saves all rollout results for this prompt to a CSV file in the
    rollout results directory under the output directory, and returns the next
    rollout ID to be used.

    Args:
        config: The configuration of this script.
        llm: The loaded vLLM model.
        tokenizer: The loaded tokenizer corresponding to the model.
        prompt_id: The ID of the prompt, as given in the training CSV.
        prompt: The prompt string for which to generate rollouts.
        correct_answer: The correct answer for the prompt, as given in the
            training CSV.
    """

    from vllm import SamplingParams  # pyright: ignore[reportMissingImports]
    from vllm.lora.request import (  # pyright: ignore[reportMissingImports]
        LoRARequest,
    )

    model_input: str = convert_prompt_into_model_input(prompt, tokenizer)
    is_correct: bool = True
    puzzle_type: str = infer_puzzle_type(prompt)

    # Generate rollouts based on the rollout settings. Each rollout setting
    # entry is a tuple constiting of "the number of rollouts" and a
    # temperature. We will be iterating over all these entries.
    print(f'Generating rollouts for prompt #{prompt_id}', end='  ')
    start_time = time.perf_counter()
    rollout_results: list[RolloutResult] = []
    group_id: int = 0
    for num_rollouts, temperature in config.model_config.rollout_settings:
        # Let LLM generate `num_rollouts` many names.
        sampling_params = SamplingParams(
            n=num_rollouts,
            temperature=temperature,
            top_p=config.model_config.top_p,
            max_tokens=config.model_config.max_tokens,
        )
        results = llm.generate(
            model_input,
            sampling_params=sampling_params,
            lora_request=LoRARequest('adapter', 1, str(config.adapter_path))
            if config.adapter_path
            else None,
            use_tqdm=False,
        )

        # Iterate over all outputs in this rollout setting entry, and add a
        # `RolloutResult` instance to `rollout_results`
        for i in range(num_rollouts):
            output = results[0].outputs[i].text
            predicted_answer: str | None = extract_final_answer(output)
            puzzle_type: str = infer_puzzle_type(prompt)
            is_correct = (
                verify(
                    correct_answer,
                    predicted_answer,
                    compare_float_closeness=should_compare_closeness(
                        puzzle_type
                    ),
                )
                if predicted_answer is not None
                else False
            )
            rollout_result = RolloutResult(
                group=group_id,
                prompt_id=prompt_id,
                puzzle_type=puzzle_type,
                correct_answer=correct_answer,
                output=output,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
            )
            rollout_results.append(rollout_result)
            group_id += 1

    elapsed_time_seconds = time.perf_counter() - start_time
    print(f'completed ({elapsed_time_seconds:2f})')

    # Save the `rollout_results` to a CSV file.
    rollout_results_df = pd.DataFrame(rollout_results)
    rollout_results_path = (
        config.output_dir / 'rollout_results' / f'{prompt_id}.csv'
    )
    rollout_results_df.to_csv(rollout_results_path, index=False)


def generate_rollouts_for_all_prompts(
    config: Config, llm: Any, tokenizer: Any, train_df: pd.DataFrame
) -> None:
    """Generates rollouts for all prompts in the training DataFrame."""

    print('\n[[ Rollouts Generation ]]')
    for _, row in train_df.iterrows():
        prompt_id = str(row['id'])

        # Skip rollout results if the results file already exists
        rollout_results_path: Path = (
            config.output_dir / 'rollout_results' / f'{prompt_id}.csv'
        )
        if rollout_results_path.exists():
            print(
                f'Rollout results file already exists: {rollout_results_path}; '
                'skipped'
            )
            continue

        prompt: str = str(row['prompt'])
        correct_answer: str = str(row['answer'])
        generate_rollouts_for_one_prompt(
            config,
            llm,
            tokenizer,
            prompt_id,
            prompt,
            correct_answer,
        )


def load_rollout_results(
    config: Config, train_df: pd.DataFrame
) -> list[RolloutResult]:
    """Loads all rollout results from storage and returns them as a list of
    `RolloutResult` instances.

    This function iterates over all prompts in the given training DataFrame, and
    for each prompt, and for each prompt, it looks for the corresponding rollout
    results CSV file in the output directory. If the file exists, it loads the
    CSV file into a pandas DataFrame, and iterates over each row in the
    DataFrame to create a `RolloutResult` instance, which is then added to the
    list of all rollout results.

    Args:
        config: The configuration of this script.
        train_df: The training data DataFrame, which contains the prompts and is
            used to determine which rollout results files to load.

    Returns:
        A list of `RolloutResult` instances containing all rollout results
        loaded from storage.
    """

    print('\n[[ Rollout Results Loading ]]')
    print('Loading all rollout results from storage.')
    all_rollout_results: list[RolloutResult] = []
    for _, train_row in train_df.iterrows():
        prompt_id = str(train_row['id'])
        rollout_results_path = (
            config.output_dir / 'rollout_results' / f'{prompt_id}.csv'
        )
        if rollout_results_path.exists():
            df = pd.read_csv(
                rollout_results_path,
                dtype={
                    'prompt_id': str,
                    'correct_answer': str,
                    'predicted_answer': str,
                },
            )

            def parse_predicted_answer(x: Any) -> str | None:
                if pd.isna(x):
                    return None
                return str(x)

            all_rollout_results.extend(
                RolloutResult(
                    group=int(str(row['group'])),
                    prompt_id=str(row['prompt_id']),
                    puzzle_type=str(row['puzzle_type']),
                    correct_answer=str(row['correct_answer']),
                    output=str(row['output']),
                    predicted_answer=parse_predicted_answer(
                        row['predicted_answer']
                    ),
                    is_correct=str(row['is_correct']).lower() == 'true',
                )
                for _, row in df.iterrows()
            )
        else:
            print(
                'Warning: Rollout results file does not exist: '
                f'{rollout_results_path}; skipped'
            )

    return all_rollout_results


def aggregate_by_prompt(
    all_rollout_results: list[RolloutResult],
) -> dict[str, list[RolloutResult]]:
    """Aggregates rollout results by prompt ID.

    This function takes a list of `RolloutResult` instances and groups them into
    a dictionary where the keys are prompt IDs and the values are lists of
    `RolloutResult` instances corresponding to that prompt ID.

    Args:
        all_rollout_results: A list of `RolloutResult` instances to be
            aggregated.

    Returns:
        A dictionary mapping prompt IDs to lists of `RolloutResult` instances
        for each prompt ID.
    """

    rollout_results_by_prompt_ids: dict[str, list[RolloutResult]] = defaultdict(
        list
    )
    for rollout_result in all_rollout_results:
        rollout_results_by_prompt_ids[rollout_result.prompt_id].append(
            rollout_result
        )
    return rollout_results_by_prompt_ids


@dataclass(frozen=True)
class PerPromptResult:
    """Aggregated rollout results for a single prompt ID."""

    # The ID of the prompt.
    prompt_id: str

    # The type of the puzzle given by infer_puzzle_type().
    puzzle_type: str

    # The number of correct rollouts for this prompt ID. A rorrect rollout
    # refers to a rollout that generates a correct final answer.
    num_correct: int

    # The list of rollout results in the group.
    groups: list[RolloutResult]

    # The indexes of the rollouts that generated the majority answer.
    majority_answer_indexes: set[int]

    # The majority answer across all rollouts for this prompt ID.
    majority_answer: str

    # The number of rollouts that generated the majority answer.
    majority_count: int

    # Whether the majority answer is correct.
    is_majority_correct: bool


def analyze_per_prompt(
    rollout_results_by_prompt_ids: dict[str, list[RolloutResult]],
) -> list[PerPromptResult]:
    """Analyzes rollout results on a per-prompt basis.

    This function takes a dictionary of rollout results grouped by prompt ID,
    and for each prompt ID, it calculates the number of correct rollouts,
    determines the majority answer and its count, and checks if the majority
    answer is correct.

    This function also prints the overall majority answer accuracy across all
    prompts.

    Args:
        rollout_results_by_prompt_ids: A dictionary mapping prompt IDs to lists
            of `RolloutResult` instances for each prompt ID.

    Returns:
        A list of `PerPromptResult` instances containing the analysis results
        for each prompt ID.
    """

    per_prompt_results: list[PerPromptResult] = []

    for prompt_id, rollout_results in rollout_results_by_prompt_ids.items():
        puzzle_type = rollout_results[0].puzzle_type
        correct_answer = rollout_results[0].correct_answer
        num_correct = sum(result.is_correct for result in rollout_results)

        answer_counts: dict[str, int] = defaultdict(int)
        for rollout_result in rollout_results:
            if rollout_result.predicted_answer is not None:
                answer_counts[rollout_result.predicted_answer] += 1

        if answer_counts:
            majority_answer, majority_count = max(
                answer_counts.items(), key=lambda item: item[1]
            )
            is_majority_correct = verify(
                correct_answer,
                majority_answer,
                compare_float_closeness=should_compare_closeness(puzzle_type),
            )
        else:
            majority_answer, majority_count, is_majority_correct = '', 0, False

        per_prompt_results.append(
            PerPromptResult(
                prompt_id=prompt_id,
                puzzle_type=puzzle_type,
                num_correct=num_correct,
                groups=rollout_results,
                majority_answer_indexes={
                    i
                    for i, result in enumerate(rollout_results)
                    if result.predicted_answer == majority_answer
                },
                majority_answer=majority_answer,
                majority_count=majority_count,
                is_majority_correct=is_majority_correct,
            )
        )

    print('\n[[ Per-Prompt Analysis ]]')
    majority_accuracy: float = (
        mean(result.is_majority_correct for result in per_prompt_results)
        if per_prompt_results
        else 0.0
    )
    if per_prompt_results:
        print(f'Majority answer accuracy: {majority_accuracy:.4f}')
    else:
        print('No per-prompt results available.')

    return per_prompt_results


def print_accuracy_by_type(per_prompt_results: list[PerPromptResult]) -> None:
    """Prints the accuracy of rollouts by puzzle type.

    This function takes a list of `PerPromptResult` instances, groups them by
    puzzle type, and calculates the single rollout accuracy and majority answer
    accuracy for each puzzle type. It then prints the results in a tabular
    format.
    """

    from tabulate import tabulate

    per_prompt_results_by_type: dict[str, list[PerPromptResult]] = defaultdict(
        list
    )
    for result in per_prompt_results:
        per_prompt_results_by_type[result.puzzle_type].append(result)

    table: list[list[str]] = []
    for puzzle_type in sorted(per_prompt_results_by_type.keys()):
        results = per_prompt_results_by_type[puzzle_type]
        if not results:
            table.append([puzzle_type, 'N/A', 'N/A', '0'])
            continue

        single_rollout_accuracy = sum(
            sum(group.is_correct for group in result.groups)
            for result in results
        ) / sum(len(result.groups) for result in results)
        majority_accuracy = mean(
            result.is_majority_correct for result in results
        )
        table.append(
            [
                puzzle_type,
                f'{single_rollout_accuracy:.4f}',
                f'{majority_accuracy:.4f}',
                str(len(results)),
            ]
        )

    print('\n[[ Accuracy by Puzzle Type ]]')
    print(
        tabulate(
            table,
            headers=[
                'Puzzle Type',
                'Single Rollout Accuracy',
                'Majority Answer Accuracy',
                'Num Prompts',
            ],
            tablefmt='github',
        )
    )


def print_accuracy_by_temperature(
    config: Config, all_rollout_results: list[RolloutResult]
) -> None:
    """Prints the accuracy of rollouts by temperature.

    This function takes the list of all rollout results and groups them by
    temperature based on the rollout settings specified in the config. For each
    temperature, it calculates the single rollout accuracy and majority answer
    accuracy across all prompts that have rollouts generated with that
    temperature. It then prints the results in a tabular format.

    Args:
        config: The configuration of this script, which contains the rollout
            settings that specify the temperatures for each group of rollouts.
        all_rollout_results: A list of `RolloutResult` instances containing all
            rollout results, which will be grouped by temperature for analysis.
    """

    from tabulate import tabulate

    group_to_temperature: dict[int, float] = {}
    group_id = 0
    for num_rollouts, temperature in config.model_config.rollout_settings:
        for _ in range(num_rollouts):
            group_to_temperature[group_id] = temperature
            group_id += 1

    # Aggregate rollout results by temperatures.
    rollout_results_by_temperature: dict[float, list[RolloutResult]] = (
        defaultdict(list)
    )
    for rollout_result in all_rollout_results:
        temperature = group_to_temperature[rollout_result.group]
        rollout_results_by_temperature[temperature].append(rollout_result)

    # For each temperature, calculate the single rollout accuracy and majority
    # answer accuracy.
    table: list[list[str]] = []
    for temperature in sorted(rollout_results_by_temperature.keys()):
        rollout_results = rollout_results_by_temperature[temperature]
        rollout_results_by_prompt_id: dict[str, list[RolloutResult]] = (
            defaultdict(list)
        )
        for rollout_result in rollout_results:
            rollout_results_by_prompt_id[rollout_result.prompt_id].append(
                rollout_result
            )

        majority_correct_by_prompt: list[bool] = []
        for prompt_rollout_results in rollout_results_by_prompt_id.values():
            answer_counts: dict[str, int] = defaultdict(int)
            for rollout_result in prompt_rollout_results:
                if rollout_result.predicted_answer is not None:
                    answer_counts[rollout_result.predicted_answer] += 1

            if answer_counts:
                majority_answer = max(
                    answer_counts.items(), key=lambda item: item[1]
                )[0]
                majority_correct_by_prompt.append(
                    verify(
                        prompt_rollout_results[0].correct_answer,
                        majority_answer,
                        compare_float_closeness=should_compare_closeness(
                            prompt_rollout_results[0].puzzle_type
                        ),
                    )
                )
            else:
                majority_correct_by_prompt.append(False)

        table.append(
            [
                f'{temperature:.1f}',
                f'{mean(result.is_correct for result in rollout_results):.4f}',
                f'{mean(majority_correct_by_prompt):.4f}',
                str(len(rollout_results_by_prompt_id)),
                str(len(rollout_results)),
            ]
        )

    print('\n[[ Accuracy by Temperature ]]')
    print(
        tabulate(
            table,
            headers=[
                'Temperature',
                'Single Rollout Accuracy',
                'Majority Answer Accuracy',
                'Num Prompts',
                'Num Rollouts',
            ],
            tablefmt='github',
        )
    )


@dataclass(frozen=True)
class SelfDistillationRecord:
    # The ID of the prompt.
    prompt_id: str

    # The type of the puzzle given by infer_puzzle_type().
    puzzle_type: str

    # The original prompt with the instruction appended.
    prompt: str

    # The raw output generated by the LLM for this prompt.
    output: str


def build_self_distillation_dataset(
    train_df: pd.DataFrame,
    per_prompt_results: list[PerPromptResult],
    tokenizer: Any,
) -> list[SelfDistillationRecord]:
    """Builds a self-distillation dataset.

    This function iterates over all per-prompt results. For each per-prompt
    result, it collects the booleans representing whether the rollouts are
    correct. In the case where all rollouts are correct or all are incorrect,
    we skip it without creating a self-distillation record. This is inspired by
    DAPO, where such cases are considered to carry little preference signal, and
    the model cannot learn much from them.

    If there exists at least one correct rollout and one incorrect rollout, we
    check whether the majority rollout is correct. If so, we check whether any
    other rollouts give the correct answer. If yes, we rule out all the majority
    rollouts and adopt the correct rollout with the shortest CoT.
    """

    records: list[SelfDistillationRecord] = []
    num_all_correct = 0
    num_all_incorrect = 0
    prompt_by_id = {
        str(row['id']): str(row['prompt']) for _, row in train_df.iterrows()
    }

    for per_prompt_result in per_prompt_results:
        is_correct_by_group = [
            group.is_correct for group in per_prompt_result.groups
        ]

        # Inspired by DAPO: skip cases where all answers are correct or all are
        # incorrect, because they carry little preference signal.
        if all(is_correct_by_group):
            num_all_correct += 1
            continue
        if not any(is_correct_by_group):
            num_all_incorrect += 1
            continue

        # Check if the majority answer is correct. If so, check if there are
        # other rollouts that give the correct answer. We want the model to
        # learn from the minority answer with the shortest CoT.
        candidate_rollout_results: list[RolloutResult] = []
        if per_prompt_result.is_majority_correct:
            for i, group in enumerate(per_prompt_result.groups):
                if i in per_prompt_result.majority_answer_indexes:
                    continue
                if group.is_correct:
                    candidate_rollout_results.append(group)
        else:
            for group in per_prompt_result.groups:
                if group.is_correct:
                    candidate_rollout_results.append(group)

        # Extract the rollout with the shortest output length.
        if candidate_rollout_results:
            best_rollout_result = min(
                candidate_rollout_results, key=lambda result: len(result.output)
            )
            records.append(
                SelfDistillationRecord(
                    prompt_id=per_prompt_result.prompt_id,
                    puzzle_type=per_prompt_result.puzzle_type,
                    prompt=convert_prompt_into_model_input(
                        prompt_by_id[per_prompt_result.prompt_id],
                        tokenizer,
                    ),
                    output=best_rollout_result.output,
                )
            )

    print('\n[[ Self-Distillation Dataset ]]')
    print(f'Constructed {len(records)} self-distillation records.')
    print(f'Number of prompts with all correct rollouts: {num_all_correct}.')
    print(
        f'Number of prompts with all incorrect rollouts: {num_all_incorrect}.'
    )

    return records


def save_self_distillation_dataset(
    config: Config, records: list[SelfDistillationRecord]
) -> None:
    """Saves self-distillation set to the storage.

    Args:
        config: The configuration of this script.
        records: A list of self-distillation records to save.
    """

    print('\n[[ Saving Self-Distillation Dataset ]]')
    self_distillation_output_path = (
        config.output_dir / 'self_distillation_dataset.csv'
    )
    print(
        f'Saving self-distillation dataset to {self_distillation_output_path}'
    )
    pd.DataFrame([asdict(record) for record in records]).to_csv(
        self_distillation_output_path, index=False
    )
    print('Saved self-distillation dataset successfully.')


def main() -> int:
    """Main entry point of the script.

    This function orchestrates the entire workflow of the script, including
    configuration setup, model and tokenizer loading, rollout generation,
    analysis, and self-distillation dataset construction and saving.

    Returns:
        An integer exit code indicating the success or failure of the script.
    """

    config = build_config(parse_args())
    print_config(config)
    download_model_from_kaggle(config)
    download_adapter_from_kaggle(config)
    ensure_paths(config)
    print_runtime_environment()

    seed_everything(config.seed)
    train_df = load_training_data(config)
    llm, tokenizer = load_model_and_tokenizer(config)
    generate_rollouts_for_all_prompts(config, llm, tokenizer, train_df)
    all_rollout_results: list[RolloutResult] = load_rollout_results(
        config, train_df
    )

    rollout_results_by_prompt_ids = aggregate_by_prompt(all_rollout_results)
    per_prompt_results = analyze_per_prompt(rollout_results_by_prompt_ids)

    print_accuracy_by_type(per_prompt_results)
    print_accuracy_by_temperature(config, all_rollout_results)

    records = build_self_distillation_dataset(
        train_df, per_prompt_results, tokenizer
    )
    save_self_distillation_dataset(config, records)

    return 0


if __name__ == '__main__':
    # Required for Python's spawn start method. Without this guard, spawned
    # workers re-import this script and would execute `main()` recursively.
    import multiprocessing as mp

    mp.freeze_support()
    mp.set_start_method(os.environ['VLLM_WORKER_MULTIPROC_METHOD'], force=True)
    raise SystemExit(main())
