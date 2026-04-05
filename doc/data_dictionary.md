# Data Dictionary - NVIDIA Nemotron Model Reasoning Challenge

Updated: 2026-03-28

## File Overview

| File | Rows | Size | Description |
|---|---:|---:|---|
| `train.csv` | 9500 | 3.07 MB | Training set, includes answer column |
| `test.csv` | 3 | 1.46 KB | Current public test set, excludes answer column |

## Schema

### `train.csv`

| Field | Type | Description | Example |
|---|---|---|---|
| `id` | string | Unique sample identifier | `00066667` |
| `prompt` | string | Full question text | `In Alice's Wonderland, ...` |
| `answer` | string | Standard answer | `10010111` |

### `test.csv`

| Field | Type | Description | Example |
|---|---|---|---|
| `id` | string | Unique sample identifier | `00066667` |
| `prompt` | string | Full question text | `In Alice's Wonderland, ...` |

## Verified Facts

- The label column name in `train.csv` is `answer`, not `prediction`.
- The current public `test.csv` contains only 3 rows.
- All 3 `id`s in the current public `test.csv` exist in `train.csv`.
- The 3 `prompt`s in the current public `test.csv` are identical to the corresponding samples in `train.csv`.

## Task Family Distribution

Broadly categorized by the first sentence of the prompt, the training set has 6 major families:

| family | count |
|---|---:|
| `bit` | 1602 |
| `gravity` | 1597 |
| `unit` | 1594 |
| `cipher` | 1576 |
| `roman` | 1576 |
| `symbol` | 1555 |

## Answer Format Statistics

- Average length: `8.39`
- Median length: `5`
- Shortest: `1`
- Longest: `39`

Rough distribution:

| Type | Count |
|---|---:|
| number-like | 5480 |
| single-token | 2433 |
| multi-token | 1576 |
| single-letter option | 11 |

## Submission Format

The submission file should be:

```csv
id,prediction
00066667,10010111
...
```

## Current Workflow Observations

- Running `kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f <csv>` returns HTTP 400.
- The actual working path is:
  1. Prepare a Kaggle kernel.
  2. Push a new version.
  3. Notebook outputs `submission.zip`.
  4. Wait for the notebook run to complete.
  5. Submit the notebook version as a code competition entry.

Note:

- This workflow was confirmed by testing on 2026-03-28.
- This suggests the competition cannot accept direct CSV uploads via CLI, at least at this stage.
- The actual mount path for the competition source in the Kaggle environment is:
  - `/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/`

## Download Command

```bash
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/raw
```
