# Local Distillation Runners

This directory contains the first local teacher-data pipeline for `exp-012`.

## Files

- `build_holdout.py`: create a family-stratified local holdout split from `train.csv`
- `run_codex_distill.py`: call local `codex exec` to generate structured teacher supervision
- `run_longcat_distill.py`: call LongCat directly through its OpenAI-compatible API
- `run_openrouter_distill.py`: call OpenRouter directly through its OpenAI-compatible API
- `run_nemotron_super_distill.py`: batch-call Nemotron 3 Super through NVIDIA hosted NIM or OpenRouter free with rate limits and resume
- `run_parallel_thinking_distill.py`: call an OpenAI-compatible API to generate parallel-thinking supervision
- `build_rft_preference_data.py`: convert scored teacher/rollout JSONL files into RFT and DPO/ORPO preference datasets
- `teacher_response.schema.json`: strict JSON schema for Codex teacher output
- `parallel_thinking_response.schema.json`: strict JSON schema for parallel-thinking teacher output

## Suggested Workflow

1. Build a local holdout split:

```powershell
python scripts/distill/build_holdout.py
```

2. Smoke-test the Codex teacher on a few rows:

```powershell
python scripts/distill/run_codex_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family bit --limit 3 --output-jsonl data/distill/codex_smoke.jsonl --model gpt-5.4
```

3. Run a harder family:

```powershell
python scripts/distill/run_codex_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family symbol --limit 50 --output-jsonl data/distill/codex_symbol_eval.jsonl --model gpt-5.4
```

4. After validating quality, distill the full train split:

```powershell
python scripts/distill/run_codex_distill.py --input-csv data/distill/holdout_v1/train_split.csv --output-jsonl data/distill/codex_teacher_v1.jsonl --model gpt-5.4
```

## Direct LongCat Workflow

Set the API key in the shell first:

```powershell
$env:LONGCAT_API_KEY="..."
```

Smoke-test a single `symbol` row with the recommended low-randomness temperature:

```powershell
python scripts/distill/run_longcat_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family symbol --limit 1 --temperature 0.2 --stream --output-jsonl data/distill/longcat_symbol_smoke.jsonl
```

If you want diversity for self-consistency experiments, increase only `temperature` first:

```powershell
python scripts/distill/run_longcat_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family symbol --limit 3 --temperature 0.5 --output-jsonl data/distill/longcat_symbol_temp05.jsonl
```

## Direct OpenRouter Workflow

Set the API key in the shell first:

```powershell
$env:OPENROUTER_API_KEY="..."
```

Smoke-test the free Nemotron teacher on one row:

```powershell
python scripts/distill/run_openrouter_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family symbol --limit 1 --temperature 0.2 --stream --output-jsonl data/distill/openrouter_symbol_smoke.jsonl
```

The default model is:

```text
nvidia/nemotron-3-super-120b-a12b:free
```

If the model behaves better with stricter output control, add JSON mode:

```powershell
python scripts/distill/run_openrouter_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family bit --limit 3 --temperature 0.2 --json-mode --output-jsonl data/distill/openrouter_bit_jsonmode.jsonl
```

To enable reasoning with family-specific effort levels:

```powershell
python scripts/distill/run_openrouter_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --limit 20 --reasoning --auto-family-effort --stream --output-jsonl data/distill/openrouter_reasoning_eval.jsonl
```

Current built-in family effort defaults:

```text
symbol=xhigh
bit=high
cipher=high
gravity=medium
unit=medium
roman=low
```

You can override individual families explicitly:

```powershell
python scripts/distill/run_openrouter_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family symbol --limit 5 --reasoning --family-reasoning-effort symbol=xhigh --family-reasoning-effort bit=medium --output-jsonl data/distill/openrouter_symbol_xhigh.jsonl
```

## Parallel Thinking API Workflow

This runner is for `DAPO + synthetic data + parallel thinking` experiments. It keeps the existing short-rationale distillation format untouched and writes a separate JSONL file with:

- `parallel_thinking`: structured branch objects and consensus metadata
- `assistant_response`: compact JSON text with `parallel_thinking`, `consensus`, and `answer`
- `messages`: a user/assistant pair that can be converted into SFT or DAPO prompt data
- `answer_format` and `symbol_subtype`: family-aware formatting labels for filtering and sampling
- `reward`, `reward_label`, and `use_for`: coarse labels for DAPO/SFT data routing
- `verified_answer_exact`: exact-match check when the source CSV has `answer`

The prompt includes family-specific scaffolds inspired by the public CoT-selection notebooks: per-bit checks for `bit`, substitution-map checks for `cipher`, rounding-aware ratio/formula checks for `unit` and `gravity`, and numeric-like vs symbolic-like handling for `symbol`.

The default backend is OpenRouter because the current repo already uses it:

```powershell
$env:OPENROUTER_API_KEY="..."
python scripts/distill/run_parallel_thinking_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family bit --limit 3 --branches 3 --json-mode --output-jsonl data/distill/parallel_thinking_bit_smoke.jsonl
```

NVIDIA hosted NIM can be used through its OpenAI-compatible endpoint:

```powershell
$env:NVIDIA_API_KEY="..."
python scripts/distill/run_parallel_thinking_distill.py --base-url https://integrate.api.nvidia.com/v1 --api-key-env NVIDIA_API_KEY --model nvidia/nemotron-3-super-120b-a12b --input-csv data/distill/holdout_v1/eval_split.csv --family bit --limit 3 --branches 3 --json-mode --output-jsonl data/distill/nim_super_bit_smoke.jsonl
```

By default, the teacher does not receive the gold answer. This is the right mode for unbiased DAPO rollout data. Correct rows get `reward_label = positive`; wrong-but-valid rows get `reward_label = negative_answer`.

For answer-conditioned rationale generation, pass `--condition-on-gold`. Use these rows as SFT/bootstrap explanations, not as unbiased rollout samples:

```powershell
python scripts/distill/run_parallel_thinking_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family symbol --limit 3 --branches 3 --condition-on-gold --assistant-format nemotron-cot --output-jsonl data/distill/parallel_thinking_symbol_gold_cot.jsonl
```

Use `--assistant-format nemotron-cot` when preparing Nemotron-style SFT targets ending in `</think>` and `\boxed{answer}`. Keep the default `--assistant-format json` when the next stage expects structured JSON.

To use another OpenAI-compatible API, override the base URL, model, and key environment variable:

```powershell
$env:OPENAI_API_KEY="..."
python scripts/distill/run_parallel_thinking_distill.py --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --model <model-name> --input-csv data/distill/holdout_v1/eval_split.csv --limit 3 --branches 3 --output-jsonl data/distill/parallel_thinking_openai_smoke.jsonl
```

Dry-run prompt formatting without calling the API:

```powershell
python scripts/distill/run_parallel_thinking_distill.py --input-csv data/distill/holdout_v1/eval_split.csv --family bit --limit 1 --branches 3 --dry-run
```

## Nemotron 3 Super Batch Workflow

Use this runner for long-running Nemotron 3 Super teacher-data collection. It supports two provider profiles:

- `nvidia`: hosted NVIDIA NIM at `https://integrate.api.nvidia.com/v1`, model `nvidia/nemotron-3-super-120b-a12b`, default `--rpm-limit 36`.
- `openrouter-free`: OpenRouter free model `nvidia/nemotron-3-super-120b-a12b:free`, default `--rpm-limit 18` and `--daily-request-budget 900`.

The hosted API path only needs a small CPU machine. For a 72-hour online run, use about 4 vCPU, 8-16 GB RAM, and 20 GB disk. A GPU is not needed unless you self-host a NIM.

Dry-run the NVIDIA prompt without calling the API:

```powershell
$env:NVIDIA_API_KEY="..."
python scripts/distill/run_nemotron_super_distill.py --provider nvidia --input-csv data/distill/holdout_v1/eval_split.csv --family bit --limit 1 --dry-run
```

Estimate a 3-candidate hard-family run:

```powershell
python scripts/distill/run_nemotron_super_distill.py --provider nvidia --input-csv data/distill/holdout_v1/train_split.csv --family bit --family symbol --samples-per-row 3 --estimate-only
```

Run NVIDIA as the main teacher:

```powershell
python scripts/distill/run_nemotron_super_distill.py --provider nvidia --input-csv data/distill/holdout_v1/train_split.csv --family bit --family symbol --samples-per-row 3 --output-format teacher --json-mode --max-runtime-hours 71.5 --output-jsonl data/distill/nim_super_teacher_bitsymbol_3x.jsonl
```

Run OpenRouter free as a supplemental source:

```powershell
$env:OPENROUTER_API_KEY="..."
python scripts/distill/run_nemotron_super_distill.py --provider openrouter-free --input-csv data/distill/holdout_v1/train_split.csv --family bit --family symbol --samples-per-row 1 --output-format teacher --json-mode --daily-request-budget 900 --max-runtime-hours 71.5 --output-jsonl data/distill/openrouter_nemotron_super_free_bitsymbol_1x.jsonl
```

Generate parallel-thinking records only as a separate experiment:

```powershell
python scripts/distill/run_nemotron_super_distill.py --provider nvidia --input-csv data/distill/holdout_v1/eval_split.csv --family bit --limit 3 --output-format parallel --branches 3 --json-mode --output-jsonl data/distill/nim_super_parallel_bit_smoke.jsonl
```

Use `--num-shards` and `--shard-index` when splitting a long run across machines. The resume key is `(provider, model, output_format, id, candidate_index)`, so successful rows are skipped on restart while error rows are retried.

## RFT / DPO / ORPO Dataset Builder

After collecting scored teacher or rollout JSONL files, build the two offline training datasets:

```powershell
python scripts/distill/build_rft_preference_data.py --input-jsonl data/distill/nim_super_bit_eval10.jsonl --output-rft-jsonl data/distill/rft_messages_v1.jsonl --output-preference-jsonl data/distill/preference_pairs_v1.jsonl --report-json data/distill/rft_preference_report_v1.json
```

Pass multiple `--input-jsonl` arguments to merge rollouts from different models, temperatures, or attempts:

```powershell
python scripts/distill/build_rft_preference_data.py --input-jsonl data/distill/nim_super_bit_eval10.jsonl --input-jsonl data/distill/codex_smoke_gpt54.jsonl --output-rft-jsonl data/distill/rft_merged_v1.jsonl --output-preference-jsonl data/distill/preference_merged_v1.jsonl
```

Outputs:

- RFT JSONL: `messages`, plus `prompt` and `completion`, for SFT/RFT bootstrapping.
- Preference JSONL: conversational `prompt`, `chosen`, and `rejected`, suitable for TRL `DPOTrainer` or `ORPOTrainer`.
- Report JSON: counts for parsed, skipped, positive, negative, synthetic, RFT, and preference records.

Preference pairs require at least one positive and one negative candidate for the same `id`. If you only have rejected candidates but want a first DPO/ORPO smoke test, allow a minimal gold-answer chosen response:

```powershell
python scripts/distill/build_rft_preference_data.py --input-jsonl data/distill/rollouts_v1.jsonl --synthesize-gold-chosen --output-rft-jsonl data/distill/rft_v1.jsonl --output-preference-jsonl data/distill/preference_v1.jsonl
```

Use synthesized chosen responses for plumbing and small smoke tests first; real DPO/ORPO data should preferentially pair actual correct rollouts against actual wrong rollouts.

## Notes

- The script stores both the teacher answer and a short rationale field.
- `verified_answer_exact` is computed automatically when the source CSV contains `answer`.
- The current prompt favors concise, structured reasoning rather than long chain-of-thought.
- For direct LongCat use, the default recommendation is `temperature = 0.2`.
- LongCat `error` rows are now retriable; only `status = ok` rows are skipped by resume.
- LongCat `max_tokens` is optional; if omitted, the script uses the model default from the API.
- `--stream` is recommended for LongCat Thinking smoke tests so you can observe whether the model is still emitting `reasoning_content` before the final answer.
- OpenRouter accepts the OpenAI-style `/chat/completions` schema, and this runner uses that directly.
- For OpenRouter, `--json-mode` is optional because model support can vary by provider; the fallback parser still tries to recover raw JSON from plain text output.
- OpenRouter reasoning is passed through the unified `reasoning` parameter, and the runner can store returned reasoning details in `raw_reasoning`.
