# Local Distillation Runners

This directory contains the first local teacher-data pipeline for `exp-012`.

## Files

- `build_holdout.py`: create a family-stratified local holdout split from `train.csv`
- `run_codex_distill.py`: call local `codex exec` to generate structured teacher supervision
- `run_longcat_distill.py`: call LongCat directly through its OpenAI-compatible API
- `run_openrouter_distill.py`: call OpenRouter directly through its OpenAI-compatible API
- `teacher_response.schema.json`: strict JSON schema for Codex teacher output

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
