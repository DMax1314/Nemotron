# Experiment Records - NVIDIA Nemotron Model Reasoning Challenge

Updated: 2026-03-28

## Naming Convention

`exp-{number}_{track}_{short_name}_{yyyymmdd}`

Recommended `track` names:

- `prompt`
- `solver`
- `ft`
- `data`
- `inference`
- `ensemble`

## Experiment Overview

| # | Experiment Name | Date | Category | Status | Local Metric | Kaggle Public | Key Conclusion |
|---|---|---|---|---|---|---|---|
| 001 | Prompt-only zero-shot notebook | 2026-03-28 | prompt | Invalidated | N/A | N/A | Old notebook used wrong data source and output long reasoning text; results are not a valid baseline. |
| 002 | Rule-based solver v1 | 2026-03-28 | solver | Invalid submission path | train exact match = 0.7581 | N/A | `submission.csv` / rule prediction file is not an accepted submission format for this competition. |
| 003 | No-op adapter submission demo | 2026-03-28 | ft | Complete | N/A | 0.50 | Successfully tested the adapter submission flow, but no-op adapter only serves as a very weak baseline. |
| 004 | Prompt-only zero-shot v2 | 2026-03-29 | prompt | Kernel complete | N/A | N/A | Used Qwen2.5-7B-Instruct (CPU), output long reasoning text, answer extraction failed. |
| 005 | CSV submission test | 2026-03-29 | solver | Submitted / ERROR | N/A | ERROR | Verified that CSV format is not accepted; must submit adapter zip. |

## exp-001: Prompt-only zero-shot notebook

- Date: 2026-03-28
- Owner: Codex / History session continuation
- Category: prompt
- Goal: Establish the first Kaggle notebook baseline.
- Execution:
  - Historical notebook `exp-001-prompt-zeroshot-v2` actually used the `Qwen/Qwen2.5-7B-Instruct` model.
  - `kernel-metadata.json` mounted an external dataset `sebmontreal/nvidia-nemotron-model-reasoning-challenge`.
  - Notebook outputted long reasoning text instead of short answer format.
- Issues Found:
  - Local/notebook documentation claimed Nemotron usage, but logs showed Qwen 7B actually running.
  - Notebook used a dataset source instead of the competition source.
  - `submission.csv` was generated successfully, but `kaggle competitions submit` returned HTTP 400.
  - This competition follows a code competition workflow requiring submission via notebook version.
- Conclusion:
  - The result should not be recorded as a valid `prompt-only` baseline.
  - `exp-001` needs to be redone with the correct competition source and answer extraction logic.
- Next Steps:
  - Prepare a true `prompt-only` notebook.
  - Explicitly document model choice and resource constraints.

## exp-002: Rule-based solver v1

- Date: 2026-03-28
- Owner: Codex
- Category: solver
- Goal:
  - Verify if problems can be solved directly via rule induction without relying on LLM reasoning.
  - Establish a stable baseline that can run in a code competition notebook.
- Method:
  - Categorized tasks into 6 types based on the first sentence of the prompt:
    - `bit`
    - `cipher`
    - `gravity`
    - `roman`
    - `symbol`
    - `unit`
  - `cipher`: Used in-question examples to build single-substitution constraints, then backtrack-completed using training set vocabulary.
  - `roman`: Direct conversion from Arabic to Roman numerals.
  - `gravity`: Derived `g` from `d = 0.5 * g * t^2` and rounding ranges from examples.
  - `unit`: Derived scale factors from example rounding ranges.
  - `bit`: Exhaustive search for bit-wise expressions on a limited DSL.
  - `symbol`: Not yet implemented.
- Key Configuration:

```yaml
train_path: data/raw/train.csv
test_path: data/raw/test.csv
lookup_on_test: true
bit_expression_space:
  unary: [id, not, rol1-7, ror1-7, shl1-7, shr1-7]
  binary: [xor, and, or]
  ternary: [maj, ch]
  allow_outer_not: true
```

- Local Evaluation:

| Subtask | total | solved | correct | accuracy |
|---|---:|---:|---:|---:|
| bit | 1602 | 1314 | 1297 | 0.8096 |
| cipher | 1576 | 1576 | 1576 | 1.0000 |
| gravity | 1597 | 1597 | 1307 | 0.8184 |
| roman | 1576 | 1576 | 1576 | 1.0000 |
| symbol | 1555 | 0 | 0 | 0.0000 |
| unit | 1594 | 1594 | 1446 | 0.9072 |
| overall | 9500 | - | 7202 | 0.7581 |

- Kaggle Related:
  - Locally generated `data/rule_solver_submission_public.csv`:

```csv
id,prediction
00066667,10010111
000b53cf,01000011
00189f6a,cat imagines book
```

  - Current public `test.csv` samples overlap completely with `train.csv`:
    - `test_exact_id_overlap = 3`
    - `test_exact_prompt_overlap = 3`
  - Pushed Kaggle kernel:
    - `zhendongli923/exp-002-rule-solver-v1`
    - Current version: `v4`
  - Code competition submissions:
    - `v3` submission ref: `51305266`
    - `v3` status: `ERROR`
    - `v4` submission ref: `51305823`
    - `v4` status: `ERROR`
    - output file: `submission.zip`
  - Current status:
    - Kernel v4 completed.
    - Neither submission yielded a public/private score.
    - Kaggle evaluator explicitly reported missing `adapter_config.json`.
- Conclusion:
  - This is not a "truly generalizable" final solution, but it demonstrates that at least 5 classes can be approximated with structured rules.
  - Compared to previous failed prompt-only versions, the rule solver is more suitable as a reproducible baseline.
  - However, it cannot be directly used as a valid submission because the competition requires an adapter zip, not a prediction file.
- Next Steps:
  - Implement a solver for the `symbol` class.
  - Improve boundary values and rounding strategies for `gravity` / `unit`.
  - Extend the `bit` DSL from one level of composition to two.
  - Re-run a clean `prompt-only` notebook to compare with the solver baseline.

## Future Experiment Queue

### exp-003: Symbol solver

- Goal: Tackle the `symbol` class, the only major category currently not covered.
- Direction:
  - Distinguish between numeric and symbolic sub-classes.
  - Try shortest edit rules / substring mapping / stack-based rewriting.

## exp-003: No-op adapter submission demo

- Date: 2026-03-28
- Owner: Codex
- Category: ft
- Goal:
  - Test the "adapter zip" submission channel.
  - Verify the actual submission contract required by the competition.
- Rationale:
  - Error messages from `51305266` / `51305823` both explicitly required `adapter_config.json`.
  - The popular public notebook `ryanholbrook/nvidia-nemotron-submission-demo` provides a minimal submission paradigm.
- Current Implementation:
  - Notebook: `zhendongli923/exp-003-noop-adapter-submission`
  - Current version: `v4`
  - Resources:
    - competition source: `nvidia-nemotron-model-reasoning-challenge`
    - kernel source: `ryanholbrook/nvidia-utility-script`
    - model source: `metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1`
    - machine shape: `NvidiaRtxPro6000`
- Progress:
  - `v1`: `cutlass` path error.
  - `v2`: Fixed utility path, but missing `offload_folder`.
  - `v3`: Added `offload_folder`, but triggered massive disk offloading on `P100`, resulting in `No space left on device`.
  - `v4`: Removed `offload_folder` and explicitly switched to `NvidiaRtxPro6000` + the docker image used in the demo.
  - `v4` results:
    - Notebook status: `COMPLETE`
    - Output files:
      - `adapter/README.md`
      - `adapter/adapter_config.json`
      - `adapter/adapter_model.safetensors`
      - `submission.zip`
    - Zip contents: `README.md`, `adapter_config.json`, `adapter_model.safetensors`
    - Code competition submission ref: `51310145`
    - Submission status: `COMPLETE`
    - Public score: `0.50`
- Conclusion:
  - This path has higher priority than further `submission.csv` submissions.
  - The correct adapter submission structure is now established.
  - However, the `no-op` adapter itself has very weak results and cannot serve as a competitive model solution.
  - Future LoRA / SFT baselines can reuse this submission channel directly.

### exp-004: Prompt-only zero-shot reboot

- Goal: Redo the prompt baseline on the correct competition source.
- Direction:
  - Strict short answer output.
  - Clear rationale for model selection.
  - Notebook output must be directly usable for code competition submission.

### exp-005: Prompt + solver hybrid

- Goal: Use the solver to handle structured problems and the model for samples not covered by the solver.
- Direction:
  - family classifier -> solver / model router.
  - Prioritize models for `symbol` problems.

## 2026-03-29 addendum: exp-007 LoRA training baseline

- Date: 2026-03-29
- Owner: Codex
- Category: ft
- Goal:
  - Move from the exp-003 no-op adapter demo to the first trained LoRA baseline.
  - Train on competition `train.csv` and emit a valid `submission.zip`.
- v2 failure:
  - Kaggle log shows `AutoTokenizer.from_pretrained()` treated `/kaggle/input/models/.../default/1` as a Hugging Face repo id.
  - Root cause was incorrect resolution of the local model source mount; the script did not locate the directory that actually contains `config.json`.
- v3 fixes:
  - Explicitly search under `kagglehub.model_download()` for the real model directory containing `config.json`.
  - Add `local_files_only=True` to tokenizer and model loading.
  - Run `patch_rmsnorm()` again after model import/load so the dynamically loaded Nemotron module is patched too.
  - Use `next(model.parameters()).device` for batch placement.
- Current status:
  - Kernel: `zhendongli923/exp-007-lora-training-v2`
  - Latest version: `v3`
  - Latest status: `RUNNING`

## 2026-03-29 addendum: exp-008 utility mount smoke

- Date: 2026-03-29
- Owner: Codex
- Category: infra / ft
- Purpose:
  - Verify whether `kernel_sources = ["ryanholbrook/nvidia-utility-script"]` only works in notebook mode.
- Result:
  - Kernel: `zhendongli923/exp-008-utility-mount-smoke`
  - Version: `v1`
  - Status: `COMPLETE`
  - The utility path `/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages` exists in notebook mode.
  - `import mamba_ssm` succeeded.
  - Mounted torch version: `2.12.0.dev20260324+cu128`.
  - `kagglehub.model_download()` resolved the Nemotron model root correctly.
- Conclusion:
  - The `kernel_sources` dependency chain is viable in notebook mode.
  - The repeated `exp-007` failures are specific to the script-kernel route, not to Nemotron itself.

## 2026-03-29 addendum: exp-009 notebook training baseline

- Date: 2026-03-29
- Owner: Codex
- Category: ft
- Purpose:
  - Promote the current LoRA training baseline onto the working notebook-based Kaggle route.
- Current status:
  - Kernel: `zhendongli923/exp-009-lora-training-notebook`
  - Version: `v1`
  - Status at last check: `RUNNING`

## 2026-03-29 addendum: exp-009 training run result

- Date: 2026-03-29
- Owner: Codex
- Category: ft
- Result:
  - Kernel: `zhendongli923/exp-009-lora-training-notebook`
  - Version: `v1`
  - Status: `COMPLETE`
  - Output files: `adapter/README.md`, `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`, `submission.zip`, `train_report.json`
  - Training report: `train_rows = 600`, `epochs = 1`, `steps = 150`, `mean_loss = 1.8399`, `final_loss = 2.0331`
  - Runtime log shows the full path succeeded: utility mount -> Nemotron load -> LoRA training -> zip packaging.
- Submission:
  - ref: `51315285`
  - Submitted at local check time around `2026-03-29 00:42 EDT`.
  - Final status: `COMPLETE`
  - Public score: `0.52`
  - Delta vs no-op baseline: `+0.02` over `exp-003` / `exp-006`.

## 2026-03-29 addendum: exp-010 notebook training baseline

- Date: 2026-03-29
- Owner: Codex
- Category: ft
- Purpose:
  - Push the successful notebook-based LoRA route harder after `exp-009` only reached `0.52`.
- Config delta vs `exp-009`:
  - `train_rows`: `600 -> 3000`
  - `epochs`: `1 -> 2`
  - `lora_rank`: `8 -> 16`
  - `lora_alpha`: `16 -> 32`
  - `learning_rate`: `2e-4 -> 1.5e-4`
- Expected runtime:
  - Estimated from `exp-009` throughput at roughly `8-9` hours on `NvidiaRtxPro6000`.
- Current status:
  - Kernel: `zhendongli923/exp-010-lora-training-notebook`
  - Version: `v1`
  - Kernel status: `COMPLETE`
  - Output files: `adapter/README.md`, `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`, `submission.zip`, `train_report.json`
  - Training report: `train_rows = 3000`, `epochs = 2`, `steps = 1500`, `mean_loss = 1.1577`
  - Epoch behavior: epoch 1 kept descending, epoch 2 dropped early and then plateaued around `~1.00`.
  - Submission ref: `51334036`
  - Submission status at local check time around `2026-03-29 14:44 EDT`: `PENDING`.

## 2026-03-29 addendum: exp-011 notebook training baseline

- Date: 2026-03-29
- Owner: Codex
- Category: ft
- Purpose:
  - Keep roughly the same total training exposure as `exp-010`, but shift budget from repeated epochs to broader and harder-example coverage.
- Config delta vs `exp-010`:
  - `train_rows`: `3000 -> 6000`
  - `epochs`: `2 -> 1`
  - `lora_rank`: kept at `16`
  - `learning_rate`: kept at `1.5e-4`
- Family sampling policy:
  - `symbol = 1500`
  - `bit = 1500`
  - `gravity = 1100`
  - `unit = 1100`
  - `cipher = 400`
  - `roman = 400`
- Rationale:
  - `exp-010` showed clear second-epoch plateau, so the next budget should buy diversity rather than repetition.
- Current status:
  - Kernel: `zhendongli923/exp-011-lora-training-notebook`
  - Version: `v1`
  - Status at launch check: `RUNNING`
- Completion snapshot:
  - Kernel status: `COMPLETE`
  - Output files: `submission.zip`, `adapter_config.json`, `adapter_model.safetensors`, `train_report.json`
  - Training report: `train_rows = 6000`, `epochs = 1`, `steps = 1500`, `mean_loss = 1.2386`
  - Sampled family counts: `bit = 1500`, `symbol = 1500`, `gravity = 1100`, `unit = 1100`, `cipher = 400`, `roman = 400`
  - Submission ref: `51345538`
  - Submission status at local check time around `2026-03-30 03:28 EDT`: `PENDING`.
  - Final public score at local check time around `2026-03-30 04:08 EDT`: `0.54`.

## 2026-03-30 addendum: exp-012 notebook training control

- Date: 2026-03-30
- Owner: Codex
- Category: ft
- Purpose:
  - Run a tighter control experiment after `exp-011 = 0.54`.
  - Isolate whether `all-linear` LoRA targeting helps more than the larger-sample family-balanced recipe.
- Config:
  - `train_rows = 1200`
  - `epochs = 2`
  - `batch_size = 1`
  - `grad_accum = 4`
  - `learning_rate = 1e-4`
  - `lora_rank = 16`
  - `target_modules = all-linear`
  - `packing = false`
  - Sampling mode: plain random subsample.
- Current status:
  - Kernel: `zhendongli923/exp-012-lora-training-notebook`
  - Version: `v1`
  - Kernel status: `COMPLETE`
  - Pushed successfully to Kaggle on `2026-03-30`.
- Completion snapshot:
  - Output files: `submission.zip`, `adapter/README.md`, `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`, `train_report.json`
  - Runtime log: `Loaded train rows = 9500`, `Sampled train rows = 1200`, `Prepared 1200 training examples`
  - LoRA footprint: `trainable params = 441,936,896 / 32,019,874,240` (`1.3802%`)
  - Epoch 1 average loss: `1.5830`
  - Epoch 2 average loss: `1.0631`
  - Submission ref: `51358175`
  - Submission status at local check time around `2026-03-30`: `PENDING`.
