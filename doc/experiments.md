# 实验记录 - NVIDIA Nemotron Model Reasoning Challenge

更新时间: 2026-03-28

## 命名规范

`exp-{编号}_{track}_{short_name}_{yyyymmdd}`

推荐的 `track`:

- `prompt`
- `solver`
- `ft`
- `data`
- `inference`
- `ensemble`

## 当前实验总览

| # | 实验名 | 日期 | 类别 | 状态 | 本地指标 | Kaggle Public | 关键结论 |
|---|---|---|---|---|---|---|---|
| 001 | Prompt-only zero-shot notebook | 2026-03-28 | prompt | Invalidated | N/A | N/A | 旧 notebook 挂错数据源并输出长推理文本，结果不可作为有效 baseline |
| 002 | Rule-based solver v1 | 2026-03-28 | solver | Invalid submission path | train exact match = 0.7581 | N/A | `submission.csv` / 规则预测文件不是这个比赛接受的提交格式 |
| 003 | No-op adapter submission demo | 2026-03-28 | ft | Complete | N/A | 0.50 | 已打通 adapter 提交流程，但 no-op adapter 只有很弱的基线价值 |
| 004 | Prompt-only zero-shot v2 | 2026-03-29 | prompt | Kernel complete | N/A | N/A | 使用 Qwen2.5-7B-Instruct (CPU)，输出长推理文本，答案提取失败 |
| 005 | CSV submission test | 2026-03-29 | solver | Submitted / ERROR | N/A | ERROR | 验证 CSV 格式不被接受，必须提交 adapter zip |

## exp-001: Prompt-only zero-shot notebook

- 日期: 2026-03-28
- 负责人: Codex / 历史会话接续
- 类别: prompt
- 目标: 建立第一个 Kaggle notebook baseline
- 实际执行:
  - 历史 notebook `exp-001-prompt-zeroshot-v2` 实际使用模型是 `Qwen/Qwen2.5-7B-Instruct`
  - `kernel-metadata.json` 挂载的是外部数据集 `sebmontreal/nvidia-nemotron-model-reasoning-challenge`
  - notebook 输出的是长推理文本，不是短答案格式
- 发现的问题:
  - 本地 / notebook 文档声称使用 Nemotron，但日志显示实际运行的是 Qwen 7B
  - notebook 使用了 dataset source，而不是 competition source
  - `submission.csv` 生成成功，但 `kaggle competitions submit` 返回 HTTP 400
  - 该 competition 的实际工作流更像 code competition，需要通过 notebook version 提交
- 结论:
  - 该结果不应记录为有效的 `prompt-only` baseline
  - `exp-001` 需要在正确的 competition source 和答案抽取逻辑下重新做
- 下一步:
  - 重新准备一个真正的 `prompt-only` notebook
  - 显式记录为什么选用某个模型以及对应资源约束

## exp-002: Rule-based solver v1

- 日期: 2026-03-28
- 负责人: Codex
- 类别: solver
- 目标:
  - 在不依赖大模型推理的前提下，先验证题目是否可由规则归纳直接求解
  - 建立一个可在 code competition notebook 中稳定运行的 baseline
- 方法:
  - 按 prompt 首句将任务分为 6 类:
    - `bit`
    - `cipher`
    - `gravity`
    - `roman`
    - `symbol`
    - `unit`
  - `cipher`: 用题内样例建立单表替换约束，再结合训练集词表做回溯补全
  - `roman`: 直接做阿拉伯数字到罗马数字转换
  - `gravity`: 根据 `d = 0.5 * g * t^2` 和样例舍入区间反推 `g`
  - `unit`: 根据样例舍入区间反推比例系数
  - `bit`: 在有限 DSL 上穷举位运算表达式
  - `symbol`: 暂未实现
- 关键配置:

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

- 本地评测:

| 子任务 | total | solved | correct | accuracy |
|---|---:|---:|---:|---:|
| bit | 1602 | 1314 | 1297 | 0.8096 |
| cipher | 1576 | 1576 | 1576 | 1.0000 |
| gravity | 1597 | 1597 | 1307 | 0.8184 |
| roman | 1576 | 1576 | 1576 | 1.0000 |
| symbol | 1555 | 0 | 0 | 0.0000 |
| unit | 1594 | 1594 | 1446 | 0.9072 |
| overall | 9500 | - | 7202 | 0.7581 |

- Kaggle 相关:
  - 本地生成的 `data/rule_solver_submission_public.csv` 为:

```csv
id,prediction
00066667,10010111
000b53cf,01000011
00189f6a,cat imagines book
```

  - 当前公开 `test.csv` 的 3 条样本与 `train.csv` 完全重合:
    - `test_exact_id_overlap = 3`
    - `test_exact_prompt_overlap = 3`
  - 已推送 Kaggle kernel:
    - `zhendongli923/exp-002-rule-solver-v1`
    - 当前版本: `v4`
  - code competition 提交:
    - `v3` submission ref: `51305266`
    - `v3` 状态: `ERROR`
    - `v4` submission ref: `51305823`
    - `v4` 状态: `ERROR`
    - output file: `submission.zip`
  - 当前状态:
    - kernel v4 已完成
    - 两次提交都没有产生 public / private score
    - Kaggle 评测器明确报错缺少 `adapter_config.json`
- 结论:
  - 这不是“真正泛化”的最终方案，但它说明题目中至少 5 类可以用结构化规则逼近
  - 相比先前的 prompt-only 失败版本，rule solver 更适合作为可复现 baseline
  - 但它不能直接作为当前比赛的有效提交物，因为比赛要求 adapter zip 而不是预测文件
- 下一步:
  - 实现 `symbol` 类 solver
  - 提升 `gravity` / `unit` 的边界取值与舍入策略
  - 把 `bit` DSL 从一层组合扩展到两层组合
  - 重新做一个干净的 `prompt-only` notebook，与 solver baseline 对照

## 后续实验队列

### exp-003: Symbol solver

- 目标: 攻克当前唯一完全未覆盖的大类 `symbol`
- 方向:
  - 先区分数字式和符号式子类
  - 尝试最短编辑规则 / 子串映射 / 栈式重写

## exp-003: No-op adapter submission demo

- 日期: 2026-03-28
- 负责人: Codex
- 类别: ft
- 目标:
  - 先打通“adapter zip”提交通道
  - 验证比赛真正需要的 submission contract
- 依据:
  - `51305266` / `51305823` 的错误信息都明确要求 `adapter_config.json`
  - 高票公共 notebook `ryanholbrook/nvidia-nemotron-submission-demo` 给出了最小提交范式
- 当前实现:
  - notebook: `zhendongli923/exp-003-noop-adapter-submission`
  - 当前版本: `v4`
  - 资源:
    - competition source: `nvidia-nemotron-model-reasoning-challenge`
    - kernel source: `ryanholbrook/nvidia-utility-script`
    - model source: `metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1`
    - machine shape: `NvidiaRtxPro6000`
- 当前进度:
  - `v1`: `cutlass` 路径错误
  - `v2`: 解决了 utility path，但缺少 `offload_folder`
  - `v3`: 已补 `offload_folder`，但在 `P100` 上触发大规模磁盘 offload，最终报 `No space left on device`
  - `v4`: 已去掉 `offload_folder`，并显式切到 `NvidiaRtxPro6000` + demo 同款 docker image
  - `v4` 运行结果:
    - notebook 状态: `COMPLETE`
    - 输出文件:
      - `adapter/README.md`
      - `adapter/adapter_config.json`
      - `adapter/adapter_model.safetensors`
      - `submission.zip`
    - zip 内容: `README.md`, `adapter_config.json`, `adapter_model.safetensors`
    - code competition 提交 ref: `51310145`
    - 提交状态: `COMPLETE`
    - public score: `0.50`
- 结论:
  - 这条线的优先级高于继续提交 `submission.csv`
  - 结构正确的 adapter submission 已经打通
  - 但 `no-op` adapter 本身只有很弱的效果，不能作为有竞争力的模型方案
  - 后续 LoRA / SFT baseline 可以直接复用这条提交通道

### exp-004: Prompt-only zero-shot reboot

- 目标: 在正确的 competition source 上重做 prompt baseline
- 方向:
  - 严格短答案输出
  - 明确模型选择理由
  - notebook 输出必须可直接用于 code competition submission

### exp-005: Prompt + solver hybrid

- 目标: 用 solver 兜底结构题，用模型处理 solver 未覆盖样本
- 方向:
  - family classifier -> solver / model router
  - 对 `symbol` 题优先尝试模型

## 2026-03-29 addendum: exp-007 LoRA training baseline

- date: 2026-03-29
- owner: Codex
- category: ft
- goal:
  - move from the exp-003 no-op adapter demo to the first trained LoRA baseline
  - train on competition `train.csv` and emit a valid `submission.zip`
- v2 failure:
  - Kaggle log shows `AutoTokenizer.from_pretrained()` treated `/kaggle/input/models/.../default/1` as a Hugging Face repo id
  - root cause was incorrect resolution of the local model source mount; the script did not locate the directory that actually contains `config.json`
- v3 fixes:
  - explicitly search under `kagglehub.model_download()` for the real model directory containing `config.json`
  - add `local_files_only=True` to tokenizer and model loading
  - run `patch_rmsnorm()` again after model import/load so the dynamically loaded Nemotron module is patched too
  - use `next(model.parameters()).device` for batch placement
- current status:
  - kernel: `zhendongli923/exp-007-lora-training-v2`
  - latest version: `v3`
  - latest status: `RUNNING`

## 2026-03-29 addendum: exp-008 utility mount smoke

- date: 2026-03-29
- owner: Codex
- category: infra / ft
- purpose:
  - verify whether `kernel_sources = ["ryanholbrook/nvidia-utility-script"]` only works in notebook mode
- result:
  - kernel: `zhendongli923/exp-008-utility-mount-smoke`
  - version: `v1`
  - status: `COMPLETE`
  - the utility path `/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages` exists in notebook mode
  - `import mamba_ssm` succeeded
  - mounted torch version: `2.12.0.dev20260324+cu128`
  - `kagglehub.model_download()` resolved the Nemotron model root correctly
- conclusion:
  - the `kernel_sources` dependency chain is viable in notebook mode
  - the repeated `exp-007` failures are specific to the script-kernel route, not to Nemotron itself

## 2026-03-29 addendum: exp-009 notebook training baseline

- date: 2026-03-29
- owner: Codex
- category: ft
- purpose:
  - promote the current LoRA training baseline onto the working notebook-based Kaggle route
- current status:
  - kernel: `zhendongli923/exp-009-lora-training-notebook`
  - version: `v1`
  - status at last check: `RUNNING`

## 2026-03-29 addendum: exp-009 training run result

- date: 2026-03-29
- owner: Codex
- category: ft
- result:
  - kernel: `zhendongli923/exp-009-lora-training-notebook`
  - version: `v1`
  - status: `COMPLETE`
  - output files: `adapter/README.md`, `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`, `submission.zip`, `train_report.json`
  - training report: `train_rows = 600`, `epochs = 1`, `steps = 150`, `mean_loss = 1.8399`, `final_loss = 2.0331`
  - runtime log shows the full path succeeded: utility mount -> Nemotron load -> LoRA training -> zip packaging
- submission:
  - ref: `51315285`
  - submitted at local check time around `2026-03-29 00:42 EDT`
  - final status: `COMPLETE`
  - public score: `0.52`
  - delta vs no-op baseline: `+0.02` over `exp-003` / `exp-006`

## 2026-03-29 addendum: exp-010 notebook training baseline

- date: 2026-03-29
- owner: Codex
- category: ft
- purpose:
  - push the successful notebook-based LoRA route harder after `exp-009` only reached `0.52`
- config delta vs `exp-009`:
  - `train_rows`: `600 -> 3000`
  - `epochs`: `1 -> 2`
  - `lora_rank`: `8 -> 16`
  - `lora_alpha`: `16 -> 32`
  - `learning_rate`: `2e-4 -> 1.5e-4`
- expected runtime:
  - estimated from `exp-009` throughput at roughly `8-9` hours on `NvidiaRtxPro6000`
- current status:
  - kernel: `zhendongli923/exp-010-lora-training-notebook`
  - version: `v1`
  - kernel status: `COMPLETE`
  - output files: `adapter/README.md`, `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`, `submission.zip`, `train_report.json`
  - training report: `train_rows = 3000`, `epochs = 2`, `steps = 1500`, `mean_loss = 1.1577`
  - epoch behavior: epoch 1 kept descending, epoch 2 dropped early and then plateaued around `~1.00`
  - submission ref: `51334036`
  - submission status at local check time around `2026-03-29 14:44 EDT`: `PENDING`

## 2026-03-29 addendum: exp-011 notebook training baseline

- date: 2026-03-29
- owner: Codex
- category: ft
- purpose:
  - keep roughly the same total training exposure as `exp-010`, but shift budget from repeated epochs to broader and harder-example coverage
- config delta vs `exp-010`:
  - `train_rows`: `3000 -> 6000`
  - `epochs`: `2 -> 1`
  - `lora_rank`: kept at `16`
  - `learning_rate`: kept at `1.5e-4`
- family sampling policy:
  - `symbol = 1500`
  - `bit = 1500`
  - `gravity = 1100`
  - `unit = 1100`
  - `cipher = 400`
  - `roman = 400`
- rationale:
  - `exp-010` showed clear second-epoch plateau, so the next budget should buy diversity rather than repetition
- current status:
  - kernel: `zhendongli923/exp-011-lora-training-notebook`
  - version: `v1`
  - status at launch check: `RUNNING`
