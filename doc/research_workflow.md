# 研究工作流 — NVIDIA Nemotron Model Reasoning Challenge

## 总体流程

```text
官方规则对齐 -> 数据理解 -> 快速 baseline -> 训练型 baseline -> 提升实验 -> 提交 -> 复盘
       ^                                                                  |
       └--------------------------- 迭代反馈 ------------------------------┘
```

## Phase 0: 官方信息对齐

- [ ] 通读 Kaggle `Overview / Data / Evaluation / Rules`
- [ ] 把已确认信息同步到 `doc/instruction.md`
- [ ] 建立 `doc/data_dictionary.md`
- [ ] 确认合法资源边界、提交次数限制和时间线

## Phase 1: 数据与 benchmark 理解

- [ ] 下载并清点所有官方文件
- [ ] 建立样本 schema 与提交 schema
- [ ] 分析输入长度、标签空间、任务类型和难度分布
- [ ] 确认可复现的本地验证方案
- [ ] 输出第一版误差分析模板

## Phase 2: 快速 baseline

- [ ] Zero-shot
- [ ] Few-shot
- [ ] Chain-of-thought / structured reasoning
- [ ] Self-consistency / reranking
- [ ] 记录成本、延迟和分数

## Phase 3: 训练与后训练

- [ ] Lightweight fine-tuning（LoRA / QLoRA / adapter）
- [ ] Data filtering / curation
- [ ] Synthetic data generation
- [ ] RL / preference optimization 可行性评估

## Phase 4: 提升与提交

- [ ] Verifier / judge / reranker
- [ ] 推理时扩展或集成
- [ ] 生成合法提交文件
- [ ] 提交 Kaggle 并记录 public score
- [ ] 分析线上/线下差异

## 调研重点

### 官方资源

- Nemotron 官方模型与技术博客
- Kaggle 比赛页
- Hugging Face 上的 Nemotron 模型卡与数据集卡

### 技术主题

- 推理增强 prompt 设计
- Test-time scaling
- Self-consistency / best-of-N
- Lightweight fine-tuning
- Synthetic reasoning data
- Verifier / judge / reranker
- Reward modeling / RL
- Benchmark contamination 与评测鲁棒性

## Agent 分工

| Agent | 职责 |
|-------|------|
| **Product Manager** | 对齐规则、制定里程碑、做优先级决策 |
| **Scholar** | 调研 Nemotron 资源、推理增强方法与相关论文 |
| **Data Scientist** | 数据清点、EDA、误差分析、分桶评测 |
| **MLE** | baseline、微调、推理策略与评测 |
| **SWE** | 数据/训练/评测/提交 pipeline |
| **Code Reviewer** | 正确性、复现性、规则风险审查 |
| **Intern** | 下载、跑脚本、收集结果、维护文档 |

## 每日检查清单

- [ ] `doc/experiments.md` 已更新
- [ ] `TODO.md` 已更新
- [ ] 是否发现新的官方规则信息
- [ ] 是否记录了失败样例和误差模式
- [ ] 下一步实验是否有明确目标与停止条件
