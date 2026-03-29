# NVIDIA Nemotron Model Reasoning Challenge 项目

本项目用于参加 Kaggle `NVIDIA Nemotron Model Reasoning Challenge`。

该挑战聚焦于使用 Nemotron 开源模型、数据集和训练 recipe，探索提升推理准确率的方法；评测将基于 NVIDIA Research 提供的新 reasoning benchmark。

## 目录结构

```text
Nemotron/
├── CLAUDE.md              # Claude agent 项目级指令
├── CODEX.md               # Codex agent 项目级指令
├── TODO.md                # 项目待办事项
├── .agents/               # Agent 角色定义与提示词
├── _agents/workflows/     # 工作流定义
├── doc/                   # 项目文档
├── meeting/               # 会议记录与归档
├── scripts/               # 运行脚本
└── src/                   # 源代码（按需要补充）
```

## 当前工作重点

1. 对齐 Kaggle 官方规则、数据和评测定义
2. 建立最小可复现 baseline
3. 把实验、提交和决策沉淀到本地文档

## 快速开始
连接kaggle的cli上传,所有实验全部在kaggle上运行,禁止在本地运行,本地只做代码管理和文档记录.
每天只能5次提交,所以要珍惜提交机会.

## 技术方向

- Prompt engineering
- Few-shot / CoT / self-consistency
- Data filtering / synthetic data generation
- Lightweight fine-tuning
- RL 或偏好优化
- Verifier / reranker / judge

## 协作说明

- 项目沟通与文档使用中文
- 未确认的官方细节统一标记为 `TBD`
- 任何旧项目遗留信息都不能视为当前项目事实
