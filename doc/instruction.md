# NVIDIA Nemotron Model Reasoning Challenge

## 当前已确认的信息

更新时间: 2026-03-28

### 官方定位

- 比赛托管平台: Kaggle
- 公开主办方说明: NVIDIA
- 目标: 开发能够提升 Nemotron 模型推理准确率的方法
- 评测基础: NVIDIA Research 新推理基准
- 官方鼓励的方向:
  - Prompting
  - Data filtering
  - Synthetic data generation
  - Reinforcement learning
  - Lightweight fine-tuning

### 官方资源入口

- Kaggle 比赛页: [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
- NVIDIA 活动说明: [NVIDIA Nemotron Model Reasoning Challenge · Luma](https://luma.com/5ugsphtp)
- NVIDIA Nemotron 官方仓库: [github.com/NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron)
- Nemotron 模型下载: [huggingface.co/nvidia](https://huggingface.co/nvidia)

### 可用 Nemotron 模型 (2026-03-28 确认)

| 模型 | 参数量 | 架构 | 用途 | 链接 |
|------|--------|------|------|------|
| NVIDIA-Nemotron-Nano-12B-v2 | 12B | Mamba-Transformer Hybrid | 推理/对话统一模型 | [HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2) |
| Nemotron 3 Nano | 31.6B 总 / 3.6B 活跃 | MoE Hybrid Mamba-Transformer | 边缘/PC 部署 | [HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) |
| Nemotron 3 Super | 120B 总 / 12B 活跃 | Hybrid Mamba Latent MoE | 多智能体推理 | [HuggingFace](https://huggingface.co/nvidia/Nemotron-3-Super) |

### 推理控制特性

- **Reasoning Budget**: 支持控制 thinking token 数量
- **系统提示**: `/think` 启用推理, `/no_think` 禁用推理
- **推荐参数**: temperature=0.6, top_p=0.95 (推理模式); greedy (非推理模式)
- **上下文长度**: 最长 128K tokens

### 推荐推理代码

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

messages = [
    {"role": "system", "content": "/think"},  # 或 "/no_think"
    {"role": "user", "content": "Your question here"}
]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(input_ids, max_new_tokens=1024, temperature=0.6, top_p=0.95)
response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
```

## 本仓库对赛题的工作定义

我们把当前项目拆成 5 条主线：

1. 规则对齐
   - 核对官方允许使用的数据、模型、外部资源与提交方式
2. 数据理解
   - 下载比赛文件，建立文件清单、字段字典和基本统计
3. baseline 搭建
   - 至少做出 prompt-only 和一个训练型 baseline
4. 提升实验
   - 在 prompting、数据构造、后训练、推理时扩展之间做系统比较
5. 提交闭环
   - 生成合法提交文件，记录 public/private leaderboard 表现

## 当前仍待补充的信息

以下内容必须从 Kaggle 页面同步，当前仓库不做主观猜测：

- 数据文件名与 split 定义
- 官方 evaluation metric
- 提交文件格式
- 外部数据/外部模型/联网限制
- 每日提交次数限制
- 重要时间线
- 奖励或评审机制

## 基线建议

### Baseline 0: 官方页面复刻

- 先把官方数据、评测、提交格式完整镜像到本地文档
- 目标是先消灭信息不对称，而不是立刻追求模型复杂度

### Baseline 1: Prompt-only

- Zero-shot
- Few-shot
- CoT / structured reasoning
- Self-consistency 或多样本 rerank

### Baseline 2: Lightweight fine-tuning

- LoRA / QLoRA / adapter
- 只在规则允许的前提下引入外部训练数据
- 重点关注训练成本、可复现性和泛化

### Baseline 3: 数据与后训练

- Data filtering / curation
- Synthetic reasoning data
- Reward / preference / RL 路线
- Verifier / judge / reranker

## 当前不再适用的旧假设

以下假设来自迁移前项目，已经全部作废：

- 推荐系统 / CTR / CVR 任务
- 广告、用户、商品、序列特征 schema
- GAUC / AUC / LogLoss 作为核心指标
- AngelML 或特定工业推荐平台作为默认运行环境
- OneTrans / HyFormer / InterFormer 这类推荐系统统一建模方案作为主线

## 文档维护要求

- 本文件只记录“已确认信息”和“待补充项”
- 如果信息来自外部公开页面，请注明来源和日期
- 如果还未核对到官方原文，统一写 `TBD`
