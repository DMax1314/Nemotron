# 调研报告归档：迁移前统一建模方向

本文件来自迁移前的推荐系统调研报告，主题是“统一序列建模与特征交互”，与当前 Kaggle `NVIDIA Nemotron Model Reasoning Challenge` 无关，现已归档。

## 不再适用的内容

- 推荐系统统一建模主线
- OneTrans / HyFormer / InterFormer 作为当前方向
- CTR / CVR / gAUC 等实验结论
- 旧数据 schema 与工业推荐平台假设

## 当前项目的替代方向

当前项目应优先关注：

- Nemotron 官方模型与训练 recipe
- Prompting 与 test-time scaling
- Lightweight fine-tuning
- Data filtering / synthetic data generation
- Verifier / reranker / judge
- 本地评测与 Kaggle 提交闭环

需要新的调研时，请新建 Nemotron 主题报告，不要继续扩展本文件。
