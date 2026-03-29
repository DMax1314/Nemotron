# 数据字典 - NVIDIA Nemotron Model Reasoning Challenge

更新时间: 2026-03-28

## 文件概览

| 文件 | 行数 | 大小 | 说明 |
|---|---:|---:|---|
| `train.csv` | 9500 | 3.07 MB | 训练集，包含答案列 |
| `test.csv` | 3 | 1.46 KB | 当前公开测试集，不包含答案列 |

## Schema

### `train.csv`

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `id` | string | 样本唯一标识 | `00066667` |
| `prompt` | string | 题目全文 | `In Alice's Wonderland, ...` |
| `answer` | string | 标准答案 | `10010111` |

### `test.csv`

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `id` | string | 样本唯一标识 | `00066667` |
| `prompt` | string | 题目全文 | `In Alice's Wonderland, ...` |

## 已验证事实

- `train.csv` 的标签列名是 `answer`，不是 `prediction`
- 当前公开 `test.csv` 只有 3 行
- 当前公开 `test.csv` 的 3 个 `id` 全部存在于 `train.csv`
- 当前公开 `test.csv` 的 3 条 `prompt` 与 `train.csv` 中对应样本完全一致

## 任务家族分布

按 prompt 首句粗分，训练集共有 6 个主要家族:

| family | count |
|---|---:|
| `bit` | 1602 |
| `gravity` | 1597 |
| `unit` | 1594 |
| `cipher` | 1576 |
| `roman` | 1576 |
| `symbol` | 1555 |

## 答案格式统计

- 平均长度: `8.39`
- 中位数长度: `5`
- 最短: `1`
- 最长: `39`

粗分布:

| 类型 | 数量 |
|---|---:|
| number-like | 5480 |
| single-token | 2433 |
| multi-token | 1576 |
| 单字母选项 | 11 |

## 提交格式

提交文件应为:

```csv
id,prediction
00066667,10010111
...
```

## 当前工作流观察

- 直接运行 `kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f <csv>` 返回 HTTP 400
- 实际可工作的路径是:
  1. 准备 Kaggle kernel
  2. push 新版本
  3. notebook 输出 `submission.zip`
  4. 等待 notebook run 完成
  5. 再按 code competition 方式提交 notebook version

注意:

- 上述工作流是 2026-03-28 的实测结论
- 这说明该比赛至少在 CLI 层面不能按普通 tabular competition 直接传 CSV
- competition source 在 Kaggle 运行环境中的实际挂载路径为:
  - `/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/`

## 下载命令

```bash
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/raw
```
