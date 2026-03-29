# %% [markdown]
# # NVIDIA Nemotron Model Reasoning Challenge - Prompt-Only Baseline
#
# **实验**: exp-001_prompt_zeroshot_YYYYMMDD
# **模型**: nvidia/NVIDIA-Nemotron-Nano-12B-v2
# **方法**: Zero-shot with reasoning enabled
#
# ## 使用说明
# 1. 在 Kaggle 上新建 Notebook
# 2. 添加 Competition Dataset: `nvidia-nemotron-model-reasoning-challenge`
# 3. 添加 Accelerator: GPU T4/P100 (推荐 P100 或更高)
# 4. 复制此代码到 Notebook
# 5. 根据实际数据格式调整 `load_data()` 和 `create_submission()`

# %%
!pip install -q transformers accelerate torch

# %%
import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# %% [markdown]
# ## 1. 配置

# %%
# 模型配置
MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"
USE_REASONING = True          # 启用推理模式
MAX_THINKING_TOKENS = 1024    # 推理 token 预算
MAX_NEW_TOKENS = 2048         # 最大生成 token 数
TEMPERATURE = 0.6 if USE_REASONING else 0.0
TOP_P = 0.95 if USE_REASONING else 1.0
BATCH_SIZE = 1                # 单条推理（内存受限时）
SEED = 42

# 数据路径 - 根据 Kaggle 实际路径调整
DATA_DIR = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge"
OUTPUT_DIR = "/kaggle/working"

# 设置随机种子
torch.manual_seed(SEED)

print(f"Model: {MODEL_ID}")
print(f"Reasoning: {USE_REASONING}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# %% [markdown]
# ## 2. 加载模型

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
model.eval()

print(f"Model loaded: {MODEL_ID}")
print(f"Parameters: {model.num_parameters() / 1e9:.1f}B")

# %% [markdown]
# ## 3. 数据加载

# %%
def load_data(data_dir):
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found in {data_dir}")
    return pd.read_csv(test_path)

# %%
# 加载数据
df_test = load_data(DATA_DIR)
print(f"Test samples: {len(df_test)}")
print(f"Columns: {list(df_test.columns)}")
df_test.head()

# %% [markdown]
# ## 4. 推理函数

# %%
def generate_response(prompt, reasoning=True):
    """生成单条回答"""
    # 构建消息
    system_content = "/think" if reasoning else "/no_think"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    # 应用 chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 生成
    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": TEMPERATURE > 0,
    }
    if TEMPERATURE > 0:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P
    
    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)
    
    # 解码（只取生成部分）
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 如果启用推理，提取最终答案（跳过 thinking trace）
    # TODO: 根据实际答案格式调整解析逻辑
    if reasoning and "</think>" in response:
        # 提取 </think> 之后的内容作为最终答案
        response = response.split("</think>")[-1].strip()
    
    return response


def extract_answer(response):
    """
    从模型回答中提取最终答案。
    TODO: 根据实际题目格式调整此函数。
    
    常见格式:
    - 答案在最后: "Answer: X" 或 "答案: X"
    - 选项: A/B/C/D
    - 数字: 直接数字
    """
    # 尝试提取 "Answer:" 后面的内容
    import re
    
    # 模式1: Answer: X
    match = re.search(r'(?:Answer|答案)[：:]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 模式2: 最后一个字母选项 A/B/C/D
    options = re.findall(r'\b([A-D])\b', response)
    if options:
        return options[-1]
    
    # 模式3: 最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1]
    
    # 默认返回完整回答
    return response.strip()

# %% [markdown]
# ## 5. 批量推理

# %%
def run_inference(df, prompt_col="prompt", id_col="id"):
    """批量推理"""
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        sample_id = row[id_col] if id_col in row.index else idx
        prompt = row[prompt_col]
        
        # 生成回答
        response = generate_response(prompt, reasoning=USE_REASONING)
        answer = extract_answer(response)
        
        results.append({
            "id": sample_id,
            "response": response,
            "answer": answer
        })
        
        # 打印前几个样本
        if idx < 3:
            print(f"\n--- Sample {idx} ---")
            print(f"Prompt: {prompt[:200]}...")
            print(f"Response: {response[:200]}...")
            print(f"Answer: {answer}")
    
    return pd.DataFrame(results)

# %%
# TODO: 根据实际列名调整
PROMPT_COL = "prompt"  # 或 "question", "input", "query"
ID_COL = "id"          # 或 "sample_id", "row_id"

df_results = run_inference(df_test, prompt_col=PROMPT_COL, id_col=ID_COL)

# %% [markdown]
# ## 6. 生成提交文件

# %%
def create_submission(df_results, output_path):
    submission = df_results[["id", "answer"]].copy()
    submission.columns = ["id", "prediction"]
    submission.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())
    return submission

# %%
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
submission = create_submission(df_results, SUBMISSION_PATH)

# %% [markdown]
# ## 7. 本地统计

# %%
print("\n=== 本地统计 ===")
print(f"总样本数: {len(df_results)}")
print(f"平均回答长度: {df_results['response'].str.len().mean():.0f} 字符")
print(f"答案类型分布:")
print(df_results['answer'].apply(lambda x: type(x).__name__).value_counts())

# 保存完整结果（含 reasoning trace）
full_output = os.path.join(OUTPUT_DIR, "results_full.jsonl")
df_results.to_json(full_output, orient="records", lines=True, force_ascii=False)
print(f"\n完整结果已保存: {full_output}")

# %% [markdown]
# ## 8. 清理

# %%
# 释放显存
del model
torch.cuda.empty_cache()
print("GPU memory released.")

print("\n=== 完成 ===")
print(f"提交文件: {SUBMISSION_PATH}")
print("请下载 submission.csv 并提交到 Kaggle")
