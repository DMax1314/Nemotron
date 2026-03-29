# %% [markdown]
# # NVIDIA Nemotron Model Reasoning Challenge — LoRA Fine-tuning Baseline
#
# **实验**: exp-002_ft_lora_base_YYYYMMDD
# **模型**: nvidia/NVIDIA-Nemotron-Nano-12B-v2 (base)
# **方法**: LoRA fine-tuning on competition training data
#
# ## 使用说明
# 1. 在 Kaggle 上新建 Notebook
# 2. 添加 Competition Dataset: `nvidia-nemotron-model-reasoning-challenge`
# 3. 添加 Accelerator: GPU P100/T4 (推荐 P100 或更高)
# 4. 复制此代码到 Notebook
# 5. 根据实际数据格式调整 `load_data()` 和训练配置

# %%
!pip install -q transformers accelerate torch peft bitsandbytes datasets

# %%
import os
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# %% [markdown]
# ## 1. 配置

# %%
# 模型配置
MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"
USE_QLORA = True              # 使用 QLoRA 节省显存
LORA_R = 16                   # LoRA rank
LORA_ALPHA = 32               # LoRA alpha
LORA_DROPOUT = 0.05           # LoRA dropout
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 目标模块

# 训练配置
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 2048
SEED = 42

# 数据路径
DATA_DIR = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge"
OUTPUT_DIR = "/kaggle/working"

torch.manual_seed(SEED)

print(f"Model: {MODEL_ID}")
print(f"QLoRA: {USE_QLORA}")
print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA}")

# %% [markdown]
# ## 2. 加载模型和 tokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "trust_remote_code": True,
    "device_map": "auto"
}

if USE_QLORA:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model_kwargs["quantization_config"] = bnb_config

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
model.config.use_cache = False
model.config.pretraining_tp = 1

if USE_QLORA:
    model = prepare_model_for_kbit_training(model)

print(f"Model loaded: {MODEL_ID}")

# %% [markdown]
# ## 3. 配置 LoRA

# %%
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## 4. 数据加载与预处理

# %%
def load_training_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found in {data_dir}")
    return pd.read_csv(train_path)

def format_prompt(example):
    """
    格式化训练样本。
    TODO: 根据实际数据格式调整。
    """
    # 假设有 prompt 和 answer 字段
    prompt = example.get("prompt", example.get("question", ""))
    answer = example.get("answer", example.get("response", ""))
    
    messages = [
        {"role": "system", "content": "/think"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

# %%
df_train = load_training_data(DATA_DIR)
print(f"Training samples: {len(df_train)}")
print(f"Columns: {list(df_train.columns)}")

# 转换为 Dataset
dataset = Dataset.from_pandas(df_train)

# %% [markdown]
# ## 5. Tokenize 数据

# %%
def tokenize_function(examples):
    text = format_prompt(examples)
    return tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

print(f"Tokenized samples: {len(tokenized_dataset)}")

# %% [markdown]
# ## 6. 训练配置

# %%
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    tf32=True,
    optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",
    lr_scheduler_type="cosine",
    seed=SEED,
    report_to="none"
)

# %% [markdown]
# ## 7. 训练

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

print("Starting training...")
train_result = trainer.train()

# 保存指标
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# %% [markdown]
# ## 8. 保存模型

# %%
# 保存 LoRA weights
adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

print(f"LoRA adapter saved: {adapter_path}")

# %% [markdown]
# ## 9. 合并权重（可选）

# %%
MERGE_AND_SAVE = False

if MERGE_AND_SAVE:
    from peft import AutoPeftModelForCausalLM
    
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    merged_model = merged_model.merge_and_unload()
    
    merged_path = os.path.join(OUTPUT_DIR, "merged_model")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved: {merged_path}")

# %% [markdown]
# ## 10. 推理测试

# %%
def generate_with_lora(prompt, adapter_path, max_new_tokens=512):
    """使用 LoRA adapter 进行推理"""
    from peft import PeftModel
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)
    
    messages = [
        {"role": "system", "content": "/think"},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_with_lora.device)
    
    with torch.no_grad():
        outputs = model_with_lora.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95
        )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

# %%
# 测试推理
test_prompt = "Solve: What is 2 + 2?"
response = generate_with_lora(test_prompt, adapter_path)
print(f"Prompt: {test_prompt}")
print(f"Response: {response}")

# %% [markdown]
# ## 11. 生成提交文件

# %%
def run_inference_with_lora(df, adapter_path, prompt_col="prompt", id_col="id"):
    """使用 LoRA 进行批量推理"""
    from peft import PeftModel
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)
    model_with_lora.eval()
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        sample_id = row[id_col] if id_col in row.index else idx
        prompt = row[prompt_col]
        
        messages = [
            {"role": "system", "content": "/think"},
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model_with_lora.device)
        
        with torch.no_grad():
            outputs = model_with_lora.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.95
            )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # 提取答案
        if "</think>" in response:
            answer = response.split("</think>")[-1].strip()
        else:
            answer = response.strip()
        
        results.append({"id": sample_id, "answer": answer})
    
    return pd.DataFrame(results)

# %%
# TODO: 根据实际列名调整
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
PROMPT_COL = "prompt"
ID_COL = "id"

df_results = run_inference_with_lora(df_test, adapter_path, PROMPT_COL, ID_COL)

# %%
submission = df_results[["id", "answer"]]
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
print(f"Submission saved: {len(submission)} rows")

# %% [markdown]
# ## 12. 清理

# %%
del model, model_with_lora
torch.cuda.empty_cache()
print("GPU memory released.")

print("\n=== 完成 ===")
print(f"Adapter: {adapter_path}")
print(f"Submission: {OUTPUT_DIR}/submission.csv")
