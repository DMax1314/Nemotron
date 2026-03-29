# %% [markdown]
# # NVIDIA Nemotron Model Reasoning Challenge - Prompt-Only Baseline
#
# ****: exp-001_prompt_zeroshot_YYYYMMDD
# ****: nvidia/NVIDIA-Nemotron-Nano-12B-v2
# ****: Zero-shot with reasoning enabled
#
# ## 
# 1.  Kaggle  Notebook
# 2.  Competition Dataset: `nvidia-nemotron-model-reasoning-challenge`
# 3.  Accelerator: GPU T4/P100 ( P100 )
# 4.  Notebook
# 5.  `load_data()`  `create_submission()`

# %%
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

# Install PyTorch with CUDA 11.8 support (compatible with P100)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 
    'torch', 'torchvision', 'torchaudio',
    '--index-url', 'https://download.pytorch.org/whl/cu118'])

for pkg in ['transformers', 'accelerate']:
    install(pkg)



# %%
import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# %% [markdown]
# ## 1. 

# %%
# 
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
USE_REASONING = True          # 
MAX_THINKING_TOKENS = 1024    #  token 
MAX_NEW_TOKENS = 2048         #  token 
TEMPERATURE = 0.6 if USE_REASONING else 0.0
TOP_P = 0.95 if USE_REASONING else 1.0
BATCH_SIZE = 1                # 
SEED = 42

#  -  Kaggle 
DATA_DIR = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge"
OUTPUT_DIR = "/kaggle/working"

# 
torch.manual_seed(SEED)

print(f"Model: {MODEL_ID}")
print(f"Reasoning: {USE_REASONING}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# %% [markdown]
# ## 2. 

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cpu"
)
model.eval()

print(f"Model loaded: {MODEL_ID}")
print(f"Parameters: {model.num_parameters() / 1e9:.1f}B")

# %% [markdown]
# ## 3. 

# %%
def load_data():
    import glob
    
    # Find all CSV files in /kaggle/input
    csv_files = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    print(f"Found CSV files: {csv_files}")
    
    # Look for test.csv
    test_files = [f for f in csv_files if "test" in f.lower()]
    if not test_files:
        raise FileNotFoundError("No test.csv found in /kaggle/input")
    
    test_path = test_files[0]
    print(f"Loading test data from: {test_path}")
    return pd.read_csv(test_path)

# %%
# 
df_test = load_data()
print(f"Test samples: {len(df_test)}")
print(f"Columns: {list(df_test.columns)}")
df_test.head()

# %% [markdown]
# ## 4. 

# %%
def generate_response(prompt, reasoning=True):
    """"""
    # 
    system_content = "You are a helpful assistant that solves reasoning problems step by step." if reasoning else "Answer concisely."
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    #  chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Handle different tokenizer output types
    if hasattr(input_ids, 'input_ids'):
        input_ids = input_ids.input_ids
    elif isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    
    input_ids = input_ids.to(model.device)
    
    # 
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
    
    # 
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    #  thinking trace
    # TODO: 
    if reasoning and "</think>" in response:
        #  </think> 
        response = response.split("</think>")[-1].strip()
    
    return response


def extract_answer(response):
    """
    
    TODO: 
    
    :
    - : "Answer: X"  ": X"
    - : A/B/C/D
    - : 
    """
    #  "Answer:" 
    import re
    
    # 1: Answer: X
    match = re.search(r'(?:Answer|)[:]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 2:  A/B/C/D
    options = re.findall(r'\b([A-D])\b', response)
    if options:
        return options[-1]
    
    # 3: 
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1]
    
    # 
    return response.strip()

# %% [markdown]
# ## 5. 

# %%
def run_inference(df, prompt_col="prompt", id_col="id"):
    """"""
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        sample_id = row[id_col] if id_col in row.index else idx
        prompt = row[prompt_col]
        
        # 
        response = generate_response(prompt, reasoning=USE_REASONING)
        answer = extract_answer(response)
        
        results.append({
            "id": sample_id,
            "response": response,
            "answer": answer
        })
        
        # 
        if idx < 3:
            print(f"\n--- Sample {idx} ---")
            print(f"Prompt: {prompt[:200]}...")
            print(f"Response: {response[:200]}...")
            print(f"Answer: {answer}")
    
    return pd.DataFrame(results)

# %%
# TODO: 
PROMPT_COL = "prompt"  #  "question", "input", "query"
ID_COL = "id"          #  "sample_id", "row_id"

df_results = run_inference(df_test, prompt_col=PROMPT_COL, id_col=ID_COL)

# %% [markdown]
# ## 6. 

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
# ## 7. 

# %%
print("\n===  ===")
print(f": {len(df_results)}")
print(f": {df_results['response'].str.len().mean():.0f} ")
print(f":")
print(df_results['answer'].apply(lambda x: type(x).__name__).value_counts())

#  reasoning trace
full_output = os.path.join(OUTPUT_DIR, "results_full.jsonl")
df_results.to_json(full_output, orient="records", lines=True, force_ascii=False)
print(f"\n: {full_output}")

# %% [markdown]
# ## 8. 

# %%
# 
del model
torch.cuda.empty_cache()
print("GPU memory released.")

print("\n===  ===")
print(f": {SUBMISSION_PATH}")
print(" submission.csv  Kaggle")
