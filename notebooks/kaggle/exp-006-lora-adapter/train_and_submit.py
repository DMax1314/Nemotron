import subprocess
import sys

for pkg in ["transformers", "peft", "accelerate"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os
import json
import torch
from transformers import AutoTokenizer, AutoConfig

MODEL_PATH = (
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
)
OUTPUT_DIR = "/kaggle/working"

print("Loading tokenizer and config only (not loading full model)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Creating no-op LoRA adapter...")
adapter_dir = os.path.join(OUTPUT_DIR, "adapter")
os.makedirs(adapter_dir, exist_ok=True)

adapter_config = {
    "alora_invocation_tokens": None,
    "alpha_pattern": {},
    "arrow_config": None,
    "auto_mapping": None,
    "base_model_name_or_path": MODEL_PATH,
    "bias": "none",
    "corda_config": None,
    "ensure_weight_tying": False,
    "eva_config": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 16,
    "lora_bias": False,
    "lora_dropout": 0.05,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "peft_version": "0.18.1",
    "qalora_group_size": 16,
    "r": 8,
    "rank_pattern": {},
    "revision": None,
    "target_modules": ".*\\.(in_proj|out_proj|up_proj|down_proj)$",
    "target_parameters": None,
    "task_type": "CAUSAL_LM",
    "trainable_token_indices": None,
    "use_dora": False,
    "use_qalora": False,
    "use_rslora": False,
}

with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
    json.dump(adapter_config, f, indent=2)

print("Creating dummy adapter weights...")
dummy_state_dict = {
    "base_model.model.model.layers.0.self_attn.in_proj.lora_A.weight": torch.randn(
        8, config.hidden_size, dtype=torch.bfloat16
    ),
    "base_model.model.model.layers.0.self_attn.in_proj.lora_B.weight": torch.randn(
        config.hidden_size, 8, dtype=torch.bfloat16
    ),
}

from safetensors.torch import save_file

save_file(dummy_state_dict, os.path.join(adapter_dir, "adapter_model.safetensors"))

readme_content = """---
base_model: /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- transformers
---

# No-op LoRA Adapter

This is a no-op adapter for submission format testing.
"""

with open(os.path.join(adapter_dir, "README.md"), "w") as f:
    f.write(readme_content)

print("Creating submission zip...")
import zipfile

submission_zip = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(submission_zip, "w") as zf:
    for root, dirs, files in os.walk(adapter_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, adapter_dir)
            zf.write(file_path, arcname)

print(f"Submission created: {submission_zip}")
print("Done!")
