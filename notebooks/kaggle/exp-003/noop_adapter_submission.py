import os
import site
import zipfile

import torch

utility_roots = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script",
]
for utility_root in utility_roots:
    cutlass_pkg_path = os.path.join(
        utility_root, "nvidia_cutlass_dsl", "python_packages"
    )
    if os.path.isdir(cutlass_pkg_path):
        site.addsitedir(cutlass_pkg_path)
        print(f"Added site dir: {cutlass_pkg_path}")
        break
else:
    raise FileNotFoundError("Could not locate the NVIDIA utility script packages.")

import kagglehub  # noqa: E402
import mamba_ssm  # noqa: F401,E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402


MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
OUTPUT_DIR = "/kaggle/working/adapter"
ZIP_PATH = "/kaggle/working/submission.zip"
LORA_RANK = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_path = kagglehub.model_download(MODEL_HANDLE)
print(f"Model path: {model_path}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
print("Base model loaded.")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(f"Saving adapter to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        file_path = os.path.join(OUTPUT_DIR, filename)
        archive.write(file_path, arcname=filename)

print(f"Created {ZIP_PATH}")
with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    names = archive.namelist()
    print(f"Zip contents: {names}")
    if "adapter_config.json" not in names:
        raise RuntimeError("submission.zip is missing adapter_config.json")

print("Done.")
