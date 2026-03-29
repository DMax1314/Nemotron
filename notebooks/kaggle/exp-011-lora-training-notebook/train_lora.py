import gc
import json
import os
import random
import shutil
import site
import stat
import sys
import time
import zipfile
from glob import glob

import pandas as pd


SEED = 42
MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
OUTPUT_DIR = "/kaggle/working/adapter"
ZIP_PATH = "/kaggle/working/submission.zip"
REPORT_PATH = "/kaggle/working/train_report.json"
SUBSAMPLE_SIZE = 6000
MAX_SEQ_LEN = 1024
NUM_EPOCHS = 1
GRAD_ACCUM = 4
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.01
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
FAMILY_SAMPLE_TARGETS = {
    "symbol": 1500,
    "bit": 1500,
    "gravity": 1100,
    "unit": 1100,
    "cipher": 400,
    "roman": 400,
}
FAMILY_PRIORITY = ["symbol", "bit", "gravity", "unit", "cipher", "roman"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_utility_root() -> str:
    utility_roots = [
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script",
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script",
    ]
    candidate_paths = []
    for utility_root in utility_roots:
        cutlass_pkg_path = os.path.join(
            utility_root, "nvidia_cutlass_dsl", "python_packages"
        )
        candidate_paths.append(cutlass_pkg_path)

    dynamic_matches = sorted(
        set(
            glob(
                "/kaggle/usr/lib/notebooks/**/nvidia_cutlass_dsl/python_packages",
                recursive=True,
            )
            + glob(
                "/kaggle/input/**/nvidia_cutlass_dsl/python_packages",
                recursive=True,
            )
        )
    )
    candidate_paths.extend(dynamic_matches)

    print("Utility path candidates:")
    for candidate_path in candidate_paths[:10]:
        print(f"  - {candidate_path}")

    deadline = time.time() + 120
    while time.time() < deadline:
        for cutlass_pkg_path in candidate_paths:
            if os.path.isdir(cutlass_pkg_path):
                utility_root = os.path.dirname(os.path.dirname(cutlass_pkg_path))
                site.addsitedir(cutlass_pkg_path)
                print(f"Added site dir: {cutlass_pkg_path}")
                print(f"Resolved utility root: {utility_root}")
                return utility_root

        remaining = int(max(deadline - time.time(), 0))
        print(f"Utility source not mounted yet; retrying ({remaining}s left)...")
        time.sleep(5)

    raise FileNotFoundError("Could not locate the NVIDIA utility script packages.")


def patch_triton_binaries(utility_root: str) -> None:
    src = os.path.join(
        utility_root, "triton", "backends", "nvidia", "bin", "ptxas-blackwell"
    )
    if not os.path.exists(src):
        print("ptxas-blackwell not found; skip Triton binary patch.")
        return

    dst = "/tmp/ptxas-blackwell"
    shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    import triton.backends.nvidia as nv_backend

    src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
    dst_bin = "/tmp/triton_nvidia_bin"
    shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
    for filename in os.listdir(dst_bin):
        file_path = os.path.join(dst_bin, filename)
        if os.path.isfile(file_path):
            os.chmod(
                file_path,
                os.stat(file_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
            )

    nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
    os.environ["TRITON_PTXAS_PATH"] = dst
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = dst
    print("Patched Triton PTXAS binaries.")


seed_everything(SEED)
utility_root = find_utility_root()
patch_triton_binaries(utility_root)

import kagglehub  # noqa: E402
import mamba_ssm  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def patch_rmsnorm() -> None:
    def _pure_rmsnorm_fn(
        x,
        weight,
        bias=None,
        z=None,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        upcast=True,
    ):
        del group_size, norm_before_gate
        dtype = x.dtype
        if upcast:
            x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        out = x_normed * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None:
            out = out * F.silu(z.float())
        return out.to(dtype)

    patched = 0
    for _, module in list(sys.modules.items()):
        if hasattr(module, "rmsnorm_fn"):
            module.rmsnorm_fn = _pure_rmsnorm_fn
            patched += 1
    print(f"Patched rmsnorm_fn in {patched} loaded modules.")


def disable_fast_path() -> None:
    patched = 0
    for name, module in list(sys.modules.items()):
        if "modeling_nemotron_h" in name and hasattr(module, "is_fast_path_available"):
            module.is_fast_path_available = False
            patched += 1
    print(f"Disabled Nemotron fast path in {patched} modules.")


def find_train_path() -> str:
    candidates = sorted(glob("/kaggle/input/**/train.csv", recursive=True))
    for candidate in candidates:
        if "nvidia-nemotron" in candidate.lower():
            print(f"Using train.csv: {candidate}")
            return candidate
    if candidates:
        print(f"Fallback train.csv: {candidates[0]}")
        return candidates[0]
    raise FileNotFoundError("Could not locate train.csv under /kaggle/input.")


def classify_prompt(prompt: str) -> str:
    first_line = prompt.split("\n", 1)[0]
    if "bit manipulation rule transforms 8-bit binary numbers" in first_line:
        return "bit"
    if "gravitational constant has been secretly changed" in first_line:
        return "gravity"
    if "secret unit conversion is applied to measurements" in first_line:
        return "unit"
    if "numbers are secretly converted into a different numeral system" in first_line:
        return "roman"
    if "secret encryption rules are used on text" in first_line:
        return "cipher"
    if "secret set of transformation rules is applied to equations" in first_line:
        return "symbol"
    return "other"


def build_family_balanced_sample(
    dataframe: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], dict[str, int]]:
    working = dataframe.copy()
    working["family"] = working["prompt"].map(classify_prompt)
    family_counts = working["family"].value_counts().to_dict()

    if not sample_size or len(working) <= sample_size:
        sampled = working.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sampled_counts = sampled["family"].value_counts().to_dict()
        print("Sampling skipped; using all rows.")
        return sampled, family_counts, sampled_counts, sampled_counts

    sampled_parts = []
    sampled_index = set()
    actual_targets: dict[str, int] = {}

    for family in FAMILY_PRIORITY:
        family_df = working[working["family"] == family]
        requested = FAMILY_SAMPLE_TARGETS.get(family, 0)
        actual = min(len(family_df), requested)
        actual_targets[family] = actual
        if actual == 0:
            continue
        sampled_family = family_df.sample(n=actual, random_state=seed)
        sampled_parts.append(sampled_family)
        sampled_index.update(sampled_family.index.tolist())

    sampled = pd.concat(sampled_parts, ignore_index=False)
    remaining = sample_size - len(sampled)
    if remaining > 0:
        print(f"Redistributing {remaining} leftover rows after family caps.")
        leftovers = working.loc[~working.index.isin(sampled_index)]
        extra = leftovers.sample(n=remaining, random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=False)

    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled_counts = sampled["family"].value_counts().to_dict()

    print("Full family counts:", family_counts)
    print("Requested family targets:", FAMILY_SAMPLE_TARGETS)
    print("Actual sampled family counts:", sampled_counts)
    return sampled, family_counts, sampled_counts, actual_targets


def resolve_model_path(download_root: str) -> str:
    search_roots = [download_root, os.path.dirname(download_root), "/kaggle/input/models"]
    candidates = []
    seen = set()

    for root in search_roots:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)

        if os.path.isfile(os.path.join(root, "config.json")):
            candidates.append(root)

        for config_path in glob(os.path.join(root, "**", "config.json"), recursive=True):
            candidates.append(os.path.dirname(config_path))

    unique_candidates = []
    seen_candidates = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen_candidates:
            continue
        if os.path.isfile(os.path.join(normalized, "config.json")):
            unique_candidates.append(normalized)
            seen_candidates.add(normalized)

    if not unique_candidates:
        raise FileNotFoundError(
            f"Could not locate a model directory with config.json under {download_root}."
        )

    def sort_key(path: str):
        has_tokenizer = any(
            os.path.isfile(os.path.join(path, filename))
            for filename in ("tokenizer.json", "tokenizer_config.json")
        )
        return (
            0 if path == os.path.normpath(download_root) else 1,
            0 if path.startswith(os.path.normpath(download_root)) else 1,
            0 if has_tokenizer else 1,
            path.count(os.sep),
            len(path),
        )

    unique_candidates.sort(key=sort_key)
    print("Resolved model candidates:")
    for candidate in unique_candidates[:5]:
        print(f"  - {candidate}")
    return unique_candidates[0]


def build_prompt(prompt: str) -> str:
    return prompt.strip() + "\nReturn only the final answer."


class SFTDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.rows = []
        skipped = 0

        for row in dataframe.itertuples(index=False):
            prompt = build_prompt(row.prompt)
            answer = str(row.answer).strip()
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            prompt_messages = [{"role": "user", "content": prompt}]

            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            prompt_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            prompt_len = min(len(prompt_ids), labels.shape[0])
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            if torch.all(labels == -100):
                skipped += 1
                continue

            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        print(
            f"Prepared {len(self.rows)} training examples "
            f"(skipped {skipped} after truncation/masking)."
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_path = find_train_path()
train_df = pd.read_csv(train_path)
print(f"Loaded train rows: {len(train_df)}")
train_family_counts = {}
sampled_family_counts = {}
actual_family_targets = {}
if SUBSAMPLE_SIZE:
    train_df, train_family_counts, sampled_family_counts, actual_family_targets = (
        build_family_balanced_sample(train_df, SUBSAMPLE_SIZE, SEED)
    )
    print(f"Sampled train rows: {len(train_df)}")

model_download_root = kagglehub.model_download(MODEL_HANDLE)
print(f"Model download root: {model_download_root}")
model_path = resolve_model_path(model_download_root)
print(f"Resolved model path: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

patch_rmsnorm()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
patch_rmsnorm()
disable_fast_path()
model.config.use_cache = False
print("Base model loaded.")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.print_trainable_parameters()

dataset = SFTDataset(train_df, tokenizer, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.AdamW(
    filter(lambda parameter: parameter.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

model.train()
loss_history = []
step = 0
device = next(model.parameters()).device
print(
    "Starting training: "
    f"epochs={NUM_EPOCHS}, batches={len(dataloader)}, grad_accum={GRAD_ACCUM}"
)

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    optimizer.zero_grad()

    for batch_index, batch in enumerate(dataloader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        raw_loss = outputs.loss.detach().float().item()
        running_loss += raw_loss
        loss_history.append(raw_loss)

        if batch_index % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 20 == 0:
                avg_loss = running_loss / batch_index
                print(f"epoch={epoch + 1} step={step} avg_loss={avg_loss:.4f}")

    if len(dataloader) % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    epoch_loss = running_loss / max(len(dataloader), 1)
    print(f"Finished epoch {epoch + 1}: avg_loss={epoch_loss:.4f}")

print(f"Saving adapter to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        file_path = os.path.join(OUTPUT_DIR, filename)
        archive.write(file_path, arcname=filename)

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    names = archive.namelist()
    print(f"Zip contents: {names}")
    if "adapter_config.json" not in names:
        raise RuntimeError("submission.zip is missing adapter_config.json")

report = {
    "seed": SEED,
    "model_handle": MODEL_HANDLE,
    "model_download_root": model_download_root,
    "model_path": model_path,
    "train_path": train_path,
    "train_rows": int(len(train_df)),
    "family_sample_targets": actual_family_targets,
    "family_counts_full": train_family_counts,
    "family_counts_sampled": sampled_family_counts,
    "max_seq_len": MAX_SEQ_LEN,
    "epochs": NUM_EPOCHS,
    "grad_accum": GRAD_ACCUM,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "steps": step,
    "final_loss": loss_history[-1] if loss_history else None,
    "mean_loss": sum(loss_history) / len(loss_history) if loss_history else None,
}
with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print(f"Saved training report: {REPORT_PATH}")
print(f"Created submission: {ZIP_PATH}")

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Done.")
