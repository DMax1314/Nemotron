"""
Train LoRA adapter on on-policy rollout data and produce submission.zip.

Usage:
    python scripts/train_on_policy.py
    python scripts/train_on_policy.py --data /path/to/on_policy_data.csv --model /path/to/model

The on_policy_data.csv is produced by exp-013 (on-policy rollout generation).
It uses the `cot` column (full reasoning chain) as the training target.
"""

import argparse
import gc
import json
import os
import random
import sys
import zipfile

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
SEED = 42
MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
SUBSAMPLE_SIZE = 0  # 0 = use all
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
NUM_EPOCHS = 2
GRAD_ACCUM = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Nemotron model patches (from exp-012)
# ---------------------------------------------------------------------------
def patch_rmsnorm() -> None:
    def _pure_rmsnorm_fn(
        x, weight, bias=None, z=None, eps=1e-5,
        group_size=None, norm_before_gate=True, upcast=True,
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_prompt(prompt: str) -> str:
    return prompt.strip() + "\nPlease put your final answer inside `\\boxed{}`."


class SFTDataset(Dataset):
    """Tokenize prompt+answer pairs, mask prompt tokens in labels."""

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
                messages, tokenize=False, add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
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

            self.rows.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        print(f"Prepared {len(self.rows)} training examples (skipped {skipped} after truncation/masking).")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------
def resolve_model_path(root: str) -> str:
    from glob import glob as _glob

    if os.path.isfile(os.path.join(root, "config.json")):
        return root

    candidates = []
    seen = set()
    for cfg in _glob(os.path.join(root, "**", "config.json"), recursive=True):
        d = os.path.dirname(cfg)
        normed = os.path.normpath(d)
        if normed not in seen:
            seen.add(normed)
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(f"No model directory with config.json under {root}")

    candidates.sort(key=lambda p: (p.count(os.sep), len(p)))
    print("Resolved model candidates:")
    for c in candidates[:5]:
        print(f"  - {c}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train LoRA on on-policy data")
    parser.add_argument("--data", type=str, default=None, help="Path to on_policy_data.csv")
    parser.add_argument("--model", type=str, default=None, help="Model path or Kaggle handle")
    parser.add_argument("--output", type=str, default="/kaggle/working/adapter", help="Adapter output dir")
    parser.add_argument("--zip", type=str, default="/kaggle/working/submission.zip", help="Submission zip path")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE_SIZE, help="0 = use all rows")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--use-cot", action="store_true", default=True,
                        help="Use cot column as training target (default: True)")
    parser.add_argument("--no-cot", dest="use_cot", action="store_false",
                        help="Use answer column instead of cot")
    args = parser.parse_args()

    seed_everything(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- Resolve data path ---
    if args.data:
        data_path = args.data
    else:
        # Try common locations
        candidates = [
            "/kaggle/working/on_policy_data.csv",
            "data/on_policy_data.csv",
        ]
        data_path = next((c for c in candidates if os.path.exists(c)), None)
        if data_path is None:
            raise FileNotFoundError(
                "Could not find on_policy_data.csv. Pass --data /path/to/on_policy_data.csv"
            )

    print(f"Loading data from: {data_path}")
    train_df = pd.read_csv(data_path)
    print(f"Loaded {len(train_df)} rows. Columns: {list(train_df.columns)}")

    # --- Build training target ---
    # on_policy_data.csv has: id, family, prompt, answer, cot, source, verified
    # Use cot (full reasoning) as training target for on-policy data
    if args.use_cot and "cot" in train_df.columns:
        train_df["answer"] = train_df["cot"]
        print("Using 'cot' column as training target (full reasoning chain).")
    else:
        print("Using 'answer' column as training target.")

    # Filter to verified rows if available
    if "verified" in train_df.columns:
        verified_count = train_df["verified"].sum()
        print(f"Verified rows: {verified_count}/{len(train_df)}")
        train_df = train_df[train_df["verified"] == True].reset_index(drop=True)
        print(f"After filtering to verified: {len(train_df)} rows")

    if args.subsample and len(train_df) > args.subsample:
        train_df = train_df.sample(n=args.subsample, random_state=SEED).reset_index(drop=True)
        print(f"Subsampled to {len(train_df)} rows")

    # --- Resolve model path ---
    import kagglehub

    model_handle = args.model or MODEL_HANDLE
    if os.path.isdir(model_handle):
        model_path = resolve_model_path(model_handle)
    else:
        download_root = kagglehub.model_download(model_handle)
        model_path = resolve_model_path(download_root)
    print(f"Model path: {model_path}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Load model ---
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

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    # --- Dataset & DataLoader ---
    dataset = SFTDataset(train_df, tokenizer, args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    # --- Training loop ---
    model.train()
    loss_history = []
    step = 0
    device = next(model.parameters()).device
    print(f"Training: epochs={args.epochs}, batches={len(dataloader)}, grad_accum={GRAD_ACCUM}")

    for epoch in range(args.epochs):
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            raw_loss = outputs.loss.detach().float().item()
            running_loss += raw_loss
            loss_history.append(raw_loss)

            if batch_idx % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % 20 == 0:
                    avg_loss = running_loss / batch_idx
                    print(f"  epoch={epoch + 1} step={step} avg_loss={avg_loss:.4f}")

        if len(dataloader) % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

        epoch_loss = running_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1} done: avg_loss={epoch_loss:.4f}")

    # --- Save adapter & zip ---
    output_dir = args.output
    zip_path = args.zip
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving adapter to {output_dir} ...")
    model.save_pretrained(output_dir)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in sorted(os.listdir(output_dir)):
            file_path = os.path.join(output_dir, filename)
            archive.write(file_path, arcname=filename)

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        print(f"Zip contents: {names}")
        if "adapter_config.json" not in names:
            raise RuntimeError("submission.zip is missing adapter_config.json")

    # --- Save report ---
    report = {
        "seed": SEED,
        "data_path": data_path,
        "model_path": model_path,
        "train_rows": len(train_df),
        "use_cot": args.use_cot,
        "max_seq_len": args.max_seq_len,
        "batch_size": BATCH_SIZE,
        "epochs": args.epochs,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "steps": step,
        "final_loss": loss_history[-1] if loss_history else None,
        "mean_loss": sum(loss_history) / len(loss_history) if loss_history else None,
    }
    report_path = os.path.join(os.path.dirname(zip_path), "train_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved report: {report_path}")
    print(f"Created submission: {zip_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
