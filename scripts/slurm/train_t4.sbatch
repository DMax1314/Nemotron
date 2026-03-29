#!/bin/bash
#SBATCH --job-name=taac_t4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_t4_%j.out
#SBATCH --error=logs/train_t4_%j.err

echo "=== TAAC2026 Training (T4) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Environment
module load python/3.10
module load cuda/12.1
source ~/.venvs/taac2026/bin/activate

# Arguments
CONFIG=${1:-"configs/default.yaml"}
EXP_NAME=${2:-"exp_t4"}

echo "Config: $CONFIG"
echo "Experiment: $EXP_NAME"

mkdir -p logs

# T4 has 16GB VRAM — use smaller batch size and gradient accumulation
python -m src.training.trainer \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --device cuda \
    --fp16 \
    --num-workers 2 \
    --batch-size 128 \
    --gradient-accumulation-steps 4 \
    --seed 42

echo "=== Training complete ==="
echo "End: $(date)"
