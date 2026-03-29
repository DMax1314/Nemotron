#!/bin/bash
#SBATCH --job-name=taac_l40
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_l40_%j.out
#SBATCH --error=logs/train_l40_%j.err

echo "=== TAAC2026 Training (L40) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Environment
module load python/3.10
module load cuda/12.1
source ~/.venvs/taac2026/bin/activate

# Arguments
CONFIG=${1:-"configs/default.yaml"}
EXP_NAME=${2:-"exp_l40"}

echo "Config: $CONFIG"
echo "Experiment: $EXP_NAME"

# Create log directory
mkdir -p logs

# Train
python -m src.training.trainer \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --device cuda \
    --fp16 \
    --num-workers 4 \
    --seed 42

echo "=== Training complete ==="
echo "End: $(date)"
