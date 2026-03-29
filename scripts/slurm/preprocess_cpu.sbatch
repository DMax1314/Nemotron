#!/bin/bash
#SBATCH --job-name=taac_preprocess
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

echo "=== TAAC2026 Data Preprocessing ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Environment
module load python/3.10
source ~/.venvs/taac2026/bin/activate

# Arguments
DATA_DIR=${1:-"data/raw"}
OUTPUT_DIR=${2:-"data/processed"}

echo "Input: $DATA_DIR"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Preprocess
python -m src.data.preprocessing \
    --input-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --seq-max-len 50

echo "=== Preprocessing complete ==="
echo "End: $(date)"
