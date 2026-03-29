#!/bin/bash
#SBATCH --job-name=taac_setup
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

echo "=== TAAC2026 Environment Setup ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

# Load modules (adjust for your cluster)
module load python/3.10
module load cuda/12.1

# Create virtual environment
python -m venv ~/.venvs/taac2026
source ~/.venvs/taac2026/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install \
    datasets \
    polars \
    pandas \
    pyarrow \
    scikit-learn \
    lightgbm \
    optuna \
    wandb \
    tqdm \
    ruff \
    pytest

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"

# Download demo data
python -c "
from datasets import load_dataset
ds = load_dataset('TAAC2026/data_sample_1000')
ds.save_to_disk('data/demo')
print('Demo data saved to data/demo')
"

echo "=== Setup complete ==="
