#!/bin/bash
#SBATCH --job-name=nim_super_retry
#SBATCH --partition=hawkmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "$(date) job started on $(hostname)"

cd "$HOME/Nemotron"
mkdir -p logs data/distill

if [ -f "$HOME/.nemotron_api_env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.nemotron_api_env"
  set +a
fi

if [ -z "${NVIDIA_API_KEY:-}" ]; then
  echo "NVIDIA_API_KEY is missing in the SLURM job environment."
  exit 2
fi

VENV_ROOT="/tmp/$USER/nemotron_super_${SLURM_JOB_ID:-manual}"
mkdir -p "$VENV_ROOT"
VENV_DIR="$VENV_ROOT/venv"
echo "$(date) creating venv at $VENV_DIR"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "$(date) installing Python dependencies"
python -m pip install --disable-pip-version-check openai pandas
echo "$(date) starting Nemotron Super retry runner"

python scripts/distill/run_nemotron_super_distill.py \
  --provider nvidia \
  --input-csv data/distill/holdout_v1/train_split.csv \
  --family bit --family symbol \
  --samples-per-row 3 \
  --output-format teacher \
  --json-mode \
  --max-tokens 512 \
  --rpm-limit 36 \
  --workers 16 \
  --max-runtime-hours 71.5 \
  --resume-against-jsonl data/distill/nim_super_teacher_bitsymbol_3x.jsonl \
  --output-jsonl data/distill/nim_super_teacher_bitsymbol_3x_retry.jsonl

echo "$(date) job finished"
