cd /scratch1

mkdir -p /scratch1/logs

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/scratch1/logs/self_distillation_${RUN_ID}.log"
PID_FILE="/scratch1/logs/self_distillation_${RUN_ID}.pid"

CUDA_VISIBLE_DEVICES=2,3 \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 \
nohup python -u main.py \
    --data-root . \
    --output-dir output \
    --model-path models/nemotron-3-nano-30b-a3b-bf16 \
    --train-csv-path train.csv \
    --adapter-path 'trained-adapter/kaggle/input/models/kienngx/nemotron-nano-30b-trained/triton/tinker-adapter/1' \
    --kagglehub-model-ref metric/nemotron-3-nano-30b-a3b-bf16/transformers/default \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --offset 0 \
    --puzzle-types symbol,unit \
    --seed 471 \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started PID: $(cat "$PID_FILE")"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
