#!/bin/bash

cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Starting ours annotation servers (GPU 0, GPU 1)..."

echo "  GPU 0 → port 3001"
nohup env CUDA_VISIBLE_DEVICES=0 gunicorn --workers 1 --timeout 300 \
    --bind 0.0.0.0:3001 app_multi_gpu:app \
    > $LOG_DIR/gpu_0.log 2>&1 &

echo "  GPU 1 → port 3002"
nohup env CUDA_VISIBLE_DEVICES=1 gunicorn --workers 1 --timeout 300 \
    --bind 0.0.0.0:3002 app_multi_gpu:app \
    > $LOG_DIR/gpu_1.log 2>&1 &

echo "Done. Logs in '$LOG_DIR/'. Use './stop.sh' to stop."
