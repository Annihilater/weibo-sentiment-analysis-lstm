#!/bin/bash

echo "=== 直接GPU模式启动 ==="

# 停止现有容器
echo "停止现有容器..."
docker-compose -f docker-compose-tensorflow.yml down 2>/dev/null || true

# 检查GPU是否可用
echo "检查宿主机GPU..."
nvidia-smi

# 使用直接的GPU运行方式
echo "启动GPU训练容器（直接模式）..."
docker run --rm -it \
    --gpus all \
    --name weibo-sentiment-gpu \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/logs:/workspace/logs \
    -w /workspace \
    -e PYTHONUNBUFFERED=1 \
    -e TF_CPP_MIN_LOG_LEVEL=1 \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e PYTHONPATH=/workspace \
    weibo-sentiment-analysis:gpu-latest \
    python src/main.py 