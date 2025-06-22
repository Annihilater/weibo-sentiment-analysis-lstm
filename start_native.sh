#!/bin/bash

echo "=== 宿主机原生GPU训练启动 ==="

# 检查GPU状态
echo ""
echo "检查GPU状态..."
nvidia-smi

# 设置CUDA环境变量
echo ""
echo "设置CUDA环境变量..."
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# 设置Python路径
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"

# 检查数据文件
echo ""
echo "检查数据文件..."
if [ ! -f "data/input/all_utf8.csv" ]; then
    echo "❌ 数据文件不存在: data/input/all_utf8.csv"
    exit 1
fi
echo "✅ 数据文件存在"

# 创建输出目录
echo ""
echo "创建输出目录..."
mkdir -p data/output logs/tensorboard
echo "✅ 目录创建完成"

# 测试TensorFlow GPU
echo ""
echo "测试TensorFlow GPU支持..."
python -c "
import tensorflow as tf
print('TensorFlow版本:', tf.__version__)
print('CUDA构建:', tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print('检测到GPU数量:', len(gpus))
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')
"

# 启动训练
echo ""
echo "🚀 启动微博情感分析训练..."
echo "================================"
python src/main.py 