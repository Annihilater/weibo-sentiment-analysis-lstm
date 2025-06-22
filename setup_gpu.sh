#!/bin/bash

# 设置错误时退出
set -e

echo "========================================"
echo "CUDA环境设置脚本 - RTX 4090 GPU优化"
echo "========================================"

# 检查NVIDIA驱动
echo "检查NVIDIA驱动..."
nvidia-smi

# 检查CUDA版本
echo "查找CUDA库路径..."
CUDA_PATH=""

# 搜索常见的CUDA安装路径
for path in /usr/local/cuda* /usr/local/cuda /opt/cuda; do
    if [ -d "$path" ]; then
        CUDA_PATH="$path"
        echo "找到CUDA路径: $CUDA_PATH"
        break
    fi
done

if [ -z "$CUDA_PATH" ]; then
    echo "警告: 未找到CUDA路径，将尝试使用系统默认路径"
    CUDA_PATH="/usr/local/cuda"
fi

# 设置CUDA环境变量
echo "设置CUDA环境变量..."
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 检查cuDNN
CUDNN_PATHS=(
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/cuda/lib64"
    "/usr/lib/cuda/lib64"
    "/usr/local/cudnn/lib64"
)

CUDNN_FOUND=false
for path in "${CUDNN_PATHS[@]}"; do
    if [ -f "$path/libcudnn.so" ]; then
        echo "找到cuDNN路径: $path"
        export LD_LIBRARY_PATH=$path:$LD_LIBRARY_PATH
        CUDNN_FOUND=true
        break
    fi
done

if [ "$CUDNN_FOUND" = false ]; then
    echo "警告: 未找到cuDNN库，TensorFlow可能无法使用GPU"
    echo "安装TensorFlow GPU依赖..."
    pip install --upgrade pip
    pip install nvidia-cudnn-cu12 tensorflow==2.16.1
fi

# 设置TensorFlow环境变量
echo "设置TensorFlow环境变量..."
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=1
export TF_GPU_THREAD_MODE=gpu_private
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"

# 验证CUDA安装
echo "验证CUDA安装..."
nvcc --version

# 验证TensorFlow GPU可用性
echo "验证TensorFlow GPU可用性..."
python -c "
import tensorflow as tf
print('TensorFlow版本:', tf.__version__)
print('CUDA可用:', tf.test.is_built_with_cuda())
print('GPU可用:', tf.config.list_physical_devices('GPU'))
"

# 保存环境变量到文件，方便后续加载
echo "# GPU环境变量 - 由setup_gpu.sh生成" > gpu_env.sh
echo "export CUDA_HOME=$CUDA_HOME" >> gpu_env.sh
echo "export PATH=$CUDA_HOME/bin:\$PATH" >> gpu_env.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> gpu_env.sh
echo "export TF_FORCE_GPU_ALLOW_GROWTH=true" >> gpu_env.sh
echo "export TF_GPU_ALLOCATOR=cuda_malloc_async" >> gpu_env.sh
echo "export TF_CPP_MIN_LOG_LEVEL=1" >> gpu_env.sh
echo "export TF_GPU_THREAD_MODE=gpu_private" >> gpu_env.sh
echo "export TF_ENABLE_ONEDNN_OPTS=0" >> gpu_env.sh
echo "export TF_XLA_FLAGS=\"--tf_xla_enable_xla_devices --tf_xla_auto_jit=2\"" >> gpu_env.sh

echo "========================================"
echo "CUDA环境设置完成"
echo "========================================"
echo "环境变量已保存到 gpu_env.sh"
echo "使用方法: source gpu_env.sh" 