#!/bin/bash

set -e

echo "========================================"
echo "CUDA环境设置脚本 - RTX 4090 GPU优化"
echo "========================================"

# 检查NVIDIA驱动
echo "检查NVIDIA驱动..."
nvidia-smi

# 查找系统CUDA库路径
echo "查找CUDA库路径..."
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-12.6"
    "/usr/local/cuda-12.5"
    "/usr/local/cuda-12.4"
    "/usr/local/cuda-12.3"
    "/usr/local/cuda-12.2"
    "/usr/local/cuda-12.1"
    "/usr/local/cuda-12.0"
)

CUDA_PATH=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        CUDA_PATH="$path"
        echo "找到CUDA路径: $CUDA_PATH"
        break
    fi
done

if [ -z "$CUDA_PATH" ]; then
    echo "未找到CUDA路径，尝试查找..."
    CUDA_PATH=$(find /usr -name "cuda" -type d | grep -v "samples" | head -n 1)
    if [ -z "$CUDA_PATH" ]; then
        echo "警告: 未找到CUDA路径"
    else
        echo "找到CUDA路径: $CUDA_PATH"
    fi
fi

# 设置CUDA环境变量
echo "设置CUDA环境变量..."
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# 安装TensorFlow GPU依赖
echo "安装TensorFlow GPU依赖..."
pip install --upgrade pip
pip install nvidia-cudnn-cu12
pip install tensorflow==2.16.1

# 设置TensorFlow环境变量
echo "设置TensorFlow环境变量..."
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

# 验证CUDA安装
echo "验证CUDA安装..."
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    $CUDA_HOME/bin/nvcc --version
else
    echo "警告: CUDA编译器(nvcc)未找到"
fi

# 验证TensorFlow GPU可用性
echo "验证TensorFlow GPU可用性..."
python - << EOF
import tensorflow as tf
print("TensorFlow版本:", tf.__version__)
print("CUDA可用:", tf.test.is_built_with_cuda())
print("GPU可用:", tf.config.list_physical_devices('GPU'))
if not tf.config.list_physical_devices('GPU'):
    print("警告: TensorFlow无法识别GPU")
    print("尝试手动设置可见设备...")
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
    print("重新检查GPU...")
    print("GPU可用:", tf.config.list_physical_devices('GPU'))
EOF

echo "========================================"
echo "CUDA环境设置完成"
echo "========================================"

# 保存环境变量到文件，供其他脚本使用
cat > gpu_env.sh << EOF
export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:\$LD_LIBRARY_PATH
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
EOF

echo "环境变量已保存到 gpu_env.sh"
echo "使用方法: source gpu_env.sh" 