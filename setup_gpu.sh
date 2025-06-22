#!/bin/bash

# 显示系统信息
echo "=========================================="
echo "系统信息:"
uname -a
echo "=========================================="

# 检查NVIDIA驱动和CUDA版本
echo "NVIDIA驱动和CUDA信息:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "未找到nvidia-smi命令，请确保NVIDIA驱动已正确安装"
fi
echo "=========================================="

# 检查CUDA库路径
echo "CUDA库路径:"
if [ -d "/usr/local/cuda" ]; then
    echo "找到CUDA安装路径: /usr/local/cuda"
    ls -la /usr/local/cuda/lib64/libcud*.so* 2>/dev/null || echo "未找到CUDA库文件"
else
    echo "未在默认位置找到CUDA安装"
    # 尝试查找系统中的CUDA库
    find /usr -name "libcudart.so*" 2>/dev/null || echo "未找到CUDA运行时库"
fi
echo "=========================================="

# 设置环境变量
echo "设置环境变量..."

# 如果存在CUDA安装，设置环境变量
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    echo "已设置CUDA环境变量:"
    echo "CUDA_HOME=${CUDA_HOME}"
    echo "LD_LIBRARY_PATH包含CUDA路径"
fi

# 设置TensorFlow相关环境变量
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

echo "已设置TensorFlow环境变量:"
echo "TF_FORCE_GPU_ALLOW_GROWTH=true"
echo "TF_CPP_MIN_LOG_LEVEL=0"
echo "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6"
echo "=========================================="

# 验证Python和TensorFlow
echo "验证Python和TensorFlow安装:"
if command -v python3 &> /dev/null; then
    echo "Python版本:"
    python3 --version
    
    echo "TensorFlow信息:"
    python3 -c "import tensorflow as tf; print(f'TensorFlow版本: {tf.__version__}'); print(f'GPU可用: {len(tf.config.list_physical_devices(\"GPU\")) > 0}'); print(f'CUDA已启用: {tf.test.is_built_with_cuda()}'); print('可用设备:'); [print(d) for d in tf.config.list_physical_devices()]" || echo "无法导入TensorFlow或执行测试"
else
    echo "未找到Python3"
fi
echo "=========================================="

echo "环境设置完成" 