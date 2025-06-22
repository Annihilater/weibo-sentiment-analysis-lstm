#!/bin/bash

echo "=== GPU调试诊断 ==="

echo "1. 检查宿主机GPU状态..."
nvidia-smi

echo ""
echo "2. 检查Docker GPU支持..."
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi

echo ""
echo "3. 检查NVIDIA Container Runtime..."
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 \
    /bin/bash -c "ls -la /usr/local/cuda/lib64/ | head -10"

echo ""
echo "4. 测试TensorFlow GPU基础镜像..."
docker run --rm --gpus all tensorflow/tensorflow:latest-gpu \
    python -c "
import tensorflow as tf
print('TensorFlow版本:', tf.__version__)
print('GPU设备:', tf.config.list_physical_devices('GPU'))
print('CUDA构建:', tf.test.is_built_with_cuda())
"

echo ""
echo "5. 测试自定义镜像GPU..."
docker run --rm --gpus all weibo-sentiment-analysis:gpu-latest \
    python -c "
import tensorflow as tf
print('TensorFlow版本:', tf.__version__)
print('GPU设备:', tf.config.list_physical_devices('GPU'))
print('CUDA构建:', tf.test.is_built_with_cuda())
"

echo ""
echo "6. 检查容器内CUDA库..."
docker run --rm --gpus all weibo-sentiment-analysis:gpu-latest \
    /bin/bash -c "
echo 'CUDA_HOME:' \$CUDA_HOME
echo 'LD_LIBRARY_PATH:' \$LD_LIBRARY_PATH
echo '检查CUDA库文件:'
find /usr/local -name 'libcuda*' 2>/dev/null || echo '未找到libcuda库'
echo '检查cuDNN库文件:'
find /usr -name 'libcudnn*' 2>/dev/null | head -5 || echo '未找到cuDNN库'
" 