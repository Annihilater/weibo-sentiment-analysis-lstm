#!/bin/bash

echo "=== 简单GPU测试 ==="

# 重新构建镜像
echo "重新构建GPU镜像..."
./build_gpu_image.sh

if [ $? -ne 0 ]; then
    echo "❌ 镜像构建失败"
    exit 1
fi

echo ""
echo "测试新构建的镜像..."

# 测试GPU检测
docker run --rm --gpus all weibo-sentiment-analysis:gpu-latest \
    python -c "
import os
print('=== 环境变量 ===')
print('CUDA_HOME:', os.environ.get('CUDA_HOME', 'None'))
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'None'))
print()

print('=== TensorFlow GPU检测 ===')
import tensorflow as tf
print('TensorFlow版本:', tf.__version__)
print('CUDA构建:', tf.test.is_built_with_cuda())

try:
    gpus = tf.config.list_physical_devices('GPU')
    print('检测到GPU数量:', len(gpus))
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu}')
        
    if len(gpus) > 0:
        print('✅ GPU检测成功！')
    else:
        print('❌ 未检测到GPU')
        
except Exception as e:
    print('❌ GPU检测出错:', str(e))
" 