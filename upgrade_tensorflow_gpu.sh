#!/bin/bash

echo "=== 升级TensorFlow GPU版本 ==="

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "请先激活虚拟环境:"
    echo "source enter_env.sh"
    exit 1
fi

echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"

# 检查当前TensorFlow版本
echo ""
echo "当前TensorFlow版本:"
python -c "import tensorflow as tf; print('版本:', tf.__version__); print('CUDA构建:', tf.test.is_built_with_cuda())"

echo ""
echo "1. 卸载当前TensorFlow..."
pip uninstall tensorflow -y

echo ""
echo "2. 清理缓存..."
pip cache purge

echo ""
echo "3. 安装支持GPU的TensorFlow..."
# 安装兼容CUDA 12.x的版本
pip install tensorflow==2.15.0 --no-cache-dir

echo ""
echo "4. 验证新安装的TensorFlow..."
python -c "
import tensorflow as tf
print('=== TensorFlow信息 ===')
print('版本:', tf.__version__)
print('CUDA构建:', tf.test.is_built_with_cuda())

print()
print('=== GPU检测 ===')
gpus = tf.config.list_physical_devices('GPU')
print('检测到GPU数量:', len(gpus))
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')

if len(gpus) > 0:
    print()
    print('✅ TensorFlow GPU安装成功！')
else:
    print()
    print('❌ TensorFlow仍无法检测GPU')
    print('可能需要安装不同版本的TensorFlow')
"

echo ""
echo "5. 如果成功，现在可以运行:"
echo "   ./start_native.sh" 