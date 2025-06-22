#!/bin/bash

# 快速启动脚本 - 服务器版本
# 使用方法: chmod +x quick_start.sh && ./quick_start.sh

set -e

echo "🚀 微博情感分析LSTM - 快速启动脚本"
echo "========================================"

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: conda命令未找到，请先安装Anaconda/Miniconda"
    exit 1
fi

# 创建并激活虚拟环境
ENV_NAME="weibo-sentiment-analysis-lstm-py310"
echo "📦 创建conda环境: $ENV_NAME"

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "✅ 环境已存在，直接激活"
else
    echo "🔧 创建新环境..."
    conda create -n $ENV_NAME python=3.10 -y
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "✅ 当前环境: $CONDA_DEFAULT_ENV"
echo "🐍 Python版本: $(python --version)"

# 给启动脚本添加执行权限
chmod +x server_start.sh

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU状态检查:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  警告: 未找到nvidia-smi命令"
fi

echo "========================================"
echo "🎯 准备就绪！即将启动训练程序..."
echo "========================================"

# 运行主启动脚本
./server_start.sh

echo "🎉 快速启动完成！" 