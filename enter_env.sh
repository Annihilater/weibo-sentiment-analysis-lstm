#!/bin/bash

set -e

echo "进入环境: 开始"

# 初始化conda（确保conda命令可用）
eval "$(conda shell.bash hook)"
conda activate weibo-sentiment-analysis-lstm-py310
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements-gpu.txt
echo "当前Python版本: $(python --version)"
echo "当前conda环境: $CONDA_DEFAULT_ENV"
# 安装plot_model所需的依赖
echo "安装模型可视化依赖..."
pip install pydot
if command -v apt-get &> /dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y graphviz
elif command -v yum &> /dev/null; then
    sudo yum install -y graphviz
elif command -v brew &> /dev/null; then
    brew install graphviz
else
    echo "警告: 无法安装graphviz，模型结构图可能无法生成"
fi

echo "进入环境: 完成"