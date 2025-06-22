#!/bin/bash

set -e

echo "程序: 开始"

# 创建输出目录
mkdir -p data/output

# 屏蔽 macOS 的输入法调试信息
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTHON_IGNORE_DEPRECATION=1

python -m src.main

echo "程序: 完成"