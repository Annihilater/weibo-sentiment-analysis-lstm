#!/bin/bash

set -e

echo "程序: 开始"

# 创建输出目录
mkdir -p data/output

python src/main.py

echo "程序: 完成"