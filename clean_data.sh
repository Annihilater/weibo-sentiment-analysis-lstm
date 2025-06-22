#!/bin/bash

set -e

echo "数据清洗: 开始"

# 创建输出目录
mkdir -p data/output

# 运行数据清洗脚本
python src/data_processing/clean_data.py

echo "数据清洗: 完成"