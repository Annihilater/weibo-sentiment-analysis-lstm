#!/bin/bash

# 设置错误时退出
set -e

# 显示执行的命令
set -x

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到项目根目录
cd "${SCRIPT_DIR}"

# 执行评估脚本
echo "开始执行模型评估..."
python src/server_evaluate.py

# 输出完成信息
echo "==========================================
评估完成！
==========================================
" 