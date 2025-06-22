#!/bin/bash

set -e

echo "=========================================="
echo "微博情感分析LSTM - 服务器启动脚本（GPU版）"
echo "=========================================="

# 动态检测硬件配置
echo "正在检测服务器硬件配置..."

# 检查 nvidia-smi 是否可用
if ! command -v nvidia-smi &> /dev/null
then
    echo "警告: nvidia-smi 命令未找到。无法检测 GPU 信息。"
    GPU_COUNT=0
else
    # 获取 GPU 数量
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    # 获取 GPU 型号 (假设所有 GPU 型号相同)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    # 获取单个 GPU 的总显存 (以 MiB 为单位)
    GPU_MEM_PER_CARD_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    # 计算总显存 (以 GB 为单位)
    TOTAL_GPU_MEM_GB=$(awk "BEGIN {printf \"%.1f\", $GPU_COUNT * $GPU_MEM_PER_CARD_MB / 1024}")
fi

# 获取 CPU 核心数
CPU_CORES=$(nproc)
# 获取 CPU 型号
CPU_NAME=$(grep "model name" /proc/cpuinfo | uniq | cut -d ':' -f 2- | sed 's/^[ \t]*//' || echo "未知")
# 获取总内存 (以 GB 为单位)
TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
# 获取根目录总磁盘空间
TOTAL_DISK_GB=$(df -h / | awk 'NR==2 {print $2}')

# 动态生成服务器配置信息
echo "=========================================="
echo "服务器配置 (自动检测):"
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "- GPU: ${GPU_COUNT}x ${GPU_NAME} (${TOTAL_GPU_MEM_GB}GB 显存)"
else
    echo "- GPU: 未检测到或不可用"
fi
echo "- CPU: ${CPU_CORES}核 ${CPU_NAME}"
echo "- 内存: ${TOTAL_RAM_GB}GB"
echo "- 硬盘 (根分区): ${TOTAL_DISK_GB}"
echo "=========================================="

# 动态设置要使用的GPU并导出环境变量
if [ "$GPU_COUNT" -gt 0 ]; then
    # 生成逗号分隔的 GPU 索引列表 (例如: 0,1,2,3)
    ALL_GPUS=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES=${ALL_GPUS}
    echo "已自动设置环境变量: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    # 如果没有GPU，则取消设置该变量
    unset CUDA_VISIBLE_DEVICES
    echo "未检测到GPU，不设置 CUDA_VISIBLE_DEVICES"
fi

export TF_FORCE_GPU_ALLOW_GROWTH=true  # 允许GPU内存动态增长
export TF_CPP_MIN_LOG_LEVEL=0  # 显示所有日志

# 记录开始时间
start_time=$(date +%s)

# 检查并激活conda环境
echo "正在激活conda环境..."
if ! conda info --envs | grep -q "weibo-sentiment-analysis-lstm-py310"; then
    echo "错误: conda环境 'weibo-sentiment-analysis-lstm-py310' 不存在"
    echo "请先运行: conda create -n weibo-sentiment-analysis-lstm-py310 python=3.10 -y"
    exit 1
fi

# 初始化conda（确保conda命令可用）
eval "$(conda shell.bash hook)"
conda activate weibo-sentiment-analysis-lstm-py310

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

# 设置GPU环境
echo "=========================================="
echo "设置GPU环境..."
echo "=========================================="

# 给setup_gpu.sh添加执行权限并运行
chmod +x setup_gpu.sh
./setup_gpu.sh

# 加载GPU环境变量
if [ -f "gpu_env.sh" ]; then
    echo "加载GPU环境变量..."
    source gpu_env.sh
else
    echo "警告: gpu_env.sh不存在，GPU可能无法正常工作"
fi

# 设置其他环境变量
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd):$PYTHONPATH
export TF_GPU_THREAD_MODE=gpu_private
export OMP_NUM_THREADS=16
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=16

# 创建必要的目录
echo "创建输出目录..."
mkdir -p data/output
mkdir -p logs/tensorboard
mkdir -p models

# 检查数据文件
if [ ! -f "data/input/all_utf8.csv" ] && [ ! -f "data/weibo_senti_100k.csv" ]; then
    echo "警告: 未找到数据文件 data/input/all_utf8.csv 或 data/weibo_senti_100k.csv"
    echo "请确保数据文件存在后再运行"
fi

# 显示系统资源信息
echo "=========================================="
echo "系统资源信息:"
if command -v free &> /dev/null; then
    echo "可用内存: $(free -h | awk '/^Mem:/ {print $7}')"
else
    echo "可用内存: $(vm_stat | grep "Pages free" | awk '{print $3 * 4096 / 1024 / 1024 / 1024 " GB"}')"
fi
echo "可用磁盘空间: $(df -h . | awk 'NR==2 {print $4}')"
if command -v nproc &> /dev/null; then
    echo "CPU核心数: $(nproc)"
else
    echo "CPU核心数: $(sysctl -n hw.ncpu)"
fi

# 启动TensorBoard（前台运行）
echo "=========================================="
echo "启动TensorBoard..."
echo "在新的终端窗口运行以下命令查看TensorBoard:"
echo "tensorboard --logdir=./logs/tensorboard --host=0.0.0.0 --port=6006"
echo "然后访问: http://localhost:6006"

# 修改src/server_config.py以允许CPU训练
if [ -f "src/server_config.py" ]; then
    echo "修改配置文件以支持CPU训练..."
    # 备份原始文件
    cp src/server_config.py src/server_config.py.bak
    
    # 使用sed修改配置文件，允许在没有GPU时也能运行
    sed -i.bak 's/return False/logger.warning("将使用CPU进行训练"); return True/g' src/server_config.py
    echo "配置文件修改完成"
fi

# 主程序启动
echo "=========================================="
echo "开始运行主程序（前台模式）..."
echo "=========================================="

# 在前台运行Python脚本，便于观察日志
echo "开始运行服务器..."
python src/server_main.py

# 计算运行时间
end_time=$(date +%s)
runtime=$((end_time - start_time))

# 输出运行时间
echo "=========================================="
echo "程序运行完成！"
echo "总运行时间: ${runtime}秒"
echo "==========================================" 