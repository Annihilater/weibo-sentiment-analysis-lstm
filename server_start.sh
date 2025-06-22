#!/bin/bash

set -e

echo "=========================================="
echo "微博情感分析LSTM - 服务器启动脚本（GPU版）"
echo "=========================================="

# 服务器配置信息
echo "服务器配置:"
echo "- GPU: 7x RTX 4090 (176.2GB显存)"
echo "- CPU: 112核 AMD EPYC 9354"
echo "- 内存: 420.9GB"
echo "- 硬盘: 5.3TB"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  # 使用7张GPU中的前6张
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