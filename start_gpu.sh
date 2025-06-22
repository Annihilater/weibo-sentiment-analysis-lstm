#!/bin/bash

# GPU训练启动脚本
# 使用TensorFlow官方GPU Docker镜像进行训练

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    print_success "Docker已安装"
}

# 检查Docker Compose是否安装
check_docker_compose() {
    if ! command -v docker compose &> /dev/null; then
        print_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    print_success "Docker Compose已安装"
}

# 检查NVIDIA Docker支持
check_nvidia_docker() {
    print_info "检查NVIDIA Docker支持..."
    if docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_success "NVIDIA Docker支持正常"
    else
        print_error "NVIDIA Docker支持异常，请检查nvidia-docker2是否正确安装"
        print_info "参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
}

# 检查数据文件
check_data_files() {
    print_info "检查数据文件..."
    if [ ! -f "data/input/all_utf8.csv" ]; then
        print_warning "数据文件 data/input/all_utf8.csv 不存在"
        print_info "请确保数据文件已放置在 data/input/ 目录下"
        read -p "是否继续启动？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "数据文件存在"
    fi
}

# 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."
    mkdir -p data/input data/output logs/tensorboard
    print_success "目录创建完成"
}

# 显示GPU信息
show_gpu_info() {
    print_info "服务器GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s (内存: %s MB 总计, %s MB 已用, %s MB 可用)\n", $1, $2, $3, $4, $5}'
}

# 显示使用说明
show_usage() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -d, --detach     后台运行"
    echo "  -i, --interactive 交互模式"
    echo "  -l, --logs       查看日志"
    echo "  -s, --stop       停止服务"
    echo "  -h, --help       显示帮助"
    echo ""
    echo "示例:"
    echo "  $0              # 前台启动训练"
    echo "  $0 -d           # 后台启动训练"
    echo "  $0 -i           # 交互模式"
    echo "  $0 -l           # 查看训练日志"
    echo "  $0 -s           # 停止训练"
}

# 启动训练
start_training() {
    local detach_flag=""
    if [ "$1" = "detach" ]; then
        detach_flag="-d"
        print_info "后台启动GPU训练..."
    else
        print_info "前台启动GPU训练..."
    fi
    
    docker compose -f docker-compose-tensorflow.yml up --build $detach_flag
    
    if [ "$1" = "detach" ]; then
        print_success "训练已在后台启动"
        print_info "使用 '$0 -l' 查看训练日志"
        print_info "使用 '$0 -s' 停止训练"
    fi
}

# 交互模式
interactive_mode() {
    print_info "启动交互模式..."
    docker compose -f docker-compose-tensorflow.yml run --rm weibo-sentiment-tensorflow bash
}

# 查看日志
show_logs() {
    print_info "显示训练日志..."
    docker compose -f docker-compose-tensorflow.yml logs -f weibo-sentiment-tensorflow
}

# 停止服务
stop_service() {
    print_info "停止GPU训练服务..."
    docker compose -f docker-compose-tensorflow.yml down
    print_success "服务已停止"
}

# 主函数
main() {
    print_info "=== 微博情感分析 GPU训练启动器 ==="
    
    # 解析命令行参数
    case "${1:-}" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--logs)
            show_logs
            exit 0
            ;;
        -s|--stop)
            stop_service
            exit 0
            ;;
        -i|--interactive)
            check_docker
            check_docker_compose
            check_nvidia_docker
            create_directories
            interactive_mode
            exit 0
            ;;
        -d|--detach)
            # 环境检查
            check_docker
            check_docker_compose
            check_nvidia_docker
            check_data_files
            create_directories
            show_gpu_info
            start_training "detach"
            exit 0
            ;;
        "")
            # 默认前台启动
            # 环境检查
            check_docker
            check_docker_compose
            check_nvidia_docker
            check_data_files
            create_directories
            show_gpu_info
            start_training
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 