#!/bin/bash

# 构建GPU基础镜像脚本
echo "开始构建weibo-sentiment-analysis GPU基础镜像..."

# 默认使用修改后的Dockerfile
DOCKERFILE="Dockerfile.gpu"

# 如果第一个参数是 "simple"，使用简化版本
if [ "$1" = "simple" ]; then
    DOCKERFILE="Dockerfile.gpu-simple"
    echo "使用简化版本Dockerfile: $DOCKERFILE"
fi

echo "使用Dockerfile: $DOCKERFILE"

# 构建镜像
docker build -f $DOCKERFILE -t weibo-sentiment-analysis:gpu-latest .

# 检查构建结果
if [ $? -eq 0 ]; then
    echo "✅ GPU基础镜像构建成功！"
    echo "镜像名称: weibo-sentiment-analysis:gpu-latest"
    
    # 显示镜像信息
    echo ""
    echo "镜像信息:"
    docker images | grep weibo-sentiment-analysis
    
    echo ""
    echo "现在可以使用以下命令启动训练:"
    echo "docker compose -f docker docker-compose-tensorflow.yml up"
    
    echo ""
    echo "如果需要调试，可以进入容器:"
    echo "docker run -it --rm --gpus all weibo-sentiment-analysis:gpu-latest bash"
else
    echo "❌ 镜像构建失败"
    echo ""
    echo "故障排除建议:"
    echo "1. 如果是网络问题，尝试使用简化版本:"
    echo "   ./build_gpu_image.sh simple"
    echo ""
    echo "2. 检查Docker是否支持GPU:"
    echo "   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi"
    echo ""
    echo "3. 如果问题持续，请检查网络连接和Docker配置"
    exit 1
fi 