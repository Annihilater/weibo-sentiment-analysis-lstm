#!/bin/bash

echo "=== 修复NVIDIA Docker支持 ==="

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then 
    echo "请使用sudo运行此脚本"
    echo "sudo ./fix_nvidia_docker.sh"
    exit 1
fi

echo "1. 检查当前Docker配置..."
if [ -f /etc/docker/daemon.json ]; then
    echo "当前daemon.json内容:"
    cat /etc/docker/daemon.json
else
    echo "daemon.json不存在"
fi

echo ""
echo "2. 安装/更新NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

echo ""
echo "3. 配置Docker daemon..."
nvidia-ctk runtime configure --runtime=docker

echo ""
echo "4. 重启Docker服务..."
systemctl restart docker

echo ""
echo "5. 验证安装..."
docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu20.04 nvidia-smi

if [ $? -eq 0 ]; then
    echo "✅ NVIDIA Docker支持修复成功！"
    echo ""
    echo "现在可以运行:"
    echo "chmod +x test_gpu_simple.sh"
    echo "./test_gpu_simple.sh"
else
    echo "❌ 验证失败，请检查安装"
fi 