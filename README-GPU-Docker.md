# GPU Docker 训练环境使用指南

## 概述

本项目提供了基于 TensorFlow GPU 的 Docker 环境，支持多卡GPU训练。

## 快速开始

### 1. 构建基础镜像（首次使用）

```bash
# 设置执行权限
chmod +x build_gpu_image.sh

# 构建基础镜像
./build_gpu_image.sh
```

### 2. 启动训练

```bash
# 启动GPU训练容器
docker compose -f docker docker-compose-tensorflow.yml up
```

### 3. 后台运行

```bash
# 后台运行
docker compose -f docker docker-compose-tensorflow.yml up -d

# 查看日志
docker compose -f docker docker-compose-tensorflow.yml logs -f
```

## 环境配置

### GPU 支持

- **基础镜像**: `tensorflow/tensorflow:latest-gpu`
- **支持GPU数**: 8卡 (可在docker compose.yml中调整)
- **CUDA版本**: 与TensorFlow最新版本兼容

### 预装依赖

- TensorFlow GPU (最新版)
- Keras
- NumPy, Pandas, Scikit-learn
- Matplotlib (含中文字体支持)
- pydot, graphviz (模型可视化)
- 所有requirements.txt中的依赖

## 目录结构

```
/workspace/          # 容器工作目录
├── src/            # 源代码
├── data/           # 数据目录
│   ├── input/      # 输入数据
│   └── output/     # 输出结果
└── logs/           # 训练日志
    └── tensorboard/ # TensorBoard日志
```

## 常用命令

### 镜像管理

```bash
# 查看镜像
docker images | grep weibo-sentiment

# 删除镜像
docker rmi weibo-sentiment-analysis:gpu-latest

# 重新构建
./build_gpu_image.sh
```

### 容器管理

```bash
# 停止容器
docker compose -f docker docker-compose-tensorflow.yml down

# 进入容器调试
docker compose -f docker docker-compose-tensorflow.yml exec weibo-sentiment-tensorflow bash

# 查看GPU状态
docker compose -f docker docker-compose-tensorflow.yml exec weibo-sentiment-tensorflow nvidia-smi
```

### 监控训练

```bash
# 实时查看日志
docker compose -f docker docker-compose-tensorflow.yml logs -f

# TensorBoard (在容器内)
tensorboard --logdir=/workspace/logs/tensorboard --host=0.0.0.0
```

## 性能优化

### GPU 内存管理

- 启用了GPU内存动态增长
- 可在代码中设置内存限制

### 多GPU训练

- 环境变量 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- 可根据实际GPU数量调整

## 故障排除

### GPU不可用

```bash
# 检查宿主机GPU
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 检查容器内GPU
docker compose -f docker docker-compose-tensorflow.yml exec weibo-sentiment-tensorflow python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 依赖问题

如果需要添加新依赖：

1. 修改 `requirements.txt`
2. 重新构建镜像: `./build_gpu_image.sh`

## 注意事项

1. **首次构建较慢**: 需要下载基础镜像和安装依赖
2. **后续启动很快**: 使用已构建的基础镜像
3. **代码修改实时生效**: 通过volume挂载，无需重建镜像
4. **数据持久化**: data和logs目录持久化到宿主机
