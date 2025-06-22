# GPU训练使用说明

## 前提条件

1. **NVIDIA Docker支持**

   ```bash
   # 检查nvidia-docker是否安装
   docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
   ```

2. **Docker Compose支持GPU**
   确保Docker Compose版本 >= 1.28.0

## 使用方法

### 方式1：使用便捷启动脚本（推荐）

我们提供了一个便捷的启动脚本 `start_gpu.sh`，支持多种运行模式：

```bash
# 前台启动训练（默认）
./start_gpu.sh

# 后台启动训练
./start_gpu.sh -d

# 交互模式（进入容器shell）
./start_gpu.sh -i

# 查看训练日志
./start_gpu.sh -l

# 停止训练服务
./start_gpu.sh -s

# 显示帮助
./start_gpu.sh -h
```

**脚本特性：**
- 自动检查环境依赖（Docker、NVIDIA Docker、数据文件等）
- 彩色输出，状态清晰
- 显示GPU信息
- 自动创建必要目录
- 智能错误处理

### 方式2：直接使用Docker Compose

```bash
# 启动GPU训练
docker compose -f docker docker-compose-tensorflow.yml up --build

# 后台运行
docker compose -f docker docker-compose-tensorflow.yml up --build -d

# 查看日志
docker compose -f docker docker-compose-tensorflow.yml logs -f

# 停止服务
docker compose -f docker docker-compose-tensorflow.yml down
```

### 方式3：交互式运行

```bash
# 启动容器并进入交互模式
docker compose -f docker docker-compose-tensorflow.yml run --rm weibo-sentiment-tensorflow bash

# 在容器内手动运行
pip install -r requirements.txt
python src/main.py
```

## 配置说明

### GPU配置

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`: 使用所有8张GPU
- 如需指定特定GPU，可修改为 `CUDA_VISIBLE_DEVICES=0,1`

### 内存配置

- 限制内存使用：32GB
- 预留内存：16GB
- 可根据服务器配置调整

### 卷挂载

- `./:/workspace`: 整个项目目录
- `./data:/workspace/data`: 数据目录
- `./logs:/workspace/logs`: 日志目录

## 性能监控

### 查看GPU使用情况

```bash
# 在宿主机上查看
nvidia-smi

# 在容器内查看
docker exec -it weibo-sentiment-tensorflow nvidia-smi
```

### 查看训练日志

```bash
# 实时查看日志
docker compose -f docker docker-compose-tensorflow.yml logs -f weibo-sentiment-tensorflow

# 查看TensorBoard
docker compose -f docker docker-compose-tensorflow.yml exec weibo-sentiment-tensorflow tensorboard --logdir=/workspace/logs/tensorboard --host=0.0.0.0 --port=6006
```

## 故障排除

### 常见问题

1. **GPU不可用**
   ```bash
   # 检查nvidia-docker运行时
   docker info | grep nvidia
   
   # 测试GPU访问
   docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
   ```

2. **内存不足**
   - 减少batch_size
   - 调整docker compose中的内存限制
   - 使用单GPU训练

3. **权限问题**
   ```bash
   # 确保数据目录权限正确
   sudo chown -R $USER:$USER data/ logs/
   ```

## 优化建议

1. **多GPU训练**
   - TensorFlow会自动检测并使用可用GPU
   - 可在代码中配置分布式训练策略

2. **数据预处理**
   - 使用Docker卷缓存预处理后的数据
   - 避免重复的数据处理

3. **模型保存**
   - 模型文件会自动保存到 `./data/output/` 目录
   - 在宿主机上持久化存储
