# 微博情感分析LSTM - 服务器部署指南

## 服务器配置

- **GPU**: 7x RTX 4090 (176.2GB 显存)
- **CPU**: 112核 AMD EPYC 9354
- **内存**: 420.9GB
- **硬盘**: 5.3TB
- **Docker**: v26.1.0
- **Python**: v3.11.8 (系统版本)
- **PyTorch**: v2.2.2
- **TensorFlow**: v2.16.1

## 快速部署

### 方法一：一键启动（推荐）

```bash
# 给脚本添加执行权限并启动
chmod +x quick_start.sh && ./quick_start.sh
```

### 方法二：手动步骤

```bash
# 1. 创建conda环境
conda create -n weibo-sentiment-analysis-lstm-py310 python=3.10 -y

# 2. 激活环境
conda activate weibo-sentiment-analysis-lstm-py310

# 3. 运行服务器启动脚本
chmod +x server_start.sh && ./server_start.sh
```

## 文件说明

### 核心脚本

- `quick_start.sh` - 一键快速启动脚本
- `server_start.sh` - 服务器完整启动脚本
- `manage_processes.sh` - 进程管理脚本（由server_start.sh自动生成）

### 核心代码

- `src/server_main.py` - 服务器优化版主程序
- `src/server_config.py` - 服务器配置文件
- `src/process2.py` - 原始训练代码
- `monitor_resources.py` - 资源监控脚本（自动生成）

## 性能优化特性

### GPU优化

- ✅ 多GPU并行训练 (MirroredStrategy)
- ✅ 混合精度训练 (Mixed Precision)
- ✅ XLA加速编译
- ✅ GPU内存动态增长
- ✅ CUDA优化器配置

### 模型优化

- ✅ 双向LSTM架构
- ✅ 注意力机制
- ✅ 批量归一化
- ✅ 学习率自适应调整
- ✅ 早停机制

### 训练优化

- ✅ 大批次训练 (批次大小：512 × GPU数量)
- ✅ 数据生成器
- ✅ 多进程数据处理
- ✅ 实时监控

## 监控和管理

### 查看进程状态

```bash
./manage_processes.sh status
```

### 查看训练日志

```bash
# 实时查看训练日志
./manage_processes.sh logs training

# 查看资源监控日志
./manage_processes.sh logs monitor

# 查看TensorBoard日志
./manage_processes.sh logs tensorboard
```

### 停止所有进程

```bash
./manage_processes.sh stop
```

### 访问TensorBoard

在浏览器中访问：`http://localhost:6006`

## 输出文件

### 模型文件

- `models/best_model_{epoch}_{accuracy}.h5` - 最佳模型检查点
- `models/final_optimized_model.h5` - 最终训练模型
- `model_optimized_lstm.png` - 优化后的模型结构图

### 日志文件

- `training.log` - 训练日志
- `monitor.log` - 资源监控日志
- `tensorboard.log` - TensorBoard日志
- `logs/training_log.csv` - 训练历史CSV
- `logs/training_history.pkl` - 训练历史pickle文件

### 数据文件

- `data/output/word_dict.pk` - 词典文件
- `data/output/label_dict.pk` - 标签字典文件
- `logs/confusion_matrix.npy` - 混淆矩阵

## 预期性能

### 训练参数

- **Batch Size**: 512 × 7 = 3584 (全局)
- **Epochs**: 20
- **Learning Rate**: 0.0001
- **LSTM Units**: 256 (第一层), 128 (第二层)
- **Embedding Dim**: 128

### 预期效果

- **训练速度**: 比单GPU快约5-6倍
- **内存利用**: 每块GPU约10-15GB显存使用
- **训练时间**: 预计10-20分钟完成（取决于数据集大小）
- **模型准确率**: 预期达到85%+

## 故障排除

### 常见问题

#### GPU内存不足

```bash
# 检查GPU状态
nvidia-smi

# 减少批次大小（修改 src/server_config.py）
BATCH_SIZE = 256  # 从512改为256
```

#### conda环境问题

```bash
# 重新初始化conda
conda init bash
source ~/.bashrc

# 手动创建环境
conda create -n weibo-sentiment-analysis-lstm-py310 python=3.10 -y
```

#### 依赖包冲突

```bash
# 清理pip缓存
pip cache purge

# 强制重装TensorFlow
pip uninstall tensorflow -y
pip install tensorflow[and-cuda]==2.16.1
```

### 性能调优

#### 如果GPU利用率低

修改 `src/server_config.py` 中的参数：

```python
BATCH_SIZE = 1024  # 增加批次大小
CPU_THREADS = 32   # 增加CPU线程数
```

#### 如果内存不足

```python
BATCH_SIZE = 256   # 减少批次大小
MIXED_PRECISION = True  # 确保开启混合精度
```

## 注意事项

1. **SSH连接**: 程序在后台运行，可以安全断开SSH连接
2. **数据准备**: 确保数据文件 `data/input/all_utf8.csv` 存在
3. **磁盘空间**: 训练过程会产生较多日志和模型文件，确保有足够空间
4. **网络**: TensorBoard需要6006端口，确保端口未被占用

## 联系支持

如遇问题，请检查：

1. 训练日志：`tail -f training.log`
2. 系统资源：`./manage_processes.sh logs monitor`
3. GPU状态：`nvidia-smi`
