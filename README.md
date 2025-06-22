# 微博情感分析系统

基于LSTM的微博文本情感分析系统，实现了文本的二分类（积极/消极）功能。

## 项目结构

```
weibo-sentiment-analysis-lstm/
├── data/
│   ├── input/          # 输入数据目录
│   │   ├── all.csv     # 原始数据
│   │   └── all_utf8.csv# UTF-8编码的数据
│   └── output/         # 输出数据目录
│       ├── word_dict.pk    # 字符映射字典
│       ├── label_dict.pk   # 标签映射字典
│       └── lstm_model.h5   # 训练好的模型
├── src/
│   ├── config.py           # 配置文件
│   ├── logger.py           # 日志配置
│   ├── main.py            # 主程序
│   ├── process.py         # 数据统计和可视化
│   └── process2.py        # 模型训练和预测
└── requirements.txt    # 项目依赖
```

## 数据处理流程

### 1. 数据加载和预处理 (`load_data` 函数)

#### 输入

- `filepath`: CSV文件路径 (data/input/all_utf8.csv)
- `input_shape`: 输入序列长度 (默认20，实际使用180)

#### 处理步骤

1. **数据读取**
   - 输入：UTF-8编码的CSV文件
   - 输出：包含评论文本(evaluation)和情感标签(label)的DataFrame

2. **标签和词汇提取**
   - 输入：DataFrame
   - 输出：
     - `labels`: 唯一标签列表 ['消极', '积极']
     - `vocabulary`: 唯一评论文本列表

3. **字符级特征构建**
   - 输入：评论文本列表
   - 输出：字符集合（unique characters）

4. **映射字典构建**
   - 输出：
     - `word_dictionary`: 字符到索引的映射
     - `inverse_word_dictionary`: 索引到字符的映射
     - `label_dictionary`: 标签到索引的映射
     - `output_dictionary`: 索引到标签的映射

5. **字典文件保存**
   - 输出位置：`data/output/`目录
     - `word_dict.pk`: 字符映射字典
     - `label_dict.pk`: 标签映射字典

6. **序列处理**
   - 输入：原始评论文本
   - 输出：填充后的数字序列矩阵
   - 处理：将每条评论转换为固定长度的数字序列

7. **标签处理**
   - 输入：原始标签
   - 输出：one-hot编码的标签矩阵

#### 最终输出

- `x`: 处理后的输入特征矩阵
- `y`: one-hot编码的标签矩阵
- `output_dictionary`: 用于预测时转换回原始标签
- `vocab_size`: 词汇表大小
- `label_size`: 标签类别数
- `inverse_word_dictionary`: 用于预测时转换回原始字符

### 2. 模型结构 (`create_lstm` 函数)

#### 输入参数

- `n_units`: LSTM层神经元个数
- `input_shape`: 输入序列长度
- `output_dim`: Embedding层输出维度
- `filepath`: 数据文件路径

#### 模型架构

1. **Embedding层**
   - 功能：将字符索引转换为密集向量
   - 参数：
     - input_dim: 词汇表大小 + 1
     - output_dim: 指定的输出维度
     - input_length: 序列长度
     - mask_zero: True（支持变长序列）

2. **LSTM层**
   - 功能：处理序列数据
   - 参数：
     - units: 神经元个数
     - input_shape: (batch_size, sequence_length)

3. **Dropout层**
   - 功能：防止过拟合
   - 参数：rate = 0.2

4. **Dense层**
   - 功能：输出分类结果
   - 参数：
     - units: 标签类别数
     - activation: 'softmax'

#### 输出

- 编译好的Keras模型
- 模型结构图（保存为`model_lstm.png`）

### 3. 模型训练 (`model_train` 函数)

#### 输入参数

- `input_shape`: 输入序列长度
- `filepath`: 数据文件路径
- `model_save_path`: 模型保存路径

#### 训练过程

1. 加载和预处理数据
2. 划分训练集和测试集（比例9:1）
3. 训练模型：
   - batch_size: 32
   - epochs: 5
   - optimizer: adam
   - loss: categorical_crossentropy

#### 输出

- 保存的模型文件 (lstm_model.h5)
- 测试集上的预测结果和准确率

## 使用说明

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行程序：

```bash
./start.sh
```

## 数据要求

输入CSV文件需要包含以下列：

- `evaluation`: 评论文本
- `label`: 情感标签（'积极' 或 '消极'）

## 环境要求

- Python 3.10
- CUDA支持（可选，用于GPU加速）
- 足够的内存（建议8GB以上）

## 注意事项

1. 确保输入数据为UTF-8编码
2. 模型参数可在 `process2.py` 中调整
3. 可视化结果会自动保存在输出目录
