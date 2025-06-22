# 微博情感分析 LSTM 项目

## 数据处理

### 数据编码转换

项目中包含一个数据清洗脚本，用于将 GB2312 编码的数据文件转换为 UTF-8 编码。

#### 使用方法

1. 确保原始数据文件位于 `data` 目录下
2. 运行数据清洗脚本：

```bash
python src/data_processing/clean_data.py
```

脚本会自动：

- 读取 `data/all.csv` 文件（GB2312编码）
- 创建 `data/processed` 目录（如果不存在）
- 生成 `data/processed/all_utf8.csv` 文件（UTF-8编码）

#### 自定义使用

如果需要处理其他文件，可以修改 `src/data_processing/clean_data.py` 中的 `main()` 函数：

```python
def main():
    input_file = 'data/your_input_file.csv'  # 修改输入文件路径
    output_file = 'data/processed/your_output_file.csv'  # 修改输出文件路径
    
    cleaner = DataCleaner(input_file, output_file)
    cleaner.clean()
```

#### 运行测试

运行单元测试：

```bash
python -m unittest src.data_processing.tests.test_clean_data.TestDataCleaner
```
