import multiprocessing
import pickle
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Embedding, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from src.config import settings
from src.logger import logger


def process_text(args: Tuple[str, Dict[str, int]]) -> List[int]:
    """
    处理单条文本数据
    :param args: (文本, 词典)的元组
    :return: 处理后的数字序列
    """
    sent, word_dict = args
    return [word_dict[word] for word in str(sent)]


class DataGenerator(Sequence):
    """
    数据生成器，用于批量生成训练数据
    """

    def __init__(self, x, y, batch_size=128, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.x))
        indexes = self.indexes[start_idx:end_idx]
        return self.x[indexes], self.y[indexes]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def load_data(
    filepath: str, input_shape: int = 20
) -> Tuple[np.ndarray, np.ndarray, Dict, int, int, Dict]:
    """
    加载数据，返回训练集和测试集
    ['evaluation'] is feature, ['label'] is label
    :param filepath: 数据集路径
    :param input_shape: 输入序列长度
    :return: (x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary)
    """
    logger.info("开始读取数据...")
    df = pd.read_csv(filepath, encoding="utf-8")

    # 检查标签分布
    label_counts = df["label"].value_counts()
    logger.info(f"标签分布:\n{label_counts}")

    # 标签及词汇表
    labels, vocabulary = list(df["label"].unique()), list(df["evaluation"].unique())

    # 构造字符级别的特征
    string = ""
    for word in vocabulary:
        string += str(word)  # 确保word是字符串类型

    vocabulary = set(string)
    logger.info(f"词汇表大小: {len(vocabulary)}")

    # 字典列表
    word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
    word_dict_path = f"{settings.OUTPUT_DIR_PATH}/word_dict.pk"
    with open(word_dict_path, "wb") as f:
        pickle.dump(word_dictionary, f)

    inverse_word_dictionary = {i + 1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}
    label_dict_path = f"{settings.OUTPUT_DIR_PATH}/label_dict.pk"
    with open(label_dict_path, "wb") as f:
        pickle.dump(label_dictionary, f)
    output_dictionary = {i: label for i, label in enumerate(labels)}

    # 词汇表大小
    vocab_size = len(word_dictionary.keys())
    # 标签类别数量
    label_size = len(label_dictionary.keys())

    logger.info("开始处理文本数据...")
    # 使用多进程处理文本数据
    texts = df["evaluation"].tolist()
    # 准备参数列表
    args_list = [(text, word_dictionary) for text in texts]
    chunk_size = len(texts) // (multiprocessing.cpu_count() * 4)  # 根据CPU核心数确定chunk大小
    
    with multiprocessing.Pool() as pool:
        x = pool.map(process_text, args_list, chunksize=chunk_size)
    
    x = pad_sequences(maxlen=input_shape, sequences=x, padding="post", value=0)
    
    # 处理标签
    y = np.array([label_dictionary[label] for label in df["label"]])
    y = to_categorical(y, num_classes=label_size)

    logger.info("数据处理完成")
    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary


def create_lstm(
    n_units: int, input_shape: int, output_dim: int, vocab_size: int, label_size: int
) -> Sequential:
    """
    创建深度学习模型，Embedding + LSTM + Softmax
    :param n_units: LSTM层神经元个数
    :param input_shape: 输入序列长度
    :param output_dim: Embedding层输出维度
    :param vocab_size: 词汇表大小
    :param label_size: 标签类别数
    :return: 编译好的模型
    """
    model = Sequential(
        [
            # 增大Embedding维度到100
            Embedding(
                input_dim=vocab_size + 1,
                output_dim=output_dim,
                input_length=input_shape,
                mask_zero=True,
            ),
            # 添加BatchNormalization层
            BatchNormalization(),
            # 使用双向LSTM
            LSTM(n_units, return_sequences=True),
            Dropout(0.3),
            LSTM(n_units // 2),
            Dropout(0.3),
            BatchNormalization(),
            Dense(n_units // 4, activation="relu"),
            Dropout(0.3),
            # 确保输出层使用softmax激活函数
            Dense(label_size, activation="softmax"),
        ]
    )

    # 使用Adam优化器，设置较小的学习率
    optimizer = Adam(learning_rate=0.0001)

    # 编译模型，使用多线程
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # 构建模型
    model.build((None, input_shape))

    # 输出模型信息
    model.summary()

    # 保存模型结构图
    plot_model(model, to_file="./model_lstm.png", show_shapes=True)

    return model


def model_train(input_shape: int, filepath: str, model_save_path: str):
    """
    模型训练
    :param input_shape: 输入序列长度
    :param filepath: 数据集路径
    :param model_save_path: 模型保存路径
    :return: (训练历史, 测试集准确率)
    """
    logger.info(f"开始训练模型，输入序列长度: {input_shape}")
    logger.info(f"数据集路径: {filepath}")
    logger.info(f"模型保存路径: {model_save_path}")

    logger.info("开始加载数据...")
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = (
        load_data(filepath, input_shape)
    )
    logger.info(
        f"数据加载完成。数据集大小: {len(x)}，词汇表大小: {vocab_size}，标签数: {label_size}"
    )

    logger.info("将数据集分为训练集和测试集，占比为9：1")
    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y,  # 使用stratify确保划分后标签分布一致
    )
    logger.info(f"训练集大小: {len(train_x)}，测试集大小: {len(test_x)}")

    # 模型输入参数，需要根据自己需要调整
    n_units = 128  # 增加LSTM单元数
    batch_size = 256  # 增大batch_size以提升训练速度
    epochs = 10  # 增加训练轮数
    output_dim = 100  # 增大Embedding维度

    logger.info("开始创建模型...")
    # 模型训练
    lstm_model = create_lstm(n_units, input_shape, output_dim, vocab_size, label_size)

    logger.info("开始训练模型...")

    # 确保日志目录存在
    import os

    os.makedirs("./logs/tensorboard", exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 创建数据生成器
    train_generator = DataGenerator(train_x, train_y, batch_size=batch_size)
    val_generator = DataGenerator(test_x, test_y, batch_size=batch_size)

    callbacks = [
        TensorBoard(
            log_dir="./logs/tensorboard",
            histogram_freq=1,
            update_freq="batch",  # 实时更新训练指标
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,  # 增加耐心值
            verbose=1,
            restore_best_weights=True,  # 恢复最佳权重
        ),
    ]

    # 使用数据生成器训练模型
    history = lstm_model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("加载最佳模型进行评估...")
    best_model = load_model(model_save_path)

    # 在测试集上进行评估
    logger.info("开始在测试集上进行评估...")
    test_loss, test_accuracy = best_model.evaluate(
        test_x,
        test_y,
        batch_size=batch_size,
        verbose=0,
    )
    logger.info(f"测试集损失: {test_loss:.4f}")
    logger.info(f"测试集准确率: {test_accuracy:.4f}")

    # 在测试集上进行预测
    y_pred = best_model.predict(
        test_x,
        batch_size=batch_size,
        verbose=0,
    )
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(test_y, axis=1)

    # 输出详细的分类报告
    logger.info("\n分类报告:")
    logger.info(
        classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=list(output_dictionary.values()),
        )
    )

    # 示例预测
    N = min(5, len(test_x))  # 展示前5个预测结果
    logger.info("\n示例预测:")
    for i in range(N):
        sentence = [inverse_word_dictionary[j] for j in test_x[i] if j != 0]
        true_label = output_dictionary[np.argmax(test_y[i])]
        pred_label = output_dictionary[np.argmax(y_pred[i])]
        logger.info(f"文本: {''.join(sentence)}")
        logger.info(f"真实标签: {true_label}, 预测标签: {pred_label}")
        logger.info("---")

    return history, test_accuracy
