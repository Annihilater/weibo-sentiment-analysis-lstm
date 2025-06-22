import pickle

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

from src.config import settings
from src.logger import logger


def load_data(filepath: str, input_shape: int = 20):
    """
    加载数据，返回训练集和测试集
    ['evaluation'] is feature, ['label'] is label
    :param filepath: 数据集路径
    :param input_shape: 输入序列长度
    :return:
    """
    df = pd.read_csv(filepath, encoding="utf-8")

    # 标签及词汇表
    labels, vocabulary = list(df["label"].unique()), list(df["evaluation"].unique())

    # 构造字符级别的特征
    string = ""
    for word in vocabulary:
        string += word

    vocabulary = set(string)

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
    output_dictionary = {i: labels for i, labels in enumerate(labels)}

    # 词汇表大小
    vocab_size = len(word_dictionary.keys())
    # 标签类别数量
    label_size = len(label_dictionary.keys())

    # 序列填充，按input_shape填充，长度不足的按0补充
    x = [[word_dictionary[word] for word in sent] for sent in df["evaluation"]]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding="post", value=0)
    y = [[label_dictionary[sent]] for sent in df["label"]]
    """
    to_categorical用于将标签转化为形如(nb_samples, nb_classes)
    的二值序列。
    假设num_classes = 10。
    如将[1, 2, 3,……4]转化成：
    [[0, 1, 0, 0, 0, 0, 0, 0]
     [0, 0, 1, 0, 0, 0, 0, 0]
     [0, 0, 0, 1, 0, 0, 0, 0]
    ……
    [0, 0, 0, 0, 1, 0, 0, 0]]
    """
    y = [to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(_[0]) for _ in y])

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary


def create_lstm(n_units: int, input_shape: int, output_dim: int, vocab_size: int, label_size: int):
    """
    创建深度学习模型，Embedding + LSTM + Softmax
    :param n_units: LSTM层神经元个数
    :param input_shape: 输入序列长度
    :param output_dim: Embedding层输出维度
    :param vocab_size: 词汇表大小
    :param label_size: 标签类别数
    :return:
    """
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size + 1,
            output_dim=output_dim,
            input_length=input_shape
        )
    )
    model.add(LSTM(n_units))
    model.add(Dropout(0.2))
    model.add(Dense(label_size, activation="softmax"))
    
    # 编译模型
    model.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
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
    :return:
    """
    logger.info(f"开始训练模型，输入序列长度: {input_shape}")
    logger.info(f"数据集路径: {filepath}")
    logger.info(f"模型保存路径: {model_save_path}")

    logger.info("开始加载数据...")
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = (
        load_data(filepath, input_shape)
    )
    logger.info(f"数据加载完成。数据集大小: {len(x)}，词汇表大小: {vocab_size}，标签数: {label_size}")

    logger.info("将数据集分为训练集和测试集，占比为9：1")
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.1, random_state=42
    )
    logger.info(f"训练集大小: {len(train_x)}，测试集大小: {len(test_x)}")

    # 模型输入参数，需要根据自己需要调整
    n_units = 100
    batch_size = 32
    epochs = 5
    output_dim = 20

    logger.info("开始创建模型...")
    # 模型训练
    lstm_model = create_lstm(n_units, input_shape, output_dim, vocab_size, label_size)
    
    logger.info("开始训练模型...")
    # 添加训练回调函数
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
    
    # 确保日志目录存在
    import os
    os.makedirs('./logs/tensorboard', exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        TensorBoard(log_dir='./logs/tensorboard', histogram_freq=1),  # TensorBoard可视化
        ModelCheckpoint(
            filepath=model_save_path,  # 使用传入的模型保存路径
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),  # 保存最佳模型
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1
        )  # 早停策略
    ]
    
    # 训练模型
    history = lstm_model.fit(
        train_x, 
        train_y, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1,  # 使用10%的训练数据作为验证集
        callbacks=callbacks,
        verbose=1  # 显示进度条
    )

    # 保存最终模型
    logger.info("保存模型...")
    lstm_model.save(model_save_path)
    logger.info(f"模型已保存到: {model_save_path}")

    # 测试条数
    logger.info("开始在测试集上进行评估...")
    N = test_x.shape[0]
    predict = []
    label = []
    for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
        sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model.predict(test_x[start:end], verbose=0)

        label_predict = output_dictionary[np.argmax(y_predict[0])]
        label_true = output_dictionary[np.argmax(test_y[start:end])]
        
        if start % 1000 == 0:  # 每1000条打印一次进度
            logger.info(f"评估进度: {start}/{N}")
            logger.info(f"示例预测 - 文本: {''.join(sentence)}")
            logger.info(f"真实标签: {label_true}, 预测标签: {label_predict}")
        
        predict.append(label_predict)
        label.append(label_true)

    # 预测准确率
    acc = accuracy_score(predict, label)
    logger.info("模型在测试集上的准确率: %.4f" % acc)
    
    return history, acc
