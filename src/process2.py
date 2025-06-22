import pickle

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.logger import logger


def load_data(filepath: str, input_shape: int = 20):
    """
    加载数据，返回训练集和测试集
    ['evaluation'] is feature, ['label'] is label
    :param filepath: 数据集路径
    :param input_shape: 输入序列长度
    :return:
    """
    df = pd.read_csv(filepath, encoding="gbk")

    # 标签及词汇表
    labels, vocabulary = list(df["label"].unique()), list(df["evaluation"].unique())

    # 构造字符级别的特征
    string = ""
    for word in vocabulary:
        string += word

    vocabulary = set(string)

    # 字典列表
    word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
    with open("word_dict.pk", "wb") as f:
        pickle.dump(word_dictionary, f)
    inverse_word_dictionary = {i + 1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}
    with open("label_dict.pk", "wb") as f:
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


def create_lstm(n_units: int, input_shape: int, output_dim: int, filepath: str):
    """
    创建深度学习模型，Embedding + LSTM + Softmax
    :param n_units: LSTM层神经元个数
    :param input_shape: 输入序列长度
    :param output_dim: Embedding层输出维度
    :param filepath: 数据集路径
    :return:
    """
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = (
        load_data(filepath)
    )
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size + 1,
            output_dim=output_dim,
            input_length=input_shape,
            mask_zero=True,
        )
    )
    model.add(LSTM(n_units, input_shape=(x.shape[0], x.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(label_size, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    plot_model(model, to_file="./model_lstm.png", show_shapes=True)
    # 输出模型信息
    model.summary()

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

    logger.info("将数据集分为训练集和测试集，占比为9：1")
    # input_shape=100
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = (
        load_data(filepath, input_shape)
    )
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    # 模型输入参数，需要根据自己需要调整
    n_units = 100
    batch_size = 32
    epochs = 5
    output_dim = 20

    # 模型训练
    lstm_model = create_lstm(n_units, input_shape, output_dim, filepath)
    lstm_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    # 模型保存
    lstm_model.save(model_save_path)

    # 测试条数
    N = test_x.shape[0]
    predict = []
    label = []
    for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
        logger.info(f"start:{start}, end:{end}")
        sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model.predict(test_x[start:end])

        logger.info("y_predict:", y_predict)
        label_predict = output_dictionary[np.argmax(y_predict[0])]
        label_true = output_dictionary[np.argmax(test_y[start:end])]
        
        logger.info(f"label_predict:{label_predict}, label_true:{label_true}")
        # 输出预测结果
        logger.info("".join(sentence), label_true, label_predict)
        
        predict.append(label_predict)
        label.append(label_true)

    # 预测准确率
    acc = accuracy_score(predict, label)
    logger.info("模型在测试集上的准确率:%s" % acc)