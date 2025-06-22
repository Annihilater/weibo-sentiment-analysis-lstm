"""
服务器环境主程序
针对7x RTX 4090 GPU环境优化的微博情感分析LSTM训练
"""

import os
import sys

import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
from keras.src.utils import plot_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Embedding,
    Dropout,
    BatchNormalization,
    Bidirectional,
    Attention,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.config import settings

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import logger
from src.server_config import ServerConfig
from src.process2 import load_data


def create_optimized_lstm_model(
    vocab_size: int,
    label_size: int,
    input_shape: int,
    config: dict,
    strategy: tf.distribute.Strategy = None,
) -> Sequential:
    """
    创建优化的LSTM模型，支持多GPU训练
    :param vocab_size: 词汇表大小
    :param label_size: 标签类别数
    :param input_shape: 输入序列长度
    :param config: 模型配置
    :param strategy: 分布式策略
    :return: 编译好的模型
    """
    logger.info("创建优化的LSTM模型...")

    # 创建模型函数，避免代码重复
    def build_model():
        m = Sequential(name="weibo_sentiment_lstm")

        # Embedding层
        m.add(
            Embedding(
                input_dim=vocab_size + 1,
                output_dim=config["embedding_dim"],
                input_length=input_shape,
                mask_zero=True,
                name="embedding",
            )
        )

        # BatchNormalization
        m.add(BatchNormalization(name="bn_1"))

        # 第一层双向LSTM
        m.add(
            Bidirectional(
                LSTM(
                    config["n_units"],
                    return_sequences=True,
                    dropout=config["dropout_rate"],
                    recurrent_dropout=config["recurrent_dropout_rate"],
                ),
                name="bidirectional_lstm_1",
            )
        )

        m.add(BatchNormalization(name="bn_2"))
        m.add(Dropout(config["dropout_rate"], name="dropout_1"))

        # 第二层双向LSTM
        m.add(
            Bidirectional(
                LSTM(
                    config["n_units"] // 2,
                    return_sequences=(
                        True if config.get("use_attention", False) else False
                    ),
                    dropout=config["dropout_rate"],
                    recurrent_dropout=config["recurrent_dropout_rate"],
                ),
                name="bidirectional_lstm_2",
            )
        )

        # 如果使用注意力机制
        if config.get("use_attention", False):
            m.add(GlobalMaxPooling1D(name="attention_pooling"))

        # 添加密集层
        m.add(BatchNormalization(name="bn_3"))
        m.add(Dropout(config["dropout_rate"], name="dropout_2"))
        m.add(Dense(config["n_units"] // 2, activation="relu", name="dense_1"))
        m.add(BatchNormalization(name="bn_4"))
        m.add(Dropout(config["dropout_rate"], name="dropout_3"))
        m.add(Dense(config["n_units"] // 4, activation="relu", name="dense_2"))

        # 输出层
        if ServerConfig.MIXED_PRECISION:
            # 混合精度训练时，输出层需要使用float32
            m.add(
                Dense(label_size, activation="softmax", dtype="float32", name="output")
            )
        else:
            m.add(Dense(label_size, activation="softmax", name="output"))

        # 编译模型
        optimizer = Adam(learning_rate=config["learning_rate"])
        m.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return m

    # 如果有分布式策略，在策略作用域内创建模型
    if strategy:
        with strategy.scope():
            model = build_model()
    else:
        # 如果没有分布式策略，正常创建模型
        model = build_model()

    return model


def train_with_multi_gpu(
    input_shape: int = 20,
    filepath: str = None,
    model_save_path: str = None,
    epochs: int = 10,
    batch_size: int = 512,
):
    """
    使用多GPU训练模型
    :param input_shape: 输入序列长度
    :param filepath: 数据集路径
    :param model_save_path: 模型保存路径
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :return: (训练历史, 测试集准确率)
    """
    logger.info("开始多GPU训练...")

    # 设置服务器环境
    server_config = ServerConfig()
    server_config.setup_environment()

    # 获取分布式策略
    strategy = server_config.get_distribution_strategy()

    # 加载训练数据
    logger.info("加载训练数据...")
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = (
        load_data(filepath, input_shape)
    )
    logger.info(f"数据集大小: {len(x)}")
    logger.info(f"词汇表大小: {vocab_size}")
    logger.info(f"标签数量: {label_size}")

    # 将数据集分为训练集和测试集，占比为9：1
    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y,  # 使用stratify确保划分后标签分布一致
    )
    logger.info(f"训练集大小: {len(train_x)}")
    logger.info(f"测试集大小: {len(test_x)}")

    # 模型配置
    gpu_count = len(tf.config.list_physical_devices("GPU"))
    # 根据GPU数量调整全局批次大小
    global_batch_size = batch_size * max(1, gpu_count)
    logger.info(f"全局批次大小: {global_batch_size} (单GPU: {batch_size})")

    # 模型参数配置
    model_config = {
        "n_units": 256,  # LSTM单元数
        "embedding_dim": 128,  # Embedding维度
        "dropout_rate": 0.3,  # Dropout比率
        "recurrent_dropout_rate": 0.2,  # 循环层Dropout比率
        "learning_rate": 0.001,  # 学习率
        "use_attention": True,  # 使用注意力机制
    }

    # 在策略作用域内创建模型和数据集
    with strategy.scope():
        # 创建优化的LSTM模型
        model = create_optimized_lstm_model(
            vocab_size, label_size, input_shape, model_config, strategy
        )

        # 手动构建模型，确保模型参数被初始化
        logger.info("构建模型...")
        dummy_input = np.zeros((1, input_shape), dtype=np.int32)
        model(dummy_input)

        # 创建训练和验证数据集
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_dataset = train_dataset.shuffle(len(train_x)).batch(global_batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        val_dataset = val_dataset.batch(global_batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        # 分发数据集
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # 输出模型信息
    model.summary()

    # 保存模型结构图
    try:
        plot_model(model, to_file="./model_lstm.png", show_shapes=True)
        logger.info("模型结构图已保存")
    except Exception as e:
        logger.warning(f"保存模型结构图失败: {e}")

    # 设置回调函数
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

    # 开始训练
    logger.info("开始训练...")
    history = model.fit(
        train_dist_dataset,
        epochs=epochs,
        validation_data=val_dist_dataset,
        callbacks=callbacks,
        verbose=1,
    )

    # 加载最佳模型进行评估
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


def main():
    """
    主函数
    """
    try:
        # 初始化日志
        logger.info("======================================")
        logger.info("微博情感分析LSTM - 服务器优化版本")
        logger.info("======================================")

        # 检查数据文件
        data_file = None
        if os.path.exists("data/input/all_utf8.csv"):
            data_file = "data/input/all_utf8.csv"
            logger.info(f"使用数据文件: {data_file}")
        elif os.path.exists("data/weibo_senti_100k.csv"):
            data_file = "data/weibo_senti_100k.csv"
            logger.info(f"使用数据文件: {data_file}")
        else:
            logger.error("未找到数据文件，请确保数据文件存在")
            return

        # 创建输出目录
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs/tensorboard", exist_ok=True)
        os.makedirs("data/output", exist_ok=True)

        # 开始训练
        history, accuracy = train_with_multi_gpu(
            input_shape=20,
            filepath=data_file,
            model_save_path=settings.MODEL_SAVE_FILE_PATH,
            epochs=10,
            batch_size=512,
        )

        logger.info(f"训练完成，最终准确率: {accuracy:.4f}")

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        logger.error(f"详细错误信息: ", exc_info=True)
    finally:
        logger.info("程序结束")


if __name__ == "__main__":
    main()
