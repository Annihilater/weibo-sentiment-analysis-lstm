"""
服务器环境主程序
针对7x RTX 4090 GPU环境优化的微博情感分析LSTM训练
"""

import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Embedding,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from src.config import settings
from src.logger import logger
from src.process2 import load_data
from src.server_config import ServerConfig

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_lstm(
    n_units: int, input_shape: int, output_dim: int, vocab_size: int, label_size: int
) -> tf.keras.Model:
    """
    创建深度学习模型，Embedding + LSTM + Softmax
    :param n_units: LSTM层神经元个数
    :param input_shape: 输入序列长度
    :param output_dim: Embedding层输出维度
    :param vocab_size: 词汇表大小
    :param label_size: 标签类别数
    :return: 编译好的模型
    """
    # 使用Sequential模型构建网络
    model = tf.keras.Sequential(
        [
            # Embedding层
            tf.keras.layers.Embedding(
                input_dim=vocab_size + 1,
                output_dim=output_dim,
                input_length=input_shape,
                mask_zero=True,
            ),
            # BatchNormalization层
            tf.keras.layers.BatchNormalization(),
            # 第一个LSTM层，返回序列
            tf.keras.layers.LSTM(n_units, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            # 第二个LSTM层
            tf.keras.layers.LSTM(n_units // 2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            # 全连接层
            tf.keras.layers.Dense(n_units // 4, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            # 输出层
            tf.keras.layers.Dense(label_size, activation="softmax"),
        ]
    )

    # 使用Adam优化器，设置较小的学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # 编译模型
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # 输出模型信息
    model.summary()

    return model


def train_with_multi_gpu(
    input_shape: int = 180,  # 修改为与process2.py一致的输入序列长度
    filepath: str = None,
    model_save_path: str = None,
    epochs: int = 15,  # 增加训练轮数
    batch_size: int = 256,  # 使用与process2.py相同的批次大小
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

    # 获取分布式策略
    server_config = ServerConfig()
    strategy = server_config.get_distribution_strategy()
    logger.info(f"使用分布式策略: {strategy.__class__.__name__}")

    # 检查可用GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"可用GPU数量: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}")
    else:
        logger.warning("未检测到可用GPU，将使用CPU训练")

    # 设置每个GPU的内存增长
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"已为 {gpu.name} 启用内存增长")
        except Exception as e:
            logger.warning(f"为 {gpu.name} 设置内存增长失败: {e}")

    # 确保TensorFlow能够看到GPU
    if len(tf.config.list_physical_devices("GPU")) > 0:
        logger.info("TensorFlow可以访问GPU")
    else:
        logger.warning("TensorFlow无法访问GPU，请检查CUDA和cuDNN安装")

    # 加载训练数据
    logger.info("加载训练数据...")

    # 如果filepath为None，使用默认路径
    if filepath is None:
        filepath = settings.CLEAN_DATA_FILE_PATH

    # 如果model_save_path为None，使用默认路径
    if model_save_path is None:
        model_save_path = settings.MODEL_SAVE_FILE_PATH

    # 确保输出目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 加载数据
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = (
        load_data(filepath, input_shape)
    )

    logger.info(f"数据集大小: {len(x)}")
    logger.info(f"词汇表大小: {vocab_size}")
    logger.info(f"标签数量: {label_size}")

    # 划分训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.1, random_state=42, stratify=y
    )

    logger.info(f"训练集大小: {len(train_x)}")
    logger.info(f"测试集大小: {len(test_x)}")

    # 根据GPU数量调整全局批次大小
    gpu_count = len(tf.config.list_physical_devices("GPU"))
    # 根据GPU数量调整全局批次大小
    global_batch_size = batch_size * max(1, gpu_count)
    logger.info(f"全局批次大小: {global_batch_size} (单GPU: {batch_size})")

    # 模型配置
    model_config = {
        "n_units": 128,  # LSTM单元数
        "embedding_dim": 100,  # Embedding维度
        "dropout_rate": 0.3,  # Dropout比率
        "recurrent_dropout_rate": 0.0,  # 循环Dropout比率
        "learning_rate": 0.0001,  # 学习率
    }

    # 处理数据，确保批次大小能被GPU数量整除
    if strategy is not None:
        # 确保全局批次大小能被GPU数量整除
        num_replicas = strategy.num_replicas_in_sync
        # 确保训练集大小能被全局批次大小整除
        train_samples_to_drop = len(train_x) % global_batch_size
        if train_samples_to_drop > 0:
            logger.info(
                f"为确保训练集能被全局批次大小整除，丢弃 {train_samples_to_drop} 个训练样本"
            )
            train_x = train_x[:-train_samples_to_drop]
            train_y = train_y[:-train_samples_to_drop]

        test_samples_to_drop = len(test_x) % global_batch_size
        if test_samples_to_drop > 0:
            logger.info(
                f"为确保测试集能被全局批次大小整除，丢弃 {test_samples_to_drop} 个测试样本"
            )
            test_x = test_x[:-test_samples_to_drop]
            test_y = test_y[:-test_samples_to_drop]

        logger.info(f"调整后训练集大小: {len(train_x)}")
        logger.info(f"调整后测试集大小: {len(test_x)}")

    # 创建模型和数据集
    if strategy is None:
        logger.info("使用单设备训练模式")
        # 创建模型
        model = create_lstm(
            model_config["n_units"],
            input_shape,
            model_config["embedding_dim"],
            vocab_size,
            label_size,
        )

        # 手动构建模型，确保模型参数被初始化
        logger.info("构建模型...")
        dummy_input = np.zeros((1, input_shape), dtype=np.int32)
        _ = model(dummy_input)  # 调用模型以触发构建过程

        # 再次输出模型摘要，确认模型已构建
        model.summary()

        # 创建标准数据集
        train_dataset = None  # 直接使用numpy数组训练
        val_dataset = None  # 直接使用numpy数组验证
    else:
        logger.info("使用多设备训练模式")
        # 在策略作用域内创建模型和数据集
        with strategy.scope():
            # 创建优化的LSTM模型
            model = create_lstm(
                model_config["n_units"],
                input_shape,
                model_config["embedding_dim"],
                vocab_size,
                label_size,
            )

            # 手动构建模型，确保模型参数被初始化
            logger.info("构建模型...")
            dummy_input = np.zeros((1, input_shape), dtype=np.int32)
            _ = model(dummy_input)  # 调用模型以触发构建过程

            # 再次输出模型摘要，确认模型已构建
            model.summary()

        # 创建分布式数据集
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(
            global_batch_size, drop_remainder=True
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        val_dataset = val_dataset.batch(global_batch_size, drop_remainder=True)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        # 分发数据集
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = strategy.experimental_distribute_dataset(val_dataset)
        logger.info("已分发数据集到多个设备")

    # 保存模型结构图
    try:
        logger.info("保存模型结构图...")
        tf.keras.utils.plot_model(model, to_file="./model_lstm.png", show_shapes=True)
        logger.info("模型结构图已保存到 ./model_lstm.png")
    except Exception as e:
        logger.warning(f"保存模型结构图失败: {e}")

    # 输出模型信息
    model.summary()

    # 定义回调函数
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir="./logs/tensorboard",
            histogram_freq=1,
            update_freq="batch",  # 实时更新训练指标
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,  # 增加耐心值
            verbose=1,
            restore_best_weights=True,  # 恢复最佳权重
        ),
        # 添加学习率调度器
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6,
        ),
    ]

    # 开始训练
    logger.info("开始训练模型...")

    # 根据是否使用分布式策略选择不同的训练方式
    if strategy is None:
        # 单设备训练
        history = model.fit(
            train_x,
            train_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_x, test_y),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        # 多设备训练
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=len(train_x) // global_batch_size,
            validation_steps=len(test_x) // global_batch_size,
        )

    # 加载最佳模型进行评估
    logger.info("加载最佳模型进行评估...")
    best_model = tf.keras.models.load_model(model_save_path)

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
            input_shape=180,
            filepath=data_file,
            model_save_path=settings.MODEL_SAVE_FILE_PATH,
            epochs=15,
            batch_size=256,
        )

        logger.info(f"训练完成，最终准确率: {accuracy:.4f}")

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        logger.error(f"详细错误信息: ", exc_info=True)
    finally:
        logger.info("程序结束")


if __name__ == "__main__":
    main()
