"""
服务器环境主程序
针对7x RTX 4090 GPU环境优化的微博情感分析LSTM训练
"""

import os
import sys
import traceback

# 在导入TensorFlow之前设置GPU内存增长
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf

# 在导入其他模块前设置GPU内存增长
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已启用GPU内存动态增长")
    except RuntimeError as e:
        print(f"设置GPU内存增长失败: {e}")

import numpy as np
import pandas as pd
from datetime import datetime
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
from sklearn.metrics import classification_report

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.logger import logger
from src.train_and_evaluation import load_data
from src.server_config import ServerConfig


# 尝试获取合适的序列化装饰器
def get_serializable_decorator():
    """
    获取适用于当前TensorFlow版本的序列化装饰器
    """
    # 尝试不同的导入路径
    try:
        from keras.saving import register_keras_serializable

        return register_keras_serializable(package="Custom")
    except ImportError:
        try:
            from tensorflow.keras.saving import register_keras_serializable

            return register_keras_serializable(package="Custom")
        except ImportError:
            try:
                from tensorflow.keras.utils import register_keras_serializable

                return register_keras_serializable(package="Custom")
            except ImportError:
                logger.warning("无法找到register_keras_serializable，使用空装饰器")
                return lambda x: x


@get_serializable_decorator()
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    带预热的余弦衰减学习率调度器
    """

    def __init__(
        self, initial_lr: float, target_lr: float, warmup_steps: int, decay_steps: int
    ):
        """
        初始化学习率调度器
        :param initial_lr: 初始学习率
        :param target_lr: 目标学习率（预热后的最大学习率）
        :param warmup_steps: 预热步数
        :param decay_steps: 总步数
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        """
        计算当前步数的学习率
        :param step: 当前训练步数
        :return: 学习率
        """
        # 转换为浮点数
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # 预热阶段
        warmup_progress = tf.minimum(1.0, step / warmup_steps)
        warmup_lr = (
            self.initial_lr + (self.target_lr - self.initial_lr) * warmup_progress
        )

        # 余弦衰减阶段
        decay_progress = tf.maximum(0.0, step - warmup_steps) / (
            decay_steps - warmup_steps
        )
        decay_factor = 0.5 * (1.0 + tf.cos(tf.constant(np.pi) * decay_progress))
        decay_lr = self.target_lr * decay_factor

        # 使用tf.where选择当前阶段的学习率
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        """
        获取配置信息，用于序列化
        :return: 配置字典
        """
        config = {
            "initial_lr": self.initial_lr,
            "target_lr": self.target_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
        }
        return config

    @classmethod
    def from_config(cls, config):
        """
        从配置创建实例，用于反序列化
        :param config: 配置字典
        :return: 类实例
        """
        return cls(**config)


def create_lstm(
    n_units: int,
    input_shape: int,
    output_dim: int,
    vocab_size: int,
    label_size: int,
    learning_rate: float = 0.0001,
) -> tf.keras.Model:
    """
    创建深度学习模型，Embedding + LSTM + Softmax
    :param n_units: LSTM层神经元个数
    :param input_shape: 输入序列长度
    :param output_dim: Embedding层输出维度
    :param vocab_size: 词汇表大小
    :param label_size: 标签类别数
    :param learning_rate: 学习率
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
                embeddings_regularizer=tf.keras.regularizers.l2(1e-5),  # 添加L2正则化
            ),
            # 第一个LSTM层，返回序列
            tf.keras.layers.LSTM(
                n_units,
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),  # 添加L2正则化
                recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            # 第二个LSTM层
            tf.keras.layers.LSTM(
                n_units // 2,
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            # 全连接层
            tf.keras.layers.Dense(
                n_units // 4,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            # 输出层
            tf.keras.layers.Dense(
                label_size,
                activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
        ]
    )

    # 使用Adam优化器，设置学习率
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,  # 默认值，通常不需要调整
        amsgrad=True,  # 启用AMSGrad变体
    )

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    return model


def train_with_multi_gpu(
    input_shape: int = 180,
    filepath: str = None,
    model_save_path: str = None,
    epochs: int = 15,
    batch_size: int = 64,  # 降低单GPU的批次大小
):
    """
    使用多GPU训练模型
    :param input_shape: 输入序列长度
    :param filepath: 数据集路径
    :param model_save_path: 模型保存路径
    :param epochs: 训练轮数
    :param batch_size: 每个GPU的批次大小
    :return: (训练历史, 测试集准确率)
    """
    logger.info("开始多GPU训练...")

    # 获取分布式策略
    server_config = ServerConfig()
    strategy = server_config.get_distribution_strategy()
    logger.info(f"使用分布式策略: {strategy.__class__.__name__}")

    # 检查可用GPU
    gpus = tf.config.list_physical_devices("GPU")
    gpu_count = len(gpus) if gpus else 1
    logger.info(f"可用GPU数量: {gpu_count}")

    # 根据GPU数量调整学习率和批次大小
    base_lr = 1e-4  # 基础学习率
    target_lr = base_lr * (gpu_count**0.5)  # 根据GPU数量调整目标学习率
    initial_lr = target_lr / 10  # 预热起始学习率
    global_batch_size = batch_size * gpu_count
    logger.info(f"全局批次大小: {global_batch_size} (单GPU: {batch_size})")
    logger.info(f"初始学习率: {initial_lr:.6f}, 目标学习率: {target_lr:.6f}")

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

    # 计算总步数和预热步数
    steps_per_epoch = len(train_x) // global_batch_size
    warmup_epochs = 3
    total_epochs = epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    total_steps = steps_per_epoch * total_epochs

    # 创建学习率调度器
    lr_schedule = WarmupCosineDecay(
        initial_lr=initial_lr,
        target_lr=target_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
    )

    # 模型配置
    model_config = {
        "n_units": 128,  # LSTM单元数
        "embedding_dim": 100,  # Embedding维度
        "dropout_rate": 0.3,  # Dropout比率
        "recurrent_dropout_rate": 0.0,  # 循环Dropout比率
        "learning_rate": lr_schedule,  # 使用学习率调度器
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
            model_config["learning_rate"],
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
                model_config["learning_rate"],
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
        tf.keras.utils.plot_model(
            model,
            to_file=f"{settings.OUTPUT_DIR_PATH}/model_lstm.png",
            show_shapes=True,
        )
        logger.info(f"模型结构图已保存到 {settings.OUTPUT_DIR_PATH}/model_lstm.png")
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.warning(f"保存模型结构图失败: {e}")

    # 输出模型信息
    model.summary()

    # 定义回调函数
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir="./logs/tensorboard",
            histogram_freq=1,
            update_freq="batch",
            profile_batch=0,  # 禁用性能分析以减少内存使用
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
            patience=7,  # 增加耐心值，给模型更多机会
            verbose=1,
            restore_best_weights=True,
            min_delta=1e-4,  # 添加最小改善阈值
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

    logger.info("训练完成！")

    # 手动保存最终模型的权重
    # 因为EarlyStopping的restore_best_weights=True，所以这将保存验证损失最小的模型
    final_weights_path = os.path.join(
        os.path.dirname(model_save_path), "lstm_model_final.weights.h5"
    )
    logger.info(f"正在手动保存最终模型权重到: {final_weights_path}")
    model.save_weights(final_weights_path)
    logger.info("最终模型权重已成功保存！")

    # =================================================================
    # 在训练后直接进行评估
    # =================================================================
    logger.info("======================================")
    logger.info("开始在测试集上进行最终评估...")
    logger.info("======================================")

    # 对于分布式模型，评估最好也在策略范围内进行，以确保度量标准正确计算
    # 或者，我们可以直接在主GPU上评估，因为模型权重是同步的
    # 这里我们采用更简单的方式，直接在 numpy 数据上评估
    logger.info("使用 test_x 和 test_y 进行最终评估。")

    # 评估时可以使用更大的批次大小
    evaluation_batch_size = global_batch_size * 2

    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        test_x,
        test_y,
        batch_size=evaluation_batch_size,
        verbose=1,
    )

    logger.info(f"最终评估 - 测试集损失: {test_loss:.4f}")
    logger.info(f"最终评估 - 测试集准确率: {test_accuracy:.4f}")
    logger.info(f"最终评估 - 测试集精确率: {test_precision:.4f}")
    logger.info(f"最终评估 - 测试集召回率: {test_recall:.4f}")

    # 获取预测结果以生成分类报告
    y_pred = model.predict(
        test_x,
        batch_size=evaluation_batch_size,
        verbose=1,
    )
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(test_y, axis=1)

    # 记录详细的分类报告
    logger.info("\n最终评估 - 分类报告:")
    report = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=list(output_dictionary.values()),
        digits=4,
    )
    # classification_report的输出可能包含换行符，直接打印
    for line in report.split("\n"):
        logger.info(line)

    # 记录一些预测示例
    logger.info("\n最终评估 - 示例预测:")
    num_samples_to_show = min(10, len(test_x))
    for i in range(num_samples_to_show):
        # 从id序列反向解析为句子
        sentence = "".join(
            [inverse_word_dictionary.get(j, "?") for j in test_x[i] if j != 0]
        )
        true_label = output_dictionary.get(y_true_classes[i])
        pred_label = output_dictionary.get(y_pred_classes[i])
        logger.info(f"文本: {sentence}")
        logger.info(f"真实标签: {true_label}, 预测标签: {pred_label}")
        logger.info("---")

    # =================================================================
    # 保存训练和评估结果到CSV文件
    # =================================================================
    logger.info("======================================")
    logger.info("正在将结果保存到CSV文件...")
    logger.info("======================================")

    try:
        # 获取最佳验证轮次的结果
        best_epoch_index = np.argmax(history.history["val_accuracy"])
        best_val_accuracy = history.history["val_accuracy"][best_epoch_index]
        best_val_loss = history.history["val_loss"][best_epoch_index]

        # 兼容不同Keras版本下metrics key的命名 (e.g., val_precision vs val_precision_1)
        val_precision_key = [k for k in history.history.keys() if "val_precision" in k][
            0
        ]
        val_recall_key = [k for k in history.history.keys() if "val_recall" in k][0]
        best_val_precision = history.history[val_precision_key][best_epoch_index]
        best_val_recall = history.history[val_recall_key][best_epoch_index]

        # 将分类报告解析为字典以便写入CSV
        report_dict = classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=list(output_dictionary.values()),
            digits=4,
            output_dict=True,
        )

        # 准备要写入CSV的数据
        results_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_epoch": best_epoch_index + 1,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "best_val_precision": best_val_precision,
            "best_val_recall": best_val_recall,
            "final_test_loss": test_loss,
            "final_test_accuracy": test_accuracy,
            "final_test_precision": test_precision,
            "final_test_recall": test_recall,
        }

        # 将分类报告的字典扁平化，并为key添加'report_'前缀
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    results_data[f'report_{label.replace(" ", "_")}_{metric_name}'] = (
                        value
                    )
            else:
                # for keys like 'accuracy'
                results_data[f'report_{label.replace(" ", "_")}'] = metrics

        # 创建DataFrame
        df = pd.DataFrame([results_data])

        # 定义CSV文件路径
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(
            settings.OUTPUT_DIR_PATH, f"training_report_{timestamp_str}.csv"
        )

        # 写入CSV
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"训练和评估报告已成功保存到: {csv_path}")

    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"保存报告到CSV失败: {e}", exc_info=True)

    return history


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
        if os.path.exists(f"{settings.INPUT_DATA_FILE_PATH}"):
            data_file = settings.INPUT_DATA_FILE_PATH
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
        os.makedirs(settings.DATA_DIR_PATH, exist_ok=True)

        # 开始训练
        history = train_with_multi_gpu(
            input_shape=180,
            filepath=data_file,
            model_save_path=settings.MODEL_SAVE_FILE_PATH,
            epochs=15,
            batch_size=64,
        )

        logger.info("训练过程完成")

    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"训练过程中发生错误: {e}")
        logger.error("详细错误信息: ", exc_info=True)
    finally:
        logger.info("程序结束")


if __name__ == "__main__":
    main()
