"""
服务器环境配置文件
针对7x RTX 4090 GPU环境进行优化
"""

import os
import traceback
from typing import Dict, Any

import psutil
import tensorflow as tf
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

from src.logger import logger


class ServerConfig:
    """
    服务器配置类，专门针对高性能GPU服务器环境
    """

    # GPU配置
    GPU_COUNT = 7
    GPU_MEMORY_LIMIT = 24576  # 每块RTX 4090显存24GB

    # 训练参数优化
    BATCH_SIZE = 512  # 利用大显存，增大batch size
    EPOCHS = 20
    LEARNING_RATE = 0.0001

    # 多GPU策略
    DISTRIBUTION_STRATEGY = "MirroredStrategy"  # 多GPU并行训练

    # 性能优化参数
    MIXED_PRECISION = True  # 混合精度训练
    XLA_ACCELERATION = True  # XLA加速

    # 资源限制
    CPU_THREADS = 16  # 限制CPU线程数，避免过度竞争
    MEMORY_GROWTH = True  # GPU内存动态增长

    def configure_gpu_environment(self):
        """
        配置GPU环境
        """
        logger.info("配置GPU环境...")

        # 检查GPU数量
        physical_devices = tf.config.list_physical_devices("GPU")
        if len(physical_devices) == 0:
            logger.warning("未检测到GPU设备，将使用CPU训练")
            logger.warning("如需使用GPU，请确保:")
            logger.warning("1. CUDA工具包已正确安装")
            logger.warning("2. 环境变量CUDA_HOME和LD_LIBRARY_PATH已设置")
            logger.warning("3. TensorFlow-GPU已正确安装")
            # 返回True而不是False，允许程序在CPU上继续运行
            return True

        logger.info(f"检测到 {len(physical_devices)} 个GPU设备")

        # 设置GPU内存增长
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU内存动态增长已启用")
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"设置GPU内存增长失败: {e}")
            logger.warning("将尝试继续运行，但可能会遇到内存问题")

        # 启用混合精度训练
        if self.MIXED_PRECISION:
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("混合精度训练已启用")
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.warning(f"启用混合精度训练失败: {e}")

        # 启用XLA加速
        if self.XLA_ACCELERATION:
            try:
                tf.config.optimizer.set_jit(True)
                logger.info("XLA加速已启用")
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.warning(f"启用XLA加速失败: {e}")

        # 设置TensorFlow内部优化
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        os.environ["TF_GPU_THREAD_COUNT"] = "1"
        os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 禁用oneDNN，避免警告

        return True

    def get_distribution_strategy(self):
        """
        获取分布式训练策略
        """
        # 检查是否有可用的GPU设备
        physical_devices = tf.config.list_physical_devices("GPU")

        # 如果没有GPU设备，返回None而不是使用MirroredStrategy
        if len(physical_devices) == 0:
            logger.info("未检测到GPU设备，不使用分布式策略")
            return None

        if self.DISTRIBUTION_STRATEGY == "MirroredStrategy":
            try:
                # 使用cross_device_ops参数，指定通信方式
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
                )
                logger.info(
                    f"使用MirroredStrategy，设备数量: {strategy.num_replicas_in_sync}"
                )
                return strategy
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"创建MirroredStrategy失败: {e}")
                logger.warning("将使用默认策略")
                return tf.distribute.get_strategy()
        elif self.DISTRIBUTION_STRATEGY == "MultiWorkerMirroredStrategy":
            try:
                strategy = tf.distribute.MultiWorkerMirroredStrategy()
                logger.info(
                    f"使用MultiWorkerMirroredStrategy，设备数量: {strategy.num_replicas_in_sync}"
                )
                return strategy
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"创建MultiWorkerMirroredStrategy失败: {e}")
                logger.warning("将使用默认策略")
                return tf.distribute.get_strategy()
        else:
            logger.info("使用默认策略")
            return tf.distribute.get_strategy()

    def get_optimized_model_config(self) -> Dict[str, Any]:
        """
        获取优化的模型配置
        """
        return {
            "n_units": 256,  # 增加LSTM单元数
            "embedding_dim": 128,  # 增加embedding维度
            "dropout_rate": 0.3,
            "recurrent_dropout_rate": 0.2,
            "batch_size": self.BATCH_SIZE,
            "epochs": self.EPOCHS,
            "learning_rate": self.LEARNING_RATE,
            "use_bidirectional": True,  # 使用双向LSTM
            "use_attention": True,  # 使用注意力机制
        }

    def get_training_callbacks(self, model_save_path="./models/best_model.h5"):
        """
        获取训练回调函数
        """
        callbacks = [
            TensorBoard(
                log_dir="./logs/tensorboard",
                histogram_freq=1,
                update_freq="batch",
                profile_batch="10,20",  # 性能分析
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False,
                mode="max",
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss", patience=8, verbose=1, restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
            ),
            CSVLogger("./logs/training_log.csv", append=True),
        ]

        return callbacks

    def setup_environment(self):
        """
        设置完整的环境配置
        """
        logger.info("开始设置服务器环境...")

        # 设置环境变量
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["OMP_NUM_THREADS"] = str(self.CPU_THREADS)

        # 设置CUDA环境变量
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            # 如果没有设置，默认使用所有GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(i) for i in range(self.GPU_COUNT)]
            )

        # 配置GPU
        if not self.configure_gpu_environment():
            logger.warning("GPU环境配置失败，将尝试继续运行")

        # 设置线程配置
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(16)

        logger.info("服务器环境配置完成")
        return True


def get_server_info():
    """
    获取服务器信息
    """

    info = {
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total // 1024 // 1024 // 1024,  # GB
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "tensorflow_version": tf.__version__,
    }

    return info
