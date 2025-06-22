"""
评估脚本
用于加载已训练好的模型并进行评估
"""

import os
import sys
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.logger import logger
from src.process2 import load_data
from src.server_main import WarmupCosineDecay


def evaluate_model(
    model_path: str = None,
    data_path: str = None,
    input_shape: int = 180,
    batch_size: int = 256,
):
    """
    评估已训练好的模型
    :param model_path: 模型文件路径
    :param data_path: 数据文件路径
    :param input_shape: 输入序列长度
    :param batch_size: 批次大小 (评估时可适当增大)
    """
    try:
        # 初始化日志
        logger.info("======================================")
        logger.info("微博情感分析LSTM - 模型评估")
        logger.info("======================================")

        # 检查模型文件路径
        if model_path is None:
            # 指向手动保存的最终模型
            model_path = "data/output/lstm_model_final.keras"
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return

        # 检查数据文件路径
        if data_path is None:
            if os.path.exists("data/input/all_utf8.csv"):
                data_path = "data/input/all_utf8.csv"
            elif os.path.exists("data/weibo_senti_100k.csv"):
                data_path = "data/weibo_senti_100k.csv"
            else:
                logger.error("未找到数据文件，请确保数据文件存在")
                return

        logger.info(f"使用模型文件: {model_path}")
        logger.info(f"使用数据文件: {data_path}")

        # 设置GPU内存增长
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("已启用GPU内存动态增长")
            except RuntimeError as e:
                logger.warning(f"设置GPU内存增长失败: {e}")

        # 加载数据
        logger.info("加载数据...")
        x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = (
            load_data(data_path, input_shape)
        )

        # 划分训练集和测试集 (与训练时保持一致)
        _, test_x, _, test_y = train_test_split(
            x, y, test_size=0.1, random_state=42, stratify=y
        )

        logger.info(f"测试集大小: {len(test_x)}")

        # 加载模型
        logger.info("加载模型...")
        try:
            # 直接加载完整的Keras模型，并提供自定义的学习率调度器
            custom_objects = {"WarmupCosineDecay": WarmupCosineDecay}
            model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects
            )
            logger.info("模型加载成功. 模型摘要:")
            model.summary(print_fn=logger.info)

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            logger.error("详细错误信息: ", exc_info=True)
            return

        # 在测试集上进行评估
        logger.info("开始评估...")
        results = model.evaluate(
            test_x,
            test_y,
            batch_size=batch_size,
            verbose=1,
        )
        
        # 结果与模型编译时的metrics顺序一致
        test_loss = results[0]
        test_accuracy = results[1]
        test_precision = results[2]
        test_recall = results[3]


        logger.info(f"测试集损失: {test_loss:.4f}")
        logger.info(f"测试集准确率: {test_accuracy:.4f}")
        logger.info(f"测试集精确率: {test_precision:.4f}")
        logger.info(f"测试集召回率: {test_recall:.4f}")

        # 在测试集上进行预测
        y_pred = model.predict(
            test_x,
            batch_size=batch_size,
            verbose=1,
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

    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        logger.error("详细错误信息: ", exc_info=True)
    finally:
        logger.info("评估完成")


def main():
    """
    主函数
    """
    evaluate_model()


if __name__ == "__main__":
    main() 