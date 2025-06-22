import platform
from itertools import accumulate

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd

from src.config import settings
from src.logger import logger


def configure_matplotlib_fonts():
    """
    配置matplotlib的字体，自动适配不同操作系统的中文字体
    """
    system = platform.system()
    logger.info(f"当前操作系统: {system}")

    if system == "Windows":
        font_list = ["SimHei", "Microsoft YaHei"]
    elif system == "Linux":
        font_list = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Droid Sans Fallback"]
    elif system == "Darwin":  # macOS
        font_list = ["Arial Unicode MS", "Hiragino Sans GB", "PingFang HK"]
    else:
        font_list = []

    # 尝试设置字体
    font_found = False
    for font_name in font_list:
        try:
            font_path = font_manager.findfont(font_name)
            if font_path:
                plt.rcParams["font.family"] = font_name
                logger.info(f"成功设置字体: {font_name}")
                font_found = True
                break
        except Exception as e:
            logger.warning(f"设置字体 {font_name} 失败: {str(e)}")

    if not font_found:
        logger.warning("未找到合适的中文字体，将使用系统默认字体")
        # 使用系统默认字体
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False


def get_data():
    """
    读取数据并绘制统计图
    """
    # 配置字体
    configure_matplotlib_fonts()

    df = pd.read_csv(settings.INPUT_DATA_FILE_PATH, encoding="utf-8")
    logger.info(df.groupby("label")["label"].count())
    df["length"] = df["evaluation"].apply(lambda x: len(x))
    len_df = df.groupby("length").count()
    sent_length = len_df.index.tolist()
    sent_freq = len_df["evaluation"].tolist()

    # 绘制句子长度及出现频数统计图
    plt.figure(figsize=(10, 6))
    plt.bar(sent_length, sent_freq)
    plt.title("评论长度及出现频数统计图")
    plt.xlabel("评论长度")
    plt.ylabel("评论长度出现的频数")
    plt.show()

    return sent_freq, sent_length


def process(sent_freq, sent_length):
    """
    计算累积分布函数并绘制图表
    """
    # 配置字体
    configure_matplotlib_fonts()

    # 绘制评论长度累积分布函数(CDF)
    sent_pentage_list = [(count / sum(sent_freq)) for count in accumulate(sent_freq)]

    # 绘制CDF
    plt.figure(figsize=(10, 6))
    plt.plot(sent_length, sent_pentage_list)

    # 寻找分位点为quantile的评论长度
    quantile = 0.9
    for length, per in zip(sent_length, sent_pentage_list):
        if round(per, 2) == quantile:
            index = length
            break
    logger.info("分位点为%s的微博长度:%d." % (quantile, index))

    # 绘制评论长度累积分布函数图
    plt.plot(sent_length, sent_pentage_list)
    plt.hlines(quantile, 0, index, colors="c", linestyles="dashed")
    plt.vlines(index, 0, quantile, colors="c", linestyles="dashed")
    plt.text(0, quantile, str(quantile))
    plt.text(index, 0, str(index))
    plt.title("评论长度累积分布函数图")
    plt.xlabel("评论长度")
    plt.ylabel("评论长度累积频率")
    plt.show()
