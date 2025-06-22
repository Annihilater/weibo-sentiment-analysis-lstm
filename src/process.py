from itertools import accumulate

import matplotlib.pyplot as plt
import pandas as pd

from src.config import settings
from src.logger import logger


def get_data():
    """
    读取数据
    设置matplotlib绘图时的字体，这个一般都没有吧
    my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\Songti.ttc")
    统计句子长度及长度出现的频数
    """
    df = pd.read_csv(settings.INPUT_DATA_FILE_PATH, encoding="utf-8")
    logger.info(df.groupby("label")["label"].count())
    df["length"] = df["evaluation"].apply(lambda x: len(x))
    len_df = df.groupby("length").count()
    sent_length = len_df.index.tolist()
    sent_freq = len_df["evaluation"].tolist()

    # 绘制句子长度及出现频数统计图
    plt.bar(sent_length, sent_freq)
    plt.title("评论长度及出现频数统计图")
    plt.xlabel("评论长度")
    plt.ylabel("评论长度出现的频数")
    plt.rcParams["font.sans-serif"] = ["Simhei"]
    plt.show()


def process(sent_freq, sent_length):
    """
    计算累积分布函数
    """
    # 绘制评论长度累积分布函数(CDF)
    sent_pentage_list = [(count / sum(sent_freq)) for count in accumulate(sent_freq)]

    # 绘制CDF
    plt.plot(sent_length, sent_pentage_list)

    # 寻找分位点为quantile的评论长度
    quantile = 0.9
    # print(list(sent_pentage_list))
    for length, per in zip(sent_length, sent_pentage_list):
        if round(per, 2) == quantile:
            index = length
            break
    print("\n分位点为%s的微博长度:%d." % (quantile, index))

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
