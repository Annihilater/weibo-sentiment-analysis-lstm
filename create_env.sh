#!/bin/bash

set -e

echo "创建环境: 开始"

conda create -n weibo-sentiment-analysis-lstm-py310 python=3.10 -y

echo "创建环境: 完成"