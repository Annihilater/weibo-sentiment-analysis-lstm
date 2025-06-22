#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于 loguru 实现的日志模块
使用示例:
```python
from src.logger import logger

logger.info("普通信息")
logger.error("错误信息")
logger.debug("调试信息")

# 结构化日志记录
logger.info("用户登录", user_id="12345", ip="192.168.1.1")
```
"""

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

from src.config import settings


class LoggerFactory:
    def __init__(
        self,
        log_dir: str = None,
        log_level: str = None,
        log_file: str = None,
        error_log_file: str = None,
        rotation: str = None,
        retention: str = None,
    ):
        # 从配置文件中读取日志配置，如果没有配置则使用默认值
        self.log_dir = Path(log_dir or settings.LOG_DIR)
        self.log_level = log_level or settings.LOG_LEVEL
        self.log_file = log_file or settings.LOG_FILE
        self.error_log_file = error_log_file or settings.LOG_ERROR_FILE
        self.rotation = rotation or settings.LOG_ROTATION
        self.retention = retention or settings.LOG_RETENTION
        self._logger = _loguru_logger
        self._configured = False
        # 打印日志级别，用于调试
        print(f"初始化日志系统，日志级别: {self.log_level}")
        self._configure()

    def _configure(self):
        if self._configured:
            return
        self._logger.remove()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 控制台日志格式（带颜色标签）
        console_log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<cyan>{process}</cyan>:<cyan>{thread}</cyan> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        # 文件日志格式（不带颜色标签）
        file_log_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{process}:{thread}"
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        # 控制台
        self._logger.add(
            sys.stderr, format=console_log_format, level=self.log_level, colorize=True
        )
        # 普通日志文件
        self._logger.add(
            self.log_dir / self.log_file,
            format=file_log_format,
            level=self.log_level,
            rotation=self.rotation,
            retention=self.retention,
            encoding="utf-8",
        )
        # 错误日志
        self._logger.add(
            self.log_dir / self.error_log_file,
            format=file_log_format,
            level="ERROR",
            rotation=self.rotation,
            retention=self.retention,
            encoding="utf-8",
        )
        self._configured = True

    @property
    def logger(self):
        return self._logger

    @property
    def current_log_level(self):
        """获取当前日志级别"""
        return self.log_level


# 实例化全局 logger
_logger_factory = LoggerFactory()
logger = _logger_factory.logger


def get_current_log_level() -> str:
    """
    获取当前日志级别

    :return: 当前日志级别
    """
    return _logger_factory.current_log_level
