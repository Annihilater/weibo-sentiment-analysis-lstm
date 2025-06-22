import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    应用配置类，使用pydantic_settings进行环境变量管理
    """

    DATA_DIR_PATH: Optional[str] = Field(default="data", description="数据目录路径")
    INPUT_DIR_PATH: Optional[str] = Field(
        default="data/input", description="数据目录路径的输入目录"
    )
    INPUT_DATA_FILE_PATH: Optional[str] = Field(
        default="data/input/all_utf8.csv", description="数据文件路径"
    )
    OUTPUT_DIR_PATH: Optional[str] = Field(
        default="data/output", description="数据目录路径的输出目录"
    )

    # 应用配置
    DEBUG: Optional[bool] = Field(default=False, description="是否开启调试模式")

    # 日志设置
    LOG_LEVEL: Optional[str] = Field(
        default="INFO",
        description="日志记录级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    LOG_DIR: Optional[str] = Field(default="logs", description="日志文件存储目录")
    LOG_FILE: Optional[str] = Field(default="app.log", description="普通日志文件名")
    LOG_ERROR_FILE: Optional[str] = Field(
        default="app_error.log", description="错误日志文件名"
    )
    LOG_ROTATION: Optional[str] = Field(
        default="00:00", description="日志文件轮转时间 (如 '00:00' 表示每天零点)"
    )
    LOG_RETENTION: Optional[str] = Field(
        default="30 days", description="日志文件保留时间 (如 '30 days')"
    )
    ENCODING: Optional[str] = Field(
        default="utf-8", description="日志文件编码 (如 'utf-8', 'gbk')"
    )

    @computed_field
    @property
    def input_config_file_path(self) -> str:
        """
        获取配置文件的完整路径
        :return: 配置文件路径
        """
        return str(Path(self.INPUT_DIR_PATH) / self.INPUT_CONFIG_FILE_NAME)

    class Config:
        """
        pydantic配置类
        """

        env_file = ".env"  # 默认的环境变量文件
        env_file_encoding = "utf-8"
        case_sensitive = True  # 区分大小写


def get_settings(env_file: Optional[str] = None) -> Settings:
    """
    获取应用配置实例
    :param env_file: 环境变量文件路径，如果为None则使用默认的.env文件
    :return: 配置实例
    """
    if env_file:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    return Settings()


def check_settings(_settings: Settings) -> Settings:
    """
    检查配置是否正确
    :param _settings: 配置实例
    """
    # 确保目录存在
    os.makedirs(_settings.DATA_DIR_PATH, exist_ok=True)
    os.makedirs(_settings.INPUT_DIR_PATH, exist_ok=True)
    os.makedirs(_settings.OUTPUT_DIR_PATH, exist_ok=True)
    os.makedirs(_settings.LOG_DIR, exist_ok=True)
    return _settings


# 获取配置实例
settings = get_settings()  # 使用默认的.env文件
settings = check_settings(settings)
