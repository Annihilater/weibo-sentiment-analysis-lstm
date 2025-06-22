import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import chardet

from src.logger import logger


class DataCleaner:
    """
    数据清洗类，用于处理GB2312编码的数据文件并转换为 UTF-8 编码
    """

    def __init__(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        encoding_to: str = "utf-8",
    ):
        """
        初始化数据清洗器
        :param input_file: 输入文件路径
        :param output_file: 输出文件路径，如果为None则自动生成
        :param encoding_to: 输出文件编码
        """
        self.input_file = Path(input_file)
        self.encoding_to = encoding_to

        if output_file is None:
            # 如果没有指定输出文件，则在processed目录下创建同名文件
            self.output_file = Path("data/processed") / self.input_file.name
        else:
            self.output_file = Path(output_file)

        # 确保输出目录存在
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def analyze_file(self) -> None:
        """
        分析文件的基本信息
        """
        logger.info("\n=== 文件基本信息 ===")
        logger.info(f"文件路径: {self.input_file.absolute()}")
        logger.info(
            f"文件大小: {os.path.getsize(self.input_file) / 1024 / 1024:.2f} MB"
        )
        logger.info(f"文件是否存在: {self.input_file.exists()}")
        logger.info(f"文件是否可读: {os.access(self.input_file, os.R_OK)}")

        # 读取文件前几个字节进行分析
        with open(self.input_file, "rb") as f:
            header = f.read(32)
            logger.info("\n前32个字节的十六进制表示：")
            logger.info(" ".join(f"{b:02x}" for b in header))

    def detect_encoding(self) -> Tuple[str | None, float]:
        """
        检测文件编码
        :return: 元组(编码类型, 置信度)
        """
        logger.info("\n=== 开始检测文件编码 ===")

        # 读取文件的多个部分来检测编码
        results = []
        with open(self.input_file, "rb") as f:
            # 读取文件开头
            start_content = f.read(4096)
            if start_content:
                result = chardet.detect(start_content)
                results.append((result["encoding"], result["confidence"], "开头"))
                logger.info(
                    f"开头部分检测结果：编码={result['encoding']}, 置信度={result['confidence']:.2%}"
                )

            # 读取文件中间部分
            f.seek(os.path.getsize(self.input_file) // 2)
            mid_content = f.read(4096)
            if mid_content:
                result = chardet.detect(mid_content)
                results.append((result["encoding"], result["confidence"], "中间"))
                logger.info(
                    f"中间部分检测结果：编码={result['encoding']}, 置信度={result['confidence']:.2%}"
                )

            # 读取文件末尾部分
            f.seek(max(0, os.path.getsize(self.input_file) - 4096))
            end_content = f.read(4096)
            if end_content:
                result = chardet.detect(end_content)
                results.append((result["encoding"], result["confidence"], "末尾"))
                logger.info(
                    f"末尾部分检测结果：编码={result['encoding']}, 置信度={result['confidence']:.2%}"
                )

        # 选择置信度最高的结果
        if results:
            encoding, confidence, position = max(results, key=lambda x: x[1])
            logger.info(
                f"\n最终选择：{position}部分的编码（{encoding}），置信度：{confidence:.2%}"
            )
            return encoding, confidence

        return None, 0.0

    def clean(self) -> None:
        """
        执行数据清洗操作
        """
        try:
            # 首先分析文件基本信息
            self.analyze_file()

            # 检测文件编码
            detected_encoding, confidence = self.detect_encoding()

            # 如果检测失败，尝试常见的编码列表
            if not detected_encoding or confidence < 0.7:
                encodings_to_try = ["gb18030", "gbk", "gb2312", "utf-8", "big5"]
                logger.info(f"\n编码检测置信度较低，将尝试以下编码：{encodings_to_try}")
            else:
                # 如果检测为GB2312，也添加其超集编码
                if detected_encoding.upper() == "GB2312":
                    encodings_to_try = ["gb18030", "gbk", "gb2312"]
                    logger.info(
                        f"\n检测到GB2312编码，将优先尝试其超集编码：{encodings_to_try}"
                    )
                else:
                    encodings_to_try = [detected_encoding]

            # 尝试不同的编码读取文件
            content = None
            used_encoding = None

            for encoding in encodings_to_try:
                try:
                    logger.info(f"\n尝试使用 {encoding} 编码读取文件...")
                    with open(self.input_file, "r", encoding=encoding) as f:
                        # 先尝试读取一小部分
                        sample = f.read(1024)
                        logger.info(f"成功读取文件前1KB内容")

                        # 如果成功，读取整个文件
                        f.seek(0)
                        content = f.read()
                        used_encoding = encoding
                        logger.info(f"成功使用 {encoding} 编码读取整个文件")
                        break
                except UnicodeDecodeError as e:
                    logger.info(f"使用 {encoding} 编码读取失败：{str(e)}")
                    continue

            if content is None:
                raise UnicodeDecodeError("NONE", b"", 0, 1, "所有编码尝试均失败")

            # 写入新文件
            logger.info(f"\n开始写入转换后的文件...")
            with open(self.output_file, "w", encoding=self.encoding_to) as f:
                f.write(content)

            logger.info(f"\n=== 数据清洗完成！===")
            logger.info(f"输入文件：{self.input_file}")
            logger.info(f"输入文件编码：{used_encoding}")
            logger.info(f"输出文件：{self.output_file}")
            logger.info(f"输出文件编码：{self.encoding_to}")

        except UnicodeDecodeError as e:
            logger.error(traceback.format_exc())
            logger.info(f"\n编码错误：{e}")
            logger.info("请检查输入文件的编码格式是否正确")
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.info(f"\n发生错误：{e}")
            logger.error(traceback.format_exc())


def main():
    """
    主函数
    """
    # 设置输入输出文件路径
    input_file = "data/input/all.csv"
    output_file = "data/output/all_utf8.csv"

    # 创建清洗器并执行清洗
    cleaner = DataCleaner(input_file, output_file)
    cleaner.clean()


if __name__ == "__main__":
    main()
