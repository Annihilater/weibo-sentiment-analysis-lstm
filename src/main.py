from src.config import settings
from src.data_processing.clean_data import DataCleaner


def main():
    """
    主函数，用于执行数据清洗
    """
    input_file = settings.INPUT_DIR_PATH
    output_file = settings.OUTPUT_DIR_PATH
    encoding_to = settings.ENCODING
    cleaner = DataCleaner(input_file, output_file, encoding_to)
    cleaner.analyze_file()
    cleaner.detect_encoding()


if __name__ == "__main__":
    main()
