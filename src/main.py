from src.config import settings
from src.data_processing.clean_data import DataCleaner
from src.process import get_data
from src.process2 import model_train


def clean_data():
    """
    数据清洗
    """
    input_file = settings.INPUT_DIR_PATH
    output_file = settings.OUTPUT_DIR_PATH
    encoding_to = settings.ENCODING
    cleaner = DataCleaner(input_file, output_file, encoding_to)
    cleaner.analyze_file()
    cleaner.detect_encoding()


def main():
    """
    主函数，基于机器学习的微博情感分析可视化系统
    """
    get_data()

    input_shape = 180
    model_train(
        input_shape=input_shape,
        filepath=settings.INPUT_DIR_PATH,
        model_save_path=settings.MODEL_SAVE_FILE_PATH,
    )


if __name__ == "__main__":
    # clean_data()
    main()
