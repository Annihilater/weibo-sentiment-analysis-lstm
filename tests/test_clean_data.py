import unittest
from unittest.mock import mock_open, patch
from pathlib import Path
from ..src.data_processing.clean_data import DataCleaner


class TestDataCleaner(unittest.TestCase):
    """
    测试数据清洗类
    执行命令:
    python -m unittest src.data_processing.tests.test_clean_data.TestDataCleaner
    """
    
    def setUp(self):
        """
        测试前的准备工作
        """
        self.input_file = 'data/test.csv'
        self.output_file = 'data/processed/test_utf8.csv'
        self.test_content = '测试内容'
    
    def test_init_with_output_file(self):
        """
        测试指定输出文件的情况
        执行命令:
        python -m unittest src.data_processing.tests.test_clean_data.TestDataCleaner.test_init_with_output_file
        """
        cleaner = DataCleaner(self.input_file, self.output_file)
        self.assertEqual(str(cleaner.input_file), self.input_file)
        self.assertEqual(str(cleaner.output_file), self.output_file)
    
    def test_init_without_output_file(self):
        """
        测试不指定输出文件的情况
        执行命令:
        python -m unittest src.data_processing.tests.test_clean_data.TestDataCleaner.test_init_without_output_file
        """
        cleaner = DataCleaner(self.input_file)
        expected_output = str(Path('data/processed') / Path(self.input_file).name)
        self.assertEqual(str(cleaner.output_file), expected_output)
    
    @patch('builtins.open', new_callable=mock_open, read_data='测试内容')
    @patch('pathlib.Path.mkdir')
    def test_clean_success(self, mock_mkdir, mock_file):
        """
        测试成功清洗数据的情况
        执行命令:
        python -m unittest src.data_processing.tests.test_clean_data.TestDataCleaner.test_clean_success
        """
        cleaner = DataCleaner(self.input_file, self.output_file)
        cleaner.clean()
        
        # 验证文件操作
        mock_file.assert_any_call(self.input_file, 'r', encoding='gb2312')
        mock_file.assert_any_call(self.output_file, 'w', encoding='utf-8')
        
        # 验证写入操作
        handle = mock_file()
        handle.write.assert_called_once_with(self.test_content)
    
    @patch('builtins.open')
    @patch('pathlib.Path.mkdir')
    def test_clean_decode_error(self, mock_mkdir, mock_file):
        """
        测试编码错误的情况
        执行命令:
        python -m unittest src.data_processing.tests.test_clean_data.TestDataCleaner.test_clean_decode_error
        """
        mock_file.side_effect = UnicodeDecodeError('gb2312', b'', 0, 1, '测试错误')
        
        cleaner = DataCleaner(self.input_file, self.output_file)
        cleaner.clean()  # 应该正常处理异常，不会抛出


if __name__ == '__main__':
    unittest.main() 