import unittest
import pandas as pd
from preprocess_data import *
from analysis import *
from binary_classifier import *

truth_df = pd.DataFrame()
class TestPreprocessMethods(unittest.TestCase):

	def test_create_binary_df(self):
		likes_path = "./data_for_unit_tests/likes/"
		dislikes_path = "./data_for_unit_tests/dislikes/"
		truth_df = create_binary_df(likes_path,dislikes_path)
		self.assertIsInstance(truth_df, pd.core.frame.DataFrame)
		self.assertEqual(truth_df.size, 128)

	def test_split_data(self):
		train_df, test_df, validation_df = split_data(truth_df)
		self.assertIsInstance(split_data(truth_df), tuple)
		df_to_csv(train_df, "./data_for_unit_tests/train.csv")
		df_to_csv(validation_df, "./data_for_unit_tests/validation.csv")
		df_to_csv(test_df, "./data_for_unit_tests/test.csv")

	def test_add_none_column(self):
		df = pd.DataFrame([1,2,3])
		self.assertEqual(add_none_column(df).size, 6)

class TestBinaryClassifier(unittest.TestCase):

	def test_organize_files(self):
		csvs = ["./data_for_unit_tests/train.csv", "./data_for_unit_tests/validation.csv", "./data_for_unit_tests/test.csv"]
		self.assertEqual(len(organize_files(csvs)), 3)

if __name__ == '__main__':
    unittest.main()