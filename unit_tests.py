"""
author @leannmendoza

Unit tests for functions non-reliant on databases/pre-populated data/files
"""
import unittest
import pandas as pd
import ctypes
import os
from preprocess_data import *
from analysis import *
from binary_classifier import *

class TestPreprocessMethods(unittest.TestCase):
	""" Unit tests for functions non-reliant on databases/pre-populated data/files """

	def test_print_binary_truth_counts1(self):
		df = pd.DataFrame({'img_path':['a.jpg','b.jpg','c.jpg'],'truth':[0,1,1]})
		self.assertEqual(print_binary_truth_counts(df, "test"), (1,2))

	def test_print_binary_truth_counts2(self):
		df = pd.DataFrame({'img_path':['a.jpg','b.jpg','c.jpg'],'truth':[0,0,1]})
		self.assertEqual(print_binary_truth_counts(df, "test"), (2,1))

	def test_add_none_column(self):
		df = pd.DataFrame([1,2,3])
		self.assertEqual(add_none_column(df).size, 6)

	def test_df_to_csv_file_creation(self):
		df = pd.DataFrame({'img_path':['a.jpg','b.jpg','c.jpg'],'truth':[0,1,1]})
		csv_filename = 'temp_unit_testing.csv'
		overwrite = False
		df_to_csv(df, csv_filename, overwrite)
		self.assertTrue(os.path.exists(csv_filename))

	def test_df_to_csv_file_accuracy(self):
		df = pd.DataFrame({'img_path':['a.jpg','b.jpg','c.jpg'],'truth':[0,1,1]})
		csv_filename = 'temp_unit_testing.csv'
		overwrite = False
		df_to_csv(df, csv_filename, overwrite)
		df_upload = pd.read_csv(csv_filename)
		self.assertTrue(df.equals(df_upload))


if __name__ == '__main__':
    unittest.main()
