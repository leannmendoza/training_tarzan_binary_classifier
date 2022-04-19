"""
author @leannmendoza

These methods are used in the extraction and preprocessing of our data
for our ml functionalites

Requires python 3.8
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

LIKE_COLOR = 'green'
DISLIKE_COLOR = 'red'
def create_binary_df(class1_path, class2_path):
	"""
	params: class1_path - path to dir containing images belonging to class 1
			class2_path - path to dir containing images belonging to class 2
	returns: df - dataframe containing absolute image path and the truth (class1('1') or class2('0'))
	Reads in files in 2 directories (class1 or class2) and creates a df with 'img_path' and 'truth'
	as columns.
	"""
	full_class1_path = os.path.abspath(class1_path)
	full_class2_path = os.path.abspath(class2_path)
	image_dictionary = {}
	for file in os.listdir(class1_path):
		image_dictionary[full_class1_path + '/' + file] = "likes"
	for file in os.listdir(class2_path):
		image_dictionary[full_class2_path + '/' + file] = "dislikes"

	df = pd.DataFrame(image_dictionary.items(), columns=['img_path', 'truth'])
	df = df.sample(frac=1).reset_index(drop=True) # shuffle rows in random order
	return df

def visualize_data(df, title):
	"""
	params: df - dataframe with binary truth values
			title - title of bar plot
	returns: None
	Creates a bar plot of pandas df "truth" column to show distributation of likes vs. dislikes
	"""
	ax = df['truth'].value_counts().plot(kind='barh', color=(DISLIKE_COLOR ,LIKE_COLOR))
	ax.set_title(title)
	ax.set_xlabel('Number of Images')
	ax.set_ylabel('Truth')
	plt.show()
	return None

def print_binary_truth_counts(df, name_of_df):
	"""
	params: df - dataframe with binary truth values
			name_of_df - name of the dataframe
	returns: None
	Prints the "truth" value counts of the dataframe
	"""
	print("Truth distrubution in", name_of_df)
	print(df['truth'].value_counts())
	return None

def split_data(df):
	"""
	params: df - master dataframe containing all of the images
	returns: train, test, validation - resulting dataframes of 85-10-5% split
	Splits data by 85%, 10%, and 5% to construct the datasets to be parsed into ml model
	"""
	train, test, validation = np.split(df, [int(.85*len(df)), int(.95*len(df))]) # 85, 10, 5 split
	return train, test, validation

def add_none_column(df):
	"""
	params: df - dataframe containing only image path and truth columns
	returns: df - dataframe containing image path, truth columns, and prediction column
	Adds prediction column to pandas df to be used in model testing
	"""
	df["prediction"] = None
	return df

def show_image_with_label(path_to_image, label):
	"""
	params: path_to_image - path to image to be shown
			label - should be one of the column values in dataframe
	returns: None
	Shows image with label either like or dislike if 'truth' or 'prediction'
	in label param.
	"""
	im = PIL.Image.open(path_to_image)
	draw = ImageDraw.Draw(im)
	font = ImageFont.truetype("Arial.ttf", 30)
	color = 'white'
	draw.text((10, 10), label, color,font=font)
	im.show()
	return None

def show_image(path_to_image):
	"""
	params: path_to_image - path to image to be shown
	returns: None
	Shows image on screen
	"""
	im = PIL.Image.open(path_to_image)
	im.show()
	return None

def df_to_csv(df, csv_file_name):
	"""
	params: df - dataframe containing image data (path, truth, and/or predictions)
			csv_file_name - desired name to name the new csv file
	returns: None
	Creates a csv file from pandas dataframe
	"""
	filepath = os.path.abspath(csv_file_name)
	df.to_csv(filepath, index = False)
	return None

def main():
	likes_path = "./likes/"
	dislikes_path = "./dislikes/"
	truth_df = create_binary_df(likes_path,dislikes_path)
	train_df, test_df, validation_df = split_data(truth_df)
	visualize_data(train_df, 'training data')
	visualize_data(test_df, 'testing data')
	visualize_data(validation_df, 'validation data')
	show_image_with_label(train_df['img_path'][0], train_df['truth'][0])
	print_binary_truth_counts(train_df, "train_df")
	df_to_csv(train_df, "train.csv")
	df_to_csv(validation_df, "validation.csv")
	df_to_csv(test_df, "test.csv")

if __name__ == '__main__':
	main()