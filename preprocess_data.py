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
import seaborn as sn
from tensorflow.keras.preprocessing import image
import argparse

def create_binary_df(class1_path, class2_path, class1_label, class2_label, all_file, overwrite):
	"""
	params: class1_path - path to dir containing images belonging to class 1
			class2_path - path to dir containing images belonging to class 2
			class1_label - string name of class1 (1st of 2 classes)
			class2_label - string name of class2 (2nd of 2 classes)
			overwrite - boolean denoting if existing files should be overwritten
	returns: df - dataframe containing 2 columns; image path and the truth
	Reads in files contained in 2 directories (class1 or class2) and 
	creates a df with 'img_path' and 'truth' as columns. 
	"""
	if os.path.exists(all_file):
		if not overwrite:
			print("Reading previously saved all_file...")
			df = pd.read_csv(all_file)
			return df

	print("Constructing new all file dataframe...")
	full_class1_path = os.path.abspath(class1_path)
	full_class2_path = os.path.abspath(class2_path)
	image_dictionary = {}
	for file in os.listdir(class1_path):
		if file.lower().endswith(('.jpg', '.jpeg', '.png')):
			image_dictionary[full_class1_path + '/' + file] = class1_label
	for file in os.listdir(class2_path):
		if file.lower().endswith(('.jpg', '.jpeg', '.png')):
			image_dictionary[full_class2_path + '/' + file] = class2_label

	df = pd.DataFrame(image_dictionary.items(), columns=['img_path', 'truth'])
	df = df.sample(frac=1).reset_index(drop=True) # shuffle rows in random order
	print("Data size:", len(df.index), "total images")
	df_to_csv(df, all_file, overwrite)
	return df

def split_data(df, train_file, validation_file, test_file, overwrite):
	"""
	params: df - master dataframe containing all of the images
			overwrite - boolean denoting if existing files should be overwritten
	returns: train, test, validation - resulting dataframes of 85-10-5% split
	Splits data by 85%, 10%, and 5% to construct the train, test, validation 
	datasets, respectively, to be parsed into ml model
	"""
	if os.path.exists(train_file) and os.path.exists(validation_file) and os.path.exists(test_file):
		if not overwrite:
			print("Reading previously saved training, testing, validation data...")
			train = pd.read_csv(train_file)
			test = pd.read_csv(test_file)
			validation = pd.read_csv(validation_file)
			return train, test, validation

	train, test, validation = np.split(df, [int(.85*len(df)), int(.95*len(df))])
	df_to_csv(train, train_file, overwrite)
	df_to_csv(test, test_file, overwrite)
	df_to_csv(validation, validation_file, overwrite)
	return train, test, validation

def make_autopct(values):
	"""
	params: values
	returns: my_autopct
	Formats data for pie chart with percentage and number
	"""
	def my_autopct(pct):
		total = sum(values)
		val = int(round(pct*total/100.0))
		return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
	return my_autopct

def visualize_data(df, title, class1_label, class2_label):
	"""
	params: df - dataframe with binary truth values
			title - title of bar plot
			class1_label - string name of class1 (1st of 2 classes)
			class2_label - string name of class2 (2nd of 2 classes)
	returns: None
	Creates a bar plot of pandas df "truth" column to show distributation of likes vs. dislikes
	"""
	colors = sn.color_palette("Spectral")
	ax = plt.pie(df['truth'].value_counts(), labels = [class1_label, class2_label], colors = colors, autopct=make_autopct(df['truth'].value_counts()))
	#add overall title to replot
	plt.title(title)
	plt.show()
	return None


def print_binary_truth_counts(df, name_of_df):
	"""
	params: df - dataframe with binary truth values
			name_of_df - name of the dataframe
	returns: n_class1, n_class2 - tuple with number of images
			 in each class
	Prints the "truth" value counts of the dataframe
	"""
	print("Truth distrubution in", name_of_df)
	print(df['truth'].value_counts())
	n_class1 = df['truth'].value_counts()[0]
	n_class2 = df['truth'].value_counts()[1]
	return n_class1, n_class2

def add_none_column(df):
	"""
	params: df - dataframe containing only image path and truth columns
	returns: df - dataframe containing image path, truth columns, 
				  and prediction column
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
	img1 = image.load_img(path_to_image,target_size=(150,150))
	plt.imshow(img1)
	Y = image.img_to_array(img1)
	X = np.expand_dims(Y,axis=0)
	plt.xlabel(label,fontsize=10)
	plt.show()
	return None

def show_image(path_to_image):
	"""
	params: path_to_image - path to image to be shown
	returns: None
	Shows image on screen
	"""
	img1 = image.load_img(path_to_image,target_size=(150,150))
	plt.imshow(img1)
	Y = image.img_to_array(img1)
	X = np.expand_dims(Y,axis=0)
	plt.show()
	return None

def df_to_csv(df, csv_filename, overwrite):
	"""
	params: df - dataframe containing image data (path, truth, and/or predictions)
			csv_filename - desired name to name the new csv file
			overwrite - boolean denoting if existing files should be overwritten
	returns: None
	Creates a csv file from pandas dataframe
	"""
	if os.path.exists(csv_filename):
		if not overwrite:
			print('File already exists. No new file created.')
			return None

	filepath = os.path.abspath(csv_filename)
	print("Creating file", filepath)
	df.to_csv(filepath, index=False)
	return None

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('class1_path', type=str, help='Path to directory containing images of class1')
	parser.add_argument('class2_path', type=str, help='Path to directory containing images of class2')
	parser.add_argument('class1_label', type=str, help='Label for images of class1')
	parser.add_argument('class2_label', type=str, help='Label for images of class1')
	parser.add_argument('-a', '--all_filename', type=str, help='Filename to save all data to csv')
	parser.add_argument('-tr', '--train_filename', type=str, help='Filename to save training csv')
	parser.add_argument('-v', '--validation_filename', type=str, help='Filename to save validation csv')
	parser.add_argument('-te', '--test_filename', type=str, help='Filename to save test csv')
	parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing files (used for constructing new model)')
	parser.set_defaults(all_filename="all_file.csv")
	parser.set_defaults(train_filename="train.csv")
	parser.set_defaults(validation_filename="validation.csv")
	parser.set_defaults(test_filename="test.csv")

	args = parser.parse_args()
	c1_path = os.path.abspath(args.class1_path)
	c2_path = os.path.abspath(args.class2_path)
	c1_label = args.class1_label
	c2_label = args.class2_label
	all_file = args.all_filename
	train_file = args.train_filename
	validation_file = args.validation_filename
	test_file = args.test_filename
	overwrite = args.overwrite

	truth_df = create_binary_df(c1_path, c2_path, c1_label, c2_label, all_file, overwrite)
	train_df, test_df, validation_df = split_data(truth_df, train_file, validation_file, test_file, overwrite)
	visualize_data(train_df, 'Training data', c1_label, c2_label)
	visualize_data(validation_df, 'Validation data', c1_label, c2_label)
	visualize_data(test_df, 'Testing data', c1_label, c2_label)

	for i in range(10):
		show_image_with_label(truth_df['img_path'][i], truth_df['truth'][i])


if __name__ == '__main__':
	main()
