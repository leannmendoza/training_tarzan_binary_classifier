"""
author @leannmendoza

These methods are used in the construction of the ml model using tensorflow and keras.

Requires python 3.8
"""
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from os.path import exists
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import preprocess_data
from preprocess_data import add_none_column, df_to_csv, show_image_with_label
import argparse


def train_model(path_to_train_csv, path_to_validation_csv, model_filename, overwrite):
	"""
	params: path_to_train, path_to_validation
	return: model
	This function constructs the ml model
	"""
	if os.path.exists(model_filename):
		if not overwrite:
			print("Retrieving previously trained model...")
			model = keras.models.load_model(model_filename)
			return model

	if os.path.exists(path_to_train_csv) and os.path.exists(path_to_validation_csv):
		print("*********************************\n"\
			  "       Training Model            \n"\
			  "*********************************")
		train_df = pd.read_csv(path_to_train_csv)
		validation_df = pd.read_csv(path_to_validation_csv)

		train = ImageDataGenerator(rescale=1/255)
		validation = ImageDataGenerator(rescale=1/255)

		train_dataset = train.flow_from_dataframe(dataframe=train_df,
												target_size=(150,150),
												x_col="img_path",
												y_col="truth",
												batch_size = 32,
												validate_filenames=True,
												class_mode = 'binary')
											 
		validation_dataset = validation.flow_from_dataframe(dataframe=validation_df,
												target_size=(150,150),
												x_col="img_path",
												y_col="truth",
												batch_size =32,
												validate_filenames=True,
												class_mode = 'binary')
		print(train_dataset.class_indices)
		print(validation_dataset.class_indices)

		model = keras.Sequential()

		# Convolutional layer and maxpool layer 1
		model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
		model.add(keras.layers.MaxPool2D(2,2))

		# Convolutional layer and maxpool layer 2
		model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
		model.add(keras.layers.MaxPool2D(2,2))

		# Convolutional layer and maxpool layer 3
		model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
		model.add(keras.layers.MaxPool2D(2,2))

		# Convolutional layer and maxpool layer 4
		model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
		model.add(keras.layers.MaxPool2D(2,2))

		# This layer flattens the resulting image array to 1D array
		model.add(keras.layers.Flatten())

		# Hidden layer with 512 neurons and Rectified Linear Unit activation function 
		model.add(keras.layers.Dense(512,activation='relu'))

		# Output layer with single neuron which gives 0 for dislike or 1 for like
		#Here we use sigmoid activation function which makes our model output to lie between 0 and 1
		model.add(keras.layers.Dense(1,activation='sigmoid'))

		model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


		#steps_per_epoch = train_imagesize/batch_size

		model.fit(train_dataset,
				steps_per_epoch = 250,
				epochs = 10,
				validation_data = validation_dataset
			)

		model.save(model_filename)
	else:
		raise FileNotFoundError('Training and/or Validation file does not exist.')
	return model

def predict_image(model, filename, class1_label, class2_label):
	"""
	params: model, filename
	return: val
	See how the model performs on one image.
	"""
	img1 = image.load_img(filename,target_size=(150,150))
	plt.imshow(img1)
	Y = image.img_to_array(img1)
	X = np.expand_dims(Y,axis=0)
	val = model.predict(X)[0]
	if val >= 0.5:
		plt.xlabel(class1_label,fontsize=10)
	else:
		plt.xlabel(class2_label,fontsize=10)
	plt.show()

	return val

def predict_test(model, class1_label, class2_label, test_filename, predictions_filename, overwrite):
	"""
	params: model, test_filename, save_file=None
	return: df
	See how the model performs on the independant test set
	"""
	print("*********************************\n"\
		  "    Calculating Predictions      \n"\
		  "*********************************")
	if os.path.exists("predictions.csv"):
		if not overwrite:
			print("Retrieving previously saved predictions...")
			df = pd.read_csv(predictions_filename)
			return df
	print('Testing model performance on independant test set...')
	df = pd.read_csv(test_filename)
	add_none_column(df)
	for index, row in tqdm(df.iterrows()):
		img1 = image.load_img(row['img_path'],target_size=(150,150))
		Y = image.img_to_array(img1)
		X = np.expand_dims(Y,axis=0)
		val = model.predict(X)[0]
		# round our results (although mostly 0/1)
		if val >= 0.5:
			prediction = class1_label
		else:
			prediction = class2_label
		df.iloc[index, df.columns.get_loc('prediction')] = prediction
	print(df)
		
	df_to_csv(df, predictions_filename, overwrite)
	print('Predictions saved!')
	return df


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('class1_label', type=str, help='Label for images of class1')
	parser.add_argument('class2_label', type=str, help='Label for images of class1')
	parser.add_argument('-tr', '--train_filename', type=str, help='Filename to save training csv')
	parser.add_argument('-v', '--validation_filename', type=str, help='Filename to save validation csv')
	parser.add_argument('-te', '--test_filename', type=str, help='Filename to save test csv')
	parser.add_argument('-p', '--predictions_filename', type=str, help='Filename to save predictions csv')
	parser.add_argument('-m', '--model_filename', type=str, help='Filename to save model (.h5 file)')
	parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing files and model (used for constructing new model)')
	parser.set_defaults(train_filename="train.csv")
	parser.set_defaults(validation_filename="validation.csv")
	parser.set_defaults(test_filename="test.csv")
	parser.set_defaults(predictions_filename="predictions.csv")
	parser.set_defaults(model_filename="model.h5")

	args = parser.parse_args()
	c1_label = args.class1_label
	c2_label = args.class2_label
	train_file = args.train_filename
	validation_file = args.validation_filename
	test_file = args.test_filename
	predictions_file = args.predictions_filename
	model_file = args.model_filename
	overwrite = args.overwrite

	model = train_model(train_file, validation_file, model_file, overwrite)
	model.summary()
	predict_test(model, c1_label, c2_label, test_file, predictions_file, overwrite)
	df = pd.read_csv(test_file)
	for i in range(10):
		predict_image(model, df['img_path'][i], c1_label, c2_label)
	
if __name__ == '__main__':
	main()
