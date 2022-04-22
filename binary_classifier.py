"""
author @leannmendoza

This file contains the train_model() method which uses a tensorflow 
written with keras to train a sequential model with adam optimizer. 
Takes in pre-populated train and validation csv's to contruct the model
and saves it as an .h5 file. 

These parameters can be defined in the clargs when running this code
independantly.

On command line type $ python binary_classifer.py -h for information on clargs.

Requires python 3.8
"""
import os
from os.path import exists
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import preprocess_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse


def train_model(path_to_train_csv, path_to_validation_csv, model_filename, overwrite):
	"""
	params: path_to_train_csv - Path to csv containing training data
			path_to_validation_csv - Path to csv containing validation data
			model_filename - Path to .h5 file containing model
			overwrite - boolean denotes if model is to be overwritten if exists
	return: model - trained model
	Uses a tensorflow written with keras to train a sequential model with 
	adam optimizer. Takes in pre-populated train and validation csv's to 
	contruct the model and saves it as an .h5 file. 
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

		model.fit(train_dataset,
				steps_per_epoch = 250,
				epochs = 10,
				validation_data = validation_dataset
			)

		model.save(model_filename)
	else:
		raise FileNotFoundError('Training and/or Validation file does not exist.')
	return model


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-tr', '--train_filename', type=str, help='Filename to save training csv')
	parser.add_argument('-v', '--validation_filename', type=str, help='Filename to save validation csv')
	parser.add_argument('-m', '--model_filename', type=str, help='Filename to save model (.h5 file)')
	parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing files and model (used for constructing new model)')
	parser.set_defaults(train_filename="train.csv")
	parser.set_defaults(validation_filename="validation.csv")
	parser.set_defaults(model_filename="model.h5")

	args = parser.parse_args()
	train_file = args.train_filename
	validation_file = args.validation_filename
	model_file = args.model_filename
	overwrite = args.overwrite

	model = train_model(train_file, validation_file, model_file, overwrite)
	model.summary()
	
if __name__ == '__main__':
	main()
