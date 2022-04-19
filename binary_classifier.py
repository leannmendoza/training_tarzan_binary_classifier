"""
author @leannmendoza

These methods are used in the construction of the ml model using tensorflow and keras.

Requires python 3.8
"""
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
from preprocess_data import add_none_column, df_to_csv

def organize_files(filenames):
	"""
	params: filenames
	return: dirs
	Takes list of filenames and for each file within the filename organizes the files
	by creating directories. i.e. train/dislikes and train/dislikes. This makes it easier
	to simply pass in a directory path into the train_model function.
	"""
	dirs = []
	for file in filenames:
		dataset_type = file.split('.')[0]
		new_dataset_dir = os.getcwd() + '/' + dataset_type
		dirs.append(new_dataset_dir)
		if not os.path.exists(new_dataset_dir):
			os.makedirs(new_dataset_dir)
			os.makedirs(new_dataset_dir + '/likes/')
			os.makedirs(new_dataset_dir + '/dislikes/')
			print("The new directory", new_dataset_dir,"is created! Adding Images...")

		with open(file) as f:
			next(f)
			for line in f:
				img_path, truth = line.strip().split(',')
				img_name = img_path.split('/')[-1]
				new_dataset_dir_truth = new_dataset_dir + '/' + truth + '/'
				new_file_path = new_dataset_dir_truth + img_name
				if not os.path.exists(new_file_path): 
					shutil.copy(os.path.abspath(img_path), new_dataset_dir_truth)
			print('All images added to', dataset_type, 'directory!')
	return dirs

def train_model(path_to_train, path_to_validation):
	"""
	params: path_to_train, path_to_validation
	return: model
	This function constructs the ml model
	"""

	train = ImageDataGenerator(rescale=1/255)
	validation = ImageDataGenerator(rescale=1/255)

	train_dataset = train.flow_from_directory(directory = path_to_train, 
											target_size=(150,150),
											batch_size = 32,
											class_mode = 'binary')
										 
	validation_dataset = validation.flow_from_directory(directory = path_to_validation, 
											target_size=(150,150),
											batch_size =32,
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

	model.save("training_tarzan.h5")
	return model

def predict_image(model, filename):
	"""
	params: model, filename
	return: val
	See how the model performs on one image.
	"""
	print('Testing model performance on', filename, '...')
	img1 = image.load_img(filename,target_size=(150,150))
	plt.imshow(img1)
	Y = image.img_to_array(img1)
	X = np.expand_dims(Y,axis=0)
	val = model.predict(X)[0]
	print(val)
	if val == 1:
		plt.xlabel("likes",fontsize=30)
	elif val == 0:
		plt.xlabel("dislikes",fontsize=30)
	plt.show()

	return val

def predict_test(model, test_filename, save_file=None):
	"""
	params: model, test_filename, save_file=None
	return: df
	See how the model performs on the independant test set
	"""
	print('Testing model performance on independant test set...')
	df = pd.read_csv(test_filename)
	add_none_column(df)
	for index, row in df.iterrows():
		img1 = image.load_img(row['img_path'],target_size=(150,150))
		Y = image.img_to_array(img1)
		X = np.expand_dims(Y,axis=0)
		val = model.predict(X)[0]
		# round our results (although mostly 0/1)
		if val >= 0.5:
			prediction = 'likes'
		else:
			prediction = 'dislikes'
		df.iloc[index, df.columns.get_loc('prediction')] = prediction
	print(df)
	if save_file:
		df_to_csv(df, 'predictions.csv')
		print('Predictions saved!')
	return df


def main():
	csvs = ['train.csv', 'validation.csv', 'test.csv']
	dirs = organize_files(csvs)
	if not exists('training_tarzan.h5'):
		model = train_model(dirs[0]+'/', dirs[1]+'/')
		model.summary()
	tt_model = keras.models.load_model('training_tarzan.h5')
	predict_test(tt_model, 'test.csv', save_file = True)

if __name__ == '__main__':
	main()