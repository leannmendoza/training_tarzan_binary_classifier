"""
author @leannmendoza

These methods are used for the analysis of our ml models. Functions such as confusion matrix and auroc

Requires python 3.8
"""
import os
from tensorflow import keras
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, model_selection, svm
from tensorflow.keras.preprocessing import image
import argparse
from tqdm import tqdm

from preprocess_data import add_none_column, df_to_csv, show_image_with_label

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
	if os.path.exists(predictions_filename):
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

def confusion_matrix(predictions_file):
	"""
	params: df - dataframe that contains image path, truth and model predictions
	returns: confusion matrix with tp, tn, fp, fn results from our data
	Constructs a confusion matrix (df) from our results
	"""
	df = pd.read_csv(predictions_file)
	confusion_matrix = pd.crosstab(df['truth'], df['prediction'], rownames=['Truth'], colnames=['Predicted'])
	sn.heatmap(confusion_matrix, annot=True, fmt='g', cmap="Spectral")
	plt.show()

	return confusion_matrix

def auroc(predictions_file):
	"""
	params: df - dataframe that contains image path, truth and model predictions
	returns: auc - accuracy value of our model predictions on independant test set
	Calculates and graphs AUC 
	"""
	df = pd.read_csv(predictions_file)
	# calculate roc curve
	df["truth"] = np.where(df["truth"] == "likes", 1, 0)
	df["prediction"] = np.where(df["prediction"] == "likes", 1, 0)
	fpr, tpr, thresholds = metrics.roc_curve(df['truth'], df['prediction'])

	# calculate AUC
	auc = metrics.roc_auc_score(df['truth'], df['prediction'])
	print('AUC: %.3f' % auc)
	plt.plot(fpr, tpr)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.text(0.5, 0.5, 'AUC: %.3f' % auc , horizontalalignment='center',
			verticalalignment='center')
	plt.show()

	return auc


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('class1_label', type=str, help='Label for images of class1')
	parser.add_argument('class2_label', type=str, help='Label for images of class2')
	parser.add_argument('-te', '--test_filename', type=str, help='Filename to save test csv')
	parser.add_argument('-m', '--model_filename', type=str, help='Filename to save model (.h5 file)')
	parser.add_argument('-p', '--predictions_filename', type=str, help='Filename to save predictions csv')
	parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing files and model (used for constructing new model)')
	parser.set_defaults(test_filename="test.csv")
	parser.set_defaults(model_filename="model.h5")
	parser.set_defaults(predictions_filename="predictions.csv")

	args = parser.parse_args()
	c1_label = args.class1_label
	c2_label = args.class2_label
	test_file = args.test_filename
	predictions_file = args.predictions_filename
	model_file = args.model_filename
	overwrite = args.overwrite

	model = keras.models.load_model(model_file)
	predict_test(model, c1_label, c2_label, test_file, predictions_file, overwrite)
	df = pd.read_csv(test_file)
	for i in range(10):
		predict_image(model, df['img_path'][i], c1_label, c2_label)

	confusion_matrix(predictions_file)
	auroc(predictions_file)

if __name__ == '__main__':
	main()
