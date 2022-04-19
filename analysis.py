"""
author @leannmendoza

These methods are used for the analysis of our ml models. Functions such as confusion matrix and auroc

Requires python 3.8
"""
from tensorflow import keras
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, model_selection, svm

def confusion_matrix(df):
	"""
	params: df - dataframe that contains image path, truth and model predictions
	returns: confusion matrix with tp, tn, fp, fn results from our data
	Constructs a confusion matrix (df) from our results
	"""
	confusion_matrix = pd.crosstab(df['truth'], df['prediction'], rownames=['Truth'], colnames=['Predicted'])
	sn.heatmap(confusion_matrix, annot=True, fmt='g', cmap="Spectral")
	plt.show()

	return confusion_matrix

def auroc(df):
	"""
	params: df - dataframe that contains image path, truth and model predictions
	returns: auc - accuracy value of our model predictions on independant test set
	Calculates and graphs AUC 
	"""
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
	df = pd.read_csv('predictions.csv')
	confusion_matrix(df)
	auroc(df)

if __name__ == '__main__':
	main()