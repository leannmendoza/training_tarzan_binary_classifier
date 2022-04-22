"""
author @leannmendoza

Main driver code for this package. Simulates the user creating their ML model by walking the user through the process.
This use case is for training a model with two classes "likes" and "dislikes" where likes dataset includes images
of cute animals and flowers, and "dislikes" dataset includes images of snakes and rocks. (See README.md for explaination.)

"""
from preprocess_data import *
from binary_classifier import *
from analysis import *

def import_data(c1_path, c2_path, c1_label, c2_label, all_file, train_file, validation_file, test_file, overwrite):
	"""
	params: c1_path - path to dir containing images belonging to class 1
			c2_path - path to dir containing images belonging to class 2
			c1_label - string name of class1 (1st of 2 classes)
			c2_label - string name of class2 (2nd of 2 classes)
			all_file - path to .csv file to save all data
			train_file - path to .csv file to save all training data
			validation_file - path to .csv file to save all validation data
			test_file - path to .csv file to save all test data
			overwrite - boolean denoting if existing files should be overwritten
	returns: train_df, test_df, validation_df - dataframes containing train, test, and validation data

	Parses the image data from all_file containing 2 rows 'img_path' and 'truth', 
	randomly arranges it, then splits the data into 85-10-5 split for training, testing, and validation, 
	respectively. Returns the df for training, test, validation data with img_path and truth columns
	"""
	truth_df = create_binary_df(c1_path, c2_path, c1_label, c2_label, all_file, overwrite)
	train_df, test_df, validation_df = split_data(truth_df, train_file, validation_file, test_file, overwrite)

	return truth_df, train_df, test_df, validation_df

def create_model(train_file, validation_file, model_file, overwrite):
	"""
	params: params: train_file - Path to csv containing training data
			validation_file - Path to csv containing validation data
			model_file - Path to .h5 file containing model
			overwrite - boolean denotes if model is to be overwritten if exists
	return: model - trained keras model

	Uses a tensorflow written with keras to train a sequential model with 
	adam optimizer. Takes in pre-populated train and validation csv's to 
	contruct the model and saves it as an .h5 file. 
	"""
	model = train_model(train_file, validation_file, model_file, overwrite)
	return model

def run_analysis(trained_model, class1_label, class2_label, test_file, predictions_file, overwrite):
	"""
	params: trained_model - Path to .h5 file containing model
			class1_label - string name of class1 (1st of 2 classes)
			class2_label - string name of class2 (2nd of 2 classes)
			test_file - path to .csv file to save all test data
			predictions_file - path to .csv file to save all predictions
							   with 3 columns; img_path, truth, and prediction
			overwrite - boolean denotes if files are to be overwritten if exists
	returns: None
	Get predictions from trained model. Visualize results/predictions through
	confusion matrix, and auroc curves/accuracy.
	"""
	predict_test(trained_model, class1_label, class2_label, test_file, predictions_file, overwrite)
	confusion_matrix(predictions_file)
	auroc(predictions_file)
	return None


def sample_images(model, filename, class1_label, class2_label, n):
	"""
	params: model - Path to .h5 file containing model
			filename - Path to image file to test model performance on
			class1_label - string name of class1 (1st of 2 classes)
			class2_label - string name of class2 (2nd of 2 classes)
			n - number of images to show
	returns: None
	Display n images and the predicted result of the model
	"""
	df = pd.read_csv(filename)
	for i in range(n):
		predict_image(model, df['img_path'][i], class1_label, class2_label)
	return None


def main():
	welcome_msg = "Training Tarzan based on Netflix's Start-Up Kdrama\n" \
				  "On their first date, Dosan introduces Dalmi, who is a newbie \n" \
				  "to AI tech, to the concept of machine learning using the metaphor \n" \
				  "Tarzan. Tarzan, having never seen a human, learns Jane's ways through\n" \
				  "building experiences that he would learn from to win her heart.\n" \
				  "Tarzan gives a flower, she becomes happy. He gives her a snake, she screams.\n" \
				  "Rabbit, happy. Rock, not happy. In this program we will mimic this learning experience.\n"\
				  "through machine learning using pandas, matplotlib, numpy, and tensorflow. \n"
	print(welcome_msg)

	load_data_msg = 'First lets load our data. We have cute animals and flowers for "likes" and rocks and snakes for "dislikes"...'
	input(load_data_msg)
	c1_path = './training_tarzan_data/likes/'
	c2_path = './training_tarzan_data/dislikes/'
	c1_label = 'likes'
	c2_label = 'dislikes'
	all_file = 'all_file.csv'
	overwrite = True
	train_file = 'train.csv'
	validation_file = 'validation.csv'
	test_file = 'test.csv'
	model_file = 'model.h5'
	predictions_file = 'predictions.csv'
	n_images_to_show = 10
	truth_df, train_df, test_df, validation_df = import_data(c1_path, c2_path, c1_label, c2_label, all_file, train_file, validation_file, test_file, overwrite)

	see_data_msg = "Now that we have the data loaded, would you like to see the distributions of like and dislike?\n"
	input(see_data_msg)
	visualize_data(truth_df, 'All data', c1_label, c2_label)
	visualize_data(train_df, 'Training data', c1_label, c2_label)
	visualize_data(validation_df, 'Validation data', c1_label, c2_label)
	visualize_data(test_df, 'Testing data', c1_label, c2_label)

	see_imgs_msg = "Let's pull up some images in our data. Shall we?\n"
	input(see_imgs_msg)
	for i in range(n_images_to_show):
		show_image_with_label(train_df['img_path'][i], train_df['truth'][i])

	train_model_msg = "Now that we, have seen our data. Let's train our model using tensorflow..."
	input(train_model_msg)
	trained_model = create_model(train_file, validation_file, model_file, overwrite)

	analysis_msg = "Great now let's see how tarzan learned!"
	input(analysis_msg)
	run_analysis(trained_model, c1_label, c2_label, test_file, predictions_file, overwrite)

	sample_image_predictions = "Now let's see what tarzan thinks of items in our test set..."
	input(sample_image_predictions)
	sample_images(trained_model, predictions_file, c1_label, c2_label, n_images_to_show)

	end_msg = "Thank you for participating in this simulation with Jane and Tarzan!"
	print(end_msg)

if __name__ == '__main__':
	main()
