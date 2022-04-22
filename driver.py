"""
author @leannmendoza

Main driver code for this package. Simulates the user creating their ML model by walking the user through the process.
Options to visualize results. 
"""
from preprocess_data import *
from binary_classifier import *
from analysis import *

def import_data(c1_path, c2_path, c1_label, c2_label, all_file, train_file, validation_file, test_file, overwrite):
	"""
	params: None
	returns: train_df, test_df, validation_df
	Collects the data from preconstructed image directories into training/valid/test dfs
	"""
	truth_df = create_binary_df(c1_path, c2_path, c1_label, c2_label, all_file, overwrite)
	train_df, test_df, validation_df = split_data(truth_df, train_file, validation_file, test_file, overwrite)

	return truth_df, train_df, test_df, validation_df

def create_model(train_file, validation_file, model_file, overwrite):
	"""
	params: None
	returns: tt_model
	Creates the ML model based on data provided, saves into h5 file and gets predictions
	"""
	model = train_model(train_file, validation_file, model_file, overwrite)
	return model

def run_analysis(trained_model, class1_label, class2_label, test_file, predictions_file, overwrite):
	"""
	params: None
	returns: None
	Runs analysis in our predictions by constructing a confusion matrix as well as auroc curve
	which allows us to know how accurate our predictions are.
	"""
	predict_test(trained_model, class1_label, class2_label, test_file, predictions_file, overwrite)
	confusion_matrix(predictions_file)
	auroc(predictions_file)


def sample_images(model, filename, class1_label, class2_label, n):
	"""
	params: model
	returns: None
	Shows 5 images and the predicted result of the model
	"""
	df = pd.read_csv(filename)
	for i in range(n):
		predict_image(model, df['img_path'][i], class1_label, class2_label)


def main():
	welcome_msg = "Training Tarzan based on Netflix's Start-Up Kdrama\n" \
				  "On their first date, Dosan introduces Dalmi, who is a newbie \n" \
				  "to AI tech, to the concept of machine learning using the metaphor \n" \
				  "Tarzan. Tarzan, having never seen a human, learns Jane's ways through\n" \
				  "building expereinces that he would learn from to win her heart.\n" \
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
	overwrite = False
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
