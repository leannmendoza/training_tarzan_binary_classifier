"""
author @leannmendoza

Main driver code for this package. Simulates the user creating their ML model by walking the user through the process.
Options to visualize results. 
"""
from preprocess_data import *
from binary_classifier import *
from analysis import *

def import_data():
	"""
	params: None
	returns: train_df, test_df, validation_df
	Collects the data from preconstructed image directories into training/valid/test dfs
	"""
	likes_path = "./training_tarzan_data/likes/"
	dislikes_path = "./training_tarzan_data/dislikes/"
	truth_df = create_binary_df(likes_path,dislikes_path)
	train_df, test_df, validation_df = split_data(truth_df)
	print(train_df, test_df, validation_df)
	df_to_csv(train_df, "train.csv")
	df_to_csv(validation_df, "validation.csv")
	df_to_csv(test_df, "test.csv")

	return train_df, test_df, validation_df

def create_model():
	"""
	params: None
	returns: tt_model
	Creates the ML model based on data provided, saves into h5 file and gets predictions
	"""
	csvs = ['train.csv', 'validation.csv', 'test.csv']
	dirs = organize_files(csvs)
	if not exists('training_tarzan.h5'):
		model = train_model(dirs[0]+'/', dirs[1]+'/')
		model.summary()
	tt_model = keras.models.load_model('training_tarzan.h5')
	predict_test(tt_model, 'test.csv', save_file = True)

	return tt_model

def run_analysis():
	"""
	params: None
	returns: None
	Runs analysis in our predictions by constructing a confusion matrix as well as auroc curve
	which allows us to know how accurate our predictions are.
	"""
	df = pd.read_csv('predictions.csv')
	confusion_matrix(df)
	auroc(df)

def sample_images(model):
	"""
	params: model
	returns: None
	Shows 5 images and the predicted result of the model
	"""
	count = 0
	with open("test.csv") as f:
		next(f)
		for line in f:
			if count > 10:
				return
			img_path = line.strip().split(',')[0]
			predict_image(model, img_path)
			count += 1


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
	train_df, test_df, validation_df = import_data()

	see_data_msg = "Now that we have the data loaded, would you like to see the distributions of like and dislike? [Y/N]\n"
	see_data = input(see_data_msg)
	if see_data == 'Y':
		visualize_data(train_df, 'training data')
		visualize_data(test_df, 'testing data')
		visualize_data(validation_df, 'validation data')

	see_imgs_msg = "Let's pull up some images in our data. Shall we? [Y/N]\n"
	see_imgs = input(see_imgs_msg)
	if see_imgs== 'Y':	
		for i in range(10):
			show_image_with_label(train_df['img_path'][i], train_df['truth'][i])


	train_model_msg = "Now that we, have seen our data. Let's train our model using tensorflow..."
	input(train_model_msg)
	trained_model = create_model()

	analysis_msg = "Great now let's see how tarzan learned!"
	input(analysis_msg)
	run_analysis()

	sample_image_predictions = "Now let's see what tarzan thinks of items in our test set..."
	input(sample_image_predictions)
	sample_images(trained_model)

	end_msg = "Thank you for participating in this simulation with Jane and Tarzan!"

if __name__ == '__main__':
	main()
