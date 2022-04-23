

# Training Tarzan: A Machine Learning Simulation (Based on Netflix’s Start-Up)
**Author:** LeAnn Mendoza, MSDS Align <br>
**Course:** DS5010: Intro to Programming for Data Science <br>
**Semester:** Spring 2022 <br>
**Campus:** Northeastern University, Silicon Valley <br>


![enter image description here](https://i.pinimg.com/736x/fb/1e/61/fb1e614e08c98115853f8194c521ccfa.jpg)
# Package Description

This program simulates Do-San’s Tarzan analogy of machine learning  [(link to clip)](https://youtube.com/clip/Ugkxs_VCFjTgGugzAIkanCq1qtxetKY99Vog)  using a Tensorflow binary image classifier. The purpose of this project is to simulate the concept of machine learning using deep learning and image classification.

## Features

 1.   [Driver program](https://github.com/leannmendoza/training_tarzan_binary_classifier/blob/main/driver.py) walks the user interactively through the ML process from data preprocessing, to training, and analyzing results.
		 - [Video demo of driver.py](https://youtu.be/mr0AFYVnB3s)
 2.   Visualizations of class distributions as a pie chart

 3.   Quickly pull up on image and its label (truth and/or predictions) at any point in the process
 4.   Analysis tools and visualizations such as confusion matricies and auroc curves.
 5.   Files can be used simultaneously (using driver code) or independantly using command line arguments ($ python analysis.py -h for clargs)

## Set-up Overview

 1. Clone this repository
 2. Download image data from [google drive](https://drive.google.com/file/d/1l5wUkxZcUa2Fd6SGhqhs_sdZZesleCNk/view) (2.12 GB, n dislikes = 11,968 imgs, n likes = 13,581 imgs) and place in directory
 3. Install requirements.txt using conda
 4. Activate conda enviroment
 5. Run [driver.py](https://github.com/leannmendoza/training_tarzan_binary_classifier/blob/main/driver.py) 

## Set-up Commands

    $ cd <path>/training_tarzan_binary_classifier/
    $ conda create --name <env> --file requirements.txt
    $ conda activate <env>
    $ mv <path/to/image/data/training_tarzan_data> .
    $ python driver.py
   
    > Training Tarzan based on Netflix's Start-Up Kdrama...
    ...
    First lets load our data. We have cute animals and flowers
    for "likes" and rocks and snakes for "dislikes"...

## Video Tutorial

https://youtu.be/mr0AFYVnB3s

## Code Organization

    .
    |-README.md
    |-driver.py
    |-preprocess_data.py
    |-binary_classifier.py
    |-analysis.py
    |-requirements.txt
    |-__init__.py
    |-unit_tests.py
    |-training_tarzan_data
    | |-likes
    | | |-16209331331_343c899d38.jpg
    | | |-2ba808b11e2e7302c65c1142fae20328.jpg
    | | |-6204049536_1ac4f09232_n.jpg
    | | |...
    | |-dislikes
    | | |-16209331331_343c899d38.jpg
    | | |-2ba808b11e2e7302c65c1142fae20328.jpg
    | | |-6204049536_1ac4f09232_n.jpg
    | | |...

