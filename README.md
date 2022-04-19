# Training Tarzan: A Machine Learning Simulation (Based on Netflix’s Start-Up)
## LeAnn Mendoza

[![N|Solid](https://i2.wp.com/i.imgur.com/HBr50Ph.png?ssl=1)](https://i2.wp.com/i.imgur.com/HBr50Ph.png?ssl=1)

#### This program  simulates Do-San’s Tarzan analogy of machine learning [(link to clip)](https://youtube.com/clip/Ugkxs_VCFjTgGugzAIkanCq1qtxetKY99Vog) using a Tensorflow binary image classifier. The purpose of this project is to simulate the concept of machine learning using deep learning and image classification. 

## Features

- Driver program (driver.py) walks the user through the ML process from data preprocessing, to training, and analyzing results. 
- Visualizations of class distributions (bar graphs)
- Quickly pull up on image and its label (truth and/or predictions) at any point in the process
- Analysis tools and visualizations such as confusion matricies and auroc curves.

Built based on a kdrama example, can be applied to any domain!
Download data and unzip before running: https://drive.google.com/file/d/1l5wUkxZcUa2Fd6SGhqhs_sdZZesleCNk/view?usp=sharing
Video demo of driver.py (with previous model saved): https://youtu.be/7zyWgmHuD0w

## Code Organization

|-- training-tarzan/<br>
|   |-- __init__.py<br>
|   |-- driver.py<br>
|   |--preprocess_data.py<br>
|   |-- binary_classifier.py<br>
|   |-- analysis.py<br>
|   |-- unit_tests.py<br>
|   |-- data_for_unit_tests/<br>
|   |   |-- likes/<br>
|   |   |-- dislikes/<br>
|   |-- likes/<br>
|   |-- dislikes/ <br>
