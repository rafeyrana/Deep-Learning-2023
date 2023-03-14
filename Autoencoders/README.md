Building Height Estimation using Autoencoders
This repository contains code for the assignment 3 of the deep learning course, which involves building a deep learning model using autoencoders for building height estimation using Google Earth imagery.

Objective
The objective of this assignment is to introduce the concept of autoencoders and their use in building height estimation using single and multi-view Google Earth imagery. The dataset comprises 4394 training samples already split into 70%, 10%, and 20% for train, validation, and test sets. The focus is on urban areas of 42 Chinese cities that offer diverse buildings with varying colors, shapes, sizes, and height.

Dataset
The dataset can be downloaded from this link: https://drive.google.com/drive/folders/1a0_2Oiz5880U3u-Wx19bOkOpWeVkG4t3?usp=sharing. It contains single and multi-view Google Earth imagery of buildings in 42 Chinese cities, comprising 4 municipalities, 26 provincial capitals, and 12 large cities. The dataset is split into train, validation, and test sets.

Autoencoder for Building Height Estimation
We have implemented an autoencoder for building height estimation for single and multi-view images using the Keras deep learning library. The code for each part of the assignment is implemented in separate files:

part_a.py: Use a single view image for building height estimation with an autoencoder.
part_b.py: Utilize both view images for building height estimation using either a symmetrical pair of encoders and one decoder or two encoders and two decoders.
part_c.py: Combine building segmentation estimation with height estimation to enhance the model's performance. Repeat steps a) and b) and report the effect on RMSE
Requirements
tensorflow>=2.0
keras>=2.3.1
numpy>=1.18.5
pandas>=1.0.5
opencv-python>=4.2.0
matplotlib>=3.2.2
Running the Code
To run the code, follow these steps:

Clone the repository to your local machine.
Download the dataset from the provided link.
Open the file for the part of the assignment you want to run (part_a.py, part_b.py, or part_c.py) and update the dataset path as required.
Run the file using the command python part_a.py or python part_b.py or python part_c.py.
Results
The model's performance is evaluated using the root mean square error (RMSE) metric. The performance of each model is reported in the console after training and testing the model. The results are also saved in a .csv file for later analysis.