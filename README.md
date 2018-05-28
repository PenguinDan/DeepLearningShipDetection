# DeepLearningProject
The goal for this project was to correctly detect and classify ships in large satellite images using a Convolutional Neural Network (CNN) that was trained on 80x80 images of ship and non-ship data in conjunction with a custom object detection algorithm.  Our approach is able to efficiently and accurately detect ships in satellite images of varying size ratios. In comparison to an approach found on Kaggle.com that uses a sliding window across the entire image to detect the ships in 15 minutes, our pipeline is able to detect and output bounding boxes in under 5 seconds.

## Official Report
[Project Report](https://github.com/PenguinDan/DeepLearningShipDetection/blob/master/Report%20Folder/object-detection-classification.pdf)

## Before you Run the Project
Download the dataset from the following link:<br>
https://www.kaggle.com/rhammell/ships-in-satellite-imagery <br>
Make sure to add in the big images into a file named "images" <br>
Make sure to move the json file with small images and everything else to folder named "small_images"

## On Ubuntu, please do the following

### To install OpenCV packages for Python3
* pip3 install opencv-python

## PreProcessing Module
The main file used to preprocess the images 
* PreProcessingModule.py
 * Provides the following methods:
   * Create a bounding box around specific locations in the image
   * Denoise the image using OpenCV's kernel techniques
   * Scale the images in such a way that their dimensional features are not lost
   * Normalize image data to provide simpler data as an input to the Object Detection Algorithm
   * Binary greyscaling using a threshhold to provide simpler data as an input to the Object Detection Algorithm
  
## Object Detection Algorithm
The main file used to output region proposals from the large satellite images
* ObjectDetection.py
  * Ouputs a list of region proposals run on preprocessed data
 
## Working with the CNNs
The main files used to create, train, test, and save the CNNs were:
* CNN.py
  * For all function that manipulate the CNN model itself
* Tuning_CNN.py
  * For training and testing the many possible CNN architectures and hyperparameters
* Helper.py
  * Works with the raw image data, such as
    * Randomly dividing the data into training/validation and test data
    * Loading a given file of data
* Constants.py
  * Contains the paths to where the different CNN models are saved as well as a short description on their differences
