# DeepLearningProject

## Before you Run the Project
Make sure to add in the big images into a file named "images"
Make sure to move the json file with small images and everything else to folder named "small_images"

## On Ubuntu, please do the following

### To install OpenCV packages for Python3
* pip3 install opencv-python

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
