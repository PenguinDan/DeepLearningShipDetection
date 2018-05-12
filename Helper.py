# -*- coding: utf-8 -*-
import json
import glob
import os
import Constants
import numpy as np
from PreProcessingModule import reshape_data
from keras.preprocessing.image import array_to_img, img_to_array, load_img

#Extracts the data from the small ship images along with their labels, divdes the data into training and testing data, and saves these into two seperate files
def extract_images():
    os.chdir(Constants.SMALL_IMAGES_LOCATION)
    full_data = []
    full_labels = []

    #Iterate through all images in the small image folder
    for file in glob.glob("*.png"):
        #Extract data from individual images
        img = load_img(file)
        img_array = img_to_array(img)

        #Add data to arrays
        full_data.append(img_array.astype(int))
        full_labels.append(int(file[0]))

    full_data = np.asarray(full_data)

    #Shuffle data
    concatenated_data = np.asarray(list(zip(full_data, full_labels)))
    np.random.shuffle(concatenated_data)
    
    #Divide dataset
    test_dataset = concatenated_data[2500:]
    training_dataset = concatenated_data[:2500]

    test_data, test_labels = zip(*test_dataset)
    training_data, training_labels = zip(*training_dataset)

    #Create json objects to be exported to a file
    json_test = {}
    json_test["data"]=np.array(test_data)
    json_test["labels"]=np.array(test_labels)
    json_training = {}
    json_training["data"]=np.array(training_data)
    json_training["labels"]=np.array(training_labels)

    #Save json objects into their respective files
    test_file = open(Constants.OTHER_TEST_SMALL_IMAGE_DATASET, 'w')
    training_file = open(Constants.OTHER_TRAINING_SMALL_IMAGE_DATASET, 'w')
    
    np.set_printoptions(threshold=np.inf)
    test_file.write(str(json_test))
    training_file.write(str(json_training))

    test_file.close()
    training_file.close()

#Divides the original small ship dataset in the json file
def divide_data():
    
    #Load full dataset from file
    json_dataset = json.load(open(Constants.FULL_SMALL_IMAGE_DATASET))
    
    #Extract necessary information from json
    full_data = np.array(json_dataset['data'])
    full_labels = np.array(json_dataset['labels'])
    
    #Shuffle data
    concatenated_data = np.asarray(list(zip(full_data, full_labels)))
    np.random.shuffle(concatenated_data)
    
    #Divide dataset
    test_dataset = concatenated_data[2500:]
    training_dataset = concatenated_data[:2500]
    
    test_data, test_labels = zip(*test_dataset)
    training_data, training_labels = zip(*training_dataset)
    
    #Create json objects to be exported to a file
    json_test = {}
    json_test["data"]=np.array(test_data)
    json_test["labels"]=np.array(test_labels)
    json_training = {}
    json_training["data"]=np.array(training_data)
    json_training["labels"]=np.array(training_labels)
    
    
    #Save json objects into their respective files
    test_file = open(Constants.TEST_SMALL_IMAGE_DATASET, 'w')
    training_file = open(Constants.TRAINING_SMALL_IMAGE_DATASET, 'w')
    
    np.set_printoptions(threshold=np.inf)
    test_file.write(str(json_test))
    training_file.write(str(json_training))

    test_file.close()
    training_file.close()

#Returns the image data along with their respective labels
def load_data(data_location, data_size=(80,80)):
    
    #Load data from file
    json_dataset = json.load(open(data_location))
    
    labels = json_dataset['labels']
    data = []
    
    #Turn data into a usuable format for a cnn
    for img in json_dataset['data']:
        data.append(np.transpose(np.resize(img, (3,80,80))))

    data = reshape_data(data, data_size)
    data = np.asarray(data)

    return np.array(data), np.array(labels)







