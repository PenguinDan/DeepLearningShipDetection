# -*- coding: utf-8 -*-
import json
import Constants
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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
    
def load_data(data_location):
    
    #Load data from file
    json_dataset = json.load(open(data_location))
    
    labels = json_dataset['labels']
    data = []
    
    #Turn data into a usuable format for a cnn
    for img in json_dataset['data']:
        data.append(np.transpose(np.resize(img, (3,80,80))))

    return np.array(data), np.array(labels)
    
    


    

  
load_data(Constants.TEST_SMALL_IMAGE_DATASET)








