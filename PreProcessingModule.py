from PIL import Image, ImageEnhance
import numpy as np
import Constants
import os

#Returns a list of file names for the images
def get_unprocessed_image_files():
    
    #Stores all of the big images in this list
    image_file_list = None
    
    #Retrieve image paths
    for root, dirs, files in os.walk(Constants.IMAGE_FILE_LOCATION):
        image_file_list = files;
        
    return image_file_list;

#Return a list of image objects
def get_unprocessed_images(form='none'):
    #Retrieve image file list
    image_file_list = get_unprocessed_image_files()
    #Turn each image file item into an Image object
    image_list = []
    #Iterate through each file name and turn them into Image objects
    for file_name in image_file_list:
        #The Image object obtained from the file
        image = Image.open(Constants.IMAGE_FILE_LOCATION + file_name)
        #Append the object onto the list
        image_list.append(image)
    
    #If the form == matrix, return a matrix version of the images
    if form == 'matrix':
        matrix_image_list = []
        for image in image_list:
            matrix_image_list = np.asarray(image);
        #Returns the matrix version of each image
        return matrix_image_list
    
    #Return the list of Image objects
    return image_list
    