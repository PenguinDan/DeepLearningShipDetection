from PIL import Image, ImageEnhance
import numpy as np
import Constants
import os

##############################################################################
#Returns a list of file names for the images
##############################################################################
def get_unprocessed_image_files():
    
    #Stores all of the big images in this list
    image_file_list = None
    
    #Retrieve image paths
    for root, dirs, files in os.walk(Constants.IMAGE_FILE_LOCATION):
        image_file_list = files;
        
    return image_file_list;

##############################################################################
#Return a list of unprocessed image files
##############################################################################
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
            matrix_image = np.asarray(image);
            matrix_image_list.append(matrix_image)
        #Returns the matrix version of each image
        return matrix_image_list
    
    #Return the list of Image objects
    return image_list
    

##############################################################################
#Return a list of preprocessed image files where each image is greyscaled
#the contrast is increased by a large margin to reduce the number of 
#color variation
#############################################################################
def get_preprocessed_images(form='none', greyscale=True ,contrast=200):
    #Retrieve image file list
    image_file_list = get_unprocessed_image_files()
    #Turn each image file item into an Image object
    image_list = []
    
    #Iterate through each file name and turn them into Image objects
    #then, preprocess them before appending it to the list
    for file_name in image_file_list:
        #The Image object obtained from the file
        image = Image.open(Constants.IMAGE_FILE_LOCATION + file_name)
        #Greyscale image if the user wants to
        if greyscale == True:
            #Greyscale the image
            image = image.convert('L')
        if contrast > 0 :
            #Enhance the contrast of the image by the specified margin
            image = ImageEnhance.Contrast(image).enhance(contrast)
        #Append the object onto the list
        image_list.append(image)
    
    return image_list
    
        
    