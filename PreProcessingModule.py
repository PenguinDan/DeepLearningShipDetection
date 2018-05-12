from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import Constants
import os
import cv2

##############################################################################
#Returns a list of file names for the images
##############################################################################
def get_unprocessed_image_files():

    #Stores all of the big images in this list
    image_file_list = None

    #Retrieve image paths
    for root, dirs, files in os.walk(Constants.IMAGE_FILE_LOCATION):
        image_file_list = files

    return image_file_list


#############################################################################
#Erosion followed by dilation to remove/reduce noise in an image
#############################################################################
def cv_opening(img, kernel) :
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


#############################################################################
#Cleans the image by plugging holes and removing noise
#############################################################################
#Cleans the image and combines spaces on the image
def clean_image(img, kernel) :
    img = cv_opening(img, kernel)
    #Dilate the image in order to fill in empty spots
    dilation_kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, dilation_kernel, iterations = 2)
    #Attempt to return the objects to their normal sizes
    erosion_kernel = np.ones((3,3), np.uint8)
    img = cv2.erode(img, erosion_kernel, iterations = 1)

    return img

##############################################################################
#Return a list of preprocessed image files where each image is greyscaled
#the contrast is increased by a large margin to reduce the number of
#color variation
#############################################################################
def get_pr_images(max_images = 1,greyscale=None, greyscale_threshhold = 80):
    #Retrieve image file list
    image_file_list = get_unprocessed_image_files()
    #Place to store the Image objects
    image_list = []
    #Create a kernel to move through the image
    kernel = np.ones((3,3), np.uint8)

    #Counter to see how many images we are working on and break once we reach image_count
    counter = 0

    #Iterate through each file name and turn them into Image objects
    #then, preprocess them before appending it to the list
    for i in range(max_images):
        file_name = image_file_list[i]
        image = None
        if greyscale != None:
            #The Image object obtained from the file in greyscale mode
            image = cv2.imread(Constants.IMAGE_FILE_LOCATION + file_name, cv2.IMREAD_GRAYSCALE)
            #Check if the Image should be a binary grey scale with only 0 and 255 values
            if greyscale == 'binary':
                image[image < greyscale_threshhold] = 0
                image[image > greyscale_threshhold] = 255
        else :
            #The Image object obtained from the file in normal mode
            image = cv2.imread(Constants.IMAGE_FILE_LOCATION + file_name, cv2.IMREAD_UNCHANGED)
        #Clean the image
        image = clean_image(image, kernel)
        #Append the object onto the list
        image_list.append(image)

        #Break if we pre-processed enough images
        if counter == max_images:
            break

    #Return the list of Image objects
    return image_list

def get_upr_images(max_images = 1) :
    #Retrieve image file list
    image_file_list = get_unprocessed_image_files()
    #Stores the image objects
    image_list = []

    for i in range(max_images):
        file_name = image_file_list[i]
        #Instantiate the image
        image = cv2.imread(Constants.IMAGE_FILE_LOCATION + file_name, cv2.IMREAD_UNCHANGED)
        #Add the image to the file list
        image_list.append(image)

    return image_list

##############################################################################
#Creates a binary matrix of 0 and 1 values
#############################################################################
def normalize_image(image, reverse = False) :
    if reverse == False:
        image = image / 255
    else:
        image = image * 255

    return image

##############################################################################
#Displays an image until the user presses a key
#############################################################################
def display_image(image):
    #Create a window object
    cv2.namedWindow("image_window", cv2.WINDOW_NORMAL)
    #Show the image within that window
    cv2.imshow("image_window", image)
    #Makes the window show the image until the user presses a value
    cv2.waitKey()
    #User has pressed a value
    cv2.destroyAllWindows()

##############################################################################
#Saves an image inside of the object detection test folder
#############################################################################
def saveImage(image, file_name = "test.png"):
    cv2.imwrite(Constants.PR_SAVE_LOCATION + file_name, image)

##############################################################################
#Creates a bounding box in the given image depending on the top left coordinates
#############################################################################
def create_bbox(image, bbox_locations, box_thickness = 3):
    for x, y, width, height in bbox_locations:
        cv2.rectangle(image, (x,y), (x+width, y+height), (255, 255, 255), box_thickness)
    return image

##############################################################################
#Scales the bounding boxes
#############################################################################
def scale_bbox(image_width, image_height, bbox_locations, MAX_IMAGE_HEIGHT = 80, MAX_IMAGE_WIDTH = 80):
    new_bbox_locations = list()
    for x, y, width, height in bbox_locations:
        #Get the center of the image width wise
        horizontal_center = (2 * x + width) // 2
        #Get the center of the image height wise
        vertical_center = (2 * y + height) // 2
        if (width < MAX_IMAGE_WIDTH and height < MAX_IMAGE_HEIGHT) :
            x = horizontal_center - (MAX_IMAGE_WIDTH // 2)
            y = vertical_center - (MAX_IMAGE_HEIGHT // 2)
            #Check if the width wise boundary boxes go beyond the boundary
            if(x < 0) :
                #If the image goes behind the boundary, just set it as the boundary
                x = 0
            elif(x + MAX_IMAGE_WIDTH > image_width) :
                #If the image goes beyond the boundary, just set it as the boundary - MAX_IMAGE_WIDTH
                x = image_width - MAX_IMAGE_WIDTH
            #Check if the height wise boundary boxes go beyong the boundary
            if(y < 0) :
                #If the image goes behind the boundary
                y = 0
            elif(y + MAX_IMAGE_HEIGHT > image_height) :
                #If the image goes beyong the boundary, just set it as the boundary - MAX_IMAGE_HEIGHT
                y = image_height - MAX_IMAGE_HEIGHT
            #Set the new size of the images
            width = MAX_IMAGE_WIDTH
            height = MAX_IMAGE_HEIGHT
        else:
            #Here, either one side is greater than MAX, create a square to keep spatial details
            if(height > width):
                #Make the Width the same size as the height
                x = horizontal_center - (height // 2)
                if(x < 0):
                    #If the image goes behind the boundary, set it as the boundary
                    x = 0
                elif(x + height > image_width):
                    #If the image goes beyond the boundary, set it as the boundary - height
                    x = image_width - height
                width = height
            elif(width > height):
                #Make the height the same size as the width
                y = vertical_center - (width // 2)
                if(y < 0):
                    #If the image goes behind the boundaryu, set it as the boundary
                    y = 0
                elif(y + width > image_height):
                    #If the image goes beyond the boundary, set is as the boundary - width
                    y = image_height - width
                height = width
        #Store the newly created bounding box
        new_bbox_locations.append((x,y,width,height))
    return new_bbox_locations

##############################################################################
#Crop images from the original image
#############################################################################
def crop(image, bbox_set, set_width = 80, set_height = 80) :
    images = list()
    #Iteratively crop the images and put them into a list
    for x, y, width, height in bbox_set:
        cropped_image = image[y: y+ height, x: x+width]
        resized_image = cv2.resize(cropped_image, (set_width, set_height), interpolation = cv2.INTER_CUBIC)
        images.append(resized_image)

    return images


##############################################################################
#Reshapes all images to the specified shape
#############################################################################
def reshape_data(data, new_shape):
    new_data = []
    #Iteratively crop the images and put them into a list
    for i in range(len(data)):
        img = data[i]
        img = img.astype("uint8")
        resized_image = cv2.resize(img, new_shape, interpolation = cv2.INTER_CUBIC)

        new_data.append(resized_image)

    return new_data
