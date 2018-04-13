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
        image_file_list = files;

    return image_file_list;


#############################################################################
#Erosion followed by dilation to remove/reduce noise in an image
#############################################################################
def cv_opening(img, kernel) :
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


#############################################################################
#Dilation followed by erosion to plug in holes in the image
#############################################################################
def cv_closing(img, kernel) :
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing


#############################################################################
#Cleans the image by plugging holes and removing noise
#############################################################################
def clean_image(img, kernel):
    img = cv_closing(img, kernel)
    img = cv_opening(img, kernel)
    return img

##############################################################################
#Return a list of preprocessed image files where each image is greyscaled
#the contrast is increased by a large margin to reduce the number of
#color variation
#############################################################################
def get_pr_images(max_images = 1, greyscale=None, greyscale_threshhold = 80):
    #Retrieve image file list
    image_file_list = get_unprocessed_image_files()
    #Turn each image file item into an Image object
    image_list = []
    #Create a kernel to move through the image
    kernel = np.ones((5,5), np.uint8)

    #Counter to see how many images we are working on and break once we reach image_count
    counter = 0

    #Iterate through each file name and turn them into Image objects
    #then, preprocess them before appending it to the list
    for file_name in image_file_list:
        counter += 1
        image = None
        if greyscale != None:
            #The Image object obtained from the file in greyscale mode
            image = cv2.imread(Constants.IMAGE_FILE_LOCATION + file_name, cv2.IMREAD_GRAYSCALE)
            #Check if the Image should be a binary grey scale with only 0 and 255 values
            if greyscale == 'binary':
                #Retrieve the shape of the image
                image_height = image.shape[0]
                image_width = image.shape[1]
                #Single for loop to iterate through the matrix
                for i in range(image_height):
                    for j in range(image_width):
                        if image[i][j] < greyscale_threshhold:
                            image[i][j] = 0
                        else :
                            image[i][j] = 255
        else :
            #The Image object obtained from the file in normal mode
            image = cv2.imread(Constants.IMAGE_FILE_LOCATION + file_name, cv2.IMREAD_UNCHANGED)
        #Clean the image
        image = clean_image(image, kernel)
        #Append the object onto the list
        image_list.append(image)

        #Break if we pre-processed enough images
        if counter == max_images:
            break;

    #Return the list of Image objects
    return image_list

##############################################################################
#Creates a binary matrix of 0 and 1 values
#############################################################################
def normalize_image(image, reverse = False) :
    if reverse == False:
        image /= 255
    else:
        image *= 255

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
        cv2.rectangle(image, (x,y), (x+width, y+height), (255, 0, 0), box_thickness)
    return image
