#Begin Program
import PreProcessingModule as ppm
import CNN
import ObjectDetection as od
import Constants as const
import pdb
import numpy as np
import time

GREYSCALE_THRESHHOLD  = 104
IMAGE_COUNT = 2

#Get a list of pre-processed images
pr_image_list = ppm.get_pr_images(max_images = IMAGE_COUNT, greyscale='binary',
    greyscale_threshhold = GREYSCALE_THRESHHOLD)
#Get a list of nomral images
norm_image_list = ppm.get_upr_images(max_images = IMAGE_COUNT)
#Stores the cropped images whose shape is 80x80x3
cropped_image_list = []
#Stores the boxes for later referall to redraw the bounding boxes on 
#items that are ships, where each item is x,y,width,height
opt_bbox = None

#Loads the stored CNN model
model = CNN.load_CNN(const.PRETRAINED_CNN_MODEL_SIX)
bounded_image_list = []

#Time how long the code takes
t0 = time.time()

#Create probable bounding boxes around items in the image and put them in a list
for curr_image_index in range(len(pr_image_list)) :
    #Retrieve a processed and an unprocessed image
    curr_pr_image = pr_image_list[curr_image_index]
    curr_upr_image = norm_image_list[curr_image_index]
    #Get the height and width of the whole image
    image_height = curr_upr_image.shape[0]
    image_width = curr_upr_image.shape[1]
    #Retrieve the locations of all of the objects
    object_locations = od.detect(curr_pr_image)
    #Optimize the bounding boxes
    opt_bbox = ppm.scale_bbox(image_width, image_height, object_locations)
    #Create the bounding box around the original image
    cropped_image_list = ppm.crop(curr_upr_image, opt_bbox)

    ## Uncomment this line if working with a pretrained model
    ## as it reshapes the image_list to the proper input size
    ## for the pretrained model
    cropped_image_list = ppm.reshape_data(cropped_image_list, (224,224))

    #Turn list of images into a usable format for the cnn
    array = np.asarray(cropped_image_list)
    #Get percentage possibilities whether the object is a ship
    predictions = CNN.predict_CNN(model, array)
    #Only save objects whose probability to be a ship is over 50%
    ships = []
    for index in range(len(predictions)):
        if predictions[index]  > 0.8:
            ships.append(object_locations[index])
    
    #Create image with bounded boxes around predicted ships
    bounded_image = ppm.create_bbox(curr_upr_image, ships, box_thickness=3)
    bounded_image_list.append(bounded_image)

#Finish timing code
t1 = time.time()
print(t1-t0)

#Display one of the images
ppm.display_image(bounded_image_list[0])
ppm.display_image(bounded_image_list[1])

ppm.saveImage(bounded_image_list[0], "Pretrained Version 6 Image 0.png")
ppm.saveImage(bounded_image_list[1], "Pretrained Version 6 Image 1.png")
