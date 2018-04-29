#Begin Program
import PreProcessingModule as ppm
import CNN
import ObjectDetection as od
import Constants as const

GREYSCALE_THRESHHOLD  = 104
IMAGE_COUNT = 1

#Get a list of pre-processed images
pr_image_list = ppm.get_pr_images(max_images = IMAGE_COUNT, greyscale='binary',
    greyscale_threshhold = GREYSCALE_THRESHHOLD);
#Get a list of nomral images
norm_image_list = ppm.get_upr_images(max_images = IMAGE_COUNT)
#Stores the bounded images
bounded_image_list = []

#Create probable bounding boxes around items in the image and put them in a list
for i in range(len(pr_image_list)):
    #Retrieve the locations of all of the objects
    curr_pr_image = pr_image_list[i]
    curr_upr_image = norm_image_list[i]
    object_locations = od.detect(curr_pr_image, activation_weight_threshhold = 3000, 
                                 max_weight_threshhold = 8000, stride_size = 100)
    #Create the bounding box around the original image
    bounded_image = ppm.create_bbox(curr_pr_image, object_locations, box_thickness = 1)
    bounded_image_list.append(bounded_image)

#Display the image
ppm.display_image(bounded_image_list[0])
