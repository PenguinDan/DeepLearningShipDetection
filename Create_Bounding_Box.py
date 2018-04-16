import numpy as np
import PreProcessingModule as ppm
import matplotlib.pyplot as plt
from skimage.feature import canny
import pdb

def main():
    imgList = ppm.get_pr_images(max_images = 1, greyscale='binary', greyscale_threshhold = 104)
    shipImg = imgList[0] 
    
    ######################################################################
    # Edge-based segmentation
    # =======================
    #
    # Next, we try to delineate the contours of the shipImg using edge-based
    # segmentation. To do this, we first get the edges of features using the
    # Canny edge-detector.
        
    edges = canny(shipImg)
    
    #ppm.normalize_image(img, reverse = True)
    allPixelList = []
    
    maxXAxis = shipImg.shape[1]
    maxYAxis = shipImg.shape[0]
    
    for y in range(maxYAxis):
        for x in range(maxXAxis):
            if(edges[y][x] == True):
                allPixelList.append([x, y, 80,80])
                
#    ppm.create_bbox(allPixelList)
    ppm.display_image(shipImg)

def find_object_weight(position):
    weight = 0
    
if __name__ == "__main__":
    main()