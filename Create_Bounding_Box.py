import numpy as np
import PreProcessingModule as ppm
import matplotlib.pyplot as plt
from skimage.feature import canny
from multiprocessing import Pool
from skimage.filters import sobel
from skimage import morphology
from scipy import ndimage as ndi
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
    elevation_map = sobel(shipImg)
    markers = np.zeros_like(shipImg)
    markers[shipImg < 30] = 1
    markers[shipImg > 100] = 2
    segmentation = morphology.watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    allPixelList = []
    maxXAxis = shipImg.shape[1]
    maxYAxis = shipImg.shape[0]
    print(segmentation)
    for y in range(maxYAxis):
        currOceanAxis = find_ocean_start(segmentation, maxXAxis, y)
        for x in range(maxXAxis):
            if(edges[y][x] == True and x > currOceanAxis):
                allPixelList.append([x - 40, y, 80, 80])
    print(allPixelList.__len__())             
    ppm.create_bbox(shipImg,allPixelList, box_thickness=1)
    ppm.display_image(shipImg)

def find_ocean_start(segmentation, max_x_axis, curr_y_axis):
    currOceanAxis = max_x_axis - 1
    for x in range(max_x_axis):
        if(segmentation[curr_y_axis][x] == False and segmentation[curr_y_axis][x+30] == False and x < currOceanAxis):
            currOceanAxis = x
            break;
    return currOceanAxis

def find_weight(xAxis, yAxis, img):
    weight = 0
    y = yAxis
    maxYAxis = yAxis + 80
    for y in range(maxYAxis):
        weight += sum(img[y][xAxis - 40: xAxis + 40])
    return weight

if __name__ == "__main__":
    main()