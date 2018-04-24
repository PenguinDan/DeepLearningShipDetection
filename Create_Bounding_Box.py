import numpy as np
import PreProcessingModule as ppm
from skimage.filters import sobel
from skimage import morphology
from scipy import ndimage as ndi

#Declare golbal variable for the max index of picture
maxXAxis = 0
maxYAxis = 0
testWeight = 0
def main():
    imgList = ppm.get_pr_images(max_images = 8, greyscale='binary', greyscale_threshhold = 104)
    shipImg = imgList[0]
        
    # Region-based segmentation
    # =========================
    #
    # We therefore try a region-based method using the watershed transform.
    # First, we find an elevation map using the Sobel gradient of the image.
    elevation_map = sobel(shipImg)
    
    ######################################################################
    # Next we find markers of the background and the coins based on the extreme
    # parts of the histogram of grey values.
    markers = np.zeros_like(shipImg)
    markers[shipImg < 30] = 1
    markers[shipImg > 100] = 2
    
    ######################################################################
    # Finally, we use the watershed transform to fill regions of the elevation
    # map starting from the markers determined above:
    currImg = morphology.watershed(elevation_map, markers)
    
    
    
    currImg = ndi.binary_fill_holes(currImg - 1)
    currImg = currImg * 1
    currImg = currImg.astype('uint8')
    #Store the position of all of the bounding boxes
    objectList = []
    global maxXAxis
    global maxYAxis
    global testWeight
    maxXAxis = shipImg.shape[1]
    maxYAxis = shipImg.shape[0]
    startY = 0;
    startX = 0;
    endY = 0;
    endX = 0;
    while(startY < maxYAxis):
        #Check to make sure that y is smaller than the largest y index
        if(startY + 100 < maxYAxis):
            endY += 100
        else:
            endY += (maxYAxis - startY)
        while(startX < maxXAxis):
            #Check to make sure that x is smaller than the largest x index
            if(endX + 100 < maxXAxis):
                endX += 100
            else:
                endX += (maxXAxis - startX);
            currWeight = find_weight(currImg, startX, endX, startY, endY)
            if(currWeight < 1500 and currWeight > 200):
                objectList.extend(find_object(currImg, startX, endX, startY, endY))
                objectList.remove([])
            #Set the start X position to the end poistion of the previous iteration
            startX = endX
        #Set the start Y position to the end of the previous iteration
        startX = 0
        endX = 0
        startY = endY
    
            
    ppm.create_bbox(shipImg, objectList, box_thickness = 1)
    ppm.display_image(shipImg)
    ppm.saveImage(shipImg)
    return objectList
#Find the weight for the specific poistion
def find_weight(img, startX, endX, startY, endY):
    return np.sum(img[startY:endY,startX:endX])

#Check if there an object within the given position
def find_object(currImg, startX, endX, startY, endY):
    currYAxis = startY
    possibleLocation = [[]]
    global testWeight
    x = 0
    y = 0
    width = 0 
    length = 0
    while(currYAxis < endY):
        if(find_weight(currImg, startX, endX, currYAxis, currYAxis + 1) > 0):
            for x in range(startX, endX):
                if(currImg[currYAxis][x] == 1):
                    allPosition = find_y_axis_down(currImg, x, currYAxis)
                    allPosition.extend(find_y_axis_up(currImg, x, currYAxis))
                    if(testWeight > 200 and testWeight < 2000):
                        x, y, width, length, currImg = remove_object(currImg, allPosition)
                        possibleLocation.append([x,y,width,length])
                        testWeight = 0
                    elif(testWeight > 2000):
                        remove_object(currImg, allPosition)
                    testWeight = 0
                    break;
        currYAxis += 1
    
    return possibleLocation
                    
#Get the starting X position
def get_start_x_position(currImg, startX, endX, currY):
    currXPosition = startX
    if(currImg[currY][startX] == 1):
        while True:
            #Check to make sure is in a valid index
            if(currXPosition - 1 > 0):
                if(currImg[currY][currXPosition - 1] == 0):
                    break;
                else:
                    currXPosition -= 1          
            else:
                break
    elif(np.sum(currImg[currY][startX:endX]) > 0):
        while True:
            #Check to make sure is in a valid index
            if(currXPosition + 1 < maxXAxis):
                if(currImg[currY][currXPosition + 1] == 1):
                    currXPosition += 1
                    break;
                else:
                    currXPosition += 1
            else:
                break
        
    return currXPosition

#Get the ending X position
def get_end_x_position(currImg, startX, currY):
    global testWeight
    currXPosition = startX
    #Try to find where the X axis will end
    while True:
        if(currXPosition == maxXAxis - 1 or currImg[currY][currXPosition + 1] == 0):
            break
        else:
            currXPosition += 1
    testWeight += currXPosition - startX
    return currXPosition

#Find the rest of the object above the starting y position
def find_y_axis_up(currImg, startX, startY):
    currY = startY
    currXStart = startX
    currXEnd = get_end_x_position(currImg, startX, currY)
    indexList = [[currY, currXStart, currXEnd]]
    while True:
        currY -= 1
        if(currY > 0 and np.sum(currImg[currY][currXStart:currXEnd]) > 0):    
            currXStart = get_start_x_position(currImg, currXStart, currXEnd, currY)
            currXEnd = get_end_x_position(currImg, currXStart, currY) + 1
            if(currXEnd == -1):
                break;
            else:
                indexList.append([currY, currXStart, currXEnd])
        else:
            break
    return indexList

#Find the rest of the object below the starting y position
def find_y_axis_down(currImg, startX, startY):
    currY = startY
    currXStart = startX
    currXEnd = get_end_x_position(currImg, startX, currY)
    indexList = [[currY, currXStart, currXEnd]]
    while True:
        currY += 1
        if(currY + 1 < maxYAxis and np.sum(currImg[currY][currXStart:currXEnd]) > 0):    
            currXStart = get_start_x_position(currImg, currXStart, currXEnd, currY)
            currXEnd = get_end_x_position(currImg, currXStart, currY) + 1
            indexList.append([currY, currXStart, currXEnd])
        else:
            break
    return indexList

#Remove the object form the current image
def remove_object(currImg, indexList):
    smallestX = indexList[0][1]
    smallestY = indexList[0][0]
    biggestX = indexList[0][2]
    biggestY = indexList[0][0]
    for y, startX, endX in indexList:
        if(startX < smallestX):
            smallestX = startX
        if(biggestX < endX):
            biggestX = endX
        if(y < smallestY):
            smallestY = y
        elif(biggestY < y):
            biggestY = y
        currImg[y][startX:endX] = currImg[y][startX:endX] * 0
    if(smallestX > 5):
        smallestX -= 5
    if(smallestY > 5):
        smallestY -= 5
    
    
    return smallestX, smallestY, biggestX - smallestX + 10, biggestY - smallestY + 10, currImg
                     
if __name__ == "__main__":
    main()