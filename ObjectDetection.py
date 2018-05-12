import numpy as np
import PreProcessingModule as ppm

test_weight = 0

def detect(original_img, activation_weight_threshhold = 3000, max_weight_threshhold = 8000, 
           stride_size = 100):
    curr_img = ppm.normalize_image(original_img)
    #Store the position of all of the bounding boxes
    objectList = []
    global MAX_X_AXIS
    global MAX_Y_AXIS
    global test_weight
    #Initialize the max index for the X axis
    MAX_X_AXIS = curr_img.shape[1]
    #Initialize the max index for the Y axis
    MAX_Y_AXIS = curr_img.shape[0]
    #The starting and ending position of where the sliding windows is at
    start_x = 0;
    start_y = 0;
    end_x = 0;
    end_y = 0;
    #Start sliding windows
    while(start_y < MAX_Y_AXIS):
        #Check to make sure that start_y is smaller than the largest y index
        if(start_y + stride_size < MAX_Y_AXIS):
            end_y += stride_size
        else:
            end_y += (MAX_Y_AXIS - start_y)
        while(start_x < MAX_X_AXIS):
            #Check to make sure that start_x is smaller than the largest x index
            if(end_x + stride_size < MAX_X_AXIS):
                end_x += stride_size
            else:
                end_x += (MAX_X_AXIS - start_x);
            #Calculate the current weight for the current sliding window position
            curr_weight = find_weight(curr_img, start_x, end_x, start_y, end_y)
            if(curr_weight < activation_weight_threshhold):
                objectList.extend(find_object(original_img, curr_img, max_weight_threshhold, start_x, end_x, start_y, end_y))
                objectList.remove([])
            #Set the start X position to the end poistion of the previous iteration
            start_x = end_x
        #Reset start_x and end_x back to 0
        start_x = 0
        end_x = 0
        #Set the start_y position to end_y
        start_y = end_y
    return objectList

#Find the weight for the specific poistion
def find_weight(img, start_x, end_x, start_y, end_y):
    return np.sum(img[start_y:end_y,start_x:end_x])

#Check if there a possible object within the given position
def find_object(original_img, curr_img, max_weight_threshhold, start_x, end_x, start_y, end_y):
    possible_locations = [[]]
    global test_weight
    x = 0
    y = 0
    width = 0
    length = 0
    for curr_y_axis in range(start_y, end_y):
        if(find_weight(curr_img, start_x, end_x, curr_y_axis, curr_y_axis + 1) > 0):
            for curr_x in range(start_x, end_x):
                if(curr_img[curr_y_axis][curr_x] == 1):
                    all_positions = find_y_axis_down(curr_img, curr_x, curr_y_axis)
                    all_positions.extend(find_y_axis_up(curr_img, curr_x, curr_y_axis))
                    if(test_weight < max_weight_threshhold):
                        x, y, width, length, curr_img = remove_object(original_img, curr_img, all_positions)
                        possible_locations.append([x,y,width,length])
                        test_weight = 0
                    else:
                        remove_object(original_img, curr_img, all_positions)
                    test_weight = 0

    return possible_locations

#Find the rest of the object below the starting y position
def find_y_axis_down(curr_img, start_x, start_y):
    curr_y = start_y
    curr_x_start = start_x
    curr_x_end = get_end_x_position(curr_img, start_x, curr_y)
    indexList = [[curr_y, curr_x_start, curr_x_end]]
    while True:
        curr_y += 1
        if(curr_y + 1 < MAX_Y_AXIS and np.sum(curr_img[curr_y][curr_x_start:curr_x_end]) > 0):
            curr_x_start = get_start_x_position(curr_img, curr_x_start, curr_x_end, curr_y)
            curr_x_end = get_end_x_position(curr_img, curr_x_start, curr_y) + 1
            indexList.append([curr_y, curr_x_start, curr_x_end])
        else:
            break
    return indexList

#Find the rest of the object above the starting y position
def find_y_axis_up(curr_img, start_x, start_y):
    curr_y = start_y
    curr_x_start = start_x
    curr_x_end = get_end_x_position(curr_img, start_x, curr_y)
    indexList = [[curr_y, curr_x_start, curr_x_end]]
    while True:
        curr_y -= 1
        if(curr_y > 0 and np.sum(curr_img[curr_y][curr_x_start:curr_x_end]) > 0):
            curr_x_start = get_start_x_position(curr_img, curr_x_start, curr_x_end, curr_y)
            curr_x_end = get_end_x_position(curr_img, curr_x_end, curr_y)
            indexList.append([curr_y, curr_x_start, curr_x_end])
        else:
            break
    return indexList

#Get the starting X position
def get_start_x_position(curr_img, start_x, end_x, curr_y):
    curr_x_position = start_x
    if(curr_img[curr_y][curr_x_position] == 1):
        while True:
            #Check to make sure is in a valid index
            if(curr_x_position - 1 > 0):
                if(curr_img[curr_y][curr_x_position - 1] == 0):
                    break;
                else:
                    curr_x_position -= 1
            else:
                break
    elif(np.sum(curr_img[curr_y][start_x:end_x]) > 0):
        while True:
            #Check to make sure is in a valid index
            if(curr_x_position + 1 < MAX_X_AXIS):
                if(curr_img[curr_y][curr_x_position + 1] == 1):
                    curr_x_position += 1
                    break;
                else:
                    curr_x_position += 1
            else:
                break

    return curr_x_position

#Get the ending X position
def get_end_x_position(curr_img, curr_x_end, curr_y):
    global test_weight 
    curr_x_position = curr_x_end
    #Try to find where the X axis will end
    while True:
        if(curr_x_position == MAX_X_AXIS - 1 or curr_img[curr_y][curr_x_position + 1] == 0):
            break
        else:
            curr_x_position += 1
    test_weight += curr_x_position - curr_x_end
    return curr_x_position

#Remove the object form the current image
def remove_object(original_img, curr_img, index_list):
    y_list, start_x_list, end_x_list = zip(*index_list)
    smallestX = np.min(start_x_list)
    biggestX = np.max(end_x_list)
    smallestY = np.min(y_list)
    biggestY = np.max(y_list)
    for y, start_x, end_x in index_list:
        temp_end_x = end_x
        if(np.sum(curr_img[y][end_x:biggestX]) != 0):
            if(y > 0 and y < MAX_Y_AXIS - 1):
                while(True):
                    if(temp_end_x == MAX_X_AXIS - 1):
                        break;
                    elif(original_img[y + 1][temp_end_x] == 0 or original_img[y-1][temp_end_x] == 0):
                        temp_end_x +=1
                    else:
                        break;
            else:
                temp_end_x = biggestX
                    
        curr_img[y][start_x:temp_end_x + 1] = curr_img[y][start_x:temp_end_x + 1] * 0
        
    if(smallestX > 5):
        smallestX -= 5
    if(smallestY > 5):
        smallestY -= 5
        
    return smallestX, smallestY, biggestX - smallestX + 10, biggestY - smallestY + 10, curr_img
