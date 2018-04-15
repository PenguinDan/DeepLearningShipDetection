import numpy as np
import PreProcessingModule as ppm
import matplotlib.pyplot as plt
import pdb
import cv2

def main():
    imgList = ppm.get_pr_images(max_images = 1, greyscale='binary', greyscale_threshhold = 104)
    shipImg = imgList[0]
    hist = np.histogram(shipImg, bins=np.arange(0, 256))
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(shipImg, cmap=plt.cm.gray, interpolation='nearest')
    axes[0].axis('off')
    axes[1].plot(hist[1][:-1], hist[0], lw=2)
    axes[1].set_title('histogram of grey values')
    
    ######################################################################
    #
    # Thresholding
    # ============
    #
    # A simple way to segment the shipImg is to choose a threshold based on the
    # histogram of grey values. Unfortunately, thresholding this image gives a
    # binary image that either misses significant parts of the shipImg or merges
    # parts of the background with the shipImg:
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    
    axes[0].imshow(shipImg > 100, cmap=plt.cm.gray, interpolation='nearest')
    axes[0].set_title('shipImg > 100')
    
    axes[1].imshow(shipImg > 150, cmap=plt.cm.gray, interpolation='nearest')
    axes[1].set_title('shipImg > 150')
    
    for a in axes:
        a.axis('off')
    
    plt.tight_layout()
    
    ######################################################################
    # Edge-based segmentation
    # =======================
    #
    # Next, we try to delineate the contours of the shipImg using edge-based
    # segmentation. To do this, we first get the edges of features using the
    # Canny edge-detector.
    
    from skimage.feature import canny
    
    edges = canny(shipImg)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('Canny detector')
    ax.axis('off')
    
    ######################################################################
    # These contours are then filled using mathematical morphology.
    
    from scipy import ndimage as ndi
    
    fill_shipImg = ndi.binary_fill_holes(edges)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(fill_shipImg, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('filling the holes')
    ax.axis('off')
    
    
    ######################################################################
    # Small spurious objects are easily removed by setting a minimum size for
    # valid objects.
    
    from skimage import morphology
    
    shipImg_cleaned = morphology.remove_small_objects(fill_shipImg, 21)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(shipImg_cleaned, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('removing small objects')
    ax.axis('off')
    
    ######################################################################
    # However, this method is not very robust, since contours that are not
    # perfectly closed are not filled correctly, as is the case for one unfilled
    # coin above.
    #
    # Region-based segmentation
    # =========================
    #
    # We therefore try a region-based method using the watershed transform.
    # First, we find an elevation map using the Sobel gradient of the image.
    
    from skimage.filters import sobel
    
    elevation_map = sobel(shipImg)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(elevation_map, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('elevation map')
    ax.axis('off')
    
    ######################################################################
    # Next we find markers of the background and the shipImg based on the extreme
    # parts of the histogram of grey values.
    
    markers = np.zeros_like(shipImg)
    markers[shipImg < 30] = 1
    markers[shipImg > 150] = 2
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
    ax.set_title('markers')
    ax.axis('off')
    
    ######################################################################
    # Finally, we use the watershed transform to fill regions of the elevation
    # map starting from the markers determined above:
    
    segmentation = morphology.watershed(elevation_map, markers)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('segmentation')
    ax.axis('off')
    
    ######################################################################
    # This last method works even better, and the shipImg can be segmented and
    # labeled individually.
    
    from skimage.color import label2rgb
    
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_shipImg, _ = ndi.label(segmentation)
    image_label_overlay = label2rgb(labeled_shipImg, image=shipImg)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    axes[0].imshow(shipImg, cmap=plt.cm.gray, interpolation='nearest')
    axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
    axes[1].imshow(image_label_overlay, interpolation='nearest')
    
    for a in axes:
        a.axis('off')
    
    plt.tight_layout()
    
    
#    #ppm.normalize_image(img, reverse = True)
#    currPixelList = []
#    allPixelList = []
#    
#    xAxis = 1306
#    yAxis = 310
#    xAxisCounter = xAxis
#    yAxisCounter = yAxis
#    maxXAxis = img.shape[1]
#    maxYAxis = img.shape[0]
#    
#    allPixelList.clear()
#    xAxisCounter = xAxis
#    yAxisCounter = yAxis
#    while(1):
#        if(yAxisCounter < 319 + 80):
#            allPixelList.append(sum(img[yAxisCounter][xAxisCounter: xAxisCounter + 80]))
#            yAxisCounter += 1
#        else:
#            break
#    
#    print(sum(allPixelList))
#    
#    
#    ppm.create_bbox(img,[[1642,429,80,80]] , box_thickness = 3)
#    ppm.create_bbox(img,[[1818,1175,80,80]] , box_thickness = 3)
#    ppm.create_bbox(img,[[xAxis,yAxis,80,80]] , box_thickness = 3)
#    
#    ppm.display_image(img)
#    


    
if __name__ == "__main__":
    main()