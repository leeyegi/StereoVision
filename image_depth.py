import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from scipy import misc



dir_basic="./dataset_split_resize/"

img_dir=["disparity_0"]

for i in range(0, img_dir.__len__()):
    imgL = cv2.imread(dir_basic+img_dir[i]+"_s1_resize.jpg",0)
    imgR = cv2.imread(dir_basic+img_dir[i]+"_s1_resize.jpg",0)
    #plt.imshow(imgL)
    #plt.show()

    #plt.imshow(imgR)
    #plt.show()

    stereo = cv2.StereoBM_create(numDisparities=222, blockSize=5)
    disparity = stereo.compute(imgL,imgR)

    plt.imshow(disparity,'gray')
    plt.show()