import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from scipy import misc
import time

dir_basic="./dataset_split_resize/"

img_dir=["disparity_0", "disparity_50", "disparity_100","test3_11m_0", "test3_11m_50", "test3_11m_100", "test3_11m_150", "test3_11m_200"]
#img_dir=["disparity_0"]


def findParameter(imgL, imgR, tmp):
    numDisparitiesValue = 0

    for i in range(25):
        print("numDisparitiesValue:"+str(numDisparitiesValue))
        blockSizeValue=5
        for j in range(11):
            print("blockSizeValue:"+str(blockSizeValue))

            stereo = cv2.StereoBM_create(numDisparities=numDisparitiesValue, blockSize=blockSizeValue)
            disparity = stereo.compute(imgL, imgR)
            cv2.imwrite("./dataset_resize_result/" + img_dir[tmp]+"("+ str(numDisparitiesValue) +", "+str(blockSizeValue)+ ").jpg", disparity)
            time.sleep(0.3)
            blockSizeValue+=2
        numDisparitiesValue+=16

for tmp in range(0, img_dir.__len__()):
    imgL = cv2.imread(dir_basic+img_dir[tmp]+"_s1_resize.jpg",0)
    imgR = cv2.imread(dir_basic+img_dir[tmp]+"_s2_resize.jpg",0)

    print("i", str(tmp))
    findParameter(imgL,imgR, tmp)
