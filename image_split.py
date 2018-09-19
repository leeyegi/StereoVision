import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from PIL import Image
import cv2

#스테레오카메라로 부터 받은 3D데이터가 2장의 사진을 합쳐놓은 형태
#그러므로 이미지를 반으로 자른 후 저장

#이미지를 반으로 자르는 메소드
def image_split(img_tmp):
    width = img_tmp.shape[1]
    height = img_tmp.shape[0]
    half_width=int(width/2)

    #split 하는 코드
    #[y시작좌표:y끝좌표, x시작좌표:x끝좌표,]
    img_split1=img_tmp[0:height, 0:half_width]
    img_split2 = img_tmp[0:height, half_width:width]
    print(img_split1.shape)

    return img_split1,img_split2

#이미지 경로 - 배열에 저장
dir_basic="./dataset/"
img_dir=["test3_11m_0","test3_11m_50", "test3_11m_100", "test3_11m_150", "test3_11m_200", "disparity_100","disparity_50", "disparity_0"]
#img_dir=["test"]

#print(img_dir.__len__())

#이미지별로 이미지 자르기
for i in range(0, img_dir.__len__()):
    #이미지 불로오기
    img=misc.imread(dir_basic+img_dir[i]+".jps")
    print(img.shape)

    #이미지 크기 조정
    size=(1024,384)
    #img.thumbnail(size, Image.ANTIALIAS)
    img_re=cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    print(img.shape)

    img_tmp=img_re
    img_split1, img_split2=image_split(img_tmp)

    #이미지 저장
    misc.imsave("./dataset_split_resize/"+img_dir[i]+"_s1_resize.jpg", img_split1)
    misc.imsave("./dataset_split_resize/" + img_dir[i] + "_s2_resize.jpg", img_split2)

    '''
    print(img_dir[i])
    print(img_tmp.shape[0])  # 세로 1536
    print(img_tmp.shape[1])  # 가로 4096

    plt.imshow(img_tmp)
    plt.show()
    '''
