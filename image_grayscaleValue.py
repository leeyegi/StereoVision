#이미지를 grayscale로 변환하고 각 픽셀값을 뿌리거나 txt파일로 저장

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

#이미지 경로
dir_basic="./dataset_split_v2/"
img_dir=["10_02_desk"]

#이미지 불러옴
img1 = misc.imread(name=dir_basic+img_dir[0]+"_s1.jpg", mode='L')
img2 = misc.imread(name=dir_basic+img_dir[0]+"_s2.jpg", mode='L')

print(img1.shape)
print(img2.shape)

#txt파일로 저장
with open('img1_data.txt', 'w') as f1:
    for i in range(0,img1.shape[0]):
        for j in range(0, img1.shape[1]):
            f1.write("%s" %img1[i,j])
            f1.write(" ")
        f1.write("\n")

print("img1 data ok")

with open('img2_data.txt', 'w') as f2:
    for i in range(0,img2.shape[0]):
        for j in range(0, img2.shape[1]):
            f2.write("%s" %img2[i,j])
            f2.write(" ")
        f2.write("\n")

#그냥 뿌림
'''
for i in range(0,img2.shape[0]):
    for j in range(0, img2.shape[1]):
        print(img2[i, j], end=" ")
    print("\n")
'''
print("img2 data ok")

plt.imshow(img1, cmap='gray')
plt.show()
plt.imshow(img2,cmap='gray')
plt.show()
"""
pix1 = np.array(img1)
pix2 = np.array(img2)

plt.imshow(pix1)
plt.show()
plt.imshow(pix2)
plt.show()
"""
