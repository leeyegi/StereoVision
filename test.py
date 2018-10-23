import cv2
import numpy as np
import matplotlib.pyplot as plt

#스테레오 calibration 과 rectify를 수행하기 위한 파라미터
pattern_size=(8,5)          #체스 개수
obj_points=[]               #체스 코너 포인트
img_left_points=[]          #왼쪽이미지 체스코너 포인트
img_right_points=[]         #오른쪽이미지 체스코너 포인트

#이미지 경로
dir_basic="./dataset_split_v2/CAM00055"
calib_files=[dir_basic+"_s1.jpg",dir_basic+"_s2.jpg"]

#캘리브레이션 하기전 세팅
#(8,5)-> chess이미지의 격자수와 동일
termination=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30,0.001)
objp=np.zeros((8*5,3), np.float32)
objp[:,:2]=np.mgrid[0:8, 0:5].T.reshape(-1,2)

#이미지 불러옴
left_img = cv2.imread(calib_files[0], cv2.CV_8UC1)
right_img = cv2.imread(calib_files[1], cv2.CV_8UC1)

# ########################
#이미지 보여줌
plt.subplot(121)
plt.imshow(left_img[...,::-1], cmap='gray')
plt.subplot(122)
plt.imshow(right_img[...,::-1], cmap='gray')
plt.show()

#이미지의 크기 저장 (1536,2048)
image_size = left_img.shape


#이미지를 grayscale로 바꿈
#left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
#right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)


#체스의 코너를 찾음
find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK          #연산 플래그

#코너를 찾음 - 이미지(그레이or컬러), 패턴 사이즈,
left_found, left_corners = cv2.findChessboardCorners(left_img, pattern_size, flags = 0)
right_found, right_corners = cv2.findChessboardCorners(right_img, pattern_size, flags = 0)

#찾음 코너정보 저장
if left_found:
    cv2.cornerSubPix(left_img, left_corners, (11 ,11), (-1 ,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
if right_found:
    cv2.cornerSubPix(right_img, right_corners, (11 ,11), (-1 ,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

if left_found and right_found:
    img_left_points.append(left_corners)
    img_right_points.append(right_corners)
    obj_points.append(objp)

    #체스 코너를 그림
    #cv2.drawChessboardCorners(left_img, pattern_size, left_corners, left_found)
    #cv2.drawChessboardCorners(right_img, pattern_size, right_corners, right_found)

#카메라 캘리브레이션 수행
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objectPoints=obj_points, imagePoints=img_left_points,  imageSize=(1536,2048), cameraMatrix=None, distCoeffs=None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objectPoints=obj_points, imagePoints=img_right_points,  imageSize=(1536,2048), cameraMatrix=None, distCoeffs=None)


#스테레오 캘리브레이션 수행
stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints=obj_points
                                                                                                             ,imagePoints1=img_left_points
                                                                                                             ,imagePoints2=img_right_points
                                                                                                             ,imageSize=image_size
                                                                                                             ,criteria = stereocalib_criteria, flags=0
                                                                                                             ,cameraMatrix1=mtx_l, cameraMatrix2=mtx_r
                                                                                                             ,distCoeffs1=dist_l, distCoeffs2=dist_r)


print("rotation")
print(R)
print("transform")
print(T)

left_img = cv2.imread(calib_files[0], cv2.CV_8UC1)
right_img = cv2.imread(calib_files[1], cv2.CV_8UC1)
rectify_scale = 1  # 0=full crop, 1=no crop

R1=np.zeros((3,3)) #output 3x3 matrix
R2=np.zeros((3,3)) #output 3x3 matrix
P1=np.zeros((3,4)) #output 3x4 matrix
P2=np.zeros((3,4)) #output 3x4 matrix

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix1, cameraMatrix2=cameraMatrix2
                                                  , distCoeffs1=distCoeffs1, distCoeffs2=distCoeffs2
                                                  , imageSize=image_size, R=R, T=T, Q=None, flags=cv2.CALIB_ZERO_DISPARITY
                                                  , alpha=-1)


left_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (left_img.shape[1],left_img.shape[0]), cv2.CV_16SC2)
right_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (left_img.shape[1],left_img.shape[0]), cv2.CV_16SC2)

print("rotation")
print(R1)
print(R2)
print("transform")
print(P1)
print(P2)


pairs={'left_img': calib_files[0], 'right_img': calib_files[0], }



left_img_remap = cv2.remap(left_img, left_maps[0], left_maps[1], cv2.INTER_CUBIC)
right_img_remap = cv2.remap(right_img, right_maps[0], right_maps[1], cv2.INTER_CUBIC)


# ########################
# 이미지 보여줌
img_tmp = np.hstack((left_img_remap, right_img_remap))

for i in range(0, img_tmp.shape[1], 70):
    cv2.line(img_tmp, (i, 0), (i, img_tmp.shape[0]), (255, 0, 0), 1)
for i in range(0, img_tmp.shape[0], 70):
    cv2.line(img_tmp, (0, i), (img_tmp.shape[1], i), (255, 0, 0), 1)

img_tmp_out = cv2.resize(img_tmp, (int(img_tmp.shape[1] / 4), int(img_tmp.shape[0] / 4)))
cv2.imshow("chess coner img", img_tmp_out)
cv2.waitKey(0)

