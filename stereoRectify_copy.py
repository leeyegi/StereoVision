import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy import misc
#"""
#이미지 경로
dir_basic="./dataset_split_v2/10_30_chess"
calib_files=[dir_basic+"_s1.jpg",dir_basic+"_s2.jpg"]

#이미지 불러오기
im_left = cv2.imread(calib_files[0])
im_right = cv2.imread(calib_files[1])

#
print(im_left.shape)
print(im_right.shape)
out = np.hstack((im_left, im_right))
plt.figure(figsize=(10, 4))
plt.imshow(out[..., ::-1])
plt.show()


"""
#왼쪽이미지에 대한 체스보드 코너 정보
ret_l, corners_l = cv2.findChessboardCorners(im_left, (8,5))

#
print(corners_l.shape)
print(corners_l[0])

corners_l=corners_l.reshape(-1,2)

#
print(corners_l.shape)
print(corners_l[0])

#저장한 코너 정보 이미지와 matching해 draw
im_left_vis=im_left.copy()
cv2.drawChessboardCorners(im_left_vis, (8,5), corners_l, ret_l)
plt.imshow(im_left_vis[...,::-1], cmap='gray')
plt.show()

#오른쪽이미지에 대한 체스보드 코너 정보
ret_r, corners_r = cv2.findChessboardCorners(im_right, (8,5))
#
print(corners_r.shape)
print(corners_r[0])

corners_r=corners_r.reshape(-1,2)

#'''
print(corners_r.shape)
print(corners_r[0])
#'''

#저장한 코너 정보 이미지와 matching해 draw
im_right_vis=im_right.copy()
cv2.drawChessboardCorners(im_right_vis, (8,5), corners_r, ret_r)
plt.imshow(im_right_vis, cmap='gray')
plt.show()
"""


#camera calibration을 위한 mesh grid생성
x,y=np.meshgrid(range(8),range(5))

#
print("x:\n",x)
print("y:\n",y)
#camera calibration과 stereo calibration을 수행하기위한 파라미터 정보 저장 인스턴스 (현재 값 x)
world_points=np.hstack((x.reshape(40,1),y.reshape(40,1),np.zeros((40,1)))).astype(np.float32)
#
print(world_points)
"""
"""
#camera stereo 수행 -> 한번만 수행해도됨
#카메라의 파라미터 뽑음
_3d_points=[]
_2d_points=[]
_2d_points_left=[]
_2d_points_right=[]

img_paths_right=glob('*.jpg')
#for path in img_paths:
#im = cv2.imread(path)
ret_l, corners_l = cv2.findChessboardCorners(im_left, (8, 5))
ret_r, corners_r = cv2.findChessboardCorners(im_right, (8, 5))


if ret_l and ret_r:
    _2d_points.append(corners_l)
    _2d_points_left.append(corners_l)
    _2d_points_right.append(corners_r)
    _3d_points.append(world_points)

ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objectPoints=_3d_points, imagePoints=_2d_points, imageSize=(1536,2048), cameraMatrix=None, distCoeffs=None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objectPoints=_3d_points, imagePoints=_2d_points, imageSize=(1536,2048), cameraMatrix=None, distCoeffs=None)


print("rec:", ret_l)
print("mtx shape", mtx_l.shape)
print("mtx:", mtx_l)
print("dist shape", dist_l.shape)
print("dist:", dist_l,)
print("rvecs shape", rvecs_l[0].shape)
print("rvecs:", rvecs_l)
print("tvecs shape",tvecs_l[0].shape)
print("tvecs:",tvecs_l)

'''
#camera calibration 수행 후 카메라에 파라미터 정보
rec=1.6907108642083184
mtx=np.array([[4.55684828e+03,0.00000000e+00,1.31478006e+02],
              [0.00000000e+00,4.53512634e+03,1.46080337e+03],
              [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
dist=np.array([-0.14052829, 3.35981911, 0.10662967,-0.0340999 ,-7.83945669])
rvecs=np.array([[ 0.31298816,-0.14260164,-0.03166638],
       [ 0.30719284,-0.14435355,-0.03713684]])
tvecs=np.array([[ 3.50305434,-6.0253759 ,28.19282081]
    ,[ 3.10635711,-6.04487063,28.37522957]])
'''


#코너 정보를 저장 하기 위해 사용
all_right_corners=[]
all_left_corners=[]
all_3d_points=[]

#camera calibration을 위한 mesh grid생성
x,y=np.meshgrid(range(8),range(5))
world_points=np.hstack((x.reshape(40,1),y.reshape(40,1),np.zeros((40,1)))).astype(np.float32)

#idx=[1, 3, 6, 12, 14]

idx=[1,6,14]
valid_idxs=[]

for i in idx:

    ret_left, left_corners = cv2.findChessboardCorners(im_left, (8, 5))
    ret_right, right_corners = cv2.findChessboardCorners(im_right, (8, 5))

    if ret_left and ret_right:  # if both extraction succeeded
        #valid_idxs.append(i)
        all_right_corners.append(right_corners)
        all_left_corners.append(left_corners)
        all_3d_points.append(world_points)

    print(len(all_right_corners))
    print(len(all_left_corners))
    print(len(all_3d_points))

    print(all_right_corners[0].shape)
    print(all_left_corners[0].shape)
    print(all_3d_points[0].shape)

    print(all_right_corners[0].reshape(-1,2)[0])

print("================")
print(all_right_corners)
print("================")
print(all_left_corners)
print("================")
print(all_3d_points)


retval, _, _, _, _, R, T, E, F=cv2.stereoCalibrate(objectPoints=all_3d_points,
                                                    imagePoints1=all_left_corners,
                                                    imagePoints2=all_right_corners,
                                                    imageSize=(1536,2048),
                                                    cameraMatrix1=mtx_l, distCoeffs1=dist_l,
                                                    cameraMatrix2=mtx_r,distCoeffs2=dist_r,
                                                    flags=cv2.CALIB_FIX_INTRINSIC)

print("retval:", retval)
print("R shape", R.shape)
print("R:", R)
print("T shape", T.shape)
print("T:", T,)
print("E shape", E.shape)
print("E:", E)
print("F shape",F.shape)
print("F:",F)

'''
retval=1.6873016611324214
R=np.array([[ 9.99981418e-01,  6.04642356e-03, -7.77510741e-04],
            [-6.04226161e-03,  9.99967973e-01,  5.24825380e-03],
            [ 8.09219005e-04, -5.24345835e-03,  9.99985926e-01]])
T=np.array([[-0.3384112 ],[-0.14609308],[ 0.14703209]])
E=np.array([[ 0.00077019, -0.14626135, -0.14686268],
            [ 0.14730321, -0.00088543,  0.33829212],
            [ 0.14813513, -0.33751702, -0.00188966]])
F=np.array([[-1.64294519e-08,  3.13496529e-06,  9.69850886e-03],
            [-3.15729648e-06,  1.90691343e-08, -3.26542304e-02],
            [-9.78525666e-03,  3.25257491e-02,  1.00000000e+00]])
'''

selected_image=2
left_corners=all_left_corners[selected_image].reshape(-1,2)
right_corners=all_right_corners[selected_image].reshape(-1,2)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(im_left, cmap='gray')
plt.subplot(122)
plt.imshow(im_right,cmap='gray')
plt.show()


cv2.circle(im_left,(left_corners[0,0],left_corners[0,1]),10,(0,0,255),10)
cv2.circle(im_right,(right_corners[0,0],right_corners[0,1]),10,(0,0,255),10)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(im_left[...,::-1])
plt.subplot(122)
plt.imshow(im_right[...,::-1])
plt.show()

lines_right = cv2.computeCorrespondEpilines(all_left_corners[selected_image], 1,F)
print(lines_right.shape)
lines_right=lines_right.reshape(-1,3) #reshape for convenience
print(lines_right.shape)





R1=np.zeros((3,3)) #output 3x3 matrix
R2=np.zeros((3,3)) #output 3x3 matrix
P1=np.zeros((3,4)) #output 3x4 matrix
P2=np.zeros((3,4)) #output 3x4 matrix

print("R")
print(R)

R1, R2, P1, P2,Q,roi1,roi2=cv2.stereoRectify(cameraMatrix1=mtx_l, #intrinsic parameters of the first camera
   cameraMatrix2=mtx_r, #intrinsic parameters of the second camera
   distCoeffs1=dist_l, #distortion parameters of the first camera
   distCoeffs2=dist_r, #distortion parameters of the second camera
   imageSize=(1536,2048), #image dimensions
   R=R, #Rotation matrix between first and second cameras (returned by cv2.stereoCalibrate)
   T=T) #last 4 parameters point to inizialized output variables

R1=np.array(R1) #convert output back to numpy format
R2=np.array(R2)
P1=np.array(P1)
P2=np.array(P2)

map1_x,map1_y=cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (im_left.shape[1],im_left.shape[0]), cv2.CV_32FC1)
map2_x,map2_y=cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (im_right.shape[1],im_right.shape[0]), cv2.CV_32FC1)


print(R1)
print(R2)
im_left_remapped=cv2.remap(im_left,map1_x,map1_y,cv2.INTER_CUBIC)
im_right_remapped=cv2.remap(im_right,map2_x,map2_y,cv2.INTER_CUBIC)
out=np.hstack((im_left_remapped,im_right_remapped))


for i in range(0, out.shape[0], 100):
    cv2.line(out, (0, i), (out.shape[1], i), (0, 255, 255), 3)

plt.figure(figsize=(10, 4))
plt.imshow(out[..., ::-1])
plt.show()

#misc.imsave("left.jpg", im_left_remapped)
#misc.imsave("right.jpg", im_right_remapped)

'''
max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

disparity = stereoProcessor.compute(im_left_remapped,im_right_remapped)
cv2.filterSpeckles(disparity, 0, 4000, max_disparity)

_, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
disparity_scaled = (disparity / 16.).astype(np.uint8)

# display image

disparity_colour_mapped = cv2.applyColorMap((disparity_scaled * (256. / max_disparity)).astype(np.uint8),cv2.COLORMAP_HOT);
plt.imshow(disparity_colour_mapped[..., ::-1])
plt.show()
'''

"""
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(im_left_remapped,None)
kp2, des2 = orb.detectAndCompute(im_right_remapped,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3=im_left_remapped
img3 = cv2.drawMatches(img1=im_left_remapped, keypoints1=kp1, img2=im_right_remapped, keypoints2=kp2, matches1to2=matches[:10], outImg=img3,flags=2)
plt.imshow(img3),plt.show()

for m in range(0,10):
        left_pt = kp1[m].pt
        right_pt = kp2[m].pt
        dispartity = abs(left_pt[0] - right_pt[0])
        z = 24*4.76 / dispartity

        print("Z")
        print(z)
"""