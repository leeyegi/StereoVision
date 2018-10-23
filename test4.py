import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


#이미지 경로
dir_basic="./dataset_split_v2/CAM00055"
calib_files=[dir_basic+"_s1.jpg",dir_basic+"_s2.jpg"]

im_left = cv2.imread(calib_files[0])
im_right = cv2.imread(calib_files[1])

print(im_left.shape)
print(im_right.shape)

plt.subplot(121)
plt.imshow(im_left[...,::-1], cmap='gray')
plt.subplot(122)
plt.imshow(im_right[...,::-1], cmap='gray')
plt.show()

#왼쪽이미지에 대한 체스보드 코너 정보
ret_l, corners_l = cv2.findChessboardCorners(im_left, (8,5))
print(corners_l.shape)
print(corners_l[0])

corners_l=corners_l.reshape(-1,2)

print(corners_l.shape)
print(corners_l[0])

im_left_vis=im_left.copy()
cv2.drawChessboardCorners(im_left_vis, (8,5), corners_l, ret_l)
plt.imshow(im_left_vis[...,::-1], cmap='gray')
plt.show()

#오른쪽이미지에 대한 체스보드 코너 정보
ret_r, corners_r = cv2.findChessboardCorners(im_right, (8,5))
print(corners_r.shape)
print(corners_r[0])

corners_r=corners_r.reshape(-1,2)

print(corners_r.shape)
print(corners_r[0])

im_right_vis=im_right.copy()
cv2.drawChessboardCorners(im_right_vis, (8,5), corners_r, ret_r)
plt.imshow(im_right_vis, cmap='gray')
plt.show()


#calibration
x,y=np.meshgrid(range(8),range(5))
print("x:\n",x)
print("y:\n",y)

world_points=np.hstack((x.reshape(40,1),y.reshape(40,1),np.zeros((40,1)))).astype(np.float32)
print(world_points)

print(corners_l[0],'->',world_points[0])
print(corners_l[35],'->',world_points[35])


_3d_points=[]
_2d_points=[]
img_paths=glob('*.jpg')
for path in img_paths:
    im = cv2.imread(path)
    plt.imshow(im[...,::-1], cmap='gray')
    plt.show()
    ret, corners = cv2.findChessboardCorners(im, (8, 5))

    if ret:
        _2d_points.append(corners)
        _3d_points.append(world_points)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints=_3d_points, imagePoints=_2d_points, imageSize=(1536,2048), cameraMatrix=None, distCoeffs=None)
print("Ret:", ret)
print("Mtx:", mtx, " ----------------------------------> [", mtx.shape, "]")
print("Dist:", dist, " ----------> [", dist.shape, "]")
print("rvecs:", rvecs, " --------------------------------------------------------> [", rvecs[0].shape, "]")
print("tvecs:", tvecs, " -------------------------------------------------------> [", tvecs[0].shape, "]")


#image rectify
im=cv2.imread('CAM00054_s1.jpg')[...,::-1]
im_undistorted=cv2.undistort(im, mtx, dist)
plt.subplot(121)
plt.imshow(im)
plt.subplot(122)
plt.imshow(im_undistorted)
plt.show()


#calibration
x,y=np.meshgrid(range(8),range(5))
print("x:\n",x)
print("y:\n",y)

world_points=np.hstack((x.reshape(40,1),y.reshape(40,1),np.zeros((40,1)))).astype(np.float32)
print(world_points)

#print(corners_l[0],'->',world_points[0])
#print(corners_l[35],'->',world_points[35])

all_right_corners=[]
all_left_corners=[]
all_3d_points=[]
idx=[1, 3, 6, 12, 14]
valid_idxs=[]

for i in idx:
    im_left = cv2.imread("CAM00054_s1.jpg")  # load left and right images
    im_right = cv2.imread("CAM00054_s2.jpg")


    ret_left, left_corners = cv2.findChessboardCorners(im_left, (8, 5))
    ret_right, right_corners = cv2.findChessboardCorners(im_right, (8, 5))

    if ret_left and ret_right:  # if both extraction succeeded
        valid_idxs.append(i)
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

retval, _, _, _, _, R, T, E, F=cv2.stereoCalibrate(objectPoints=all_3d_points,
                                                    imagePoints1=all_left_corners,
                                                    imagePoints2=all_right_corners,
                                                    imageSize=(1536,2048),
                                                    cameraMatrix1=mtx, distCoeffs1=dist,
                                                    cameraMatrix2=mtx,distCoeffs2=dist,
                                                    flags=cv2.CALIB_FIX_INTRINSIC)

selected_image=2
left_im=cv2.imread("CAM00054_s1.jpg")
right_im=cv2.imread("CAM00054_s2.jpg")
left_corners=all_left_corners[selected_image].reshape(-1,2)
right_corners=all_right_corners[selected_image].reshape(-1,2)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(left_im)
plt.subplot(122)
plt.imshow(right_im)
plt.show()

cv2.circle(left_im,(left_corners[0,0],left_corners[0,1]),10,(0,0,255),10)
cv2.circle(right_im,(right_corners[0,0],right_corners[0,1]),10,(0,0,255),10)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(left_im[...,::-1])
plt.subplot(122)
plt.imshow(right_im[...,::-1])
plt.show()

lines_right = cv2.computeCorrespondEpilines(all_left_corners[selected_image], 1,F)
print(lines_right.shape)
lines_right=lines_right.reshape(-1,3) #reshape for convenience
print(lines_right.shape)


def drawLine(line, image):
    a = line[0]
    b = line[1]
    c = line[2]

    # ax+by+c -> y=(-ax-c)/b
    # define an inline function to compute the explicit relationship
    def y(x): return (-a * x - c) / b

    x0 = 0  # starting x point equal to zero
    x1 = image.shape[1]  # ending x point equal to the last column of the image

    y0 = y(x0)  # corresponding y points
    y1 = y(x1)

    # draw the line
    cv2.line(image, (x0, int(y0)), (x1, int(y1)), (0, 255, 255), 3)  # draw the image in yellow with line_width=3

drawLine(lines_right[0], right_im)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(left_im[..., ::-1])
plt.subplot(122)
plt.imshow(right_im[..., ::-1])
plt.show()

R1=np.zeros((3,3)) #output 3x3 matrix
R2=np.zeros((3,3)) #output 3x3 matrix
P1=np.zeros((3,4)) #output 3x4 matrix
P2=np.zeros((3,4)) #output 3x4 matrix

R1, R2, P1, P2,Q,roi1,roi2=cv2.stereoRectify(cameraMatrix1=mtx, #intrinsic parameters of the first camera
   cameraMatrix2=mtx, #intrinsic parameters of the second camera
   distCoeffs1=dist, #distortion parameters of the first camera
   distCoeffs2=dist, #distortion parameters of the second camera
   imageSize=(1536,2048), #image dimensions
   R=R, #Rotation matrix between first and second cameras (returned by cv2.stereoCalibrate)
   T=T) #last 4 parameters point to inizialized output variables

R1=np.array(R1) #convert output back to numpy format
R2=np.array(R2)
P1=np.array(P1)
P2=np.array(P2)

map1_x,map1_y=cv2.initUndistortRectifyMap(mtx, dist, R1, P1, (left_im.shape[1],left_im.shape[0]), cv2.CV_32FC1)
map2_x,map2_y=cv2.initUndistortRectifyMap(mtx, dist, R2, P2, (left_im.shape[1],left_im.shape[0]), cv2.CV_32FC1)

im_left=cv2.imread('CAM00054_s1.jpg')
im_right=cv2.imread('CAM00054_s2.jpg')

im_left_remapped=cv2.remap(im_left,map1_x,map1_y,cv2.INTER_CUBIC)
im_right_remapped=cv2.remap(im_right,map2_x,map2_y,cv2.INTER_CUBIC)
out=np.hstack((im_left_remapped,im_right_remapped))

plt.figure(figsize=(10,4))
plt.imshow(out[...,::-1])
plt.show()

for i in range(0, out.shape[0], 100):
    cv2.line(out, (0, i), (out.shape[1], i), (0, 255, 255), 3)

plt.figure(figsize=(10, 4))
plt.imshow(out[..., ::-1])
plt.show()