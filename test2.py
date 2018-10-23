import os.path
import numpy as np
from stereovision.calibration import StereoCalibrator, StereoCalibration
from stereovision.blockmatchers import StereoBM, StereoSGBM
import cv2

calib_dir = './dataset_split_v2/CAM00054'
#if(not os.path.exists(calib_dir)):
calibrator = StereoCalibrator(8, 5, 2, (1536, 2048))
for idx in range(1, 14):
    calibrator.add_corners((cv2.imread(calib_dir+"_s1.jpg"), cv2.imread(calib_dir+"_s2.jpg")))


calibration = calibrator.calibrate_cameras()
print ("Calibation error:", calibrator.check_calibration(calibration))
calibration.export(calib_dir)

calibration = StereoCalibration(input_folder=calib_dir)

if True:
    block_matcher = StereoBM()
else:
    block_matcher = StereoSGBM()

for idx in range(1, 14):
    image_pair = (cv2.imread('images/left%02d.jpg' %idx), cv2.imread('images/right%02d.jpg' %idx))
    rectified_pair = calibration.rectify(image_pair)
    disparity = block_matcher.get_disparity(rectified_pair)
    norm_coeff = 255 / disparity.max()
    cv2.imshow('Disparity %02d' %idx, disparity * norm_coeff / 255)

    for line in range(0, int(rectified_pair[0].shape[0] / 20)):
        rectified_pair[0][line * 20, :] = (0, 0, 255)
        rectified_pair[1][line * 20, :] = (0, 0, 255)

    cv2.imshow('Rect %02d' %idx, np.hstack(rectified_pair))
    cv2.waitKey()