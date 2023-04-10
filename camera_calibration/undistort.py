import cv2 as cv
import numpy as np
import torch
ImageSizeX = 1600
ImageSizeY = 900
CameraFOV = 150
f = ImageSizeX /(2 * np.tan(CameraFOV * np.pi / 360))
Cx = ImageSizeX / 2
Cy = ImageSizeY / 2
intrinsics = np.array([[f, 0, Cx],
    [0, f, Cy],
    [0, 0, 1 ]])

distortion_matrix = np.array([[ 0.00888296, -0.00130899,  0.00012061, -0.00338673,  0.00028834]])
newcameramtx, roi = cv.getOptimalNewCameraMatrix(intrinsics, distortion_matrix, (1600, 900), 1, (1600, 900))
# newcameramtx = np.array([[304.14395142,   0,         788.25758876,],
# [  0,        221.49429321, 449.78972161,],
# [  0,           0,           1,        ],])
#print(roi, newcameramtx)


original_img = cv.imread("test" + '.png')
mapx, mapy = cv.initUndistortRectifyMap(intrinsics, distortion_matrix, None, newcameramtx, (ImageSizeX, ImageSizeY), 5)
undistorted_img = cv.remap(original_img, mapx, mapy, cv.INTER_LINEAR)
cv.imwrite("undistort.png", undistorted_img)