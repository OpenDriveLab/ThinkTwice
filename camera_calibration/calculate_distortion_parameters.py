#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (7, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
img_path = None
print("Image Path:", img_path, "(Please Change it to your path of Carla **built from source**!!!)")

images = glob.glob(img_path+'/*.png')

ImageSizeX = 1600
ImageSizeY = 900
CameraFOV = 150
f = ImageSizeX /(2 * np.tan(CameraFOV * np.pi / 360))
Cx = ImageSizeX / 2
Cy = ImageSizeY / 2
# intrinsics = np.array([[214.35935394,   0,         800,        ],
# [  0,         214.35935394, 450,        ], 
# [  0,           0,           1,        ]])
intrinsics = np.array([[f, 0, Cx],
    [0, f, Cy],
    [0, 0, 1 ]])
print(intrinsics)

for fname in images:
    index = int()
    if fname[:-4].split("/")[-1][-4:]  < "8028" or fname[:-4].split("/")[-1][-4:] > "8068":
        continue
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    print(fname, ret)
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], intrinsics, None, flags=cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_FIX_PRINCIPAL_POINT)

print("Camera matrix : \n")
print(mtx)
print("Distortion Matrix : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)