import sys

import numpy as np
import cv2
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg


def calculate_calibration(filename, nx=9, ny=6):
    # Make a list of calibration images
    img = cv2.imread(filename)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    objpoints = []
    imgpoints = []
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if not ret:
        return None

    imgpoints.append(corners)
    objpoints.append(objp)
    #img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    ret, mtx, dist, a, b = cv2.calibrateCamera(
        objpoints, imgpoints,
        gray.shape[::-1], None, None
    )
    #dst = cv2.undistort(img, mtx, dist, None, mtx)
    return img, corners, mtx, dist

def get_chessboard_corners_corners(corners, nx=9, ny=6):
    corners = np.reshape(corners, (ny, nx, 1, 2))
    src = np.float32([corners[0, 0], corners[0, nx-1], corners[ny-1, nx-1], corners[ny-1, 0]])
    x1 = corners[0, 0][0][0]
    y1 = corners[0, 0][0][1]
    x2 = corners[ny-1, nx-1][0][0]
    y2 = (x2 - x1) / nx * ny + y1
    dst = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    return src, dst

def warp_perspective(img, src_corners, dest_corners):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src_corners, dest_corners)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def ahead_to_top_down(img):
    y1 = 450
    y2 = 600
    x1_b = 395
    x2_b = 908

    x1_t = 599
    x2_t = 681
    src = np.float32([[x1_t, y1], [x2_t, y1], [x2_b, y2], [x1_b, y2]])
    dst = np.float32([[x1_b, y1], [x2_b, y1], [x2_b, y2], [x1_b, y2]])
    return warp_perspective(img, src, dst)


def pipeline(img, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('undist.png', undistorted)
    top_down = ahead_to_top_down(undistorted)
    cv2.imwrite('topdown.png', top_down)

def undistort_and_warp_chess(filename):
    img, c_corners, cam_matrix, distortion = calculate_calibration(filename)
    undistorted = cv2.undistort(img, cam_matrix, distortion, None, cam_matrix)
    src, dst = get_chessboard_corners_corners(c_corners)
    warped_image = warp_perspective(undistorted, src, dst)
    return warped_image

#pipeline(sys.argv[1])
img, c_corners, cam_matrix, distortion = calculate_calibration(sys.argv[1])

pipeline(cv2.imread(sys.argv[2]), cam_matrix, distortion)


#warpedd = undistort_and_warp_chess(sys.argv[1])
#cv2.imwrite(sys.argv[2], warpedd)
