import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
    y2 = 675
    x1_b = 286
    x2_b = 1024

    x1_t = 596
    x2_t = 686

    x1_out = 440
    x2_out = 850
    src = np.float32([[x1_t, y1], [x2_t, y1], [x2_b, y2], [x1_b, y2]])
    dst = np.float32([[x1_out, y1], [x2_out, y1], [x2_out, y2], [x1_out, y2]])
    return warp_perspective(img, src, dst)

def crop(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

def histogram(img, y_start, y_end):
    if y_start <= 1.0 and y_end <= 1.0:
        y_start = int(img.shape[0]*y_start)
        y_end = int(img.shape[0]*y_end*0.999)
    histogram = np.sum(img[int(y_start):int(y_end),:], axis=0)
    return histogram / np.max(histogram)

def histogram_max(hist, start, end):
    if start <= 1.0 and end <= 1.0:
        start = int(hist.shape[0] * start)
        end = int(hist.shape[0] * end * 0.999)
    return start + np.argmax(hist[start:end])

def find_line(img, bin_stop, bin_size, left_start, right_start):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = []
    for start in np.arange(1.0-bin_size, bin_stop, -1*bin_size):
        end = start + bin_size
        hist = histogram(gray, start, end)
        hmax = histogram_max(hist, left_start, right_start)
        pts.append((hmax, (gray.shape[0]*(start + end)/2)))

    return pts

def find_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_stop = 0.2
    bin_size = 0.1
    rect_side = img.shape[0]*bin_size/2

    # left side
    left_pts = find_line(img, bin_stop, bin_size, 0.1, 0.5)
    for center in left_pts:
        pt1 = (int(center[0]-rect_side), int(center[1]-rect_side))
        pt2 = (int(center[0]+rect_side), int(center[1]+rect_side))
        cv2.rectangle(img, pt1, pt2, (255, 255, 255))

    # right side
    right_pts = find_line(img, bin_stop, bin_size, 0.5, 0.9)
    for center in right_pts:
        pt1 = (int(center[0]-rect_side), int(center[1]-rect_side))
        pt2 = (int(center[0]+rect_side), int(center[1]+rect_side))
        cv2.rectangle(img, pt1, pt2, (255, 255, 255))


def pipeline(img, outfilename, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    #cv2.imwrite('undist.png', undistorted)
    top_down = ahead_to_top_down(undistorted)
    cropped = crop(top_down, 0, 250, 1280, 680)
    find_lines(cropped)
    cv2.imwrite(outfilename, cropped)

def undistort_and_warp_chess(filename):
    img, c_corners, cam_matrix, distortion = calculate_calibration(filename)
    undistorted = cv2.undistort(img, cam_matrix, distortion, None, cam_matrix)
    src, dst = get_chessboard_corners_corners(c_corners)
    warped_image = warp_perspective(undistorted, src, dst)
    return warped_image

#pipeline(sys.argv[1])
img, c_corners, cam_matrix, distortion = calculate_calibration(sys.argv[1])

for file in sys.argv[2:]:
    outfile_name = 'output/top_down_' + file.split('/')[-1]
    print(outfile_name)
    pipeline(cv2.imread(file), outfile_name, cam_matrix, distortion)


#warpedd = undistort_and_warp_chess(sys.argv[1])
#cv2.imwrite(sys.argv[2], warpedd)
