""" basic utility functions for processing images """

import numpy as np
import cv2

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_polyfit(img, polyfit, color=[0, 255, 0], width=3):
    extend = int(width/2)
    for y in range(0, img.shape[0]-1):
        x = min(max(polyfit[0]*y*y + polyfit[1]*y + polyfit[2], 0), img.shape[1]-1)
        img[y][x-extend:x+extend] = color

def fill_between_polys(img, poly_left, poly_right, color=[0, 255, 0]):
    for y in range(0, img.shape[0]-1):
        xl = min(max(poly_left[0]*y*y + poly_left[1]*y + poly_left[2], 0), img.shape[1]-1)
        xr = min(max(poly_right[0]*y*y + poly_right[1]*y + poly_right[2], 0), img.shape[1]-1)
        img[y][xl:xr] = color

def draw_at_position(inset, onto, at):
    """ draw `inset` over `onto` at the position `at` = (x, y) """
    x_start = at[0]
    y_start = at[1]
    x_end = x_start + inset.shape[1]
    y_end = y_start + inset.shape[0]
    onto[y_start:y_end, x_start:x_end] = inset

def calculate_calibration(filename, nx=9, ny=6):
    """ calculates the camera distorion given a filename with a 9x6
    checkerboard """
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

    ret, mtx, dist, a, b = cv2.calibrateCamera(
        objpoints, imgpoints,
        gray.shape[::-1], None, None
    )
    return img, corners, mtx, dist
