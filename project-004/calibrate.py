""" pipeline to find lane lines """
import sys

import numpy as np
import cv2
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import image_util as iu
#import matplotlib.image as mpimg


def create_perspective_transforms():
    """ creates a cv2 perspective transform based on coordinates taken from
    straight ahead images """
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
    return cv2.getPerspectiveTransform(src, dst), \
        cv2.getPerspectiveTransform(dst, src)

M_ahead_to_top_down, M_top_down_to_ahead = create_perspective_transforms()


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

def get_chessboard_corners_corners(corners, nx=9, ny=6):
    """ returns the chessboard corners as src and dst arrays """
    corners = np.reshape(corners, (ny, nx, 1, 2))
    src = np.float32([corners[0, 0], corners[0, nx-1], corners[ny-1, nx-1], corners[ny-1, 0]])
    x1 = corners[0, 0][0][0]
    y1 = corners[0, 0][0][1]
    x2 = corners[ny-1, nx-1][0][0]
    y2 = (x2 - x1) / nx * ny + y1
    dst = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    return src, dst

def warp_perspective(img, transform):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, transform, img_size, flags=cv2.INTER_LINEAR)
    return warped

def ahead_to_top_down(img):
    return warp_perspective(img, M_ahead_to_top_down)

def top_down_to_ahead(img):
    return warp_perspective(img, M_top_down_to_ahead)

def crop(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

def histogram(img, y_start, y_end):
    """ calculates a histogram of a row of an image """
    if y_start <= 1.0 and y_end <= 1.0:
        y_start = int(img.shape[0]*y_start)
        y_end = int(img.shape[0]*y_end*0.999)
    histogram = np.sum(img[int(y_start):int(y_end), :], axis=0)
    return histogram / np.max(histogram)

def histogram_max(hist, start, end):
    if start <= 1.0 and end <= 1.0:
        start = int(hist.shape[0] * start)
        end = int(hist.shape[0] * end * 0.999)
    return start + np.argmax(hist[start:end])

def find_line(img, bin_start, bin_stop, bin_size, left_start, right_start):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img[:, :, 0] # red channel
    pts = []
    for start in np.arange(bin_start-bin_size, bin_stop, -1*bin_size):
        end = start + bin_size
        hist = histogram(gray, start, end)
        hmax = histogram_max(hist, left_start, right_start)
        pts.append((hmax, (gray.shape[0]*(start + end)/2)))

    return pts

def find_side_line(img, center_start, start_width=0.06, bin_size=0.1):
    # left side
    found_center = np.array(find_line(img, 1.0, 0.69, 0.3, center_start-start_width, center_start+start_width))[0][0] / img.shape[1]
    #print(found_center)
    new_width = start_width*0.75
    left_pts = np.array(find_line(img, 1.0, 0.7, bin_size, found_center-new_width, found_center+new_width))
    left_x_mid = np.average(left_pts[:, 0])/img.shape[1]
    #print(left_x_mid)
    left_pts = np.append(left_pts, np.array(find_line(img, 0.7, 0.5999, bin_size, left_x_mid-start_width, left_x_mid+start_width)), 0)
    #print(left_pts)
    xs = left_pts[:, 0]
    ys = left_pts[:, 1]
    pl = np.polyfit(ys, xs, 2)
    y = 0.55*img.shape[0]
    #print(y)
    left_x_proj = (pl[0]*y*y + pl[1]*y + pl[2])/img.shape[1]
    #print(left_x_proj)
    left_pts = np.append(left_pts, np.array(find_line(img, 0.6, 0.4999, bin_size, left_x_proj-start_width, left_x_proj+start_width)), 0)
    xs = left_pts[:, 0]
    ys = left_pts[:, 1]
    pl = np.polyfit(ys, xs, 2)
    return pl, left_pts

def coords_of_non_zero(img):
    xs = []
    ys = []

    for y in range(0, img.shape[0]-1):
        for x in range(0, img.shape[1]-1):
            if np.sum(img[y][x]) > 0:
                xs.append(x)
                ys.append(y)

    return xs, ys

def mask_for_poly(img, polyfit, y_min=0.3, y_max=0.98, width=0.15, color=(255, 255, 255)):
    if y_min < 1 and y_max <= 1.0:
        y_min = int(img.shape[0]*y_min)
        y_max = int(img.shape[0]*y_max)
    offset = int(img.shape[1]*width/2.)

    mask = np.zeros_like(img)
    for y in range(y_min, y_max-1):
        x = polyfit[0]*y*y + polyfit[1]*y + polyfit[2]
        x = int(min(img.shape[1], max(0, x)))
        x_l = x - offset
        x_r = x + offset
        for x_set in range(x_l, x_r-1):
            mask[y][x_set] = color

    return mask

def find_lines(img):
    global pl_avg, pr_avg
    pl, left_pts = find_side_line(img, 0.34)
    pr, right_pts = find_side_line(img, 0.67)
    #print('pl', pl, type(pl))
    if pl_avg is None:
        pl_avg = pl
    if pr_avg is None:
        pr_avg = pr

    pl_avg = pl_avg*mwa_alpha + pl*(1.-mwa_alpha)
    pr_avg = pr_avg*mwa_alpha + pr*(1.-mwa_alpha)


    bin_size = 0.1
    rect_side = img.shape[0]*bin_size/2
    for center in left_pts:
        pt1 = (int(center[0]-rect_side), int(center[1]-rect_side))
        pt2 = (int(center[0]+rect_side), int(center[1]+rect_side))
        cv2.rectangle(img, pt1, pt2, (255, 255, 255))

    for center in right_pts:
        pt1 = (int(center[0]-rect_side), int(center[1]-rect_side))
        pt2 = (int(center[0]+rect_side), int(center[1]+rect_side))
        cv2.rectangle(img, pt1, pt2, (255, 255, 255))

    filtered = filter_old(img)
    #x_max = filtered.shape[1]
    #x_mid = int(x_max / 2)
    #y_max = filtered.shape[0]
    #y_mid = int(y_max / 2)

    #left_mask = np.array([[(x_max*0.2, y_mid), (x_mid, y_mid), (x_max*0.4, y_max), (x_max*0.3, y_max)]], dtype=np.int32)

    mask = mask_for_poly(filtered, pl_avg)

    cv2.addWeighted(mask, 0.1, img, 1., 0, img)

    masked_filtered_left = cv2.bitwise_and(filtered, mask)

    nonz = np.nonzero(cv2.cvtColor(masked_filtered_left, cv2.COLOR_BGR2GRAY))
    xls = nonz[1]
    yls = nonz[0]
    pfl = np.polyfit(yls, xls, 2)
    iu.draw_polyfit(img, pfl, color=[255, 0, 255])

    alpha = 0.8
    cv2.addWeighted(masked_filtered_left, alpha, img, 1., 0, img)

    return pl_avg, pr_avg

#def filter(img, h_center=98):
#    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
#    h_channel = hsv[:, :, 0]
#    h_channel = np.square(255. - (np.absolute(h_channel-h_center)/180.*255.)) / 255.
#    return np.array(np.dstack(( h_channel, h_channel, h_channel))).astype(np.uint8)

def filter_old(img, s_thresh=(170, 255), h_thresh=(18, 24), sx_thresh=(20, 100)):
#def filter(img, s_thresh=(120, 255), h_thresh=(10, 34), sx_thresh=(10, 150)):
    img = np.copy(img)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:, :, 0]
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold H channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( h_binary, np.zeros_like(sxbinary), np.zeros_like(s_binary)))
    color_binary = np.array(np.dstack((h_binary, sxbinary, s_binary))*255).astype(np.uint8)
    color_binary = cv2.cvtColor(color_binary, cv2.COLOR_RGB2BGR)
    return color_binary

def pipeline_debug(image):
    """Just a simply wrapper to call the pipeline function with the debug flag
    set to True since the `fl_image` function just passes the image"""
    return pipeline(image, True)

#def pipeline_on_image(img, outfilename, mtx, dist):

pl_avg = np.array([0., 0., 470.])
pr_avg = np.array([0., 0., 810.])
MWA_MAX = 0.8
mwa_alpha = 0.
frame = 0
def pipeline(img, debug=False):
    global mwa_alpha, frame
    frame = frame + 1
    mwa_alpha = MWA_MAX - (MWA_MAX/(frame + 0.2))

    mtx = cam_matrix
    dist = distortion
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    top_down = ahead_to_top_down(undistorted)
    cropped = crop(top_down, 0, 250, 1280, 680)

    #filtered = filter_old(cropped)
    #find_lines(filtered)
    pl, pr = find_lines(cropped)
    iu.draw_polyfit(cropped, pl)
    iu.draw_polyfit(cropped, pr)
    #for y in range(0, cropped.shape[0]-1):
    #    xl = min(max(pl[0]*y*y + pl[1]*y + pl[2], 0), cropped.shape[1]-1)
    #    xr = min(max(pr[0]*y*y + pr[1]*y + pr[2], 0), cropped.shape[1]-1)
    #    cropped[y][xl] = [0, 255, 0]
    #    cropped[y][xr] = [0, 255, 0]
    return cropped

def undistort_and_warp_chess(filename):
    img, c_corners, cam_matrix, distortion = calculate_calibration(filename)
    undistorted = cv2.undistort(img, cam_matrix, distortion, None, cam_matrix)
    src, dst = get_chessboard_corners_corners(c_corners)
    warped_image = warp_perspective(undistorted, src, dst)
    return warped_image

def process_video(in_filename, out_filename, debug=False):
    """read in a video, processes it with the provided debug setting, and then
    write it back out"""

    clip2 = VideoFileClip(in_filename)
    if debug:
        clip = clip2.fl_image(pipeline_debug)
    else:
        clip = clip2.fl_image(pipeline)

    clip.write_videofile(out_filename, audio=False)

calib_img, c_corners, cam_matrix, distortion = calculate_calibration(sys.argv[1])

#for file in sys.argv[2:]:
#    outfile_name = 'output/top_down_' + file.split('/')[-1]
#    print(outfile_name)
#    cv2.imwrite(outfile_name, pipeline(cv2.imread(file)))

process_video(sys.argv[2], sys.argv[3])

