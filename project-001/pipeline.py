# NOTE: drop me into the CarND-LaneLines-P1 directory and run `python pipeline.py`
# results will be written out to disk

import math
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy
import cv2
from moviepy.editor import VideoFileClip

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, m_min=0.2):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    # filter out lines that are too horizontal, this helps remove lines that
    # aren't helpful in identifying the lane lines
    filtered_lines = []
    for line in lines:
        l = line[0]
        m = abs((l[3]-l[1]) / (l[2]-l[0]))
        if m > m_min:
            filtered_lines.append(line)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, filtered_lines, thickness=1)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def to_color(img, color=[255, 255, 255]):
    """Converts a grayscale image to color"""
    return np.dstack((img if color[0] == 255 else img*0,
                      img if color[1] == 255 else img*0,
                      img if color[2] == 255 else img*0))

def split_XYs_out_from_pixels(pixel_coords):
    """Converts an array of pixels of the form [[[x1, y1]], [[x2, y2]]...] into
    separate x and y arrays of the form [x1, x2, ...] and [y1, y2, ...]"""
    xs = []
    ys = []

    if pixel_coords is None:
        return xs, ys

    for coord in pixel_coords:
        xs.append(coord[0][0])
        ys.append(coord[0][1])

    return xs, ys

def fit_line_to_points(img):
    """Fits a regression line to the non-zero pixels in a image"""
    y_max=img.shape[0]
    pixels = cv2.findNonZero(img)
    xs, ys = split_XYs_out_from_pixels(pixels)

    # x = m*y + b, y as function of x because we want to draw a line from bottom
    # of image to the middle (so y is the input)
    m, b, r_value, p_value, std_err = scipy.stats.linregress(ys, xs)

    x1 = m*y_max + b
    x2 = m*y_max*0.60 + b

    return int(x1), int(y_max), int(x2), int(y_max*0.60), abs(r_value)


def split_image(img, x_pos=0.5):
    """Splits the image into two parts, dividing at x_pos"""
    x_min = 0
    x_mid = int(img.shape[1]*x_pos)
    x_max = img.shape[1]
    y_min = 0
    y_max = img.shape[0]
    left =  np.array([[(x_min,y_max),(x_min, y_min), (x_mid,y_min),(x_mid,y_max)]], dtype=np.int32)
    right = np.array([[(x_mid,y_max),(x_mid, y_min), (x_max,y_min),(x_max,y_max)]], dtype=np.int32)
    left_image =  region_of_interest(img, left)
    right_image = region_of_interest(img, right)
    return left_image, right_image

def get_regression_lines(img):
    """Calculates two linear regression lines, one for the left half of the
    image and one for the right"""

    # first, split the image into left a right sections
    left_image, right_image = split_image(img)

    x1_l, y1_l, x2_l, y2_l, R_l = fit_line_to_points(left_image)
    x1_r, y1_r, x2_r, y2_r, R_r = fit_line_to_points(right_image)

    # return the two lines in the same format as is used by the Hough lines
    return [[[x1_l, y1_l, x2_l, y2_l]], [[x1_r, y1_r, x2_r, y2_r]]]

def get_polyfit_from_points(img):
    """Fit a second order polynomial to the nonzero points in an image"""
    y_max=img.shape[0]
    pixels = cv2.findNonZero(img)
    xs, ys = split_XYs_out_from_pixels(pixels)
    p = np.polyfit(ys, xs, 2)
    return p

def get_polyfit(img):
    """Splits an image in two and returns a second order polynomial fit to the
    left and right sides"""
    left_image, right_image = split_image(img)

    p_l = get_polyfit_from_points(left_image)
    p_r = get_polyfit_from_points(right_image)
    y_max = img.shape[0]
    y_min = y_max*0.60
    lines = []

    inc = 10
    for y in range(int(y_min)+inc, int(y_max), inc):
        y_0 = y-inc
        x_l_0 = p_l[0]*y_0*y_0 + p_l[1]*y_0 + p_l[2]
        x_l = p_l[0]*y*y + p_l[1]*y + p_l[2]
        lines.append([[int(x_l_0), y_0, int(x_l), y]])
        x_r_0 = p_r[0]*y_0*y_0 + p_r[1]*y_0 + p_r[2]
        x_r = p_r[0]*y*y + p_r[1]*y + p_r[2]
        lines.append([[int(x_r_0), y_0, int(x_r), y]])

    return lines


def get_lane_area_mask(image):
    """Get a mask of the area of the image most likely to contain the lane
    lines"""
    mask = np.zeros_like(image)
    ignore_mask_color = 255
    imshape = image.shape
    y_min =imshape[0]*0.61
    y_max =imshape[0]

    # the driver in these videos seems to drive towards the left side of the
    # lane, so we shift our mask area to the right to compensate
    right_shift = 30 # pixels

    # create the trapezoidal mask capturing all of the area of the lines
    x1 = imshape[1]*0.08 + right_shift
    x2 = imshape[1]*0.44 + right_shift
    x3 = imshape[1]*0.56 + right_shift
    x4 = imshape[1]*0.95 + right_shift
    vertices = np.array([[(x1,y_max),(x2, y_min), (x3, y_min), (x4,y_max)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # remove the inner portion of the mask, which is the space between the
    # lines, as this usually just contains noise
    y_min =imshape[0]*0.69
    y_max =imshape[0]
    x1 = imshape[1]*0.30 + right_shift
    x2 = imshape[1]*0.46 + right_shift
    x3 = imshape[1]*0.54 + right_shift
    x4 = imshape[1]*0.75 + right_shift
    vertices = np.array([[(x1,y_max),(x2, y_min), (x3, y_min), (x4,y_max)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 0)

    return mask

def get_lane_masked_edges(image, mask, kernel_size=5, low_threshold=50, high_threshold=100):
    """Returns Canny edges for the lane lines falling within the mask"""
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size)
    edges = canny(blur_gray, low_threshold, high_threshold)
    return cv2.bitwise_and(edges, mask)

def get_hough_lines(masked_edges, rho=2, theta=np.pi/180, threshold=5, min_line_len=10, max_line_gap=2):
    """Returns the Hough lines for the provided edges"""
    all_line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    return grayscale(all_line_img)

def process_video(in_filename, out_filename, debug=False):
    """read in a video, processes it with the provided debug setting, and then
    write it back out"""

    clip2 = VideoFileClip(in_filename)
    if debug:
        clip = clip2.fl_image(pipeline_debug)
    else:
        clip = clip2.fl_image(pipeline)

    clip.write_videofile(out_filename, audio=False)

def pipeline_debug(image):
    """Just a simply wrapper to call the pipeline function with the debug flag
    set to True since the `fl_image` function just passes the image"""
    return pipeline(image, True)

def pipeline(image, debug=False):
    """Takes in a image and returns that same image with the lanes lines
    overlaid in red.

    The strategy is to:

    1. Mask out an area that the lane lines are likely to be in
    2. Find the Canny edges in that area
    3. Calculate Hough lines from the Canny edges
    4. Fit a best-fit line to the Hough line pixels on the left and right sides

    If the debug flag is set, we also display some experimental information
    including attempting to fit a second order polynomial to the Hough line
    pixels.
    """
    # get the mask area
    mask = get_lane_area_mask(grayscale(image))

    # get the Canny edges
    masked_edges = get_lane_masked_edges(image, mask)

    # get Hough lines from the Canny edges
    hough_gray = get_hough_lines(masked_edges)

    # get regression lines from the hough lines
    reg_lines = get_regression_lines(hough_gray)

    # draw regression lines
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, reg_lines, thickness=6)
    image = weighted_img(line_img, image, 0.8, 1)

    if not debug:
        return image

    # The code below is all only run if the debug flag is set to True

    # Experimenting with polyfit curves instead of just lines
    hough_polyfit_lines = get_polyfit(hough_gray)
    canny_polyfit_lines = get_polyfit(masked_edges)
    curve_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(curve_img, hough_polyfit_lines, color=[0,255,0], thickness=6)
    draw_lines(curve_img, canny_polyfit_lines, color=[0,0,255], thickness=6)
    image = weighted_img(curve_img, image, 0.8, 1)

    # draw the edge mask lightly
    image = weighted_img(image, to_color(mask), α=0.1, β=1)

    # draw the canny edges
    blue_edges = to_color(masked_edges, [0, 0, 255])
    image = weighted_img(image, blue_edges, α=0.9, β=0.9)

    # draw the hough lines
    green_hough = to_color(hough_gray, [255, 255, 0])
    image = weighted_img(image, green_hough, α=0.9, β=0.7)

    return image


# process the images in the test_images directory and put output into the
# test_output directory
def process_test_images():
    os.makedirs("test_output", exist_ok=True)
    for filename in os.listdir("test_images/"):
        image = mpimg.imread("test_images/" + filename)
        line_image = pipeline(image)
        mpimg.imsave("test_output/" + filename, line_image)

# process each of the videos
process_video('challenge.mp4', 'challenge_output.mp4')
process_video('solidWhiteRight.mp4', 'white.mp4')
process_video('solidYellowLeft.mp4', 'yellow.mp4')

