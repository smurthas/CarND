""" pipeline to find lane lines """
import sys

import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import image_util as iu


# As taken from example top down images and US highway specs of 3.7m lane width
# and 3m white dashed line length
m_px_x = 3.7/420.
m_px_y = 3. / 63.


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

    y1_out = 100
    src = np.float32([[x1_t, y1], [x2_t, y1], [x2_b, y2], [x1_b, y2]])
    dst = np.float32([[x1_out, y1_out], [x2_out, y1_out], [x2_out, y2], [x1_out, y2]])
    return cv2.getPerspectiveTransform(src, dst), \
        cv2.getPerspectiveTransform(dst, src)

M_ahead_to_top_down, M_top_down_to_ahead = create_perspective_transforms()

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

def mask_for_poly(img, polyfit, y_min=0.2, y_max=0.99, width=0.10, color=(255, 255, 255)):
    """ creates a mask that follows the polynomial from the bottom of the image
    towards to top, stopping at y_min """
    if y_min < 1 and y_max <= 1.0:
        y_min = int(img.shape[0]*y_min)
        y_max = int(img.shape[0]*y_max)
    base_width_pix = int(img.shape[1]*width/2.)

    mask = np.zeros_like(img)
    for y in range(y_min, y_max-1):
        offset = int(base_width_pix * (1 - y/y_max + y_min/y_max))
        x = polyfit[0]*y*y + polyfit[1]*y + polyfit[2]
        x = int(min(img.shape[1], max(0, x)))
        x_l = x - offset
        x_r = x + offset
        mask[y][x_l:x_r-1] = color

    return mask

def radius_of_curvature(polyfit, y_eval):
    return ((1 + (2*polyfit[0]*y_eval + polyfit[1])**2)**1.5) / np.absolute(2*polyfit[0])


def get_line_from_filtered(filtered, poly, m_px_y, m_px_x):
    y_eval = filtered.shape[0]-1 - 50

    mask = mask_for_poly(filtered, poly)
    masked_filtered = cv2.bitwise_and(filtered, mask)
    nonz = np.nonzero(cv2.cvtColor(masked_filtered, cv2.COLOR_BGR2GRAY))
    pf = np.polyfit(nonz[0], nonz[1], 2)
    pf_m = np.polyfit(nonz[0]*m_px_y, nonz[1]*m_px_x, 2)
    r_curv = radius_of_curvature(pf_m, y_eval*m_px_y)

    return mask, masked_filtered, pf, r_curv

def find_lines(img, debug=False):
    global pl_avg, pr_avg, r_curv_avg

    filtered = threshold_binary_filter(img)

    mask_left, masked_filtered_left, pfl, r_curv_l = \
        get_line_from_filtered(filtered, pl_avg, m_px_y, m_px_x)
    mask_right, masked_filtered_right, pfr, r_curv_r = \
        get_line_from_filtered(filtered, pr_avg, m_px_y, m_px_x)

    pl_avg = pl_avg*mwa_alpha + pfl*(1.-mwa_alpha)
    pr_avg = pr_avg*mwa_alpha + pfr*(1.-mwa_alpha)
    r_curv_avg = r_curv_avg*mwa_alpha + (r_curv_l + r_curv_r)/2.*(1.-mwa_alpha)

    if debug:
        cv2.imwrite('frame_' + str(frame) + '_filtered.png', cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
        alpha = 0.8
        img = cv2.addWeighted(masked_filtered_left, alpha, img, 1., 0)
        img = cv2.addWeighted(masked_filtered_right, alpha, img, 1., 0)

        img = cv2.addWeighted(mask_left, 0.1, img, 1., 0)
        img = cv2.addWeighted(mask_right, 0.1, img, 1., 0)

        masks = cv2.addWeighted(mask_left, 0.5, mask_right, 0.5, 0)
        filtered_m = cv2.addWeighted(masked_filtered_left, 0.5, masked_filtered_right, 0.5, 0)
        filtered_m = cv2.addWeighted(filtered_m, 0.7, masks, 0.2, 0)
        iu.draw_polyfit(filtered_m, pl_avg)
        iu.draw_polyfit(filtered_m, pr_avg)
        cv2.imwrite('frame_' + str(frame) + '_multi.png', cv2.cvtColor(filtered_m, cv2.COLOR_RGB2BGR))

    return pl_avg, pr_avg, r_curv_avg, mask_left, mask_right, masked_filtered_left, masked_filtered_right

def threshold_binary_filter(img, s_thresh=(170, 255), h_thresh=(18, 24), sx_thresh=(20, 100)):
    """ create a binary image from a combination of the saturation and hue
    channels, as well as a sobel_x threshold """

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
    color_binary = np.array(np.dstack((h_binary, sxbinary, s_binary))*255).astype(np.uint8)
    color_binary = cv2.cvtColor(color_binary, cv2.COLOR_RGB2BGR)
    return color_binary



pl_avg = np.array([0., 0., 470.])
pr_avg = np.array([0., 0., 880.])
r_curv_avg = 0
MWA_MAX = 0.9
mwa_alpha = 0.
frame = 0

def pipeline(img, debug=False):
    """ pipeline to detect lane lines in images """
    global mwa_alpha, frame
    frame = frame + 1
    mwa_alpha = MWA_MAX - (MWA_MAX/(frame + 0.2))

    undistorted = cv2.undistort(img, cam_matrix, distortion, None, cam_matrix)
    top_down = ahead_to_top_down(undistorted)
    cropped = crop(top_down, 0, 50, 1280, 680)

    pl, pr, r_curv, mask_left, mask_right, masked_filtered_left, masked_filtered_right = \
        find_lines(cropped, debug)

    y_pos = cropped.shape[0] - 10
    x_l = pl[0]*y_pos**2 + pl[1]*y_pos + pl[2]
    x_r = pr[0]*y_pos**2 + pr[1]*y_pos + pr[2]
    lateral_position = -1.0 * m_px_x * ((x_l + x_r) / 2.0 - cropped.shape[1]/2.0)

    polys = np.zeros_like(cropped)
    iu.draw_polyfit(polys, pl, color=[255, 0, 0], width=19)
    iu.draw_polyfit(polys, pr, color=[0, 0, 255], width=19)
    iu.fill_between_polys(polys, pl, pr)
    polys_full = np.zeros_like(top_down)
    drop_poly_top = 50
    polys_full[50 + drop_poly_top:680] = polys[drop_poly_top:]
    polys_ahead = top_down_to_ahead(polys_full)
    undistorted = cv2.addWeighted(polys_ahead, 0.7, undistorted, 1.0, 1.4)

    inset_w = 320
    inset_h = 107
    margin = 10
    x_start = 0
    inset_size = (inset_w, inset_h)

    overlay_box = np.copy(undistorted)
    cv2.rectangle(overlay_box, (0, 0), (1280, inset_h + 20), (0, 0, 0), thickness=-1)
    undistorted = cv2.addWeighted(overlay_box, 0.6, undistorted, 0.4, 0)

    overlay = np.copy(undistorted)
    masks = cv2.resize(cv2.addWeighted(mask_left, 0.5, mask_right, 0.5, 0), inset_size)

    inset_top_down = cv2.resize(cropped, inset_size)
    inset_top_down = cv2.addWeighted(masks, 0.5, inset_top_down, 1.0, 0)
    iu.draw_at_position(inset_top_down, overlay, (x_start + margin, margin))

    inset_polys = cv2.resize(polys, inset_size)
    iu.draw_at_position(inset_polys, overlay, (x_start + 2*margin + inset_w, margin))

    inset_binary = cv2.addWeighted(masked_filtered_left, 0.5, masked_filtered_right, 0.5, 0)
    inset_binary = cv2.resize(inset_binary, inset_size)
    inset_binary = cv2.addWeighted(inset_binary, 0.7, masks, 0.2, 0)
    iu.draw_at_position(inset_binary, overlay, (x_start + 3*margin + 2*inset_w, margin))

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    x_text = x_start + 4*margin + 3*inset_w
    cv2.putText(overlay, 'Radius: %.1fm' % r_curv, (x_text, 40), font_face, 0.75, \
        (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, 'Offset: %.2fm' % lateral_position, (x_text, 80), \
        font_face, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    undistorted = cv2.addWeighted(overlay, 0.9, undistorted, 0.1, 0)

    if debug:
        cv2.imwrite('frame_' + str(frame) + '_undist.png', cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR))
        cv2.imwrite('frame_' + str(frame) + '_dist.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite('frame_' + str(frame) + '_top_down.png', cv2.cvtColor(top_down, cv2.COLOR_RGB2BGR))

    return undistorted

def pipeline_debug(image):
    """Just a simply wrapper to call the pipeline function with the debug flag
    set to True since the `fl_image` function just passes the image"""
    return pipeline(image, True)

def process_video(in_filename, out_filename, debug=False):
    """read in a video, processes it with the provided debug setting, and then
    write it back out"""

    clip2 = VideoFileClip(in_filename)
    if debug:
        clip = clip2.fl_image(pipeline_debug)
    else:
        clip = clip2.fl_image(pipeline)

    clip.write_videofile(out_filename, audio=False)

calib_img, c_corners, cam_matrix, distortion = iu.calculate_calibration(sys.argv[1])
cv2.imwrite('calib_undistored.png', cv2.undistort(cv2.imread(sys.argv[1]), cam_matrix, distortion, None, cam_matrix))

process_video(sys.argv[2], sys.argv[3])
