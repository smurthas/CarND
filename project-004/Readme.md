**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./assets/distortion_correction.png "Distortion Correction"
[image2]: ./assets/frame_distortion_correction.gif "Video Frame Distortion Correction"
[image3]: ./assets/binary_threshold.png "Binary Example"
[image4]: ./assets/top_down.png "Top Down"
[image5]: ./assets/filtered_masked_polys.gif "Filter and fit"
[image6]: ./assets/full_frame.png "Full video frame"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You are reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this is in the `calculate_calibration` function of
`image_utils.py`. It converts the calibration image to grayscale, then searches
for chessboard corners. If it finds them, it calcuates a camera distortion from
them and returns it. This function is called from `pipeline.py` before
processing any frames.

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

At the beginning of the `pipeline` function (in `pipeline.py`), the passed image
is first undistorted by calling the `cv2.undistort` function.

![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image
in the `threshold_binary_filter` function of `pipeline.py` using a Sobel
gradient filter in the X direction, a hue filter centered around yellow, and a
filter for high saturation values.

Note, I did the perspective transform prior to generating the thresholded binary
image, so the example below is from the top down view.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transforms (note: performed BEFORE the threshold binary image
creation) and done via the `ahead_to_top_down` and `top_down_to_ahead` functions
in `pipeline.py`, which in turn call the `cv2.warpPerspective` function with a
set of `src` and `dst` points that I gathered by hand from a straight ahead
frame of the `project_video.mp4` video.

The points used are as follows:

| Source        | Destination   |
|:-------------:|:-------------:|
| 596, 450      | 440, 100      |
| 686, 675      | 850, 675      |
| 1024, 675     | 850, 675      |
| 286, 450      | 440, 000      |

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane line pixels are identified in the `find_lines` function of `pipeline.py` be
first filtering to the thresholds described in step 2, then masking to an area
surrounding the previous lane line (which was bootstrapped from simply a
straight-ahead vertical line), and then a 2nd order polynomial was fit to the
non-zero points.

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature was calculated in the `get_line_from_filtered` function
of `pipeline.py` by first transforming the thresholded points from pixels to
meters using conversions I created by measuring the width of the lanes and the
length of the white dashes in pixels and then calculating a unit conversion
given that the width of a highway lane is 3.7m and the length of a white dash is
3m.

Next, I fit another second order polynomial to the meters points and then
calcuated its radius of curvature at a point close to the bottom of the image.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

To highlight the lanes in the orignal image, the polynomials (calcuated from the
top-down perspective) are first draw on a blank image and then that image is
transformed to the "ahead" perspective using the `top_down_to_ahead` function
(in `pipeline.py`)

Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a [link to my video result](https://youtu.be/qTHuEPN8ZMs)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a few major challenges in this project:

1. The radius of curvature always seems to low. From satelite imagery, it is
clear that the radii of the turns the car is taking are around 1000m, however my
calculations always result in a number about 1/3 to 1/4 of that. I double
checked the math, tried a few different perspective transforms, and even plotted
a circle of that radius on the image to be sure, all of which confirmed to me
that I had the calculation correct, however the values are still too low.

2. The pipeline will likely fail in areas with lines near to and parallel to the
actual lane lines. Since the binary images is a result of the logical OR of the
three filters, areas of strong yellow hue, large horizontal gradient, or high
saturation would introduce noise into the polynomial fit. Since the algorithm
agressively smoothes to previous lines (using a moving weighted average), once
the algorithm gets diverted from a lane line, it will tend to fit to the noise
rather than the actual line. This could be improved by checking R-values of the
fit, or by periodically confirming the quality of the fit with a sanity check.

