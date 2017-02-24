## Vehicle Detection and Tracking

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./assets/windows.png
[image2]: ./assets/example-frame-1.png
[image3]: ./assets/example-frame-2.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features` method of the
`VehicleClassifer` class.

I explored using different values of pixels per cell, cells per block,
orientations, and channels. For each, I varied it across a range (while holding
other params constant) and recorded precision and recall values by testing it on
ground truth of the project video that I captured by hand.

I found the following values provided a useful balance of precision and recall,
ultimately resulting in a 94% precision and 93% recall on the project video. I
calculated precision as the percentage of detections that overlapped at least
50% with a ground truth box, and recall as the percentage of ground truth boxes
cover at least 50% by detections.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the provided "vehicles" and "non-vehicles" data
sets, augmenting them by fliping the vehicle images horizontally. I experimented
with jittering the images and zooming and cropping, but I found that all of
those other options actually made the performance of the classifier worse on the
project video, which I attribute to overfitting.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search on line 126 of the `detect_vehicles`
method in `VehicleTracker.py`. It projects 9 lines from the bottom outside of
the frame to the middle, and places boxes of decreasing size along the
lines.

Here you can see the boxes here:

![detection windows][image1]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As I alluded to above, I experimented with many of the parameters. First, I
tried adding and removing the color histogram, spatial binning, and HOG feature
extraction, and found that including all three resulted in the highest score on
the test data (which was 20% of the data, held back from training). I also
experimented with different color spaces, and found that YUV proved better than
RGB and HSV given my algorithm, sliding windows, etc.

Ultimately, my configuration resulted in a 98.9% accuracy test score.

Here are some example images, which include my debug overlay at the top. The
left image has hot windows in red, the center one has the thresholded heat map,
and the right one has the sliding windows. Finally, there is some text with
various classifier parameters. I found this very helpful when cycling through
many different configurations as it was otherwise had to remember which test
video was which.

![example frame 1][example-frame-1]
![example frame 2][example-frame-2]


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here is a [link to my video result](https://youtu.be/4hv3oSlEYt8)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The algorithm records the positions of positive detections in each frame of the
video. From the positive detections, it creates a heatmap that is averaged over
time and then thresholds that averaged map to identify vehicle positions.
It then uses `scipy.ndimage.measurements.label()` to identify individual blobs
in the heatmap and rejects those that are not square-ish or are too small.

>>>>>>
TODO: FINISH ME STARTING HERE
>>>>>>

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

