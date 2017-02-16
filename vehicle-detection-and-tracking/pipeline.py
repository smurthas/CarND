""" pipeline to find lane lines """
import sys
import glob

import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split

#import image_util as iu

from VehicleTracker import VehicleTracker

tracker = VehicleTracker()


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """ draw bounding boxes on an image """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    #print('bboxes', bboxes)
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def pipeline(img):
    """ pipeline to detect lane lines in images """
    #print(img)
    detections = tracker.detect_vehicles(img)
    #print(detections)
    img = draw_boxes(img, detections)
    return img

def process_video(in_filename, out_filename):
    """read in a video, processes it with the provided debug setting, and then
    write it back out"""

    print(in_filename)
    clip2 = VideoFileClip(in_filename)
    print(clip2)
    clip = clip2.fl_image(pipeline)
    print(clip)
    clip.write_videofile(out_filename, audio=False)


print('Loading training data...')
features = []
subsample = 10
cars = list(glob.iglob('data/vehicles/**/*.png', recursive=True))[0:subsample]
noncars = list(glob.iglob('data/non-vehicles/**/*.png', recursive=True))[0:subsample]
y = np.hstack((np.ones(len(cars)), np.zeros(len(noncars))))
for car in cars:
    features.append(cv2.imread(car))
for noncar in noncars:
    features.append(cv2.imread(noncar))

X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=1337)

print('Training classifier...')
tracker.train(X_train, y_train)

print('Processing video...')
infile = sys.argv[1]
outfile = sys.argv[2]
process_video(infile, outfile)
