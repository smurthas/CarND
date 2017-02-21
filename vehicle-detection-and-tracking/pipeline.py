""" pipeline to find lane lines """
import sys
import glob

import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle

from VehicleTracker import VehicleTracker
from HUDOverlay import HUDOverlay

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
    """ draw bounding boxes on an image """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    #print('bboxes', bboxes)
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        ul = (int(bbox[0][0]), int(bbox[0][1]))
        lr = (int(bbox[1][0]), int(bbox[1][1]))
        cv2.rectangle(imcopy, ul, lr, color, thick)
    # Return the image copy with boxes drawn
    return imcopy


tracker = VehicleTracker()
overlay = HUDOverlay()

def pipeline(img):
    """ pipeline to detect lane lines in images """
    detections = tracker.detect_vehicles(img)
    img = draw_boxes(img, detections)
    windows_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    i = 0
    for windows in tracker.windows_sets:
        windows_img = draw_boxes(windows_img, windows, color=(i, 255, 255))
        i += 180/len(tracker.windows_sets)
    windows_img = cv2.cvtColor(windows_img, cv2.COLOR_HSV2RGB)
    heat_map = np.ndarray.astype(np.dstack((tracker.heat_map*16,
                                            tracker.heat_map*16,
                                            tracker.heat_map*16)), np.uint8)
    #print(type(heat_map), type(img), heat_map.shape, img.shape)
    heat_map = cv2.addWeighted(img, 0.7, heat_map, 1.0, 0.)
    heat_map_w_boxes = draw_boxes(heat_map, tracker.hot_windows, color=255)
    hud_imgs = [heat_map_w_boxes,
                tracker.thresholded_heatmap*16.,
                windows_img]
                #tracker.labels_map*80.]
    texts = ['HM Alpha: %.2f' % tracker.heat_map_alpha,
             'HM Tresh: %.2f' % tracker.heat_map_threshold,
             'Windows: %d, %d' %(len(tracker.windows_sets), len(tracker.windows))
            ]
    img = overlay.add_to(img, hud_imgs, texts, 0.5)
    #img = cv2.addWeighted(img, 1.0, labels, 1.0, 0.)
    return img

def process_video(classifier_file, in_filename, out_filename):
    """read in a video, processes it with the provided debug setting, and then
    write it back out"""

    with open(classifier_file, 'rb') as fid:
        tracker.clf = pickle.load(fid)

    print('Processing video...')
    print(in_filename)
    clip2 = VideoFileClip(in_filename)
    clip = clip2.fl_image(pipeline)
    clip.write_videofile(out_filename, audio=False)


def train(classifier_file):
    """ trains a classifier and saves it """
    print('Loading training data...')
    features = []
    subsample = None
    cars = list(glob.iglob('data/vehicles/**/*.png', recursive=True))[0:subsample]
    noncars = list(glob.iglob('data/non-vehicles/**/*.png', recursive=True))[0:subsample]
    y = []
    #y = np.hstack((np.ones(len(cars)), np.zeros(len(noncars))))
    for car in cars:
        im = cv2.cvtColor(cv2.imread(car), cv2.COLOR_BGR2RGB)
        features.append(im)
        y.append(1)
        features.append(cv2.flip(im, flipCode=1))
        y.append(1)
        zm = cv2.resize(im, (88, 88))
        features.append(zm[0:64, 0:64])
        #features.append(zm[12:76, 0:64])
        features.append(zm[24:88, 0:64])
        y.append(1)
        #y.append(1)
        y.append(1)
        features.append(zm[0:64, 12:76])
        #features.append(zm[12:76, 12:76])
        features.append(zm[24:88, 12:76])
        y.append(1)
        #y.append(1)
        y.append(1)
        features.append(zm[0:64, 24:88])
        #features.append(zm[12:76, 24:88])
        features.append(zm[24:88, 24:88])
        y.append(1)
        #y.append(1)
        y.append(1)

    for noncar in noncars:
        features.append(cv2.cvtColor(cv2.imread(noncar), cv2.COLOR_BGR2RGB))
        y.append(0)

    features, y = shuffle(features, y, random_state=1337)

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=1337)

    print('Training classifier on %d images, holding back %d for testing...'%(len(X_train), len(X_test)))
    tracker.train(X_train, y_train)

    print('Test Accuracy of SVC = ', round(tracker.clf.score(X_test, y_test), 4))
    with open(classifier_file, 'wb') as fid:
        pickle.dump(tracker.clf, fid)

command = sys.argv[1]

if command == 'detect':
    clffile = sys.argv[2]
    infile = sys.argv[3]
    outfile = sys.argv[4]
    process_video(clffile, infile, outfile)
elif command == 'train':
    clffile = sys.argv[2]
    train(clffile)
