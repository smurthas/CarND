""" score a classifier """

import numpy as np
import sys

import pickle
import cv2
import numpy as np
from shapely.geometry import box

from VehicleTracker import VehicleTracker


def box_from_bbox(bbox):
    """ creates a shapely box from a bounding box """
    x1 = min([bbox[0][0], bbox[1][0]])
    y1 = min([bbox[0][1], bbox[1][1]])
    x2 = max([bbox[0][0], bbox[1][0]])
    y2 = max([bbox[0][1], bbox[1][1]])
    return box(x1, y1, x2, y2)

def percent_intersection(base_box, others):
    bbox = box_from_bbox(base_box)
    union = box(0, 0, 0, 0)
    for other in others:
        union = union.union(box_from_bbox(other))
    return bbox.intersection(union).area / bbox.area

def boxes_covered(base_boxes, other_boxes, min_intersection=0.5):
    """ returns the percentage of the GT area covered by the detections """
    if len(base_boxes) < 1:
        return 1.0
    count = 0
    for bbox in base_boxes:
        if percent_intersection(bbox, other_boxes) >= min_intersection:
            count += 1
    return count / len(base_boxes)

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=2):
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


def slice_out_bboxes(img, bboxes):
    imgs = []
    for bbox in bboxes:
        x1 = max(min([bbox[0][0], bbox[1][0]]), 0)
        y1 = max(min([bbox[0][1], bbox[1][1]]), 0)
        x2 = min(max([bbox[0][0], bbox[1][0]]), img.shape[1] - 1)
        y2 = min(max([bbox[0][1], bbox[1][1]]), img.shape[0] - 1)
        imgs.append(img[y1:y2, x1:x2])
    return imgs



clf_file = sys.argv[1]
out_dir = sys.argv[2]
min_dec = sys.argv[3]
data_files = sys.argv[4:]
print(clf_file)
print(data_files)

clf = None
data = None

with open(clf_file, 'rb') as fid:
    clf = pickle.load(fid)

for fn in data_files:
    with open(fn, 'rb') as fid:
        this_data = pickle.load(fid)
        if data is None:
            data = this_data
        else:
            data = np.concatenate((data, this_data))

print(str(len(data)) + ' frames.')

tracker = VehicleTracker()
tracker.clf = clf
num = 0
recalls = []
precisions = []
for frame in data:
    num += 1
    if num < 150:
        continue
    img_data = frame[0]
    detections = tracker.detect_vehicles(img_data, min_dec)
    gt = frame[1]
    recall = 0.
    precision = 0.

    if len(gt) > 0:
        recall = boxes_covered(gt, tracker.hot_windows)
        recalls.append(recall)

    if len(detections) > 0:
        if len(gt) < 1:
            gt = []

        precision = boxes_covered(tracker.hot_windows, gt)
        precisions.append(precision)

    print('frame %d, hot windows: %d, recall: %f, precision %f'%
          (num, len(tracker.hot_windows), recall, precision))

    img_data = draw_boxes(img_data, gt, color=(0, 255, 0))
    img_data = draw_boxes(img_data, tracker.hot_windows, color=(255, 0, 0))
    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_dir + '/' + str(num) + '.png', img_data)

print('Totals')
print('Precision: %f'%np.average(precisions))
print('Recall:    %f'%np.average(recalls))
