import sys
import csv

import cv2
import numpy as np
import pickle
from moviepy.editor import VideoFileClip

gt_file = sys.argv[1]
video_file = sys.argv[2]
out_file = sys.argv[3]


def read_in_ground_truth(filename):
    gt = {}
    with open(filename, newline='') as csvfile:
        gt_reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        for line in gt_reader:
            frame = int(line[0])
            bbox = (np.ndarray.tolist(np.ndarray.astype(np.array((line[1:3])), int)),
                    np.ndarray.tolist(np.ndarray.astype(np.array((line[3:5])), int)))
            if frame not in gt:
                gt[frame] = []
            gt[frame].append(bbox)
    return gt

frames = []

filenum = 0
frame_num = 1
def pipeline(img):
    global frames, filenum, frame_num

    if len(frames) == 600:
        pickle.dump(frames, open(out_file + '_' + str(filenum) + '.p', 'wb'))
        frames = []
        filenum += 1
    frame = [img]
    bboxes = []
    if frame_num in GT:
        bboxes = GT[frame_num]
        img = np.copy(img)
        for bbox in bboxes:
            print(bbox)
            cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (255, 0, 0), 2)
    frame.append(bboxes)
    frames.append(frame)
    frame_num += 1
    return img

print(video_file)

GT = read_in_ground_truth(gt_file)
clip2 = VideoFileClip(video_file)
clip = clip2.fl_image(pipeline)
clip.write_videofile('blargh.mp4', audio=False)
pickle.dump(frames, open(out_file + '_' + str(filenum) + '.p', 'wb'))
