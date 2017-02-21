""" Class to detect vehicles over multiple frames in a video """

import numpy as np
from VehicleClassifier import VehicleClassifier
from scipy.ndimage.measurements import label

class VehicleTracker:
    """ Class to detect vehicles over multiple frames in a video """
    def __init__(self, y_min=0.55, heat_map_alpha=0.9, heat_map_threshold=3.5):
        self.y_min = y_min
        self.clf = VehicleClassifier(color_space='YUV')
        self.windows = []
        self.windows_sets = []
        self.hot_windows = []
        self.heat_map = None
        self.frame_heatmap = None
        self.heat_map_alpha = heat_map_alpha

        self.heat_map_threshold = heat_map_threshold
        self.detections = []
        self.thresholded_heatmap = None
        self.labels_map = None
        self.labels_count = 0

    def slide_window(self, img, x_start_stop=None, y_start_stop=None,
                     xy_window=(128, 128), xy_overlap=(0.7, 0.7)):
        """ slides a window of a given size across the image, returning a list
        of window bboxes """
        if x_start_stop is None:
            x_start_stop = [None, None]
        if y_start_stop is None:
            y_start_stop = [None, None]
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] is None:
            y_start_stop[0] = self.y_min * img.shape[0]
        if y_start_stop[1] is None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1

        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        these_windows = []
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                self.windows.append(((startx, starty), (endx, endy)))
                these_windows.append(((startx, starty), (endx, endy)))
        self.windows_sets.append(these_windows)

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, min_dec=1.0):
        #1) Create an empty list to receive positive detection windows
        self.hot_windows = []
        #2) Iterate over all windows in the list
        for window in self.windows:
            #3) Predict using classifier
            test_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            if self.clf.predict(test_img, min_dec):
                self.hot_windows.append(window)

    def update_heatmap(self, img):
        self.frame_heatmap = np.zeros((img.shape[0], img.shape[1]))
        for box in self.hot_windows:
            self.frame_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    def detect_vehicles(self, img, min_dec=1.0):
        """ detects vehicles in an image """
        first_y = self.y_min*img.shape[0]
        # Initialize a list to append window positions to
        self.windows = []
        self.windows_sets = []

        self.slide_window(img, y_start_stop=[first_y, first_y + 170], xy_window=(256, 256))
        y_start_stop = [first_y+60, first_y+60 + 170]
        self.slide_window(img, x_start_stop=[0, img.shape[1]*0.3],
                          y_start_stop=y_start_stop, xy_window=(256, 256))
        self.slide_window(img, x_start_stop=[img.shape[1]*0.7, None],
                          y_start_stop=y_start_stop, xy_window=(256, 256))

        self.slide_window(img, y_start_stop=[first_y-20, first_y-20 + 130], xy_window=(192, 192))
        self.slide_window(img, x_start_stop=[0, img.shape[1]*0.4],
                          y_start_stop=[first_y+37, first_y+37 + 130],
                          xy_window=(192, 192))
        self.slide_window(img, x_start_stop=[img.shape[1]*0.6, None],
                          y_start_stop=[first_y+37, first_y+37 + 130],
                          xy_window=(192, 192))

        self.slide_window(img, x_start_stop=[0.1*img.shape[1], 0.9*img.shape[1]],
                          y_start_stop=[first_y-20, first_y-20 + 137], xy_window=(136, 136))
        self.slide_window(img, x_start_stop=[0.1*img.shape[1], 0.9*img.shape[1]],
                          y_start_stop=[first_y-20, first_y-20 + 137], xy_window=(136, 136))

        y_start_stop = [first_y, first_y+155]
        self.slide_window(img, x_start_stop=[0.18*img.shape[1], 0.82*img.shape[1]],
                          y_start_stop=y_start_stop, xy_window=(104, 104))
        #print(len(self.windows))

        self.search_windows(img, min_dec)
        #print('after search, HW: %d'%len(self.hot_windows))

        self.update_heatmap(img)
        if self.heat_map is None:
            self.heat_map = np.zeros((img.shape[0], img.shape[1]))

        self.heat_map = (self.heat_map_alpha * self.heat_map) +\
                        ((1.0-self.heat_map_alpha)*
                         np.ndarray.astype(self.frame_heatmap, np.float32))
        self.thresholded_heatmap = np.copy(self.heat_map)
        self.thresholded_heatmap[self.thresholded_heatmap < self.heat_map_threshold] = 0
        labels = label(self.thresholded_heatmap)
        self.labels_map = labels[0]
        self.labels_count = labels[1]

        self.detections = []
        for label_i in range(self.labels_count):
            label_mask = np.copy(self.labels_map)
            label_mask[label_mask != (label_i+1)] = 0
            label_coords = np.nonzero(label_mask)
            x1 = min(label_coords[1])
            y1 = min(label_coords[0])
            x2 = max(label_coords[1])
            y2 = max(label_coords[0])
            bbox = ((x1, y1), (x2, y2))
            #print(bbox)
            self.detections.append(bbox)

        return self.detections

    def train(self, x_train, y_train):
        return self.clf.train(x_train, y_train)
