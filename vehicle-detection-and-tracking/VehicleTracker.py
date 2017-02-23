""" Class to detect vehicles over multiple frames in a video """
#from math import pow

import numpy as np
from VehicleClassifier import VehicleClassifier
from scipy.ndimage.measurements import label


def bounded_point(x, y, shape=(1280, 720)):
    x = min(max(int(x), 0), shape[0])
    y = min(max(int(y), 0), shape[1])
    return (x, y)

class VehicleTracker:
    """ Class to detect vehicles over multiple frames in a video """
    def __init__(self, y_min=0.55, heat_map_alpha=0.9, heat_map_threshold=0.5):
        self.y_min = y_min
        self.clf = VehicleClassifier()
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

    def slide_windows_linear(self, point_close, point_far,
                             close_size=(256, 256), far_size=(88, 88),
                             n_windows=12, power_factor=0.7):
        """ slide windows along a line, with decreasing size """
        x_fac = (point_far[0] - point_close[0]) / pow(n_windows-1, power_factor)
        y_fac = (point_far[1] - point_close[1]) / pow(n_windows-1, power_factor)
        #x_step = (point_far[0] - point_close[0]) / (n_windows-1)
        #y_step = (point_far[1] - point_close[1]) / (n_windows-1)

        x_size_step = (far_size[0] - close_size[0]) / (n_windows-1)
        y_size_step = (far_size[1] - close_size[1]) / (n_windows-1)

        these_windows = []
        for i in range(n_windows):
            pf = pow(i, power_factor)
            #center = (point_close[0] + (x_step*i), point_close[1] + (y_step*i))
            center = (point_close[0] + (pf*x_fac), point_close[1] + (pf*y_fac))
            extent = ((close_size[0] + (x_size_step*i))/2.,
                      (close_size[1] + (y_size_step*i))/2.)
            p1 = bounded_point(center[0] - extent[0], center[1] - extent[1])
            p2 = bounded_point(center[0] + extent[0], center[1] + extent[1])
            #print(i, center, extent, p1, p2)
            self.windows.append((p1, p2))
            these_windows.append((p1, p2))
        self.windows_sets.append(these_windows)

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

    def detect_vehicles(self, img, min_dec=0.35):
        """ detects vehicles in an image """
        # Initialize a list to append window positions to
        self.windows = []
        self.windows_sets = []

        # starting at far-right, moving inwards
        self.slide_windows_linear((1280, 440), (940, 435), close_size=(120, 120))
        self.slide_windows_linear((1280, 475), (910, 435), close_size=(180, 180))
        self.slide_windows_linear((1280, 520), (815, 435), close_size=(280, 280))
        self.slide_windows_linear((960, 510), (755, 435), close_size=(220, 220), n_windows=10)

        # center
        self.slide_windows_linear((640, 500), (640, 430), close_size=(400, 400), n_windows=6)

        # from center, moving left
        self.slide_windows_linear((320, 510), (525, 435), close_size=(220, 220), n_windows=10)
        self.slide_windows_linear((0, 520), (465, 435), close_size=(280, 280))
        self.slide_windows_linear((0, 475), (370, 435), close_size=(180, 180))
        self.slide_windows_linear((0, 440), (340, 435), close_size=(120, 120))


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
            x = x2 - x1
            y = y2 - y1
            area = x * y
            aspect_ratio = max(x, y) / min(x, y)

            if area > 1024 and aspect_ratio < 2.2:
                self.detections.append(bbox)
            else:
                print('rejected bbox:', bbox, x, y, area, aspect_ratio)

        return self.detections

    def train(self, x_train, y_train):
        return self.clf.train(x_train, y_train)
