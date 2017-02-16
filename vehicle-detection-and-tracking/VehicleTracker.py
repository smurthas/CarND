""" Class to detect vehicles over multiple frames in a video """

#from sklearn.preprocessing import StandardScaler

from VehicleClassifier import VehicleClassifier

class VehicleTracker:
    """ Class to detect vehicles over multiple frames in a video """
    def __init__(self, y_min=0.58):
        self.y_min = y_min
        self.clf = VehicleClassifier()
        self.windows = []
        self.hot_windows = []

    def get_windows(self):
        self.windows = [((930, 375), (1070, 515))]

    def check_windows(self, img):
        self.hot_windows = []
        for window in self.windows:
            roi = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            if self.clf.predict(roi) == 1:
                self.hot_windows.append(window)

    def detect_vehicles(self, img):
        self.get_windows()
        self.check_windows(img)
        return self.hot_windows

    def train(self, x_train, y_train):
        self.clf.train(x_train, y_train)
