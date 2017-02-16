""" A class to classify vehicles """

import numpy as np
import cv2
from sklearn.svm import LinearSVC


class VehicleClassifier:
    """ A class to classify vehicles """

    def __init__(self):
        self.clf = LinearSVC()
        self.bin_spatial_size = (32, 32)
        self.color_hist_nbins = 32

    # Define a function to compute binned color features
    def bin_spatial(self, img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.bin_spatial_size).ravel()
        # Return the feature vector
        return features

    def color_hist(self, img, bins_range=(0, 256)):
        """ Compute color histogram features """
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.color_hist_nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.color_hist_nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.color_hist_nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def extract_features(self, img, cspace='RGB', hist_range=(0, 256)):
        """ Extract features from an image """
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(img)

        # Apply bin_spatial() to get spatial color features
        spatial_features = self.bin_spatial(feature_image)
        # Apply color_hist() also with a color space option now
        hist_features = self.color_hist(feature_image, bins_range=hist_range)
        # Append the new feature vector to the features list
        return np.concatenate((spatial_features, hist_features))

    def train(self, x_train, y_train):
        """ train the classifier """
        x_train_features = []
        for img in x_train:
            x_train_features.append(self.extract_features(img))
        self.clf.fit(x_train_features, y_train)

    def predict(self, img):
        features = [self.extract_features(img)]
        return self.clf.predict(features)[0]
