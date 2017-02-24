""" A class to classify vehicles """

import numpy as np
import cv2
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

class VehicleClassifier:
    """ A class to classify vehicles """

    def __init__(self, color_space='YUV'):
        self.clf = LinearSVC()

        self.color_space = color_space
        self.bin_spatial_size = (16, 16)
        self.color_hist_nbins = 12

        self.hog_pix_per_cell = 8
        self.hog_cell_per_block = 2
        self.hog_orient = 9
        self.hog_channel = 'ALL'

        self.scaler = StandardScaler()

        self.train_sample_count = 0
        self.feature_vector_length = 0

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

    def get_hog_features(self, img, vis=False, feature_vec=True):
        """ returns HOG features and visualization """
        # Call with two outputs if vis==True
        pixels_per_cell = (self.hog_pix_per_cell, self.hog_pix_per_cell)
        cells_per_block = (self.hog_cell_per_block, self.hog_cell_per_block)
        if vis is True:
            features, hog_image = hog(img, orientations=self.hog_orient,
                                      pixels_per_cell=pixels_per_cell,
                                      cells_per_block=cells_per_block,
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.hog_orient,
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features


    def extract_features(self, img):
        """ Extract features from an image """

        if img.shape != (64, 64, 3):
            img = cv2.resize(img, (64, 64))

        # apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)

        spatial_features = self.bin_spatial(feature_image)

        # Apply color_hist()
        hist_features = self.color_hist(feature_image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(self.get_hog_features(feature_image[:, :, channel],
                                                          vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = self.get_hog_features(feature_image[:, :, self.hog_channel],
                                                 vis=False, feature_vec=True)

        return np.concatenate((spatial_features, hist_features, hog_features))

    def train(self, X_train, y_train):
        """ train the classifier """
        X_train_features = []
        for img in X_train:
            X_train_features.append(self.extract_features(img))

        self.train_sample_count += len(X_train_features)
        self.feature_vector_length = len(X_train_features[0])
        print('Feature Vector Length:', self.feature_vector_length)
        self.scaler = self.scaler.fit(X_train_features)
        self.clf.fit(self.scaler.transform(X_train_features), y_train)

    def score(self, X_test, y_test):
        """ extracts features and then tests """
        X_test_features = []
        for img in X_test:
            X_test_features.append(self.extract_features(img))
        return self.clf.score(self.scaler.transform(X_test_features), y_test)

    def predict(self, img, min_dec=1.0):
        features = self.scaler.transform([self.extract_features(img)])
        pred = self.clf.predict(features)[0]
        return pred == 1 and self.clf.decision_function(features)[0] >= min_dec

    def __str__(self):
        self_str = 'color_space=%s\nbin_spatial_size=%s\ncolor_hist_nbins=%d\n\
hog_pix_per_cell=%d\nhog_cell_per_block=%d\nhog_orient=%d\nhog_channel=%s\ntrain_sample_count=%d'%\
            (self.color_space, self.bin_spatial_size, self.color_hist_nbins,
             self.hog_pix_per_cell, self.hog_cell_per_block, self.hog_orient,
             self.hog_channel, self.train_sample_count)
        return self_str
