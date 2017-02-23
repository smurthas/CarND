""" video overlay heads up display """

import cv2
import numpy as np

class HUDOverlay:
    """ video overlay heads up display """
    def __init__(self, height=200, margin=10):
        self.height = height
        self.margin = margin
        self.imgs_height = height - 2*margin

    def add_to(self, target_img, imgs, texts, alpha):
        """ add an overlay to the target image with the passed HUD images """
        overlay = np.copy(target_img)
        overlay[0:self.height] = np.zeros((self.height, target_img.shape[1], 3))
        overlay = cv2.addWeighted(target_img, 1.0-alpha, overlay, alpha, 0.)
        x = self.margin
        y = self.margin
        for img in imgs:
            resize_to = (int(self.imgs_height/img.shape[0]*img.shape[1]),
                         int(self.imgs_height))
            img = cv2.resize(img, resize_to)
            img = np.ndarray.astype(img, np.uint8)
            y1 = y+resize_to[1]
            x1 = x+resize_to[0]
            img[img > 255] = 255
            if len(img.shape) == 2:
                img = np.dstack((img, img, img))
            overlay[y:y1, x:x1] = img
            cv2.rectangle(overlay, (x, y), (x1, y1), (255, 255, 255), 1)
            x += resize_to[0] + self.margin

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        y += 30
        for text in texts:
            cv2.putText(overlay, text, (x, y), font_face, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            y += 30
            #x += resize_to[0] + self.margin

        return overlay
