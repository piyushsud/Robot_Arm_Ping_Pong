import cv2
import numpy as np

class ColorChecker:

    def __init__(self):
        self.lower_orange = np.array([6, 70, 130])
        self.upper_orange = np.array([18, 255, 255])


    def is_valid(self, img):

        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masked_image = cv2.inRange(image_hsv, self.lower_orange, self.upper_orange)


        ind = np.where(masked_image > 0)

        # print(ind[0])


        if len(ind[0]) > 0:
            return True
        else:
            return False



