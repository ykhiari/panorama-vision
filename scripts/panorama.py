import numpy as np
import cv2
import imutils

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)
        self.cachedH = None