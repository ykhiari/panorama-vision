import numpy as np
import cv2
import imutils

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)
        self.cachedH = None
    
    self.stitch(self, images, ratio=0.75, reprojThresh=0.4):
        (imageB, imageA) = images
        if self.cachedH is None:
            # detect keypoints and extract
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
            # if the match is None then there is no enough matched keypoints
            if M is None:
                return None
            # cache the homography matrix
            self.cachedH = M[1]
        result = cv2.wrapPerspective(imageA, self.cachedH, (imageA.shape[1]+imageB.shape[1],imageA.shape[0]))
        result[0:imageB.shape[0],0:imageB.shape[1]] = imageB
        return result