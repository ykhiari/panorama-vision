import cv2

class PedistrianDetector:
    
    def __init__(self, cascadePath):
        self.pedestrianCascade = cv2.CascadeClassifier(cascadePath)
        
    def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
        rects = self.pedestrianCascade.detectMultiScale(image, scaleFactor=scaleFactor,
                        minNeighbors=minNeighbors, minSize=minSize,flags=cv2.CASCADE_SCALE_IMAGE)
        return rects
    
    