from scripts.pedestriandetector import PedistrianDetector
from scripts.panorama import Stitcher
import numpy as np
import argparse
import datetime
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--cascade", required=True, help="path to the haar cascade")
args = vars(ap.parse_args())

stitcher = Stitcher()
pd = PedistrianDetector(args["cascade"])

print("[INFO] Start streaming...")
leftstream = cv2.VideoCapture(0)
rightstream = cv2.VideoCapture(1)
time.sleep(2)

while True:
    _, left = leftstream.read()
    _, right = rightstream.read()
    # left = imutils.resize(left, width=500)
    # right = imutils.resize(right, width=500)
    
    result = stitcher.stitch([left,right])
    if result is None:
        print("[INFO] homography could not be computed")
        break
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    pedistrianRects = pd.detect(gray)
    for (x, y, w, h) in pedistrianRects:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(result, ts, (10, result.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Result", result)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

print("[INFO] cleaning up...")
cv2.destroyAllWindows()
leftstream.release()
rightstream.release()