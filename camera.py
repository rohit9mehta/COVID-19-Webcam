from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import sys

absolute_dir = open("absolute_path.txt", "r").read()
# replace with path to folder
sys.path.append(absolute_dir)
from faceDetection import detector

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# defining face detector
cascPath = absolute_dir+ "haarcascade_frontalface_default.xml"
cascPathHand = absolute_dir + "haarcascade_hand3.xml"
# cascPathHandFist = "haarcascade_hand_default.xml" 
# NEW CLASSFIERS TO TEST (UNCOMMENT THIS)

cascPathHandFist2 = absolute_dir + "haarcascade_fist.xml"
# cascPathGlasses = "haarcascade_eyeglasses.xml"
# cascPathPalm = "haarcascade_palm.xml"
# cascPathClosedPalm = "haarcascade_closed_palm.xml"

#handCascadeFist2 = cv2.CascadeClassifier(cascPathHandFist2)
# glassesCascade = cv2.CascadeClassifier(cascPathGlasses)
# palmCascade = cv2.CascadeClassifier(cascPathPalm)
# closedPalmCascade = cv2.CascadeClassifier(cascPathClosedPalm)

faceCascade = cv2.CascadeClassifier(cascPath)
handCascade = cv2.CascadeClassifier(cascPathHand)
handCascadeFist2 = cv2.CascadeClassifier(cascPathHandFist2)

ds_factor=0.6

class VideoCamera(object):
    def __init__(self):
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        #vs = WebcamVideoStream(src=0).start()
        fps = FPS().start()
        
        while fps._numFrames < args["num_frames"]:
            frame = detector(self.video)
            frame = imutils.resize(frame, width=400)
            success, image = self.video.read()
            image = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
            interpolation=cv2.INTER_AREA)
               # grab the frame from the stream and resize it to have a maximum
               # width of 400 pixels
               # check to see if the frame should be displayed to our screen
            if args["display"] > 0:
                ret, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()
                # update the FPS counter
            fps.update()
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
