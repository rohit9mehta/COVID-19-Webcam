import cv2
import sys
import imutils

absolute_dir = open("absolute_path.txt", "r").read()
# replace with path to folder
sys.path.append(absolute_dir)
from faceDetection import detector


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
        frame = detector(self.video)
        frame = imutils.resize(frame, width=400)
        success, image = self.video.read()
        image = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
        interpolation=cv2.INTER_AREA)                    
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)
        # hand_rects = handCascade.detectMultiScale(gray, 1.3, 5)
        # fist_rects = handCascadeFist2.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            break
        # for (x, y, w, h) in hand_rects:
        #     x2 = x
        #     y2 = y
        #     x3 = x+w
        #     y3 = y+h
        #     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #     break

        # for (x, y, w, h) in fist_rects:
        #     x4 = x
        #     y4 = y
        #     x5 = x+w
        #     y5 = y+h
        #     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #     break
        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

