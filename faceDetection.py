from __future__ import print_function

import numpy as np
import cv2 as cv
import re

cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
fistCascade = cv.CascadeClassifier("fist.xml")
handCascade = cv.CascadeClassifier("haarcascade_hand3.xml")

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects
    
""" def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
"""

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
def clock():
    return cv.getTickCount() / cv.getTickFrequency()

def detector(video):
    cam = create_capture(video, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('messi.jpg')))
    while True: 
        #print(video.read())
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)
        fist = detect(gray, fistCascade)
        hands = detect(gray, handCascade)
        frame = img.copy()
        draw_rects(frame, rects, (0, 255, 0))
        draw_rects(frame, fist, (0, 0, 255))
        draw_rects(frame, hands, (30, 140, 20))
        #cv.imshow('facedetect', frame)

        

    print('Done')
    return frame
    
def create_capture(source, fallback):
    source = 0
    cap = None
    cap = cv.VideoCapture(source)
    params = {}

    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        if fallback is not None:
            return create_capture(fallback, None)
    return cap

"""" TODO
def intersectCheck(x0, y0, x1, y1, x2, y2, x3, y3):
    #left side intersect
    if (x3 >= x0 and x2 < x0):
        if (y3 >= y1 and y2 <= y1) or (y2 <=y0 and y3>= y0):
            return True
    #right side intersect
    if (x3 > x1 and x2 <= x1):
        if (y3 >= y1 and y2 <= y1) or (y2 <=y0 and y3>= y0):
            return True
    else:
        return False
""" 



if __name__ == '__main__':
    video = cv.VideoCapture(0)
    detector(video)
    cv.destroyAllWindows()