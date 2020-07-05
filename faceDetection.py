import cv2
# import sys
import numpy as np
import math

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
handCascade = cv2.CascadeClassifier("haarcascade_hand3.xml")
fistCascade = cv2.CascadeClassifier("haarcascade_fist.xml")
fingerCascade = cv2.CascadeClassifier("finger.xml")

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def detector(video):
    cam = create_capture(video, fallback='synth:bg={}:noise=0.05'.format(cv2.samples.findFile('messi.jpg')))
    # Capture frame-by-frame
    frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detect(gray, faceCascade)
    hands = detect(gray, handCascade)
    fist = detect(gray, fistCascade)
    finger = detect(gray, fingerCascade)
    vis = frame.copy()
    
    x0, x1, y0, y1, x2, x3, y2, y3, x4, x5, y4, y5, x6, x7, y6, y7, x10, x11, y10, y11 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    for (x, y, w, h) in faces:
        x0 = x
        y0 = y
        x1 = w
        y1 = h
        rect1 = cv2.rectangle(vis, (x, y), (w, h), (0, 255, 0), 2)
    
    for (x, y, w, h) in hands:
        x2 = x
        y2 = y
        x3 = w
        y3 = h
        rect2 = cv2.rectangle(vis, (x, y), (w, h), (255, 0, 0), 2)

    for (x, y, w, h) in fist:
        x4 = x
        y4 = y
        x5 = w
        y5 = h
        rect3 = cv2.rectangle(vis, (x, y), (w, h), (30, 140, 20), 2)
    
    for (x, y, w, h) in finger:
        x6 = x
        y6 = y
        x7 = w
        y7 = h
        rect3 = cv2.rectangle(vis, (x, y), (w, h), (70, 14, 140), 2)

    if (intersectCheck(x0, y0, x1, y1, x2, y2, x3, y3) or intersectCheck(x0, y0, x1, y1, x4, y4, x5, y5) 
        or intersectCheck(x0, y0, x1, y1, x6, y6, x7, y7)):
        sign = cv2.putText(vis,'PLEASE DO NOT TOUCH YOUR FACE!',(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
    return vis

def create_capture(source, fallback):
    source = 0
    params = {}

    cap = None
    if source == 'synth':
        Class = classes.get(params.get('class', None), VideoSynthBase)
        try: cap = Class(**params)
        except: pass
    else:
        cap = cv2.VideoCapture(source)
        if 'size' in params:
            w, h = map(int, params['size'].split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        if fallback is not None:
            return create_capture(fallback, None)
    return cap

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
