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

def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    cam = create_capture(video_src, fallback=None)

    while True:
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)
        fist = detect(gray, fistCascade)
        hands = detect(gray, handCascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        draw_rects(vis, fist, (0, 0, 255))
        draw_rects(vis, hands, (30, 140, 20))
        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break

    print('Done')
    
def create_capture(source, fallback):
    source = str(source).strip()

    source = re.sub(r'(^|=)([a-zA-Z]):([/\\a-zA-Z0-9])', r'\1?disk\2?\3', source)
    chunks = source.split(':')
    chunks = [re.sub(r'\?disk([a-zA-Z])\?', r'\1:', s) for s in chunks]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )

    cap = None
    cap = cv.VideoCapture(source)
    if 'size' in params:            
        w, h = map(int, params['size'].split('x'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
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
    main()
    cv.destroyAllWindows()



