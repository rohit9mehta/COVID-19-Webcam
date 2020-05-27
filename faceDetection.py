# import cv2
# import sys
# import logging as log
# import datetime as dt

# #Getting inputs from the system
# # imagePath = sys.argv[0]
# # cascadePath = sys.argv[1]
# cascadePath = "haarcascade_frontalface_default.xml"
# videoInput = cv2.VideoCapture(0) #Setting video source to webcam

# #Creating the haar feature characteristic (under Viola-Jones Object Detection Framework)
# cvCascadeClassifier = cv2.CascadeClassifier(cascadePath)

# videoOn = True
# while videoOn:
#     if (videoInput.isOpened()):
#         print("Camera Working.")
#     else:
#         print('System Error. Camera not loading. Video capture failed.')

#     videoFrame = videoInput.read() # .read() returns a video frame as well as a
#                                         #return code(which can be ignored for webcam
#                                         # video inputs but I'm just including it for now)
#     colorSpaceConversionCode = cv2.COLOR_BGR2GRAY
#     grayScale = cv2.cvtColor(videoFrame, colorSpaceConversionCode) # converting BGR color scale to solely grayscale
#     # detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects
#     # @brief Detects objects of different sizes in the input image.
#     # The detected objects are returned as a list of rectangles.
#     detectObjects = cvCascadeClassifier.detectMultiScale(
#         grayScale, scaleFactor = 1.5, minNeighbors = 4, minSize = (40, 40)
#     )

#     #Draw a BOX (rectangle?) around the detected face. cv2.rectangle(image, start_point, end_point, color, thickness)
#     image = videoFrame
#     color = (16, 202, 41)
#     thickness = 2
#     for (x, y) in  detectObjects:
#         start_point = (x, y)
#         end_point = (2 * x, 2 * y) # x + x is length and y + y is breadth
#         videoFrame = cv2.rectangle(image, start_point, end_point, color, thickness)

#     log.info("Detected face(s): "+ str(len(faces)) + " at time: " + str(dt.datetime.now()))

#     # Window name in which image is displayed
#     window_name = 'Chrome'
#     # Displaying the image
#     cv2.imshow(window_name, videoFrame)

#     if cv2.waitKey(1) & 0b11111111 == ord('x'):
#         break
#     cv2.imshow(window_name, videoFrame)

# videoInput.release() # Very important!!! It gets rid of pointer that
#                      # points to the memory locaion of the video source.
#                      # Privacy++
# print("Program terminated")
# cv2.destroyAllWindows() #shuts down all cv stuff

import cv2
# import sys
import numpy as np
import math

def detector(video):
    cascPath = "haarcascade_frontalface_default.xml"
    cascPathHand = "haarcascade_hand3.xml"
    # cascPathHandFist = "haarcascade_hand_default.xml" 
    # NEW CLASSFIERS TO TEST (UNCOMMENT THIS)

    cascPathHandFist2 = "haarcascade_fist.xml"
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


    # video_capture = cv2.VideoCapture(0)

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

    # Capture frame-by-frame
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    hands = handCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    fists = handCascadeFist2.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # fists2 = handCascadeFist2.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(60, 60),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )

    # palm = palmCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(60, 60),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )

    # closedPalm = closedPalmCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(60, 60),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )


    # Draw a rectangle around the faces
    # x6, x7, y6, y7 = 0,0,0,0
    x0, x1, y0, y1, x2, x3, y2, y3, x4, x5, y4, y5, x8, x9, y8, y9, x10, x11, y10, y11 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    for (x, y, w, h) in faces:
        x0 = x
        y0 = y
        x1 = x+w
        y1 = y+h
        rect1 = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for (x, y, w, h) in hands:
        x2 = x
        y2 = y
        x3 = x+w
        y3 = y+h
        rect2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in fists:
        x4 = x
        y4 = y
        x5 = x+w
        y5 = y+h
        rect3 = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # for (x, y, w, h) in fists2:
    #     x6 = x
    #     y6 = y
    #     x7 = x+w
    #     y7 = y+h
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # for (x, y, w, h) in palm:
    #     x8 = x
    #     y8 = y
    #     x9 = x+w
    #     y9 = y+h
    #     rect4 = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # for (x, y, w, h) in closedPalm:
    #     x10 = x
    #     y10 = y
    #     x11 = x+w
    #     y11 = y+h
    #     rect5 = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    if (intersectCheck(x0, y0, x1, y1, x2, y2, x3, y3) or intersectCheck(x0, y0, x1, y1, x4, y4, x5, y5)):
        sign = cv2.putText(frame,'PLEASE DO NOT TOUCH YOUR FACE!',(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
    # Display the resulting frame
    # return cv2.imshow('frame',frame)
    return frame

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


    # When everything is done, release the capture
    # video.release()
    # cv2.destroyAllWindows()