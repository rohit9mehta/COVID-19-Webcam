# import cv2
# import sys
# import logging as log
# import datetime as dt

# #Getting inputs from the system
# # imagePath = sys.argv[0]
# # cascadePath = sys.argv[1]
# cascadePath = "haarcascade_frontalface_default.xml"
# videoInput = cv2.VideoCapture(0) #Setting video source to webcam

# #Creating the haar feature charcteristic (under Viola-Jones Object Detection Framework)
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
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()