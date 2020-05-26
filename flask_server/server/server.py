from flask import Flask, render_template, Response
from camera import Camera
import cv2
import sys
# replace with path to folder
sys.path.append('/Users/Rohit/Desktop/Corona')
from faceDetection import detector

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)


    def __del__(self):
        self.video.release()

    def get_frame(self):
        frame = detector(self.video)
        success, frame = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', touches = num_times())

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/num_times')
def num_times():
    return 0

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
