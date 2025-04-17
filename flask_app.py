import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, render_template, Response
import threading
import os

app = Flask(__name__)

# Emotion labels (DeepFace handles this internally, but we use this if needed)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables for video capture and threading
cap = None
frame = None
lock = threading.Lock()
stop_camera = False

def generate_frames():
    global cap, frame, stop_camera
    cap = cv2.VideoCapture(0)
    stop_camera = False

    while not stop_camera:
        success, current_frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop and predict emotion using DeepFace
            face_img = current_frame[y:y + h, x:x + w]

            try:
                analysis = DeepFace.analyze(
                    img_path=face_img,
                    actions=['emotion'],
                    enforce_detection=False
                )
                emotion = analysis[0]['dominant_emotion']
                confidence = analysis[0]['emotion'][emotion]
                cv2.putText(current_frame, f"{emotion} ({confidence:.1f}%)",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except Exception as e:
                print("Emotion analysis failed:", str(e))

        with lock:
            frame = current_frame.copy()

        ret, buffer = cv2.imencode('.jpg', current_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    global cap, stop_camera
    stop_camera = True
    if cap is not None:
        cap.release()
    return "Camera stopped"

if __name__ == '__main__':
    # Create templates and static folders if they don't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)