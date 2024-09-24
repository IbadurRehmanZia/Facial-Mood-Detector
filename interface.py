import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import model_from_json
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model for face detection
face_model = YOLO('yolov8n-face.pt')

# Load Emotion Detection model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(model_json)
emotion_model.load_weights("emotiondetector.h5")

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def process_image(file_path):
    img = cv2.imread(file_path)
    if img is not None:
        image = cv2.resize(img, (900, 600))
        face_results = face_model.predict(image, conf=0.30)
        for info in face_results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1

                face_region = image[y1:y2, x1:x2]
                face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                face_region_resized = cv2.resize(face_region_gray, (48, 48))
                img = extract_features(face_region_resized)
                pred = emotion_model.predict(img)
                prediction_label = labels[pred.argmax()]

                cv2.putText(image, f'{prediction_label}', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        return img

def process_video(file_path):
    cap_video = cv2.VideoCapture(file_path)
    while cap_video.isOpened():
        ret, frame = cap_video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (700, 500))

        face_results = face_model.predict(frame, conf=0.30)
        for info in face_results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1

                face_region = frame[y1:y2, x1:x2]
                face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                face_region_resized = cv2.resize(face_region_gray, (48, 48))
                img = extract_features(face_region_resized)
                pred = emotion_model.predict(img)
                prediction_label = labels[pred.argmax()]

                cv2.putText(frame, f'{prediction_label}', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Face Detection with Emotion', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def open_webcam():
    webcam = cv2.VideoCapture(0)  # Change '0' to the appropriate webcam index if multiple cameras are connected
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        frame = cv2.resize(frame, (700, 500))

        face_results = face_model.predict(frame, conf=0.30)
        for info in face_results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1

                face_region = frame[y1:y2, x1:x2]
                face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                face_region_resized = cv2.resize(face_region_gray, (48, 48))
                img = extract_features(face_region_resized)
                pred = emotion_model.predict(img)
                prediction_label = labels[pred.argmax()]

                cv2.putText(frame, f'{prediction_label}', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Face Detection with Emotion', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def upload_image():
    file_path = filedialog.askopenfilename()
    img = process_image(file_path)
    if img is not None:
        panel.img = img
        panel.config(image=img)

def upload_video():
    file_path = filedialog.askopenfilename()
    process_video(file_path)

def switch_to_webcam():
    open_webcam()

root = tk.Tk()
root.title("Face Detection with Emotion Recognition")

panel = tk.Label(root)
panel.pack(padx=10, pady=10)

button_image = tk.Button(root, text="Upload Image", command=upload_image)
button_image.pack(pady=5)

button_video = tk.Button(root, text="Upload Video", command=upload_video)
button_video.pack(pady=5)

button_webcam = tk.Button(root, text="Open Webcam", command=switch_to_webcam)
button_webcam.pack(pady=5)

root.mainloop()
