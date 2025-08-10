import cv2
import numpy as np
import app as st
from keras.models import model_from_json
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import streamlit as st


# Load the model (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    with open("model.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("emotiondetector.h5")
    return model

model = load_model()

# Emotion labels
labels = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Feature extraction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Streamlit UI
st.title("🎭 Real-Time Emotion Detector")
st.write("This app detects emotions from your webcam in real time.")

# Video processing class
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            try:
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_features = extract_features(roi_gray)
                prediction = model.predict(img_features)
                label = labels[prediction.argmax()]
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except:
                continue

        return img

# Stream webcam in browser
webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)
