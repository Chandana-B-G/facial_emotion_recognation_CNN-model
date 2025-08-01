 import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

 # Load pre-trained emotion model
model = load_model("./facial_emotion_recognation_model.h5")
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Contempt"]
[21:55, 10/04/2025] Kalyan: # Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (48, 48))
        roi_color = img_to_array(roi_color) / 255.0
        roi_color = np.expand_dims(roi_color, axis=0)

        # Predict emotion
        preds = model.predict(roi_color)
        emotion = class_labels[np.argmax(preds)]

        # Draw green box around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
