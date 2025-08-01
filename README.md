# facial_emotion_recognation_CNN-model
A deep learning-based facial emotion recognition system using CNNs and OpenCV. Trained on the AffectNet dataset to detect 8 emotions in real-time via webcam. Includes data preprocessing, model training, evaluation (confusion matrix, ROC), and live emotion prediction using a saved model.


🧠 Facial Emotion Recognition using CNN and OpenCV
This project implements a deep learning-based Facial Emotion Recognition (FER) system using Convolutional Neural Networks (CNN) trained on the AffectNet dataset. The model is capable of detecting and classifying facial expressions in real-time video streams into one of eight emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise, and Contempt.

🚀 Features
📦 Preprocessing using Keras' ImageDataGenerator with real-time data augmentation

🧠 CNN architecture with three convolutional blocks and dropout for regularization

📊 Model evaluation using learning curves, confusion matrix, classification report, and ROC curves

🎥 Real-time face and emotion detection using OpenCV

💾 Trained model saved and re-used for deployment

🛠️ Tech Stack
Python, TensorFlow, Keras, NumPy, Matplotlib, Seaborn, OpenCV

Dataset: AffectNet

📷 Real-Time Demo
After training, the model is integrated with OpenCV to detect faces and classify their emotional state through the webcam. Emotions are overlaid on live video feed in real-time.

