# Face & Emotion Detection System

This project is an AI-powered application that detects faces and classifies emotions from images, videos, and live webcam feeds. Built with a combination of deep learning models and a user-friendly interface, it offers a flexible and intuitive tool for emotion recognition.

## Key Features:
- **Face Detection**: Powered by a YOLOv8n model to accurately detect faces in various media (images, videos, live feeds).
- **Emotion Classification**: A custom-trained CNN model classifies emotions such as happiness, sadness, anger, and more from detected faces.
- **Tkinter Interface**: A simple and easy-to-use GUI built with Tkinter, offering three detection modes:
  - Image-based emotion detection
  - Video-based emotion detection
  - Live webcam-based emotion detection

## Tech Stack:
- **YOLOv8n**: For real-time face detection
- **CNN**: For emotion classification
- **Tkinter**: For the graphical user interface (GUI)
- **Python**: Core programming language

## How It Works:
1. **Select Mode**: Choose between image, video, or live webcam feed.
2. **Face Detection**: The system identifies the face using YOLOv8n.
3. **Emotion Detection**: A CNN classifies the emotion from the detected face.
4. **Results**: Displayed in the interface for easy visualization.

This project demonstrates the integration of advanced machine learning models with a simple interface, making emotion detection accessible and easy to use across different media types.

