# Face-Detection-model
This project is an AI-based exam monitoring system built using Python, Streamlit, Computer Vision, and Deep Learning. The system allows students to be registered using their facial images and then verifies their identity during an online exam using face recognition techniques.

The application uses YOLOv8 for person detection to ensure only one candidate is present in front of the camera and DeepFace with FaceNet embeddings to verify whether the detected face matches the registered student. If an unknown person or multiple people are detected, the system flags the activity and logs it as a suspicious event.

This project demonstrates the use of real-time face recognition, object detection, AI-based monitoring, and automated logging to help improve the integrity of online examinations. It is built with Streamlit for the user interface and uses OpenCV for real-time webcam processing.
