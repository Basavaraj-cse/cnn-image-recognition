CNN Image Recognition and YOLO Object Detection

About

This project combines MobileNetV2 image classification and YOLO object detection using Python.

Features

- Classifies images using MobileNetV2.
- Detects objects using YOLOv8.
- Displays prediction confidence.
- Supports webcam-based object detection in the live detection script.

Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- Ultralytics YOLOv8

Installation

Install the required libraries:

pip install opencv-python tensorflow ultralytics

Run the Project

Run the image recognition script:

python image_test.py

Run the webcam detection script:

python cnn_imagerec.py

Note

The YOLO model file "yolov8n.pt" must be available locally to run detection. Predictions may not always be accurate.
