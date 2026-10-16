import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

classifier = MobileNetV2(weights="imagenet")
detector = YOLO(r"C:\Users\Basavaraj\Downloads\students.csv\yolov8n.pt")
camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()
mode = "yolo"

while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        break

    if mode == "yolo":
        result = detector(frame, verbose=False)[0]
        frame = result.plot()

        count = len(result.boxes)
        cv2.putText(
          frame,
          f"Objects detected: {count}",
          (10, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.8,
          (0, 255, 0),
          2
       )
    else:
        img = preprocess_input(
            np.expand_dims(
                cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB).astype("float32"), 0
            )
        )
        _, name, score = decode_predictions(classifier.predict(img, verbose=0), top=1)[0][0]
        label = f"MobileNet: {name.replace('_', ' ')} ({score:.1%})"

        cv2.rectangle(frame, (5, 5), (460, 55), (0, 0, 0), -1)
        cv2.putText(
        frame,
        label,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
        )

    cv2.imshow("m=MobileNet, y=YOLO, q=quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    mode = {"m": "yolo", "y": "yolo"}.get(chr(key), mode)

camera.release()
cv2.destroyAllWindows()
