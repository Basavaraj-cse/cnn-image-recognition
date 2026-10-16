import cv2
from tkinter import Tk, filedialog
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
import numpy as np

# Load pretrained models
classifier = MobileNetV2(weights="imagenet")
detector = YOLO(r"C:\Users\Basavaraj\Downloads\students.csv\yolov8n.pt")

# Select an image
root = Tk()
root.withdraw()
path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
)
root.destroy()

if not path:
    print("No image selected.")
    raise SystemExit

image = cv2.imread(path)

if image is None:
    print("Could not open the selected image.")
    raise SystemExit

# Classify the image
resized = cv2.resize(image, (224, 224))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
img = preprocess_input(np.expand_dims(rgb.astype("float32"), axis=0))

predictions = classifier.predict(img, verbose=0)
_, name, score = decode_predictions(predictions, top=1)[0][0]
print(f"MobileNet prediction: {name.replace('_', ' ')} ({score:.1%})")

# Detect objects
result = detector(image, conf=0.10, verbose=False)[0]
detected_image = result.plot()
cv2.rectangle(detected_image, (5, 5), (460, 45), (0, 0, 0), -1)
cv2.putText(
    detected_image,
    f"MobileNet: {name.replace('_', ' ')} ({score:.1%})",
    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
print(f"YOLO objects detected: {len(result.boxes)}")

cv2.imshow("Image Recognition - YOLO", detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()