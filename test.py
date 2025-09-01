import cv2
import time
import urllib.request
import os
from ultralytics import YOLO

# Download sample 720p video if not already present
url = "https://download.samplelib.com/mp4/sample-5s.mp4"
video_path = "sample_720p.mp4"
if not os.path.exists(video_path):
    print("Downloading sample video...")
    urllib.request.urlretrieve(url, video_path)

# Load YOLO model
model = YOLO("yolo11m.pt")

# Open video
cap = cv2.VideoCapture(video_path)
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run inference
    _ = model(frame, verbose=False)
    frame_count += 1

end_time = time.time()
cap.release()

# FPS calculation
fps = frame_count / (end_time - start_time)
print(f"Processed {frame_count} frames")
print(f"FPS: {fps:.2f}")
