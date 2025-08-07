import os
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
VIDEO_PATH = os.getenv("VIDEO_PATH")

if not MODEL_PATH or not VIDEO_PATH:
    raise ValueError("MODEL_PATH or VIDEO_PATH is missing in .env file")

# load YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
    print(f"Loaded YOLO model from: {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit()

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Failed to open video: {VIDEO_PATH}")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = frame_count / fps

print(f"Video Loaded: {VIDEO_PATH}")
print(f"Frame count: {frame_count}")
print(f"FPS: {fps}")
print(f"Duration: {duration:.2f} seconds")

cap.release()
