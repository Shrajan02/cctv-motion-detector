import cv2
from ultralytics import YOLO

VIDEO_PATH = "your_video.mp4"  
model = YOLO("yolov8m.pt")  

# load video
cap = cv2.VideoCapture(VIDEO_PATH)
detected_car = None

# read first few frames to find the car
for i in range(30):
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    results = model(frame)
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if model.names[int(cls)] == "car":
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                detected_car = (x1, y1, x2, y2)
                print(f"Car detected at: {detected_car}")
                break
        if detected_car:
            break

if not detected_car:
    print("No car detected in the first 30 frames!")
else:
    # draw the ROI around the car
    x1, y1, x2, y2 = detected_car
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("car_detected.jpg", frame)
    print("Saved detection as car_detected.jpg")

cap.release()
