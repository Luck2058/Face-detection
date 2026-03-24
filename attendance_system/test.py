import cv2
import time

cap = cv2.VideoCapture(0)
start = time.time()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frame_count % 30 == 0:
        fps = frame_count / (time.time() - start)
        print(f"FPS: {fps:.2f}")