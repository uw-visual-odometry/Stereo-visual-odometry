import cv2
import os

video_path = "testgood1.mp4"
output_folder_left = "frames/left"
output_folder_right = "frames/right"

os.makedirs(output_folder_left, exist_ok=True)
os.makedirs(output_folder_right, exist_ok=True)

mid = 1440

cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    left_img = frame[:, :mid, :]
    right_img = frame[:, mid:, :]

    left_frame = os.path.join(output_folder_left, f"frame_{frame_count:04d}.jpg")
    right_frame = os.path.join(output_folder_right, f"frame_{frame_count:04d}.jpg")

    cv2.imwrite(left_frame, left_img)
    cv2.imwrite(right_frame, right_img)

    frame_count += 1

cap.release()
