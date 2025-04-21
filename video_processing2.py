# process 2 file (left & right, seperate files)
import cv2
import os

video_path_left = "/home/sysop/bag/mp4/Collection3/straight_data_closer_left.mp4"
video_path_right = "/home/sysop/bag/mp4/Collection3/straight_data_closer_right.mp4"
output_folder_left = "frames/left"
output_folder_right = "frames/right"

os.makedirs(output_folder_left, exist_ok=True)
os.makedirs(output_folder_right, exist_ok=True)

def extract(cap, path):

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_path = os.path.join(path, f"frame_{frame_count:04d}.jpg")

        cv2.imwrite(output_path, frame)

        frame_count += 1

    cap.release()

extract(cv2.VideoCapture(video_path_left), output_folder_left)
extract(cv2.VideoCapture(video_path_right), output_folder_right)
