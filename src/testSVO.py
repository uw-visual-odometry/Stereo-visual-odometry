import cv2
import os

def video_to_png(video_path, output_folder):
    """
    Extracts frames from a video and saves them as PNG images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the output folder where PNG images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while frame_count < 12:
        success, frame = video_capture.read()
        if not success:
            break
            
        if success and (frame_count == 1 or frame_count == 6):     
           output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
           cv2.imwrite(output_path, frame)
        frame_count += 1
    
    video_capture.release()

# Example usage:
video_path = "/home/sysop/aquarium_jetson_nano/testgood1.mp4"
output_folder = "output_frames"
video_to_png(video_path, output_folder)

def splitLR(imagePath, idx):
    img = cv2.imread(imagePath)
    
    height, width, channels = img.shape
    
    midpoint = width // 2
    
    left_half = img[:, :midpoint]
    right_half = img[:, midpoint:]
    
    cv2.imwrite('left_half' + str(idx) + '.jpg', left_half)
    cv2.imwrite('right_half' + str(idx) + '.jpg', right_half)
# or
# cv2.imshow('Left Half', left_half)
# cv2.imshow('Right Half', right_half)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

idx = 1
splitLR("/home/sysop/Stereo-visual-odometry/src/output_frames/frame_0001.png", idx)
idx += 1
splitLR("/home/sysop/Stereo-visual-odometry/src/output_frames/frame_0006.png", idx)     

#imgPath = "/home/sysop/Stereo-visual-odometry/src/output_frames/frame_0001.png"
#image = cv2.imread(imgPath)
#cv2.imshow('Left Half', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

'''
# Load images
img1 = cv2.imread('right_half1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('right_half2.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocess images
img1 = cv2.resize(img1, (512, 512))
img2 = cv2.resize(img2, (512, 512))
img1 = cv2.GaussianBlur(img1, (5, 5), 0)
img2 = cv2.GaussianBlur(img2, (5, 5), 0)

# Detect features and compute descriptors using ORB
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match features using brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw top matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display matches
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''    
