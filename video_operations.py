import cv2

# show mapClique
img = cv2.imread('src/mapClique.png')
print(img.shape)
w, h, c = img.shape
for i in range(w):
    for j in range(h):
        isZero = True
        for k in range(c):
            if img[i, j, k] != 0:
                isZero = False
                break
        if not isZero:
            print(w, h, img[w, h])


# # get frame numbers
# video_path = "testgood1.mp4"
#
# cap = cv2.VideoCapture(video_path)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frame_count)
# cap.release()
#
# # get frame shape
# img = cv2.imread('./frames1/frame_0000.jpg')
# print(img.shape)
# # cv2.imwrite('./out_test.jpg', img)
#
#
# # split to left and right
# height, width, _ = img.shape
# mid = width // 2  # 2880 / 2 = 1440
#
# left_img = img[:, :mid, :]
# right_img = img[:, mid:, :]
#
# cv2.imwrite("out_test_left.jpg", left_img)
# cv2.imwrite("out_test_right.jpg", right_img)