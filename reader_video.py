import cv2

#Установка размера кадра
width_video = 1280
height_video = 720

cap = cv2.VideoCapture("input_video/people_video.mp4")
while True:
    seccess, img = cap.read()
    video = cv2.resize(img, (width_video, height_video))
    cv2.imshow('India', video)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break