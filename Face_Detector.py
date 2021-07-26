import cv2
from random import randrange


trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# img = cv2.imread('IU.jpg')
webcam = cv2.VideoCapture(0)
while True:
    succesful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),
                                                  randrange(256), randrange(256)), 3)

    cv2.imshow('Face Detector', frame)
    cv2.waitKey(1)


# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# # print(face_coordinates)
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
#                                             randrange(256), randrange(256)), 3)

# cv2.imshow('Face Detector', img)
# cv2.waitKey()


# print("Code Completed")
