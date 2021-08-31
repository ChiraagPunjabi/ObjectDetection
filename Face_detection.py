import cv2

image = cv2.imread("C:/Users/Singh/Downloads/fd.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

faces = face_cascade.detectMultiScale(gray)

for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

    cv2.imwrite("face_detected.jpg", image)
