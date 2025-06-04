import cv2

classifier = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(grey_frame, scaleFactor=1.5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("Face", frame)
    cv2.waitKey(1)
    
    if cv2.getWindowProperty("Face", cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()
