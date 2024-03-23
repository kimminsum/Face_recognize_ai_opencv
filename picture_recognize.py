import cv2

# load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')



# load the picture
img = cv2.imread('/home/kar7mp5/Projects/Face_recognize_ai_opencv/209389_58874_433.jpg')

# image preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find the face
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # find eyes
    roi_color = img[y:y + h, x:x + w]
    roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# print result
cv2.imshow('image', img)

key = cv2.waitKey(0)
cv2.destroyAllWindows()