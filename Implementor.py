import pickle

import cv2.data
import numpy as np
from skimage.transform import resize

cap = cv2.VideoCapture(0)
MODEL = pickle.load(open("model.p", "rb"))
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    flat_data = []
    ret, frame = cap.read()
    img_resized = resize(frame, (15, 15, 3))

    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        print('Male')
        cv2.putText(frame,
                    'Male',
                    (50, 50),
                    font, 1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_4)
    else:
        print('Female')
        cv2.putText(frame,
                    'Female',
                    (50, 50),
                    font, 1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray_image = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
