from Cam.recognition import haarcascade
import cv2
import os.path


class PyWhoIs(object):
    def __init__(self, classifier=haarcascade.frontal_face):
        self._classifier = classifier

    def detectFaces(self, frame):
        if not os.path.isfile(self._classifier):
            raise ValueError('Could not find the classifier file xml. Exiting...')

        face_cascade = cv2.CascadeClassifier(self._classifier)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame
