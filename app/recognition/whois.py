from app.recognition import haarcascade
import cv2
import os.path


class PyWhoIs(object):
    def __init__(self):
        self._classifier_face = haarcascade.frontal_face
        self._classifier_eye = haarcascade.eye_cascade
        self._model = cv2.face.LBPHFaceRecognizer_create()
        self._model.read('train/model_2018-07-02.xml')
        self._names = ['irene', 'mara', 'rafael']

    def detectFaces(self, frame, recognition=True):
        if not os.path.isfile(self._classifier_face):
            raise ValueError('Could not find the classifier file xml. Exiting...')

        face_cascade = cv2.CascadeClassifier(self._classifier_face)
        #eye_cascade = cv2.CascadeClassifier(self._classifier_eye)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if recognition:
                self.recognitionFaces(frame, x, y, w, h)
            # roi_gray = gray[y:y + h, x:x + w]
            # eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
            # for(ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        return frame

    def recognitionFaces(self, frame, x, y, w, h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[x:x+w, y:y+h]
        roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
        params = self._model.predict(roi)
        print("Label: {}, Confidence: {}".format(params[0], params[1]))
        cv2.putText(frame, self._names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)


