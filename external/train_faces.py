import cv2
from app.recognition.haarcascade import frontal_face
import os
from datetime import datetime
import numpy as np


def store_face():
    face_cascade = cv2.CascadeClassifier('../' + frontal_face)
    name = input('Enter the unique name of person: ')
    os.mkdir('train/{}'.format(name))

    camera = cv2.VideoCapture('http://192.168.0.101:4747/mjpegfeed')
    count = 0
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

            f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

            cv2.imwrite('../train/{}/{}_{}.pgm'.format(name, name, count), f)
            count += 1

        cv2.imshow("camera", frame)
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def read_faces(path=os.path.normpath(os.getcwd() + os.sep + os.pardir + "/train")):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print("I/O error({0}): {1}")
            c += 1
    return [X, y]


def train_face():
    names = ['irene', 'mara', 'rafael']
    name_file = str(datetime.now().date())
    [X,y] = read_faces()
    Y = np.asanyarray(y, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))
    model.save('../train/model_{}.xml'.format(name_file))
    print('Saved with name: {}'.format(name_file))


def check_option():
    op = int(input('Select option...\n1 ) Store new face to database\n2 ) Train model with faces stored\n>>> '))
    if op != 1 and op != 2:
        raise RuntimeError('Invalid option: {}\nQuiting...'.format(op))

    if op == 1:
        store_face()
    else:
        train_face()


if __name__ == "__main__":
    check_option()
