from app.cam import PyCam

if __name__ == '__main__':
    PyCam(channel='http://192.168.0.101:4747/mjpegfeed').run()
