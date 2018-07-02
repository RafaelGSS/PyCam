import cv2
from app.managers import WindowManager, CaptureManager
from app import filters
from app.recognition.whois import PyWhoIs


class PyCam(object):

    def __init__(self, channel=0, name='PyCam', recognizerFace=PyWhoIs()):
        self._windowManager = WindowManager(name, self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(channel), self._windowManager, True)

        self._curveFilter = None
        self._enabledFilter = False

        self._recognizer = recognizerFace
        self._enabledDetectFaces = False

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if self._enabledFilter:
                self.applyFilter(frame)

            if self._enabledDetectFaces:
                self._recognizer.detectFaces(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    # Recognition functions
    def detectFaces(self):
        self._enabledDetectFaces = not self._enabledDetectFaces

    # Filter functions
    def applyFilter(self, frame):
        if self._curveFilter is not None:
            self._curveFilter.apply(frame, frame)

    def enabledFilter(self):
        return self._enabledFilter

    def setFilter(self, curveFilter=filters.BGRPortraCurveFilter()):
        self._curveFilter = curveFilter
        self._enabledFilter = True

    def removeFilter(self):
        self._enabledFilter = False
        self._curveFilter = None

    # Callback to window manager
    def onKeypress(self, keycode):
        if keycode == 32:  # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # escape
            self._windowManager.destroyWindow()
        elif keycode == 102:  # f
            self.detectFaces()
        # 1 ~ 7
        elif keycode == 48:
            self.setFilter(curveFilter=filters.BGRPortraCurveFilter())
        elif keycode == 49:
            self.setFilter(curveFilter=filters.BGRCrossProcessCurveFilter())
        elif keycode == 50:
            self.setFilter(curveFilter=filters.BGRVelviaCurveFilter())
        elif keycode == 51:
            self.setFilter(curveFilter=filters.EmbossFilter())
        elif keycode == 52:
            self.setFilter(curveFilter=filters.FindEdgesFilter())
        elif keycode == 53:
            self.setFilter(curveFilter=filters.SharpenFilter())
        elif keycode == 54:
            self.setFilter(curveFilter=filters.VStrokeEdges())
        elif keycode == 55:
            self.setFilter(curveFilter=filters.BlurFilter())
        elif keycode == 103:  # g
            self.removeFilter()

