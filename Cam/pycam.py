import cv2
from Cam.managers import WindowManager, CaptureManager
from Cam import filters


class PyCam(object):

    def __init__(self, channel=0, name='PyCam'):
        self._windowManager = WindowManager(name, self.onKeypress)
        self._captureManager = CaptureManager(
        cv2.VideoCapture(channel), self._windowManager, True)
        self._curveFilter = None
        self._functionFilter = None
        self._enabledFilter = False

    def enabledFilter(self):
        return self._enabledFilter

    def setFilter(self, curveFilter=filters.BGRPortraCurveFilter(), functionFilter=filters.strokeEdges):
        self._curveFilter = curveFilter
        self._functionFilter = functionFilter
        self._enabledFilter = True

    def removeFilter(self):
        self._enabledFilter = False
        self._curveFilter = None
        self._functionFilter = None

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if self._enabledFilter:
                self.applyFilter(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def applyFilter(self, frame):
        if self._functionFilter is not None:
            self._functionFilter(frame, frame)
        if self._curveFilter is not None:
            self._curveFilter.apply(frame, frame)

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
        # 1 ~ 7
        elif keycode == 48:
            self.setFilter(curveFilter=filters.BGRPortraCurveFilter(), functionFilter=None)
        elif keycode == 49:
            self.setFilter(curveFilter=filters.BGRCrossProcessCurveFilter(), functionFilter=None)
        elif keycode == 50:
            self.setFilter(curveFilter=filters.BGRVelviaCurveFilter(), functionFilter=None)
        elif keycode == 51:
            self.setFilter(curveFilter=filters.EmbossFilter(), functionFilter=None)
        elif keycode == 52:
            self.setFilter(curveFilter=filters.FindEdgesFilter(), functionFilter=None)
        elif keycode == 53:
            self.setFilter(curveFilter=filters.SharpenFilter(), functionFilter=None)
        elif keycode == 54:
            self.setFilter(curveFilter=None, functionFilter=filters.strokeEdges)
        elif keycode == 55:
            self.setFilter(curveFilter=filters.BlurFilter(), functionFilter=None)
        elif keycode == 103:  # g
            self.removeFilter()

