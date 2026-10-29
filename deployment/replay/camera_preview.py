"""Show live camera frames without blocking acquisition or robot control."""
import multiprocessing as mp
import os
import queue
import sys
from pathlib import Path


def _configure_qt_fonts():
    """OpenCV wheels can point Qt at a font directory they do not ship."""
    configured = os.environ.get('QT_QPA_FONTDIR')
    if configured and Path(configured).is_dir():
        return
    for candidate in ('/usr/share/fonts/truetype/dejavu',
                      '/usr/share/fonts/truetype/liberation2',
                      '/usr/share/fonts/truetype/freefont'):
        if Path(candidate).is_dir():
            os.environ['QT_QPA_FONTDIR'] = candidate
            return


def _show_frames(frames, stop):
    import cv2
    # Set this after importing cv2, which may overwrite QT_QPA_FONTDIR.
    _configure_qt_fonts()
    window = 'Policy camera (Q closes image only)'
    opened = False
    try:
        while not stop.is_set():
            try:
                frame = frames.get(timeout=0.03)
            except queue.Empty:
                frame = None
            if frame is not None:
                cv2.imshow(window, frame)
                opened = True
            if opened:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except Exception as exc:
        print(f'Camera preview unavailable: {exc}', file=sys.stderr, flush=True)
    finally:
        if opened:
            cv2.destroyAllWindows()


class CameraPreview:
    def __init__(self):
        context = mp.get_context('spawn')
        self.frames = context.Queue(maxsize=1)
        self.stop = context.Event()
        self.process = context.Process(target=_show_frames, args=(self.frames, self.stop),
                                       name='policy-camera-preview', daemon=True)
        self.started = False

    def start(self):
        if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            print('No graphical display; camera preview disabled.', file=sys.stderr)
            return
        self.process.start()
        self.started = True

    def publish(self, bgr):
        if not self.started or not self.process.is_alive():
            return
        try:
            self.frames.put_nowait(bgr.copy())
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(bgr.copy())
            except queue.Full:
                pass

    def close(self):
        self.stop.set()
        if self.started:
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        self.frames.cancel_join_thread()
        self.frames.close()
