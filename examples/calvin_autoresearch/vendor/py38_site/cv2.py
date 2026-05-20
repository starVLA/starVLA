"""Tiny cv2 compatibility shim for headless CALVIN evaluation.

The shared CALVIN environment imports cv2 at module import time, but the formal
non-GUI rollout path only needs the import to succeed. A few common functions
are provided for debug paths without requiring libGL.
"""

import numpy as np

__version__ = "headless-shim"

INTER_LINEAR = 1
FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16
COLOR_GRAY2BGR = 8
COLOR_RGB2BGR = 4


def resize(image, dsize, interpolation=INTER_LINEAR):
    try:
        from PIL import Image

        pil_image = Image.fromarray(np.asarray(image))
        resized = pil_image.resize(tuple(dsize), Image.BILINEAR)
        return np.asarray(resized, dtype=np.asarray(image).dtype)
    except Exception:
        arr = np.asarray(image)
        width, height = dsize
        y_idx = np.linspace(0, arr.shape[0] - 1, height).astype(np.int64)
        x_idx = np.linspace(0, arr.shape[1] - 1, width).astype(np.int64)
        return arr[y_idx][:, x_idx]


def imshow(*_args, **_kwargs):
    return None


def waitKey(_delay=0):
    return -1


def destroyAllWindows():
    return None


def putText(image, *_args, **_kwargs):
    return image


def cvtColor(image, code):
    arr = np.asarray(image)
    if code == COLOR_GRAY2BGR and arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if code == COLOR_RGB2BGR and arr.ndim == 3:
        return arr[:, :, ::-1]
    return arr


def GaussianBlur(image, *_args, **_kwargs):
    return image


def imread(*_args, **_kwargs):
    return None


def imwrite(*_args, **_kwargs):
    return False


def VideoWriter_fourcc(*_args):
    return 0


class VideoWriter:
    def __init__(self, *_args, **_kwargs):
        pass

    def write(self, *_args, **_kwargs):
        return None

    def release(self):
        return None


class KeyPoint:
    def __init__(self, x=0, y=0, _size=1, size=None, **_kwargs):
        self.pt = (x, y)
        self.size = _size if size is None else size


def drawKeypoints(image, *_args, **_kwargs):
    return image


def circle(image, *_args, **_kwargs):
    return image
