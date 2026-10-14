"""Histograms, contrast stretching, equalisation and thresholding on uint8 images."""
import numpy as np


def histogram(img, bins=256):
    """Count how many pixels hold each grey level."""
    h = np.zeros(bins, dtype=np.int64)
    np.add.at(h, img.ravel().astype(np.int64), 1)
    return h


def cumulative(h):
    return np.cumsum(h)


def stretch(img, lo=None, hi=None):
    """Linearly map [lo, hi] to [0, 255]. Defaults to the image min and max."""
    f = img.astype(np.float64)
    lo = f.min() if lo is None else lo
    hi = f.max() if hi is None else hi
    if hi <= lo:
        return img.copy()
    return np.clip((f - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def equalisation_lut(img):
    """Look-up table that makes the cumulative histogram as close to a straight line as possible."""
    h = histogram(img)
    c = cumulative(h)
    cmin = c[c > 0][0]
    n = img.size
    if n == cmin:
        return np.arange(256, dtype=np.uint8)
    return np.clip(np.round((c - cmin) * 255.0 / (n - cmin)), 0, 255).astype(np.uint8)


def equalise(img):
    return equalisation_lut(img)[img]


def threshold(img, t):
    return (img > t).astype(np.uint8) * 255


def otsu(img):
    """Threshold that maximises the between-class variance."""
    h = histogram(img).astype(np.float64)
    p = h / h.sum()
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mt * w - m) ** 2 / (w * (1 - w))
    s = np.nan_to_num(s)
    return int(np.argmax(s))
