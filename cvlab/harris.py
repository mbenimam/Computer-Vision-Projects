"""Harris corner detector: gradients -> structure tensor -> response -> non-max suppression."""
import numpy as np
from .filters import convolve, gaussian_kernel, SOBEL_X, SOBEL_Y


def structure_tensor(img, sigma=1.5):
    ix = convolve(img, SOBEL_X)
    iy = convolve(img, SOBEL_Y)
    n = int(2 * round(3 * sigma) + 1)
    g = gaussian_kernel(n, sigma)
    return convolve(ix * ix, g), convolve(ix * iy, g), convolve(iy * iy, g), ix, iy


def response(img, k=0.05, sigma=1.5):
    ixx, ixy, iyy, _, _ = structure_tensor(img, sigma)
    det = ixx * iyy - ixy ** 2
    tr = ixx + iyy
    return det - k * tr ** 2


def local_maxima(r, size=5, rel=0.01):
    h = size // 2
    p = np.pad(r, h, mode="constant", constant_values=-np.inf)
    H, W = r.shape
    mx = np.max(np.stack([p[i:i + H, j:j + W] for i in range(size) for j in range(size)]), axis=0)
    return np.argwhere((r == mx) & (r > rel * r.max()))


def corners(img, k=0.05, sigma=1.5, rel=0.01, size=7):
    r = response(img, k, sigma)
    return local_maxima(r, size, rel), r
