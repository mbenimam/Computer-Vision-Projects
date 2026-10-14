"""Padding, 2D convolution, classic kernels, gradients and a median filter."""
import numpy as np


def pad(img, k, mode="edge"):
    """Add k pixels on every side. mode: 'zero', 'edge' (replicate) or 'reflect'."""
    m = {"zero": "constant", "edge": "edge", "reflect": "symmetric"}[mode]
    return np.pad(img, k, mode=m)


def convolve(img, kernel, mode="edge"):
    """True convolution (kernel flipped) with an odd-sized kernel, same output size."""
    f = img.astype(np.float64)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(f, ((ph, ph), (pw, pw)), mode={"zero": "constant", "edge": "edge", "reflect": "symmetric"}[mode])
    kf = kernel[::-1, ::-1]
    out = np.zeros_like(f)
    H, W = f.shape
    for i in range(kh):
        for j in range(kw):
            out += kf[i, j] * p[i:i + H, j:j + W]
    return out


def mean_kernel(n):
    return np.ones((n, n)) / (n * n)


def gaussian_kernel(n, sigma):
    r = np.arange(n) - n // 2
    g = np.exp(-(r ** 2) / (2.0 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)


def sobel(img):
    gx = convolve(img, SOBEL_X)
    gy = convolve(img, SOBEL_Y)
    return gx, gy, np.hypot(gx, gy)


def median_filter(img, n=3):
    f = img.astype(np.float64)
    r = n // 2
    p = np.pad(f, r, mode="edge")
    H, W = f.shape
    stack = np.stack([p[i:i + H, j:j + W] for i in range(n) for j in range(n)])
    return np.median(stack, axis=0)


def salt_and_pepper(img, frac, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    m = rng.random(img.shape)
    out[m < frac / 2] = 0
    out[m > 1 - frac / 2] = 255
    return out


def gaussian_noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
