"""The 2D discrete Fourier transform from its definition, plus frequency-domain filters."""
import numpy as np


def dft_matrix(n):
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n)


def dft2(img):
    """F = W_h . f . W_w : the DFT as two matrix products (O(n^3)), no FFT."""
    f = img.astype(np.float64)
    return dft_matrix(f.shape[0]) @ f @ dft_matrix(f.shape[1])


def idft2(F):
    h, w = F.shape
    Wh = np.conj(dft_matrix(h)) / h
    Ww = np.conj(dft_matrix(w)) / w
    return (Wh @ F @ Ww).real


def shift(F):
    return np.fft.fftshift(F)


def log_magnitude(F):
    return np.log1p(np.abs(shift(F)))


def radial_mask(shape, radius, kind="low", soft=0.0):
    """Centred mask on a shifted spectrum. kind: low, high or band (radius=(r1,r2))."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    d = np.hypot(yy - h // 2, xx - w // 2)
    if kind == "band":
        r1, r2 = radius
        m = ((d >= r1) & (d <= r2)).astype(np.float64)
    else:
        m = (d <= radius).astype(np.float64)
        if kind == "high":
            m = 1 - m
    if soft > 0:
        k = np.exp(-(np.arange(-8, 9) ** 2) / (2 * (soft) ** 2))
        k /= k.sum()
        for ax in (0, 1):
            m = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, m)
    return m


def filter_frequency(img, mask):
    F = shift(dft2(img))
    return idft2(np.fft.ifftshift(F * mask)), F * mask
