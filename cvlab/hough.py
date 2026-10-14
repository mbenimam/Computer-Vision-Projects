"""Hough transform for straight lines, with the (rho, theta) accumulator exposed."""
import numpy as np
from .filters import sobel


def edges(img, frac=0.25):
    _, _, m = sobel(img)
    return m > frac * m.max()


def accumulator(edge_mask, n_theta=180, n_rho=None):
    """Every edge pixel votes for all lines through it: rho = x cos(t) + y sin(t)."""
    h, w = edge_mask.shape
    diag = int(np.ceil(np.hypot(h, w)))
    n_rho = n_rho or 2 * diag + 1
    thetas = np.deg2rad(np.arange(n_theta) * 180.0 / n_theta)
    rhos = np.linspace(-diag, diag, n_rho)
    acc = np.zeros((n_rho, n_theta), dtype=np.int64)
    ys, xs = np.nonzero(edge_mask)
    cos, sin = np.cos(thetas), np.sin(thetas)
    for x, y in zip(xs, ys):
        r = x * cos + y * sin
        idx = np.round((r + diag) * (n_rho - 1) / (2 * diag)).astype(int)
        acc[idx, np.arange(n_theta)] += 1
    return acc, rhos, thetas


def peaks(acc, n=5, nms=9, min_votes=1):
    a = acc.copy()
    out = []
    for _ in range(n):
        i = np.unravel_index(np.argmax(a), a.shape)
        if a[i] < min_votes:
            break
        out.append((i[0], i[1], int(a[i])))
        r0, r1 = max(0, i[0] - nms), i[0] + nms + 1
        t0, t1 = max(0, i[1] - nms), i[1] + nms + 1
        a[r0:r1, t0:t1] = 0
    return out


def line_points(rho, theta, shape):
    """Two end points of the line inside the image (x, y order) or None."""
    h, w = shape
    c, s = np.cos(theta), np.sin(theta)
    pts = []
    if abs(s) > 1e-9:
        for x in (0, w - 1):
            y = (rho - x * c) / s
            if 0 <= y <= h - 1:
                pts.append((x, y))
    if abs(c) > 1e-9:
        for y in (0, h - 1):
            x = (rho - y * s) / c
            if 0 <= x <= w - 1:
                pts.append((x, y))
    return (pts[0], pts[-1]) if len(pts) >= 2 else None
