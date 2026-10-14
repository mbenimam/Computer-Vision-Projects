import numpy as np
import pytest
from scipy.signal import convolve2d
from scipy.ndimage import median_filter as sp_median
from cvlab import histogram as H, filters as F, fourier as FT, harris as HR, hough as HG, quadtree as Q, huffman as HF

rng = np.random.default_rng(1)
IMG = rng.integers(0, 256, (32, 40)).astype(np.uint8)


def test_histogram_counts_every_pixel():
    h = H.histogram(IMG)
    assert h.sum() == IMG.size and np.array_equal(h, np.bincount(IMG.ravel(), minlength=256))


def test_stretch_hits_full_range():
    low = (IMG // 4 + 50).astype(np.uint8)
    s = H.stretch(low)
    assert s.min() == 0 and s.max() == 255


def test_equalise_flattens_cumulative_histogram():
    dark = (IMG // 8).astype(np.uint8)
    e = H.equalise(dark)
    c = np.cumsum(H.histogram(e)) / e.size
    used = np.nonzero(H.histogram(e))[0]
    ideal = (used + 1) / 256
    assert np.abs(c[used] - ideal).max() < 0.2 and e.max() == 255


def test_otsu_separates_two_modes():
    a = np.concatenate([np.full(500, 40), np.full(500, 200)]).reshape(10, 100).astype(np.uint8)
    assert 40 <= H.otsu(a) < 200


@pytest.mark.parametrize("n", [3, 5])
def test_convolve_matches_scipy(n):
    k = rng.random((n, n))
    ref = convolve2d(IMG.astype(float), k, mode="same", boundary="symm")
    assert np.allclose(F.convolve(IMG, k, "reflect"), ref)


def test_gaussian_kernel_sums_to_one_and_is_symmetric():
    k = F.gaussian_kernel(7, 1.4)
    assert abs(k.sum() - 1) < 1e-12 and np.allclose(k, k.T)


def test_median_matches_scipy_and_kills_salt_pepper():
    assert np.allclose(F.median_filter(IMG, 3), sp_median(IMG.astype(float), size=3, mode="nearest"))
    flat = np.full((40, 40), 100, np.uint8)
    noisy = F.salt_and_pepper(flat, 0.05, seed=2)
    assert F.psnr(flat, F.median_filter(noisy, 3).astype(np.uint8)) > F.psnr(flat, noisy)


def test_dft_matches_fft_and_roundtrips():
    X = IMG[:16, :20].astype(float)
    assert np.allclose(FT.dft2(X), np.fft.fft2(X))
    assert np.allclose(FT.idft2(FT.dft2(X)), X)


def test_sinusoid_has_two_spikes():
    n = 32
    x = np.cos(2 * np.pi * 4 * np.arange(n) / n)[None, :].repeat(n, 0)
    mag = np.abs(FT.shift(FT.dft2(x)))
    top = np.argsort(mag.ravel())[-2:]
    assert {int(t % n) for t in top} == {n // 2 - 4, n // 2 + 4}


def test_lowpass_removes_high_frequencies():
    n = 64
    yy, xx = np.mgrid[:n, :n]
    img = 100 + 50 * np.sin(2 * np.pi * 2 * xx / n) + 30 * np.sin(2 * np.pi * 20 * yy / n)
    out, _ = FT.filter_frequency(img, FT.radial_mask(img.shape, 6, "low"))
    assert np.abs(out - (100 + 50 * np.sin(2 * np.pi * 2 * xx / n))).max() < 1e-6


def test_harris_finds_four_square_corners():
    img = np.zeros((40, 40)); img[10:30, 10:30] = 255
    pts, _ = HR.corners(img, rel=0.2)
    for c in [(10, 10), (10, 29), (29, 10), (29, 29)]:
        assert min(np.hypot(*(p - np.array(c))) for p in pts) <= 3
    assert len(pts) <= 8


def test_hough_recovers_a_line():
    m = np.zeros((60, 60), bool)
    for x in range(60):
        m[x // 2 + 10, x] = True
    acc, rhos, thetas = HG.accumulator(m)
    r, t, v = HG.peaks(acc, 1)[0]
    assert v >= 50
    slope = -np.cos(thetas[t]) / np.sin(thetas[t])
    assert abs(slope - 0.5) < 0.05


def test_quadtree_partitions_image_and_respects_tolerance():
    img = np.zeros((64, 64)); img[:, 32:] = 200; img[10:20, 10:20] = 90
    root = Q.split(img, tol=5, min_size=2)
    ls = Q.leaves(root)
    assert sum(l.h * l.w for l in ls) == img.size
    assert all(l.std <= 5 or l.h <= 2 for l in ls)
    assert np.abs(Q.render(ls, img.shape) - img).max() < 1e-9


def test_merge_joins_neighbours_with_similar_means():
    img = np.zeros((32, 32)); img[:, 16:] = 100; img[8:12, 8:12] = 5
    ls = Q.leaves(Q.split(img, tol=0.5, min_size=2))
    labels = Q.merge(ls, tol=10)
    assert len(set(labels)) == 2 and len(ls) > 2


def test_huffman_is_prefix_free_optimal_and_lossless():
    data = np.array([0] * 50 + [1] * 25 + [2] * 15 + [3] * 10)
    table, freqs = HF.encode(data)
    cs = list(table.values())
    assert not any(a != b and b.startswith(a) for a in cs for b in cs)
    assert HF.entropy(data) <= HF.avg_length(table, freqs) < HF.entropy(data) + 1
    bits = HF.encode_bits(data, table)
    assert HF.decode_bits(bits, table) == data.tolist()


def test_quadtree_depth_limit_gives_coarser_partition():
    img = rng.integers(0, 256, (64, 64)).astype(float)
    root = Q.split(img, tol=1, min_size=2)
    assert len(Q.leaves(root, 0)) == 1 and len(Q.leaves(root, 1)) == 4
    assert len(Q.leaves(root, 2)) == 16 and Q.depth(root) == 5
