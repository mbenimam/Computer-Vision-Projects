"""Figure builders for the demo. Pure numpy + matplotlib, so they run in the browser (Pyodide)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from matplotlib.colors import ListedColormap

from cvlab import histogram as H, filters as F, fourier as FT, harris as HR, hough as HG, quadtree as Q, huffman as HF

BG = "#0e1117"
FG = "#e6e6e6"
ACC = "#ffb000"
ACC2 = "#4fc3f7"
ACC3 = "#ff5c8a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": "#444", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": FG,
})

KERNELS = {
    "mean (box blur)": lambda n: F.mean_kernel(n),
    "gaussian blur": lambda n: F.gaussian_kernel(n, max(n / 4.0, 0.8)),
    "sobel x (vertical edges)": lambda n: F.SOBEL_X,
    "sobel y (horizontal edges)": lambda n: F.SOBEL_Y,
    "laplacian (edges, all directions)": lambda n: F.LAPLACIAN,
    "sharpen": lambda n: F.SHARPEN,
}


def to_gray(a, max_side=160):
    from PIL import Image
    if a is None:
        raise ValueError("no image")
    a = np.asarray(a)
    if a.ndim == 3:
        a = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    s = max(im.size) / float(max_side)
    if s > 1:
        im = im.resize((max(8, int(round(im.size[0] / s))), max(8, int(round(im.size[1] / s)))), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _img(ax, a, title="", cmap="gray", vmin=None, vmax=None):
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def _note(ax, text, **kw):
    ax.axis("off")
    ax.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=10, color=FG,
            transform=ax.transAxes, linespacing=1.55, **kw)


def _fig(rows, cols, w=12, h=None):
    h = h or 4.0 * rows
    fig, axs = plt.subplots(rows, cols, figsize=(w, h), constrained_layout=True)
    return fig, np.atleast_2d(axs)


# ------------------------------------------------------------------ histograms
def histogram_figure(img, lo_pct=1.0, hi_pct=99.0):
    g = to_gray(img, 256)
    lo, hi = np.percentile(g, [lo_pct, hi_pct])
    s = H.stretch(g, lo, hi)
    e = H.equalise(g)
    t = H.otsu(g)
    fig, axs = _fig(3, 3, 12, 11)
    for ax, a, name in zip(axs[0], (g, s, e), ("original", f"stretched [{lo:.0f}, {hi:.0f}] → [0, 255]", "equalised")):
        _img(ax, a, name, vmin=0, vmax=255)
    for ax, a, c in zip(axs[1], (g, s, e), (ACC2, ACC, ACC3)):
        h = H.histogram(a)
        ax.fill_between(np.arange(256), h, color=c, alpha=0.8, step="mid")
        ax.set_xlim(0, 255); ax.set_yticks([]); ax.set_xlabel("grey level")
        ax2 = ax.twinx()
        ax2.plot(np.cumsum(h) / h.sum(), color="white", lw=1.6)
        ax2.set_ylim(0, 1.02); ax2.set_ylabel("cumulative", color="white")
        ax2.spines[:].set_visible(False)
    ax = axs[2, 0]
    x = np.arange(256)
    ax.plot(x, x, "--", color="#777", label="identity")
    ax.plot(x, np.clip((x - lo) * 255 / max(hi - lo, 1), 0, 255), color=ACC, lw=2, label="stretch")
    ax.plot(x, H.equalisation_lut(g), color=ACC3, lw=2, label="equalise (= cumulative histogram)")
    ax.set_title("the transfer curve: input level → output level")
    ax.set_xlabel("input"); ax.set_ylabel("output"); ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 255); ax.set_ylim(0, 255)
    ax = axs[2, 1]
    h = H.histogram(g)
    ax.fill_between(x, h, color=ACC2, alpha=0.7, step="mid")
    ax.axvline(t, color="white", lw=2)
    ax.set_title(f"Otsu threshold = {t}"); ax.set_yticks([]); ax.set_xlabel("grey level")
    _img(axs[2, 2], H.threshold(g, t), f"pixels > {t}", vmin=0, vmax=255)
    fig.suptitle("Contrast: stretch pulls the used range to full scale, equalise spreads the mass evenly", color=ACC)
    return fig


# ------------------------------------------------------------------ convolution
def _annotate(ax, a, fmt="{:.0f}", color_by=True):
    n, m = a.shape
    vmax = np.abs(a).max() or 1
    for i in range(n):
        for j in range(m):
            v = a[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8 if n > 3 else 10,
                    color="black" if (color_by and abs(v) > 0.55 * vmax and v > 0) else "white")
    ax.set_xticks([]); ax.set_yticks([])


def convolution_figure(img, kernel_name, size, fx, fy):
    g = to_gray(img, 160)
    size = int(size) | 1
    k = KERNELS[kernel_name](size)
    kh = k.shape[0]
    out = F.convolve(g, k)
    H_, W_ = g.shape
    r = kh // 2
    px = int(np.clip(round(fx * (W_ - 1)), r, W_ - 1 - r))
    py = int(np.clip(round(fy * (H_ - 1)), r, H_ - 1 - r))
    patch = g[py - r:py + r + 1, px - r:px + r + 1].astype(float)
    kf = k[::-1, ::-1]
    prod = patch * kf
    fig, axs = _fig(2, 3, 12, 8.4)
    _img(axs[0, 0], g, f"input · pixel ({px},{py})", vmin=0, vmax=255)
    axs[0, 0].add_patch(Rectangle((px - r - .5, py - r - .5), kh, kh, fill=False, ec=ACC, lw=2))
    axs[0, 1].imshow(patch, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axs[0, 1].set_title("the window under the kernel")
    _annotate(axs[0, 1], patch)
    axs[0, 2].imshow(k, cmap="RdBu_r", vmin=-np.abs(k).max(), vmax=np.abs(k).max(), interpolation="nearest")
    axs[0, 2].set_title("the kernel (weights)")
    _annotate(axs[0, 2], k, "{:.2f}" if kh > 3 or k.max() < 1 else "{:.0f}", color_by=False)
    axs[1, 0].imshow(prod, cmap="RdBu_r", vmin=-np.abs(prod).max() or -1, vmax=np.abs(prod).max() or 1, interpolation="nearest")
    axs[1, 0].set_title(f"window × kernel → sum = {prod.sum():.1f}")
    _annotate(axs[1, 0], prod, "{:.0f}" if kh <= 3 else "{:.0f}", color_by=False)
    show = out if kernel_name.startswith(("mean", "gauss", "sharp")) else np.abs(out)
    vmax = 255 if show is out else max(np.percentile(show, 99.5), 1)
    _img(axs[1, 1], np.clip(show, 0, vmax), f"output · value at the marked pixel = {out[py, px]:.1f}", vmin=0, vmax=vmax)
    axs[1, 1].plot([px], [py], "o", mfc="none", mec=ACC, ms=11, mew=2)
    P = 64
    resp = np.abs(np.fft.fftshift(np.fft.fft2(k, (P, P))))
    axs[1, 2].imshow(resp, cmap="magma", extent=[-.5, .5, -.5, .5])
    axs[1, 2].set_title("what the kernel does to frequencies\n(centre = flat areas, edges = fine detail)")
    axs[1, 2].set_xlabel("horizontal frequency"); axs[1, 2].set_ylabel("vertical frequency")
    fig.suptitle("Convolution: slide the window, multiply by the weights, add up", color=ACC)
    return fig


def denoise_figure(img, noise, amount, method, size):
    g = to_gray(img, 160)
    size = int(size) | 1
    noisy = F.salt_and_pepper(g, amount / 100.0) if noise == "salt & pepper" else F.gaussian_noise(g, amount)
    if method == "mean":
        out = F.convolve(noisy, F.mean_kernel(size))
    elif method == "gaussian":
        out = F.convolve(noisy, F.gaussian_kernel(size, max(size / 4.0, 0.8)))
    else:
        out = F.median_filter(noisy, size)
    out = np.clip(out, 0, 255)
    fig, axs = _fig(2, 3, 12, 8.2)
    _img(axs[0, 0], g, "clean", vmin=0, vmax=255)
    _img(axs[0, 1], noisy, f"noisy · PSNR {F.psnr(g, noisy):.1f} dB", vmin=0, vmax=255)
    _img(axs[0, 2], out, f"{method} {size}×{size} · PSNR {F.psnr(g, out):.1f} dB", vmin=0, vmax=255)
    h, w = g.shape
    y0, x0 = h // 3, w // 3
    sl = (slice(y0, y0 + 32), slice(x0, x0 + 32))
    for ax, a, t in zip(axs[1], (g, noisy, out), ("clean (zoom)", "noisy (zoom)", "filtered (zoom)")):
        _img(ax, a[sl], t, vmin=0, vmax=255)
    fig.suptitle("Averaging blurs edges; the median removes outliers without smearing them", color=ACC)
    return fig


# ------------------------------------------------------------------ fourier
def fourier_figure(img, mode, radius, keep_pct):
    g = to_gray(img, 128).astype(np.float64)
    Fm = FT.dft2(g)
    S = FT.shift(Fm)
    spec = np.log1p(np.abs(S))
    if mode == "keep the strongest coefficients":
        mag = np.abs(S)
        thr = np.percentile(mag, 100 - keep_pct)
        mask = (mag >= thr).astype(float)
    else:
        kind = {"low-pass": "low", "high-pass": "high", "band-pass": "band"}[mode]
        rad = (radius * 0.5, radius) if kind == "band" else radius
        mask = FT.radial_mask(g.shape, rad, kind, soft=1.5)
    out = FT.idft2(np.fft.ifftshift(S * mask))
    kept = (np.abs(S * mask) ** 2).sum() / (np.abs(S) ** 2).sum() * 100
    fig, axs = _fig(2, 3, 12, 8.4)
    _img(axs[0, 0], g, "image (space)", vmin=0, vmax=255)
    _img(axs[0, 1], spec, "its spectrum, log magnitude\n(centre = 0 Hz, far = fine detail)", cmap="magma")
    ov = np.dstack([spec / spec.max()] * 3) * 0.45
    ov[..., 1] += mask * 0.55
    axs[0, 2].imshow(np.clip(ov, 0, 1), interpolation="nearest")
    axs[0, 2].set_title(f"mask: green = kept ({mask.mean() * 100:.1f}% of coefficients)")
    axs[0, 2].set_xticks([]); axs[0, 2].set_yticks([])
    _img(axs[1, 0], np.clip(out, 0, 255), f"result · {kept:.1f}% of the energy kept", vmin=0, vmax=255)
    _img(axs[1, 1], np.log1p(np.abs(S * mask)), "spectrum after masking", cmap="magma")
    d = g - out
    _img(axs[1, 2], d, "what was removed (error)", cmap="RdBu_r", vmin=-np.abs(d).max(), vmax=np.abs(d).max())
    fig.suptitle("Fourier: an image is a sum of waves. Keep some waves, drop others", color=ACC)
    return fig


def sinusoid_figure(u, v, phase):
    n = 64
    yy, xx = np.mgrid[:n, :n]
    a = np.cos(2 * np.pi * (u * xx + v * yy) / n + np.deg2rad(phase))
    S = np.fft.fftshift(np.fft.fft2(a))
    fig, axs = _fig(1, 3, 12, 4.4)
    _img(axs[0, 0], a, f"cos(2π(u·x + v·y)/N + φ), u={u}, v={v}", vmin=-1, vmax=1)
    mag = np.abs(S)
    axs[0, 1].imshow(mag, cmap="magma", interpolation="nearest")
    axs[0, 1].set_title("spectrum: two spikes, mirrored through the centre")
    axs[0, 1].set_xticks([]); axs[0, 1].set_yticks([])
    wl = n / np.hypot(u, v) if (u or v) else float("inf")
    _note(axs[0, 2], f"frequency      : ({u}, {v}) cycles per image\nwavelength     : {wl:.1f} px\n"
                  f"direction      : {np.degrees(np.arctan2(v, u)):.0f}° (wave travels along it)\n\n"
                  "The spike distance from the centre = how fast\nthe wave oscillates. Its angle = which way.\n"
                  "Phase only moves the bright bands; the spectrum\nmagnitude does not change.")
    return fig


# ------------------------------------------------------------------ harris
def harris_figure(img, k, sigma, thresh_pct):
    g = to_gray(img, 160).astype(np.float64)
    ixx, ixy, iyy, ix, iy = HR.structure_tensor(g, sigma)
    R = ixx * iyy - ixy ** 2 - k * (ixx + iyy) ** 2
    pts = HR.local_maxima(R, 7, thresh_pct / 100.0)
    h, w = g.shape
    r = 6
    inner = np.zeros_like(R, bool)
    inner[r:h - r, r:w - r] = True
    trace = ixx + iyy
    ci = np.unravel_index(np.argmax(np.where(inner, R, -np.inf)), R.shape)
    ei = np.unravel_index(np.argmin(np.where(inner, R, np.inf)), R.shape)
    fi = np.unravel_index(np.argmin(np.where(inner, trace, np.inf)), R.shape)
    fig = plt.figure(figsize=(12, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    _img(ax, g, f"{len(pts)} corners found", vmin=0, vmax=255)
    if len(pts):
        ax.plot(pts[:, 1], pts[:, 0], "o", mfc="none", mec=ACC3, ms=9, mew=1.6)
    for (yy, xx), c in zip((ci, ei, fi), (ACC3, ACC2, "#9e9")):
        ax.plot([xx], [yy], "s", mfc="none", mec=c, ms=10, mew=2)
    _img(fig.add_subplot(gs[0, 1]), np.hypot(ix, iy), "gradient magnitude\n(edges light up, corners sit on edges)", cmap="viridis")
    m = max(np.percentile(np.abs(R), 99.5), 1e-9)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("rbk", ["#3b7bff", "#05070c", "#ff3b5c"])
    _img(fig.add_subplot(gs[0, 2]), R, "Harris response R = det(M) − k·trace(M)²\nred = corner, blue = edge, black = flat", cmap=cm, vmin=-m, vmax=m)
    cap = 6.5
    c0 = 3.0 * np.sqrt(max(np.linalg.eigvalsh(np.array([[ixx[ci], ixy[ci]], [ixy[ci], iyy[ci]]]))[0], 1e-9))
    for col, (name, (yy, xx), c) in enumerate((("strongest corner", ci, ACC3), ("strongest edge", ei, ACC2), ("flattest area", fi, "#9e9"))):
        ax = fig.add_subplot(gs[1, col])
        M = np.array([[ixx[yy, xx], ixy[yy, xx]], [ixy[yy, xx], iyy[yy, xx]]])
        lam, vec = np.linalg.eigh(M)
        lam = np.maximum(lam, 1e-9)
        patch = g[yy - r:yy + r + 1, xx - r:xx + r + 1]
        ax.imshow(patch, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax1, ax2 = min(c0 / np.sqrt(lam[1]), cap), min(c0 / np.sqrt(lam[0]), cap)
        ang = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
        ax.add_patch(Ellipse((r, r), 2 * ax2, 2 * ax1, angle=ang - 90, fill=False, ec=c, lw=2.4))
        ax.plot([r], [r], "+", color=c, ms=10)
        kind = ("FLAT" if lam[1] < 0.01 * max(trace.max(), 1e-9) else "EDGE" if lam[0] < 0.05 * lam[1] else "CORNER")
        ax.set_title(f"{name}: pixel ({xx},{yy}) → {kind}\nλ1={lam[1]:,.0f}  λ2={lam[0]:,.0f}  R={R[yy, xx]:,.0f}")
    fig.suptitle("Harris: a corner is where the image changes in two directions at once · ellipse = how far you can slide before the patch changes", color=ACC, fontsize=10)
    return fig


# ------------------------------------------------------------------ hough
def hough_figure(img, edge_frac, n_lines, n_curves, seed=3):
    g = to_gray(img, 128).astype(np.float64)
    E = HG.edges(g, edge_frac)
    acc, rhos, thetas = HG.accumulator(E, 180)
    pk = HG.peaks(acc, int(n_lines), 14)
    h, w = g.shape
    fig, axs = _fig(2, 3, 12, 8.4)
    _img(axs[0, 0], g, "image", vmin=0, vmax=255)
    _img(axs[0, 1], E, f"edge pixels: {int(E.sum())}", cmap="gray")
    th_deg = np.degrees(thetas)
    axs[0, 2].imshow(acc, cmap="inferno", aspect="auto", extent=[0, 180, rhos[-1], rhos[0]])
    axs[0, 2].set_title("accumulator: every edge pixel votes for\nevery line through it; bright = many votes")
    axs[0, 2].set_xlabel("θ (degrees)"); axs[0, 2].set_ylabel("ρ (pixels)")
    cols = plt.cm.cool(np.linspace(0, 1, max(len(pk), 2)))
    diag = (len(rhos) - 1) // 2
    for (ri, ti, v), c in zip(pk, cols):
        axs[0, 2].plot(th_deg[ti], rhos[ri], "o", mfc="none", mec=c, ms=14, mew=2)
    _img(axs[1, 0], g, "detected lines", vmin=0, vmax=255)
    for (ri, ti, v), c in zip(pk, cols):
        seg = HG.line_points(rhos[ri], thetas[ti], g.shape)
        if seg:
            axs[1, 0].plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=c, lw=2.2)
    axs[1, 0].set_xlim(-.5, w - .5); axs[1, 0].set_ylim(h - .5, -.5)
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(E)
    ax = axs[1, 1]
    if len(xs):
        sel = rng.choice(len(xs), size=min(int(n_curves), len(xs)), replace=False)
        for i in sel:
            ax.plot(th_deg, xs[i] * np.cos(thetas) + ys[i] * np.sin(thetas), color="#9ad", alpha=0.5, lw=1)
    for (ri, ti, v), c in zip(pk, cols):
        ax.plot(th_deg[ti], rhos[ri], "o", mfc="none", mec=c, ms=14, mew=2)
    ax.set_xlim(0, 180); ax.set_title(f"{min(int(n_curves), len(xs))} edge pixels drawn as curves:\nlines are where curves cross")
    ax.set_xlabel("θ (degrees)"); ax.set_ylabel("ρ (pixels)")
    txt = "peak    ρ(px)   θ(°)  votes\n" + "\n".join(
        f"#{i + 1:<3d}  {rhos[r]:7.1f}  {th_deg[t]:5.0f}  {v:5d}" for i, (r, t, v) in enumerate(pk))
    _note(axs[1, 2], txt + "\n\nline: x·cosθ + y·sinθ = ρ\nvotes ≈ number of edge pixels\non that line.")
    fig.suptitle("Hough: a line becomes one point in (θ, ρ) space, so finding lines = finding bright spots", color=ACC)
    return fig


# ------------------------------------------------------------------ quadtree
def quadtree_figure(img, split_tol, max_depth, merge_tol):
    g = to_gray(img, 128).astype(np.float64)
    root = Q.split(g, tol=split_tol, min_size=2)
    full = Q.leaves(root)
    ls = Q.leaves(root, int(max_depth))
    rec = Q.render(ls, g.shape)
    labels = Q.merge(ls, merge_tol)
    lab, regions = Q.render_regions(ls, labels, g.shape)
    dmap = np.zeros(g.shape)
    for l in ls:
        dmap[l.y:l.y + l.h, l.x:l.x + l.w] = np.log2(g.shape[0] / max(l.h, 1))
    fig, axs = _fig(2, 3, 12, 8.4)
    _img(axs[0, 0], g, f"image {g.shape[1]}×{g.shape[0]}", vmin=0, vmax=255)
    _img(axs[0, 1], g, f"split to depth {int(max_depth)}: {len(ls)} blocks", vmin=0, vmax=255)
    for l in ls:
        axs[0, 1].add_patch(Rectangle((l.x - .5, l.y - .5), l.w, l.h, fill=False, ec=ACC, lw=0.7))
    _img(axs[0, 2], dmap, "block depth (small blocks where detail is)", cmap="plasma")
    _img(axs[1, 0], rec, f"each block → its mean\n{g.size / len(ls):.0f} pixels per block on average", vmin=0, vmax=255)
    rng = np.random.default_rng(5)
    pal = ListedColormap(rng.random((max(labels) + 1, 3)) * 0.8 + 0.2)
    _img(axs[1, 1], lab, f"merge: {len(set(labels))} regions from {len(ls)} blocks", cmap=pal, vmin=0, vmax=max(labels))
    _img(axs[1, 2], regions, "regions painted with their mean", vmin=0, vmax=255)
    d = Q.depth(root)
    fig.suptitle(f"Split & merge: cut where it's not uniform, then glue similar neighbours · full tree depth {d}, {len(full)} leaves", color=ACC)
    return fig


# ------------------------------------------------------------------ huffman
def _weights(t, freqs):
    if not isinstance(t, tuple):
        return freqs[t]
    return sum(_weights(c, freqs) for c in t if c is not None)


def _draw_tree(ax, tree, table, freqs, label):
    pos = {}
    counter = [0]

    def lay(t, d):
        if not isinstance(t, tuple):
            x = counter[0]; counter[0] += 1
            pos[id(t)] = (x, -d); return x
        xs = [lay(c, d + 1) for c in t if c is not None]
        x = sum(xs) / len(xs); pos[id(t)] = (x, -d); return x

    def draw(t, d):
        x, y = pos[id(t)]
        if not isinstance(t, tuple):
            ax.text(x, y, f"{label(t)}\n{freqs[t]}", ha="center", va="center", fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc=ACC, ec="none"))
            return
        wgt = _weights(t, freqs)
        ax.text(x, y, str(wgt), ha="center", va="center", fontsize=7.5, color=FG,
                bbox=dict(boxstyle="circle,pad=0.25", fc="#2a3140", ec="#667"))
        for bit, c in zip("01", t):
            if c is None:
                continue
            cx, cy = pos[id(c)]
            ax.plot([x, cx], [y, cy], color="#778", lw=1, zorder=0)
            ax.text((x + cx) / 2, (y + cy) / 2 + 0.08, bit, color=ACC2, fontsize=9, ha="center", fontweight="bold")
            draw(c, d + 1)

    lay(tree, 0); draw(tree, 0)
    ax.axis("off")


def huffman_figure(img, source, text, levels):
    if source == "text":
        data = [ord(c) for c in (text or "abracadabra")][:300]
        label = lambda s: repr(chr(s))[1:-1] if chr(s) != " " else "␣"
        picture = None
        title = "your text"
    else:
        g = to_gray(img, 128)
        q = int(round(np.log2(levels)))
        data = (g >> (8 - q)).ravel().tolist()
        label = lambda s: str(s)
        picture = (g >> (8 - q)) * (255 // (levels - 1))
        title = f"image quantised to {levels} grey levels"
    table, freqs = HF.encode(data)
    ent = HF.entropy(data)
    avg = HF.avg_length(table, freqs)
    tree = HF.build_tree(freqs)
    fixed = max(1, int(np.ceil(np.log2(max(len(freqs), 2)))))
    fig = plt.figure(figsize=(12, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axT = fig.add_subplot(gs[0, :])
    if len(freqs) > 1:
        _draw_tree(axT, tree, table, freqs, label)
    axT.set_title(f"Huffman tree for {title}: rare symbols sit deep (long codes), frequent ones near the root")
    axP = fig.add_subplot(gs[1, 0])
    if picture is not None:
        _img(axP, picture, f"quantised image · {len(freqs)} symbols", vmin=0, vmax=255)
    else:
        order = sorted(freqs, key=lambda s: -freqs[s])
        axP.bar(range(len(order)), [freqs[s] for s in order], color=ACC)
        axP.set_xticks(range(len(order))); axP.set_xticklabels([label(s) for s in order], fontsize=8)
        axP.set_title("symbol counts")
    axB = fig.add_subplot(gs[1, 1])
    vals = [fixed, ent, avg]
    names = ["fixed-length\ncode", "entropy\n(the floor)", "Huffman\naverage"]
    axB.bar(names, vals, color=[ACC2, "white", ACC3])
    for i, v in enumerate(vals):
        axB.text(i, v + 0.05, f"{v:.2f}", ha="center")
    axB.set_title("bits per symbol"); axB.set_ylim(0, max(vals) * 1.25)
    axT2 = fig.add_subplot(gs[1, 2])
    order = sorted(table, key=lambda s: (len(table[s]), table[s]))[:14]
    rows = [f"{label(s):>5} {freqs[s]:>7}  {table[s]}" for s in order]
    total = len(data) * fixed
    comp = sum(len(table[s]) * c for s, c in freqs.items())
    _note(axT2, "symbol  count   code\n" + "\n".join(rows) + ("\n…" if len(table) > 14 else "") +
          f"\n\nraw  : {total:,} bits\nhuff : {comp:,} bits  ({100 * comp / total:.0f}%)")
    fig.suptitle("Huffman coding: give frequent symbols short codes, no code is a prefix of another", color=ACC)
    return fig
