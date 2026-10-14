"""Render the README figures with the same code the demo runs."""
import pathlib
import sys

import numpy as np
from PIL import Image

root = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(root), str(root / "demo")]
import figures as Fg  # noqa: E402

load = lambda n: np.asarray(Image.open(root / "demo" / "samples" / f"{n}.png"))
out = root / "docs" / "assets"
jobs = {
    "histograms": lambda: Fg.histogram_figure(load("muscle")),
    "convolution": lambda: Fg.convolution_figure(load("cameraman"), "sobel x (vertical edges)", 3, 0.5, 0.45),
    "fourier": lambda: Fg.fourier_figure(load("cameraman"), "low-pass", 12, 5),
    "harris": lambda: Fg.harris_figure(load("house"), 0.05, 1.5, 5),
    "hough": lambda: Fg.hough_figure(load("university"), 0.3, 5, 50),
    "quadtree": lambda: Fg.quadtree_figure(load("cameraman"), 14, 6, 25),
    "huffman": lambda: Fg.huffman_figure(load("cameraman"), "image", "", 8),
}
for name, fn in jobs.items():
    fn().savefig(out / f"{name}.png", dpi=80)
    print(name)
