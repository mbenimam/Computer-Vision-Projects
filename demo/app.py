"""Image processing from scratch: runs in the visitor's browser (Gradio-Lite on Pyodide)."""
import base64
import io

import gradio as gr
import numpy as np
from PIL import Image

import figures as Fg
import shell

SAMPLES = {
    "cameraman": "cameraman.png", "muscle fibres (low contrast)": "muscle.png", "corridor (lines)": "corridor.png",
    "house (corners)": "house.png", "coins": "coins.png", "rectangle": "rectangle.png", "prism": "prism.png",
    "office": "office.png", "university facade (lines)": "university.png", "cuttlefish": "cuttlefish.png",
    "angiogram": "angiogram.png",
}


def sample(name):
    from samples_data import SAMPLES as DATA
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(DATA[SAMPLES[name]]))).convert("L"))


S = gr.Slider
TOPICS = [
    dict(title="Histograms", takes_image=True, fn=Fg.histogram_figure,
         idea="A histogram counts how many pixels have each grey level. A dull image uses a narrow slice of [0, 255]. "
              "*Stretching* maps that slice to the full range; *equalising* uses the cumulative histogram as the mapping, "
              "so every level ends up used about equally.",
         try_="pick *muscle fibres*, then drag the clipping.",
         controls=lambda: [S(0, 20, 1, step=0.5, label="Clip darkest pixels (%)"),
                           S(80, 100, 99, step=0.5, label="Clip brightest pixels (%)")]),
    dict(title="Convolution", takes_image=True, fn=Fg.convolution_figure,
         idea="One output pixel is a small grid of weights (the *kernel*) laid over the input, multiplied cell by cell, "
              "then summed. Equal weights blur; weights of opposite sign find edges.",
         try_="move the marked pixel and read the arithmetic panel.",
         controls=lambda: [gr.Dropdown(list(Fg.KERNELS), value="sobel x (vertical edges)", label="Kernel"),
                           S(3, 9, 3, step=2, label="Kernel size (blurs)"),
                           S(0, 1, 0.5, step=0.01, label="Pixel, horizontal"), S(0, 1, 0.45, step=0.01, label="Pixel, vertical")]),
    dict(title="Noise and filters", takes_image=True, fn=Fg.denoise_figure,
         idea="Averaging neighbours cancels random noise but also blurs edges. The *median* takes the middle value of the "
              "neighbourhood, so a lone white or black speck is simply outvoted.",
         try_="salt & pepper at 20%, then compare mean and median.",
         controls=lambda: [gr.Radio(["salt & pepper", "gaussian"], value="salt & pepper", label="Noise"),
                           S(1, 40, 10, step=1, label="Amount (% of pixels, or σ)"),
                           gr.Radio(["mean", "gaussian", "median"], value="median", label="Filter"),
                           S(3, 9, 3, step=2, label="Filter size")]),
    dict(title="Fourier transform", takes_image=True, fn=Fg.fourier_figure,
         idea="An image is a sum of 2D waves. The spectrum shows how much of each: slow waves (coarse shapes) near the "
              "centre, fast waves (edges, texture) far out. Erase part of it and see what the picture loses.",
         try_="high-pass, then keep only 1% of the strongest coefficients.",
         controls=lambda: [gr.Radio(["low-pass", "high-pass", "band-pass", "keep the strongest coefficients"],
                                    value="low-pass", label="Keep"),
                           S(2, 60, 12, step=1, label="Radius in the spectrum (px)"),
                           S(0.5, 30, 5, step=0.5, label="Strongest coefficients kept (%)")]),
    dict(title="One wave, two spikes", takes_image=False, fn=Fg.sinusoid_figure,
         idea="A single 2D wave has exactly two mirrored spikes in its spectrum. Their distance from the centre is the "
              "frequency; their direction is the direction of the wave. The phase only slides the stripes.",
         try_="raise *u* and watch the spikes move outward.",
         controls=lambda: [S(0, 16, 5, step=1, label="u (horizontal cycles)"), S(0, 16, 3, step=1, label="v (vertical cycles)"),
                           S(0, 360, 0, step=15, label="Phase φ (°)")]),
    dict(title="Harris corners", takes_image=True, fn=Fg.harris_figure,
         idea="Slide a small window around a pixel. On flat ground nothing changes; along an edge it changes only across "
              "the edge; at a corner it changes whichever way you go. The *structure tensor* captures that with two "
              "eigenvalues, drawn as the ellipse.",
         try_="the *house* sample, then lower the threshold.",
         controls=lambda: [S(0.02, 0.1, 0.05, step=0.005, label="k (edge vs corner)"), S(0.8, 3, 1.5, step=0.1, label="Window σ"),
                           S(0.5, 30, 5, step=0.5, label="Threshold (% of max response)")]),
    dict(title="Hough lines", takes_image=True, fn=Fg.hough_figure,
         idea="A line is x·cosθ + y·sinθ = ρ. Each edge pixel votes for every (θ, ρ) of a line through it; real lines "
              "collect many votes and become bright spots in the accumulator.",
         try_="*university facade* or *corridor*.",
         controls=lambda: [S(0.1, 0.8, 0.3, step=0.05, label="Edge threshold"), S(1, 10, 5, step=1, label="Lines to find"),
                           S(5, 150, 50, step=5, label="Vote curves drawn")]),
    dict(title="Quadtree split and merge", takes_image=True, fn=Fg.quadtree_figure,
         idea="Cut a block in four whenever it is not uniform enough, recursively: big blocks where nothing happens, tiny "
              "ones around detail. Then *merge* touching blocks of similar brightness into regions.",
         try_="lower the maximum depth one step at a time.",
         controls=lambda: [S(2, 40, 14, step=1, label="Split tolerance (std)"), S(1, 7, 6, step=1, label="Maximum depth"),
                           S(1, 60, 25, step=1, label="Merge tolerance")]),
    dict(title="Huffman coding", takes_image=True, fn=lambda im, s, t, lv: Fg.huffman_figure(im, s, t, int(lv)),
         idea="Entropy is the fewest bits per symbol any code can average. Huffman builds the best *prefix* code by "
              "repeatedly joining the two rarest symbols.",
         try_="switch to *text* and type something repetitive.",
         controls=lambda: [gr.Radio(["image", "text"], value="image", label="Encode"),
                           gr.Textbox("abracadabra alakazam", label="Text"),
                           gr.Radio([2, 4, 8, 16], value=8, label="Grey levels (image mode)")]),
]

demo = shell.build(
    app_title="Image processing from scratch",
    tagline="Classical computer vision in plain numpy, running in your browser. Nothing is uploaded.",
    links_md="[Source code](https://github.com/mbenimam/Computer-Vision-Projects) · ← → keys switch topics",
    topics=TOPICS, image_value=sample("cameraman"), sample_names=list(SAMPLES), load_sample=sample)

demo.launch()
