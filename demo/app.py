"""Gradio app. In the deployed site it runs inside the visitor's browser (Gradio-Lite on Pyodide)."""
import os
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import figures as Fg

SAMPLES = {
    "cameraman": "cameraman.png", "muscle fibres (low contrast)": "muscle.png", "corridor (lines)": "corridor.png",
    "house (corners)": "house.png", "coins": "coins.png", "rectangle": "rectangle.png", "prism": "prism.png",
    "office": "office.png", "university facade (lines)": "university.png", "cuttlefish": "cuttlefish.png",
    "angiogram": "angiogram.png",
}


def sample(name):
    import base64, io
    from samples_data import SAMPLES as DATA
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(DATA[SAMPLES[name]]))).convert("L"))


def safe(fn):
    def run(*a):
        try:
            return fn(*a)
        except Exception as e:  # show the reason in the plot area instead of a stack trace
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis("off"); ax.text(0.5, 0.5, f"{type(e).__name__}: {e}", ha="center", va="center", wrap=True)
            return fig
    return run


INTRO = """# 👁️ Image processing from scratch, in your browser
Every algorithm below is written from first principles in numpy ([source](https://github.com/mbenimam/Computer-Vision-Projects/tree/master/cvlab)) and runs **inside this page** (Python compiled to WebAssembly).
🔒 **Your image never leaves your device**: it is not uploaded, there is no server that computes anything. Pick a sample or drop your own below, then play with the sliders in each tab."""

with gr.Blocks(title="Image processing from scratch", analytics_enabled=False,
                   theme=gr.themes.Soft(primary_hue="amber", font=["ui-sans-serif", "system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"])) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        pick = gr.Dropdown(list(SAMPLES), value="cameraman", label="Sample image", scale=1)
        src = gr.Image(value=sample("cameraman"), label="Your image (stays in this tab)", type="numpy", image_mode="L", height=240, scale=2)
    pick.change(sample, pick, src)

    def tab(title, blurb, controls, fn):
        with gr.Tab(title):
            gr.Markdown(blurb)
            comps = controls()
            plot = gr.Plot(label="")
            ins = [src] + comps
            out = safe(fn)
            triggers = [src.change] + [(c.release if isinstance(c, gr.Slider) else c.change) for c in comps]
            gr.on(triggers=triggers, fn=out, inputs=ins, outputs=plot, show_progress="minimal")
            demo.load(out, ins, plot)
        return plot

    tab("1 · Histograms",
        "**Idea.** A histogram counts how many pixels have each grey level. A dull image uses only a narrow slice of `[0, 255]`. "
        "*Stretching* maps that slice to the full range; *equalising* uses the cumulative histogram as the mapping so every level is used about equally. "
        "Try the **muscle fibres** sample and drag the percentiles.",
        lambda: [gr.Slider(0, 20, 1, step=0.5, label="clip lowest % of pixels"), gr.Slider(80, 100, 99, step=0.5, label="clip highest % of pixels")],
        Fg.histogram_figure)

    tab("2 · Convolution",
        "**Idea.** To compute one output pixel, centre a small grid of weights (the *kernel*) on the input pixel, multiply it with the values under it, and add. "
        "The weights decide the effect: all equal blurs, opposite signs find edges. Move the marked pixel and watch the arithmetic.",
        lambda: [gr.Dropdown(list(Fg.KERNELS), value="sobel x (vertical edges)", label="kernel"),
                 gr.Slider(3, 9, 3, step=2, label="kernel size (blurs only)"),
                 gr.Slider(0, 1, 0.5, step=0.01, label="pixel position →"), gr.Slider(0, 1, 0.45, step=0.01, label="pixel position ↓")],
        Fg.convolution_figure)

    tab("3 · Noise & filters",
        "**Idea.** Averaging neighbours cancels random noise but also blurs edges. The *median* replaces a pixel by the middle value of its neighbourhood, so a lone white or black speck is simply outvoted.",
        lambda: [gr.Radio(["salt & pepper", "gaussian"], value="salt & pepper", label="noise"),
                 gr.Slider(1, 40, 10, step=1, label="amount (% pixels, or σ)"),
                 gr.Radio(["mean", "gaussian", "median"], value="median", label="filter"),
                 gr.Slider(3, 9, 3, step=2, label="filter size")],
        Fg.denoise_figure)

    tab("4 · Fourier",
        "**Idea.** Any image is a sum of 2D waves. The spectrum shows how much of each wave there is: slow waves (coarse shapes) near the centre, fast waves (edges, texture) far out. "
        "Erase part of the spectrum and see what the picture loses. The DFT here is the plain matrix definition, no FFT.",
        lambda: [gr.Radio(["low-pass", "high-pass", "band-pass", "keep the strongest coefficients"], value="low-pass", label="what to keep"),
                 gr.Slider(2, 60, 12, step=1, label="radius in the spectrum (pixels)"),
                 gr.Slider(0.5, 30, 5, step=0.5, label="% of coefficients (strongest mode)")],
        Fg.fourier_figure)

    with gr.Tab("4b · One wave"):
        gr.Markdown("A single 2D wave and its spectrum: always exactly two mirrored spikes. Their distance from the centre is the frequency, their direction is the wave's direction.")
        u = gr.Slider(0, 16, 5, step=1, label="u (cycles across, horizontal)")
        v = gr.Slider(0, 16, 3, step=1, label="v (cycles across, vertical)")
        ph = gr.Slider(0, 360, 0, step=15, label="phase φ (degrees)")
        p2 = gr.Plot(label="")
        f2 = safe(Fg.sinusoid_figure)
        gr.on([s.release for s in (u, v, ph)], f2, [u, v, ph], p2, show_progress="minimal")
        demo.load(f2, [u, v, ph], p2)

    tab("5 · Harris corners",
        "**Idea.** Slide a small window around a pixel. On flat ground nothing changes, along an edge it changes only across the edge, at a corner it changes whichever way you go. "
        "The *structure tensor* summarises this with two numbers (eigenvalues); the ellipse below shows how far you can move before the patch looks different.",
        lambda: [gr.Slider(0.02, 0.1, 0.05, step=0.005, label="k (edge vs corner penalty)"),
                 gr.Slider(0.8, 3, 1.5, step=0.1, label="σ window size"),
                 gr.Slider(0.5, 30, 5, step=0.5, label="detection threshold (% of the max response)")],
        Fg.harris_figure)

    tab("6 · Hough lines",
        "**Idea.** A line is `x·cosθ + y·sinθ = ρ`. Each edge pixel votes for every (θ, ρ) pair of a line that could pass through it; real lines collect many votes and become bright spots in the accumulator. "
        "Try the **university facade** or **corridor**.",
        lambda: [gr.Slider(0.1, 0.8, 0.3, step=0.05, label="edge threshold (fraction of max gradient)"),
                 gr.Slider(1, 10, 5, step=1, label="number of lines"),
                 gr.Slider(5, 150, 50, step=5, label="curves to draw")],
        Fg.hough_figure)

    tab("7 · Quadtree split & merge",
        "**Idea.** Cut the image into four quarters whenever a block is not uniform enough, recursively: big blocks where nothing happens, tiny ones around detail. "
        "Then *merge* touching blocks with similar brightness to get regions. The block count is also a crude compression ratio.",
        lambda: [gr.Slider(2, 40, 14, step=1, label="split tolerance (grey-level std)"),
                 gr.Slider(1, 7, 6, step=1, label="maximum depth"),
                 gr.Slider(1, 60, 25, step=1, label="merge tolerance (grey levels)")],
        Fg.quadtree_figure)

    tab("8 · Huffman coding",
        "**Idea.** Entropy is the least number of bits per symbol any code can average. Huffman builds the best *prefix* code by repeatedly joining the two rarest symbols. "
        "Switch to *text* and type something repetitive.",
        lambda: [gr.Radio(["image", "text"], value="image", label="encode"),
                 gr.Textbox("abracadabra alakazam", label="text", lines=1),
                 gr.Radio([2, 4, 8, 16], value=8, label="grey levels (image mode)")],
        lambda im, s, t, lv: Fg.huffman_figure(im, s, t, int(lv)))
    gr.Markdown("Built by [Mounib Benimam](https://github.com/mbenimam) · [source and notebooks](https://github.com/mbenimam/Computer-Vision-Projects)")


demo.launch()
