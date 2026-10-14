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


CSS = """
.gradio-container {max-width: 1500px !important;}
#sidebar {position: sticky; top: 12px; align-self: flex-start;}
#nav .wrap {display: flex; flex-direction: column; gap: 4px;}
#nav label {width: 100%; padding: 10px 12px; border-radius: 8px;}
#nav label.selected {border-left: 4px solid var(--primary-500);}
#plot-panel {min-height: 520px;}
.idea {font-size: 0.97rem; line-height: 1.5;}
footer {display: none !important;}
"""

TOPICS = []  # (title, blurb, controls factory, function, takes_image)


def topic(title, blurb, controls, fn, takes_image=True):
    TOPICS.append((title, blurb, controls, fn, takes_image))


topic("Histograms",
      "**Idea.** A histogram counts how many pixels have each grey level. A dull image uses only a narrow slice of `[0, 255]`. "
      "*Stretching* maps that slice to the full range; *equalising* uses the cumulative histogram as the mapping, so every level is used about equally.\n\n"
      "**Try:** pick *muscle fibres* and drag the clipping.",
      lambda: [gr.Slider(0, 20, 1, step=0.5, label="clip lowest % of pixels"), gr.Slider(80, 100, 99, step=0.5, label="clip highest % of pixels")],
      Fg.histogram_figure)
topic("Convolution",
      "**Idea.** One output pixel = a small grid of weights (the *kernel*) laid over the input, multiplied cell by cell, then summed. "
      "Equal weights blur; opposite signs find edges.\n\n**Try:** move the marked pixel, and read the arithmetic panel.",
      lambda: [gr.Dropdown(list(Fg.KERNELS), value="sobel x (vertical edges)", label="kernel"),
               gr.Slider(3, 9, 3, step=2, label="kernel size (blurs only)"),
               gr.Slider(0, 1, 0.5, step=0.01, label="pixel position →"), gr.Slider(0, 1, 0.45, step=0.01, label="pixel position ↓")],
      Fg.convolution_figure)
topic("Noise and filters",
      "**Idea.** Averaging neighbours cancels random noise but also blurs edges. The *median* takes the middle value of the neighbourhood, so a lone white or black speck is simply outvoted.\n\n"
      "**Try:** salt & pepper at 20%, then compare mean and median.",
      lambda: [gr.Radio(["salt & pepper", "gaussian"], value="salt & pepper", label="noise"),
               gr.Slider(1, 40, 10, step=1, label="amount (% pixels, or σ)"),
               gr.Radio(["mean", "gaussian", "median"], value="median", label="filter"),
               gr.Slider(3, 9, 3, step=2, label="filter size")],
      Fg.denoise_figure)
topic("Fourier transform",
      "**Idea.** An image is a sum of 2D waves. The spectrum shows how much of each: slow waves (coarse shapes) near the centre, fast waves (edges, texture) far out. "
      "Erase part of the spectrum and see what the picture loses. The DFT here is the plain matrix definition, no FFT.\n\n**Try:** high-pass, then keep only 1% of the strongest coefficients.",
      lambda: [gr.Radio(["low-pass", "high-pass", "band-pass", "keep the strongest coefficients"], value="low-pass", label="what to keep"),
               gr.Slider(2, 60, 12, step=1, label="radius in the spectrum (pixels)"),
               gr.Slider(0.5, 30, 5, step=0.5, label="% of coefficients (strongest mode)")],
      Fg.fourier_figure)
topic("One wave, two spikes",
      "**Idea.** A single 2D wave has exactly two mirrored spikes in its spectrum. Their distance from the centre is the frequency; their direction is the wave's direction. "
      "The phase only slides the stripes.\n\n**Try:** raise *u* and watch the spikes move outward.",
      lambda: [gr.Slider(0, 16, 5, step=1, label="u (cycles across, horizontal)"),
               gr.Slider(0, 16, 3, step=1, label="v (cycles across, vertical)"),
               gr.Slider(0, 360, 0, step=15, label="phase φ (degrees)")],
      Fg.sinusoid_figure, takes_image=False)
topic("Harris corners",
      "**Idea.** Slide a small window around a pixel. On flat ground nothing changes; along an edge it changes only across the edge; at a corner it changes whichever way you go. "
      "The *structure tensor* summarises that with two eigenvalues, drawn as the ellipse.\n\n**Try:** the *house* sample, then lower the threshold.",
      lambda: [gr.Slider(0.02, 0.1, 0.05, step=0.005, label="k (edge vs corner penalty)"),
               gr.Slider(0.8, 3, 1.5, step=0.1, label="σ window size"),
               gr.Slider(0.5, 30, 5, step=0.5, label="detection threshold (% of the max response)")],
      Fg.harris_figure)
topic("Hough lines",
      "**Idea.** A line is `x·cosθ + y·sinθ = ρ`. Each edge pixel votes for every (θ, ρ) of a line through it; real lines collect many votes and become bright spots.\n\n"
      "**Try:** *university facade* or *corridor*.",
      lambda: [gr.Slider(0.1, 0.8, 0.3, step=0.05, label="edge threshold (fraction of max gradient)"),
               gr.Slider(1, 10, 5, step=1, label="number of lines"),
               gr.Slider(5, 150, 50, step=5, label="curves to draw")],
      Fg.hough_figure)
topic("Quadtree split and merge",
      "**Idea.** Cut a block in four whenever it is not uniform enough, recursively: big blocks where nothing happens, tiny ones around detail. "
      "Then *merge* touching blocks of similar brightness into regions. The block count is also a crude compression ratio.\n\n**Try:** lower the maximum depth step by step.",
      lambda: [gr.Slider(2, 40, 14, step=1, label="split tolerance (grey-level std)"),
               gr.Slider(1, 7, 6, step=1, label="maximum depth"),
               gr.Slider(1, 60, 25, step=1, label="merge tolerance (grey levels)")],
      Fg.quadtree_figure)
topic("Huffman coding",
      "**Idea.** Entropy is the fewest bits per symbol any code can average. Huffman builds the best *prefix* code by repeatedly joining the two rarest symbols.\n\n"
      "**Try:** switch to *text* and type something repetitive.",
      lambda: [gr.Radio(["image", "text"], value="image", label="encode"),
               gr.Textbox("abracadabra alakazam", label="text", lines=1),
               gr.Radio([2, 4, 8, 16], value=8, label="grey levels (image mode)")],
      lambda im, s, t, lv: Fg.huffman_figure(im, s, t, int(lv)))

NAMES = [f"{i + 1}  ·  {t[0]}" for i, t in enumerate(TOPICS)]

with gr.Blocks(title="Image processing from scratch", analytics_enabled=False, css=CSS,
               theme=gr.themes.Soft(primary_hue="amber", font=["ui-sans-serif", "system-ui", "sans-serif"],
                                    font_mono=["ui-monospace", "monospace"])) as demo:
    gr.Markdown("# 👁️ Image processing from scratch\n"
                "Classical computer vision written in plain numpy and running **inside this page**. "
                "🔒 Your image never leaves your device. "
                "[Source](https://github.com/mbenimam/Computer-Vision-Projects)")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=270, elem_id="sidebar"):
            src = gr.Image(value=sample("cameraman"), label="Image (drop your own)", type="numpy", image_mode="L", height=210)
            pick = gr.Dropdown(list(SAMPLES), value="cameraman", label="or choose a sample")
            nav = gr.Radio(NAMES, value=NAMES[0], label="Topics", elem_id="nav")
        with gr.Column(scale=4):
            panels, runs = [], []
            for i, (title, blurb, controls, fn, takes_image) in enumerate(TOPICS):
                with gr.Column(visible=(i == 0)) as panel:
                    gr.Markdown(f"## {title}")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, min_width=260):
                            gr.Markdown(blurb, elem_classes="idea")
                            comps = controls()
                        with gr.Column(scale=3, elem_id=f"plot-panel-{i}"):
                            plot = gr.Plot(label="", show_label=False)
                panels.append(panel)
                inputs = [nav, src] + comps

                def make(name, fn, takes_image):
                    safe_fn = safe(fn)

                    def run(current, img, *vals):
                        if current != name:
                            return gr.update()
                        return safe_fn(img, *vals) if takes_image else safe_fn(*vals)
                    return run

                run = make(NAMES[i], fn, takes_image)
                triggers = [src.change, nav.change] + [(c.release if isinstance(c, gr.Slider) else c.change) for c in comps]
                gr.on(triggers=triggers, fn=run, inputs=inputs, outputs=plot, show_progress="minimal")
                if i == 0:
                    demo.load(run, inputs, plot)
            with gr.Row():
                prev_b = gr.Button("‹ Previous topic", size="sm")
                next_b = gr.Button("Next topic ›", size="sm", variant="primary")

    pick.change(sample, pick, src)
    nav.change(lambda cur: [gr.update(visible=(n == cur)) for n in NAMES], nav, panels)

    def step(cur, d):
        return NAMES[(NAMES.index(cur) + d) % len(NAMES)]
    prev_b.click(lambda c: step(c, -1), nav, nav)
    next_b.click(lambda c: step(c, 1), nav, nav)

demo.launch()
