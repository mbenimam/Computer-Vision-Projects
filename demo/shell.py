"""Shared page shell for the browser demos: topic navigation | figure canvas | inspector.

The figure is the focus. Navigation sits on the left, the explanation and the parameters of the
current topic sit on the right, next to what they change. Only one topic is computed at a time.
"""
import gradio as gr
import matplotlib.pyplot as plt

BG = "#0e1117"
PANEL = "#121722"
LINE = "#222a37"
MUTED = "#9aa4b2"
ACCENT = "#ffb000"

CSS = f"""
:root, .dark {{ --body-background-fill: {BG}; }}
body, .gradio-container {{ background: {BG} !important; }}
.gradio-container {{ max-width: none !important; padding: 0 !important; margin: 0 !important; }}
footer {{ display: none !important; }}
main.app, main.fillable, .main, .contain, .wrap.svelte-czcr5b {{ max-width: none !important; padding: 0 !important; margin: 0 !important; gap: 0 !important; }}

/* three columns: nav | canvas | inspector */
#shell {{ gap: 0 !important; min-height: 100vh; flex-wrap: nowrap !important; align-items: stretch; }}
#nav-col {{ flex: 0 0 236px !important; min-width: 236px !important; background: {PANEL}; border-right: 1px solid {LINE};
           position: sticky; top: 0; height: 100vh; overflow-y: auto; padding: 18px 12px !important; gap: 4px !important; }}
#canvas {{ flex: 1 1 auto !important; min-width: 0 !important; padding: 14px 22px 10px !important; gap: 6px !important; }}
#inspector {{ flex: 0 0 330px !important; min-width: 330px !important; background: {PANEL}; border-left: 1px solid {LINE};
             position: sticky; top: 0; height: 100vh; overflow-y: auto; padding: 20px 18px 24px !important; gap: 0 !important; }}
#inspector > .column, #inspector > div {{ gap: 6px !important; flex: 0 0 auto !important; }}
#inspector .prose, #inspector .md {{ margin: 0 !important; }}
.insp-group {{ padding-bottom: 16px !important; margin-bottom: 14px !important; border-bottom: 1px solid {LINE} !important; }}

/* quiet blocks: no cards inside the panels */
#nav-col .block, #inspector .block, #canvas .block, #inspector .form, #canvas .form {{
  background: transparent !important; border: none !important; box-shadow: none !important; }}
#inspector .block {{ padding: 4px 0 !important; }}

/* brand */
#brand h3 {{ font-size: 15px; margin: 0 0 2px; color: #fff; }}
#brand p {{ font-size: 12.5px; color: {MUTED}; margin: 0; line-height: 1.45; }}
.section-title p, .section-title {{ font-size: 11px !important; letter-spacing: .08em; text-transform: uppercase; color: {MUTED} !important;
  margin: 0 0 4px !important; font-weight: 600; }}
#nav-col .section-title {{ margin-top: 16px !important; }}

/* topic navigation: a list, not radio buttons */
#nav > [data-testid="block-info"] {{ display: none; }}
#nav .wrap {{ display: flex; flex-direction: column; gap: 2px; }}
#nav input[type=radio] {{ display: none; }}
#nav label {{ width: 100%; background: transparent !important; border: none !important; box-shadow: none !important;
  padding: 8px 10px !important; border-radius: 6px; color: #b9c1cc; font-size: 14px; cursor: pointer; }}
#nav label:hover {{ background: #1a2130 !important; color: #fff; }}
#nav label.selected {{ background: #1e2635 !important; color: #fff; box-shadow: inset 3px 0 0 {ACCENT} !important; }}
#nav label span {{ margin: 0 !important; }}
#links p {{ font-size: 12.5px; color: {MUTED}; margin-top: 14px; }}

/* canvas header */
#topbar {{ align-items: center; gap: 8px !important; flex-wrap: nowrap !important; }}
#heading {{ flex: 1 1 auto; min-width: 0; }}
#heading h2 {{ font-size: 21px; margin: 0; color: #fff; font-weight: 650; }}
#heading .step {{ color: {MUTED}; font-weight: 500; font-size: 13px; margin-right: 8px; }}
button.navbtn {{ flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; padding: 5px 12px !important;
  font-size: 13px !important; background: transparent !important; border: 1px solid #2c3546 !important; color: #d3d9e1 !important;
  border-radius: 6px !important; box-shadow: none !important; }}
button.navbtn:hover {{ background: #1a2130 !important; border-color: #3a4558 !important; }}

/* the figure fills the canvas */
.fig, .fig > div, .fig .plot-container {{ background: {BG} !important; border: none !important; min-height: 0 !important; padding: 0 !important; }}
.fig [data-testid="block-label"] {{ display: none !important; }}
.fig img {{ display: block; margin: 0 auto; width: auto; max-width: 100%; height: auto;
  max-height: calc(100vh - 78px); object-fit: contain; }}

/* inspector content */
.idea p, .idea li {{ font-size: 14px; line-height: 1.55; color: #d6dce4; margin: 0 0 8px; }}
.try p {{ font-size: 13px; line-height: 1.5; color: #e9e2cf; margin: 0; padding: 8px 10px; border-radius: 6px;
  background: rgba(255,176,0,.08); border-left: 3px solid {ACCENT}; }}
#inspector [data-testid="block-info"], #inspector label > span:first-child {{ font-size: 13px !important; color: #c6cdd6 !important;
  font-weight: 500 !important; background: none !important; padding: 0 !important; }}
#inspector .head {{ align-items: baseline; }}
#inspector .tab-like-container {{ border: none !important; background: transparent !important; box-shadow: none !important; }}
#inspector .head input[type=number] {{ width: 64px !important; background: transparent !important; border: none !important;
  box-shadow: none !important; text-align: right; color: #fff; font-variant-numeric: tabular-nums; padding: 0 !important; }}
#inspector .reset-button, #inspector .min_value, #inspector .max_value {{ display: none !important; }}
#inspector input[type=range] {{ accent-color: {ACCENT}; }}
#inspector .image-container {{ background: #0b0e14; border-radius: 8px; }}
#inspector .wrap-inner, #inspector input[type=text], #inspector textarea {{ background: #0b0e14 !important; border: 1px solid {LINE} !important;
  box-shadow: none !important; }}
#inspector .wrap-inner input {{ border: none !important; }}

/* narrow screens: nav becomes a scrolling strip, then figure, then inspector */
@media (max-width: 1000px) {{
  #shell {{ flex-direction: column !important; }}
  #nav-col, #inspector {{ position: static; height: auto; flex: none !important; min-width: 0 !important; width: 100% !important;
    border: none; }}
  #nav .wrap {{ flex-direction: row; overflow-x: auto; flex-wrap: nowrap; }}
  #nav label {{ white-space: nowrap; width: auto; }}
  #canvas {{ padding: 10px !important; }}
  .fig img {{ max-height: none; width: 100%; }}
  #links {{ display: none !important; }}
}}
"""

# Left/right arrow keys move between topics (ignored while typing in a field).
KEYS_JS = """
() => {
  document.addEventListener('keydown', (e) => {
    const t = e.target; if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    const id = e.key === 'ArrowRight' ? 'next-topic' : e.key === 'ArrowLeft' ? 'prev-topic' : null;
    if (id) document.getElementById(id)?.click();
  });
}
"""


def _safe(fn):
    def run(*a):
        try:
            return fn(*a)
        except Exception as e:  # show the reason where the figure would be
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis("off"); ax.text(0.5, 0.5, f"{type(e).__name__}: {e}", ha="center", va="center", wrap=True)
            return fig
    return run


def build(app_title, tagline, links_md, topics, image_value, sample_names, load_sample):
    """topics: list of dicts with title, idea, try_, controls (callable returning components), fn, takes_image."""
    names = [t["title"] for t in topics]
    n = len(names)
    theme = gr.themes.Base(primary_hue="amber", neutral_hue="slate",
                           font=["ui-sans-serif", "system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"])

    def heading(name):
        i = names.index(name)
        return f"## <span class='step'>{i + 1} / {n}</span>{name}"

    with gr.Blocks(title=app_title, analytics_enabled=False, css=CSS, theme=theme, js=KEYS_JS, fill_height=True) as demo:
        with gr.Row(elem_id="shell", equal_height=False):
            with gr.Column(elem_id="nav-col", scale=0):
                gr.Markdown(f"### {app_title}\n{tagline}", elem_id="brand")
                gr.Markdown("Topics", elem_classes="section-title")
                nav = gr.Radio(names, value=names[0], show_label=False, elem_id="nav", container=False)
                gr.Markdown(links_md, elem_id="links")

            with gr.Column(elem_id="canvas", scale=1):
                with gr.Row(elem_id="topbar", equal_height=True):
                    title = gr.Markdown(heading(names[0]), elem_id="heading")
                    prev_b = gr.Button("← Previous", elem_id="prev-topic", elem_classes="navbtn", size="sm", scale=0)
                    next_b = gr.Button("Next →", elem_id="next-topic", elem_classes="navbtn", size="sm", scale=0)
                plots = []
                for i, t in enumerate(topics):
                    plots.append(gr.Plot(show_label=False, elem_classes="fig", visible=(i == 0), container=False))

            with gr.Column(elem_id="inspector", scale=0):
                ideas, params = [], []
                for i, t in enumerate(topics):
                    with gr.Column(visible=(i == 0), elem_classes="insp-group") as g:
                        gr.Markdown("How it works", elem_classes="section-title")
                        gr.Markdown(t["idea"], elem_classes="idea")
                        if t.get("try_"):
                            gr.Markdown("**Try:** " + t["try_"], elem_classes="try")
                    ideas.append(g)
                with gr.Column(visible=topics[0]["takes_image"], elem_classes="insp-group") as image_group:
                    gr.Markdown("Input image", elem_classes="section-title")
                    src = gr.Image(value=image_value, show_label=False, type="numpy", image_mode="L", height=150,
                                   sources=["upload", "clipboard"], show_download_button=False, show_fullscreen_button=False)
                    pick = gr.Dropdown(sample_names, value=sample_names[0], label="Sample")
                comps_per_topic = []
                for i, t in enumerate(topics):
                    with gr.Column(visible=(i == 0)) as g:
                        gr.Markdown("Parameters", elem_classes="section-title")
                        comps_per_topic.append(t["controls"]())
                    params.append(g)

        # one compute function per topic; it does nothing unless its topic is on screen
        for i, t in enumerate(topics):
            safe_fn = _safe(t["fn"])

            def run(current, img, *vals, _name=names[i], _fn=safe_fn, _img=t["takes_image"]):
                if current != _name or (_img and img is None):
                    return gr.update()
                return _fn(img, *vals) if _img else _fn(*vals)

            comps = comps_per_topic[i]
            inputs = [nav, src] + comps
            triggers = [nav.change] + ([src.change] if t["takes_image"] else []) + \
                       [(c.release if isinstance(c, gr.Slider) else c.change) for c in comps]
            gr.on(triggers=triggers, fn=run, inputs=inputs, outputs=plots[i], show_progress="minimal")
            if i == 0:
                demo.load(run, inputs, plots[i])

        def show(current):
            k = names.index(current)
            return ([heading(current), gr.update(visible=topics[k]["takes_image"])]
                    + [gr.update(visible=(j == k)) for j in range(n)] * 3)

        nav.change(show, nav, [title, image_group] + plots + ideas + params, show_progress="hidden")
        pick.change(load_sample, pick, src)
        prev_b.click(lambda c: names[(names.index(c) - 1) % n], nav, nav)
        next_b.click(lambda c: names[(names.index(c) + 1) % n], nav, nav)
    return demo
