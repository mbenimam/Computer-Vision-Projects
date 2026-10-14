"""Assemble the static site in site/: index.html + demo code + cvlab + samples."""
import html as htmllib
import json
import pathlib
import shutil

root = pathlib.Path(__file__).resolve().parent.parent
site = root / "site"
site.mkdir(exist_ok=True)
for child in site.iterdir():  # empty the folder in place so a running static server keeps its cwd
    shutil.rmtree(child) if child.is_dir() else child.unlink()
(site / "cvlab").mkdir(parents=True)
# Binary files cannot go through <gradio-file>, so the samples ship inside a python module.
import base64
data = {p.name: base64.b64encode(p.read_bytes()).decode() for p in sorted((root / "demo" / "samples").glob("*.png"))}
(site / "samples_data.py").write_text("SAMPLES = " + json.dumps(data) + "\n")
files = ["cvlab/" + p.name for p in (root / "cvlab").glob("*.py")]
for p in (root / "cvlab").glob("*.py"):
    shutil.copy(p, site / "cvlab" / p.name)
shutil.copy(root / "demo" / "figures.py", site / "figures.py")
shutil.copy(root / "demo" / "app.py", site / "app.py")

import subprocess
import sys
import tempfile

LITE_VERSION = "5.45.0"
# Gradio-Lite is self-hosted (the page loads no third-party script) and patched in one place.
# gradio 5.45 resolves its dependencies from live PyPI at start-up, and releases made after it
# need a newer typing-extensions than Pyodide 0.27 ships. Pre-installing the pure-Python
# dependency set as it was on 2025-09-20 (from `uv pip compile --exclude-newer`) fixes that.
PINS = ("aiofiles==24.1.0 annotated-types==0.7.0 anyio==4.10.0 certifi==2025.8.3 charset-normalizer==3.4.3 "
        "fastapi==0.117.1 filelock==3.19.1 fsspec==2025.9.0 groovy==0.1.2 h11==0.16.0 httpcore==1.0.9 "
        "httpx==0.28.1 huggingface-hub==0.35.0 idna==3.10 jinja2==3.1.6 pydub==0.25.1 python-multipart==0.0.20 "
        "requests==2.32.5 safehttpx==0.1.6 semantic-version==2.10.0 sniffio==1.3.1 starlette==0.48.0 "
        "tomlkit==0.13.3 tqdm==4.67.1 typing-inspection==0.4.1 urllib3==2.5.0 websockets==15.0.1").split()
cache = pathlib.Path(tempfile.gettempdir()) / f"gradio-lite-{LITE_VERSION}"
if not cache.exists():
    cache.mkdir()
    subprocess.run(["npm", "pack", f"@gradio/lite@{LITE_VERSION}", "--silent"], cwd=cache, check=True)
    subprocess.run(["tar", "xzf", next(cache.glob("*.tgz"))], cwd=cache, check=True)
shutil.copytree(cache / "package" / "dist", site / "lite", ignore=shutil.ignore_patterns("*.map", "lite.html"))
patched = 0
for w in (site / "lite" / "assets").glob("webworker-*.js"):
    t = w.read_text()
    needle = "await E(s,i,r),"
    if needle in t:
        pre = "await(async()=>{try{await i.install(%s)}catch(e){console.warn('pins',e)}})()," % json.dumps(PINS)
        w.write_text(t.replace(needle, pre + needle, 1))
        patched += 1
if not patched:
    sys.exit("webworker patch point not found: gradio-lite layout changed")
lite = "./lite"
def gradio_file(name, path, entry=False):
    # inline content: <gradio-file url=...> hung the Gradio-Lite worker in testing
    body = htmllib.escape(path.read_text(), quote=False)
    return f'    <gradio-file name="{name}"{" entrypoint" if entry else ""}>\n{body}\n    </gradio-file>'


tags = [gradio_file("app.py", site / "app.py", entry=True)]
tags += [gradio_file(f, site / f) for f in files]
tags += [gradio_file("figures.py", site / "figures.py"), gradio_file("samples_data.py", site / "samples_data.py")]
tags = "\n".join(tags)
html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image processing from scratch</title>
  <meta name="description" content="Interactive classical computer vision (histograms, convolution, Fourier, Harris, Hough, quadtree, Huffman) running entirely in your browser. Nothing is uploaded.">
  <script type="module" crossorigin src="{lite}/lite.js"></script>
  <link rel="stylesheet" href="{lite}/lite.css" />
  <style>body{{margin:0;background:#0e1117}}</style>
</head>
<body>
  <gradio-lite>
    <gradio-requirements>
numpy
matplotlib
pillow
    </gradio-requirements>
{tags}
  </gradio-lite>
</body>
</html>
"""
(site / "index.html").write_text(html)
print("site ready:", site, len(list(site.rglob('*'))), "entries")
