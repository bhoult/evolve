"""Renders assets/icon.json with ray, a separate SDF ray tracer (RAY=path/to/ray, default
~/programming/ai/ray/target/release/ray), at 1024 px and writes assets/icon.png: 256 px, a
rounded square with transparent corners. Needs Pillow. usage: python3 tools/make-icon.py"""
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ray = os.environ.get("RAY", os.path.expanduser("~/programming/ai/ray/target/release/ray"))
with tempfile.TemporaryDirectory() as tmp:
    shot = os.path.join(tmp, "icon.png")
    subprocess.run([ray, os.path.join(root, "assets/icon.json"), "--out", shot, "--size", "1024x1024", "--frames", "32"], check=True)
    big = Image.open(shot).convert("RGBA")
# The corners: a rounded square 4 px in from each edge, with a 48 px radius (at 256 px).
mask = Image.new("L", big.size, 0)
ImageDraw.Draw(mask).rounded_rectangle([16, 16, big.width - 17, big.height - 17], radius=192, fill=255)
big.putalpha(mask)
big.resize((256, 256), Image.LANCZOS).save(os.path.join(root, "assets/icon.png"), optimize=True)
print("wrote assets/icon.png")
