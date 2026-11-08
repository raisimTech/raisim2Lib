#!/usr/bin/env python3
"""Checks that every figure drawn on white is marked as a diagram.

A plot or diagram on a white background shows as a bright box on the site's dark
theme. Marking it with `{.diagram}` (Markdown) or class="diagram" (<img>) makes
the theme invert it there (see .vitepress/theme/styles/docs.css). This test finds
the figures that pages use whose border is mostly white and fails for any that
is not marked.

Raster images are read with Pillow; without it, only SVG figures are checked.

Run with: python3 -m unittest discover -s docs/scripts
"""

import re
import unittest
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # Pillow is optional; the raster check is skipped without it.
    Image = None

DOCS = Path(__file__).resolve().parent.parent

# ![alt](src){attrs} with an alt that may span lines, and <img ...> tags.
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)(\{[^}\n]*\})?", re.S)
_HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.S)
_WHITE_RECT = re.compile(
    r"<rect\b[^>]*\bfill=\"(?:#fff|#ffffff|#fcfcfc|#fdfdfd|#fefefe|white)\"", re.I)


def figures():
    """(page, image path, marked as diagram) for every local image a page uses."""
    for page in sorted((DOCS / "sections").rglob("*.md")):
        text = page.read_text(encoding="utf-8")
        found = [(m.group(1), ".diagram" in (m.group(2) or "")) for m in _MARKDOWN_IMAGE.finditer(text)]
        for tag in _HTML_IMAGE.findall(text):
            src = re.search(r'\bsrc="([^"]+)"', tag)
            classes = re.search(r'\bclass="([^"]*)"', tag)
            if src:
                found.append((src.group(1), bool(classes) and "diagram" in classes.group(1).split()))
        for src, marked in found:
            if not re.match(r"^[a-z]+:", src):
                yield page.relative_to(DOCS).as_posix(), (page.parent / src).resolve(), marked


def svg_on_white(path):
    return bool(_WHITE_RECT.search(path.read_text(encoding="utf-8", errors="replace")))


def raster_on_white(path):
    """True when most of the image's border is near-white."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    border = ([pixels[x, 0] for x in range(0, width, 4)] + [pixels[x, height - 1] for x in range(0, width, 4)]
              + [pixels[0, y] for y in range(0, height, 4)] + [pixels[width - 1, y] for y in range(0, height, 4)])
    white = sum(1 for r, g, b in border if min(r, g, b) > 235)
    return white > len(border) / 2


class DocsFiguresTest(unittest.TestCase):
    def check(self, suffixes, on_white):
        unmarked = sorted({f"{page}: {path.name}" for page, path, marked in figures()
                           if path.suffix.lower() in suffixes and not marked and on_white(path)})
        self.assertEqual(unmarked, [], "figures on white need {.diagram} or class=\"diagram\"")

    def test_pages_have_figures_and_some_are_marked(self):
        found = list(figures())
        self.assertGreater(len(found), 100)
        self.assertTrue(any(marked for _, _, marked in found))
        for _, path, _ in found:
            self.assertTrue(path.is_file(), path)

    def test_svg_figures_on_white_are_marked(self):
        self.check({".svg"}, svg_on_white)

    @unittest.skipIf(Image is None, "Pillow is not installed")
    def test_raster_figures_on_white_are_marked(self):
        self.check({".png", ".jpg", ".jpeg", ".webp"}, raster_on_white)


if __name__ == "__main__":
    unittest.main()
