#!/usr/bin/env python3
"""Checks the picture of the card a shared link shows on Facebook and X.

Every page used to show the site's picture, so a link to an example showed the RaiSim
banner instead of the example. A page now shows the first picture of its text that the
networks can show (.vitepress/socialImage.mjs), with its size, which Facebook needs to
show the picture on the first share of a link.

Run with: python3 -m unittest discover -s docs/scripts
"""

import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
MODULE = DOCS / ".vitepress" / "socialImage.mjs"


def png(width, height):
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\0\0\0"


def gif(width, height, padding=0):
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\0" * (4 + padding)


def jpeg(width, height):
    exif = b"Exif\0\0" + b"\0" * 300
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\0" * 9
    return b"\xff\xd8" + app1 + b"\xff\xdb\x00\x04\0\0" + sof + b"\xff\xd9"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class SocialImageTest(unittest.TestCase):
    def pick(self, content, files):
        with tempfile.TemporaryDirectory(prefix="raisim-social-image-") as directory:
            root = Path(directory)
            for name, data in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(data)
            script = ("import { firstContentImage } from " + json.dumps(MODULE.as_uri()) + ";\n"
                      "const [content, outDir] = JSON.parse(process.argv[1]);\n"
                      "console.log(JSON.stringify(firstContentImage(content, outDir, '/') ?? null));\n")
            run = subprocess.run(["node", "--input-type=module",
                                  "-e", script, json.dumps([content, str(root)])],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            return json.loads(run.stdout)

    def test_first_picture_of_the_text_that_a_card_can_show(self):
        content = ('<header><img class="VPImage logo" src="/logo-mark.svg"><img src="/nav.png"></header>'
                   '<div style="position:relative;" class="vp-doc _sections_examples_tank">'
                   '<h1>Tank</h1><img src="/assets/diagram.Ab1.svg" alt="diagram">'
                   '<img src="/assets/icon.Cd2.png" alt="icon">'
                   '<img src="https://example.com/remote.jpg" alt="remote">'
                   '<img src="/assets/huge.Ef3.gif" alt="huge">'
                   '<img src="/assets/tank%20survival.Gh4.jpg" alt="A Leopard 2 at the enemy&#39;s &amp; river">'
                   '<img src="/assets/later.Ij5.png" alt="later"></div>')
        files = {"nav.png": png(1200, 630), "assets/diagram.Ab1.svg": b"<svg/>",
                 "assets/icon.Cd2.png": png(64, 64), "assets/huge.Ef3.gif": gif(1200, 630, 5 * 1024 * 1024),
                 "assets/tank survival.Gh4.jpg": jpeg(1600, 900), "assets/later.Ij5.png": png(1280, 720)}
        self.assertEqual(self.pick(content, files),
                         {"path": "/assets/tank%20survival.Gh4.jpg", "width": 1600, "height": 900,
                          "type": "image/jpeg", "alt": "A Leopard 2 at the enemy's & river"})

    def test_png_and_gif_sizes_are_read(self):
        content = '<div class="vp-doc"><img src="/a.gif" alt=""><img src="/b.png" alt="b"></div>'
        self.assertEqual(self.pick(content, {"a.gif": gif(640, 360), "b.png": png(800, 600)}),
                         {"path": "/a.gif", "width": 640, "height": 360, "type": "image/gif", "alt": ""})
        self.assertEqual(self.pick(content, {"a.gif": gif(100, 100), "b.png": png(800, 600)})["path"], "/b.png")

    def test_pages_without_text_or_a_suitable_picture_keep_the_site_picture(self):
        self.assertIsNone(self.pick('<div class="VPHome"><img src="/hero.png"></div>', {"hero.png": png(1920, 1080)}))
        self.assertIsNone(self.pick('<div class="vp-doc"><img src="/missing.png"></div>', {}))


if __name__ == "__main__":
    unittest.main()
