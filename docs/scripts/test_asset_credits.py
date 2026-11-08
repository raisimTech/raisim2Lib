#!/usr/bin/env python3
"""Checks that the site obeys the licenses of the third-party assets it shows.

Renders of CC BY 4.0 models (the city cars, the off-road buggy and its venue
props, the warehouse models, the YCB objects, MetalRoughSpheres) and the TCP
viewer's Flaticon icons must credit their authors: every page that embeds such
a render links to sections/AssetCredits.md, and so do the example gallery, the
home page and the README. DamagedHelmet (CC BY-NC 4.0, non-commercial only) and
AntiqueCamera (UX3D trademark) must not appear in anything the site publishes.

Run with: python3 -m unittest discover -s docs/scripts
"""

import re
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
LIB = DOCS.parent
SECTIONS = DOCS / "sections"
CREDITS = SECTIONS / "AssetCredits.md"
# Media files whose pictures show credited assets.
CREDITED_MEDIA = re.compile(
    r"rsc/docs/(image|video)/[^ )\"']*(city|offroad_buggy|deformable_terrain_(car|loading|suspension)|warehouse|ycb|"
    r"complete_showcase|visual_asset_support|coacd|pbr_material_grid|tcp_viewer)"
)
# Models whose renders cannot be published on a commercial product's site.
UNPUBLISHABLE = re.compile(r"DamagedHelmet|AntiqueCamera|07_pbr_material_maps")


class AssetCreditsTest(unittest.TestCase):
    def test_pages_with_credited_renders_link_the_credits(self):
        missing = []
        for page in SECTIONS.rglob("*.md"):
            text = page.read_text(encoding="utf-8")
            if page != CREDITS and CREDITED_MEDIA.search(text) and "AssetCredits.md" not in text:
                missing.append(page.relative_to(DOCS).as_posix())
        self.assertEqual(sorted(missing), [], "pages that show CC BY assets without crediting them")

    def test_the_gallery_home_page_and_readme_link_the_credits(self):
        theme = DOCS / ".vitepress" / "theme" / "components"
        for path in (theme / "ExampleGallery.vue", theme / "home" / "HomeFooter.vue"):
            self.assertIn("/sections/AssetCredits.html", path.read_text(encoding="utf-8"), path.name)
        self.assertIn("docs/sections/AssetCredits.md", (LIB / "README.md").read_text(encoding="utf-8"))
        self.assertIn("AssetCredits.md", (DOCS / ".vitepress" / "sidebar.mts").read_text(encoding="utf-8"))

    def test_the_credits_name_every_cc_by_author(self):
        text = CREDITS.read_text(encoding="utf-8")
        for notice in ("city", "offroad_buggy", "warehouse"):
            body = (LIB / "rsc" / notice / "ATTRIBUTION.md").read_text(encoding="utf-8")
            sketchfab = body[body.index("## Sketchfab"):]
            for author in re.findall(r'\) by ([^(]+?) \(https://sketchfab\.com/', sketchfab):
                self.assertIn(author.strip(), text, f"{notice}: {author}")
        for name in ("YCB Object and Model Set", "Analytical Graphics", "UIcons by Flaticon"):
            self.assertIn(name, text)

    def test_cc_by_asset_folders_keep_their_notice(self):
        for folder in ("city", "offroad_buggy", "warehouse", "ycb"):
            self.assertTrue((LIB / "rsc" / folder / "ATTRIBUTION.md").is_file(), folder)

    def test_nothing_published_shows_unlicensed_models(self):
        sources = [*(LIB / "examples" / "src").rglob("*.cpp"), *(DOCS / "image_generators").glob("*.cpp")]
        pages = [*SECTIONS.rglob("*.md"), LIB / "README.md"]
        found = []
        for path in sources:
            # Source comments may name the models to explain why they are left out.
            code = re.sub(r"//[^\n]*", "", path.read_text(encoding="utf-8", errors="ignore"))
            if UNPUBLISHABLE.search(code):
                found.append(path.relative_to(LIB).as_posix())
        for page in pages:
            for line in page.read_text(encoding="utf-8").splitlines():
                if UNPUBLISHABLE.search(line) and ("](" in line or "src=" in line):
                    found.append(page.relative_to(LIB).as_posix())
        self.assertEqual(sorted(set(found)), [])


if __name__ == "__main__":
    unittest.main()
