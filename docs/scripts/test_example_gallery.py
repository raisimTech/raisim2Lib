#!/usr/bin/env python3
"""Checks the example gallery: sections/examples/gallery.json against the example
pages and the clips in rsc/docs/video/examples.

Every example page appears in exactly one gallery category, and a category is
small enough to show all of its clips at once; every listed example has a page; an example has a clip and a poster unless it is in the "console" map
(no window); every clip belongs to a listed example; and the recording
settings in image_generators/example_clips/clips.json cover exactly the clips.

Run with: python3 -m unittest discover -s docs/scripts
"""

import json
import re
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
EXAMPLES = DOCS / "sections" / "examples"
CLIPS = DOCS.parent / "rsc" / "docs" / "video" / "examples"
RECORDING = DOCS / "image_generators" / "example_clips" / "clips.json"
# Settings record_example_clips.py reads.
RECORDING_KEYS = {"kind", "binary", "args", "viewer_args", "start", "seconds", "load", "fps", "every_frame",
                  "size", "no_minimize", "image", "crop"}
CROP_KEYS = {"trim", "dur", "zoom", "cx", "cy"}
# Pages under sections/examples that are not examples themselves.
NOT_EXAMPLES = {"overview"}
# A tile clip is a few seconds of 960x600 video; keep the page light.
MAX_CLIP_BYTES = 2 * 1024 * 1024
# The gallery shows one category at a time; Chrome ran out of memory on a page
# with every clip at once.
MAX_TILES_PER_CATEGORY = 12
# Category tabs sit two to a row on phones, an icon left of the title; a longer
# title is cut off there.
MAX_CATEGORY_TITLE = 10
ICONS = DOCS / ".vitepress" / "theme" / "exampleCategoryIcons.ts"
SIDEBAR = DOCS / ".vitepress" / "sidebar.mts"


class ExampleGalleryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gallery = json.loads((EXAMPLES / "gallery.json").read_text(encoding="utf-8"))
        cls.listed = [path for category in cls.gallery["categories"] for path in category["examples"]]

    def test_every_example_page_is_listed_once(self):
        self.assertEqual(len(self.listed), len(set(self.listed)), "an example is listed twice")
        pages = {page.relative_to(EXAMPLES).with_suffix("").as_posix() for page in EXAMPLES.rglob("*.md")}
        # "This page moved" stubs that only point elsewhere.
        stubs = {page for page in pages if page.startswith("current/")}
        self.assertEqual(sorted(pages - stubs - NOT_EXAMPLES - set(self.listed)), [], "pages missing from gallery.json")
        self.assertEqual(sorted(set(self.listed) - pages), [], "gallery.json lists pages that do not exist")

    def test_deformable_examples_have_their_own_category(self):
        categories = {category["id"]: category for category in self.gallery["categories"]}
        sources = DOCS.parent / "examples" / "src" / "server" / "deformable"
        expected = {f"server/{source.stem}" for source in sources.glob("*.cpp")}
        self.assertEqual(set(categories["deformable"]["examples"]), expected)
        self.assertIn("rayrai/rayrai_deformable_terrain", categories["terrain"]["examples"])

    def test_console_examples_are_listed(self):
        self.assertEqual(sorted(set(self.gallery["console"]) - set(self.listed)), [])

    def test_examples_with_a_window_have_a_clip_and_a_poster(self):
        missing = []
        for path in self.listed:
            name = path.rsplit("/", 1)[-1]
            has_media = (CLIPS / f"{name}.mp4").is_file() and (CLIPS / f"{name}.jpg").is_file()
            if path in self.gallery["console"]:
                self.assertFalse(has_media, f"{path} is a console example but has a clip")
            elif not has_media:
                missing.append(name)
        self.assertEqual(missing, [], "clips or posters missing in rsc/docs/video/examples")

    def test_every_clip_belongs_to_a_listed_example_and_is_small(self):
        names = {path.rsplit("/", 1)[-1] for path in self.listed}
        for clip in CLIPS.glob("*"):
            if clip.suffix in (".mp4", ".jpg"):
                self.assertIn(clip.stem, names, f"{clip.name} is not used by the gallery")
            if clip.suffix == ".mp4":
                self.assertLessEqual(clip.stat().st_size, MAX_CLIP_BYTES, clip.name)

    def test_every_clip_has_recording_settings(self):
        recording = json.loads(RECORDING.read_text(encoding="utf-8"))
        clips = {path.rsplit("/", 1)[-1] for path in self.listed if path not in self.gallery["console"]}
        self.assertEqual(sorted(clips - set(recording)), [], "examples missing from clips.json")
        self.assertEqual(sorted(set(recording) - clips), [], "clips.json records examples without a tile")
        for name, spec in recording.items():
            self.assertIn(spec.get("kind"), ("rayrai", "server", "comparison"), name)
            self.assertLessEqual(set(spec), RECORDING_KEYS, name)
            self.assertLessEqual(set(spec.get("crop", {})), CROP_KEYS, name)
            if spec["kind"] == "comparison":
                self.assertIn("image", spec, name)

    def test_categories_are_small_and_described(self):
        categories = self.gallery["categories"]
        self.assertTrue(categories)
        self.assertEqual(len({category["id"] for category in categories}), len(categories), "duplicate category id")
        for category in categories:
            self.assertTrue(category["title"] and category["description"], category["id"])
            self.assertTrue(category["examples"], category["id"])
            self.assertLessEqual(len(category["examples"]), MAX_TILES_PER_CATEGORY, category["id"])
            self.assertLessEqual(len(category["title"]), MAX_CATEGORY_TITLE, category["id"])

    def test_every_category_has_an_icon(self):
        icons = set(re.findall(r"^  (\w+): \[", ICONS.read_text(encoding="utf-8"), re.M))
        self.assertEqual(sorted({category["id"] for category in self.gallery["categories"]} - icons), [])

    def test_the_sidebar_lists_every_example_once_under_its_type(self):
        text = SIDEBAR.read_text(encoding="utf-8")
        entries = re.findall(r"\{ text: '((?:[^'\\]|\\.)*)', link: '/sections/examples/([^']+)\.md' \}", text)
        linked = [path for _, path in entries if path != "overview"]
        self.assertEqual(sorted(linked), sorted(self.listed), "sidebar and gallery.json list different examples")
        for label, path in entries:
            # The type is the subtree the entry sits in, not part of its label.
            self.assertNotRegex(label, r"(?i)\b(example|workflow):|^rayrai\b", path)

    def test_the_pages_show_one_gallery(self):
        for page in (DOCS / "sections" / "Examples.md", EXAMPLES / "overview.md"):
            text = page.read_text(encoding="utf-8")
            self.assertEqual(text.count("<ExampleGallery"), 1, page)
            self.assertEqual(text.count("<ExampleGallery />"), 1, f"{page}: the gallery takes no attributes")
            first_section = text.find("\n## ")
            self.assertTrue(first_section < 0 or text.index("<ExampleGallery") < first_section,
                            f"{page}: the gallery comes first, above the page's sections")


if __name__ == "__main__":
    unittest.main()
