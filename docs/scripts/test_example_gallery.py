#!/usr/bin/env python3
"""Checks the example gallery: sections/examples/gallery.json against the example
pages and the clips in rsc/docs/video/examples.

Every example page appears in exactly one gallery category, and a category is
small enough to show all of its clips at once; every listed example has a page; an example has a clip and a poster unless it is in the "console" map
(no window); every clip belongs to a listed example; and the recording
settings in image_generators/example_clips/clips.json cover exactly the clips.

Run with: python3 -m unittest discover -s docs/scripts
"""

import importlib.util
import json
import os
import re
import unittest
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

DOCS = Path(__file__).resolve().parent.parent
EXAMPLES = DOCS / "sections" / "examples"
CLIPS = DOCS.parent / "rsc" / "docs" / "video" / "examples"
RECORDING = DOCS / "image_generators" / "example_clips" / "clips.json"
# Settings record_example_clips.py reads.
RECORDING_KEYS = {"kind", "binary", "args", "viewer_args", "start", "seconds", "load", "fps", "every_frame",
                  "size", "no_minimize", "image", "crop", "keys"}
CROP_KEYS = {"trim", "dur", "zoom", "cx", "cy"}
# Pages under sections/examples that are not examples themselves.
NOT_EXAMPLES = {"overview"}
# A tile clip is a few seconds of 960x600 video; keep the page light.
MAX_CLIP_BYTES = 3 * 1024 * 1024
# The gallery shows one category at a time; Chrome ran out of memory on a page
# with every clip at once.
MAX_TILES_PER_CATEGORY = 12
# Category tabs sit two to a row on phones, an icon left of the title; a longer
# title is cut off there.
MAX_CATEGORY_TITLE = 10
ICONS = DOCS / ".vitepress" / "theme" / "exampleCategoryIcons.ts"
SIDEBAR = DOCS / ".vitepress" / "sidebar.mts"


def is_held_key(held):
    """A clips.json "keys" entry: [SDL key name, from, to] in seconds, which framecap reads as NAME@FROM-TO."""
    return (isinstance(held, list) and len(held) == 3 and isinstance(held[0], str) and bool(re.fullmatch(r"[^@,]+", held[0]))
            and all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in held[1:]) and 0 <= held[1] < held[2])


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

    def test_every_rayrai_example_target_has_a_page(self):
        cmake = (DOCS.parent / "examples" / "CMakeLists.txt").read_text(encoding="utf-8")
        targets = set(re.findall(r"create_rayrai_executable\((\w+)\s", cmake))
        pages = {page.stem for page in EXAMPLES.rglob("*.md")}
        # Documented elsewhere: the tendon examples page, the Blender import page and a debug tool.
        elsewhere = {"rayrai_tendons", "rayrai_blue_wall_scene", "rayrai_heightmap_replacement"}
        self.assertEqual(sorted(targets - pages - elsewhere), [], "rayrai example targets without a page")

    def test_gallery_keys_accept_vitepress_relative_and_absolute_paths(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is needed to check the VitePress data loader")
        module = (DOCS / ".vitepress/theme/data/examplePaths.mjs").as_uri()
        script = ("import {examplePageKey} from " + json.dumps(module) + ";"
                  "import assert from 'node:assert/strict';"
                  "for(const path of ['sections/examples/rayrai/demo.md',"
                  "'/tmp/site/sections/examples/rayrai/demo.md',"
                  "'C:/site/sections/examples/rayrai/demo.md'])"
                  "assert.equal(examplePageKey(path),'rayrai/demo');")
        subprocess.run([node, "--input-type=module", "-e", script], check=True)

    def test_deformable_examples_have_their_own_category(self):
        categories = {category["id"]: category for category in self.gallery["categories"]}
        sources = DOCS.parent / "examples" / "src" / "server" / "deformable"
        expected = {f"server/{source.stem}" for source in sources.glob("*.cpp")}
        expected.add("rayrai/rayrai_fem_workbench")
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
            for held in spec.get("keys", []):
                self.assertTrue(is_held_key(held), f"{name}: keys entries are [SDL key name, from, to] with 0 <= from < to: {held}")

    def test_doc_images_never_inherit_the_shared_capture(self):
        documented = set(re.findall(r"RAYRAI_EXAMPLE_(CAPTURE\w*)", (DOCS / "sections/rayrai/Capture.md").read_text(encoding="utf-8")))
        runner = (DOCS / "image_generators/run_doc_image_generator.cmake").read_text(encoding="utf-8")
        unset = set(re.search(r"foreach\(_capture ([^)]*)\)", runner).group(1).split())
        self.assertTrue(documented)
        self.assertLessEqual(documented, unset, "run_doc_image_generator.cmake lets the shell's capture variables through")

    def test_held_keys_reach_framecap_as_name_at_from_to(self):
        spec = importlib.util.spec_from_file_location("record_example_clips", RECORDING.with_name("record_example_clips.py"))
        recorder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recorder)
        self.assertEqual(recorder.key_script([["W", 20, 40], ["Left Shift", 25.5, 26]]), "W@20-40,Left Shift@25.5-26")
        self.assertEqual(recorder.key_script([]), "")
        buggy = {"start": 25, "seconds": 10, "crop": {"dur": 6.0, "trim": 1.0}, "keys": [["W", 20, 40], ["A", 2, 5]]}
        self.assertEqual(recorder.timing(buggy), (25, 10, buggy["keys"]))
        # Capped at 10 s: the 7 s the tile uses, recorded from 3 s, with W still held from 5 s before.
        self.assertEqual(recorder.timing(buggy, 10), (3, 7.0, [["W", 0, 18]]))
        self.assertEqual(recorder.timing({"start": 2, "seconds": 8}, 10), (2, 6.0, []))
        launched = []

        class Finished:
            pid = 0

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

        recorder.start = lambda command, cwd, env, log: launched.append(env) or Finished()
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            recorder.record_window("buggy", {"kind": "rayrai", **buggy}, work, work / "raw.mp4", work / "hook.so", work, 10)
        self.assertEqual([(env["FRAMECAP_START"], env["FRAMECAP_SECONDS"], env["FRAMECAP_KEYS"]) for env in launched],
                         [("3.0", "7.0", "W@0-18.0")])
        for name, spec in json.loads(RECORDING.read_text(encoding="utf-8")).items():
            start, seconds, _ = recorder.timing(spec, 10)
            if spec["kind"] != "comparison":
                self.assertLessEqual(start + seconds, max(10, seconds), name)
        self.assertTrue(is_held_key(["W", 20, 40]) and is_held_key(["Left Shift", 0, 0.5]))
        for malformed in (["W", 40, 20], ["W@x", 1, 2], ["W,A", 1, 2], ["W", "1", 2], ["W", 1], "W@1-2"):
            self.assertFalse(is_held_key(malformed), malformed)
        os.environ["RAYRAI_EXAMPLE_CAPTURE"] = "inherited.png"
        try:
            self.assertNotIn("RAYRAI_EXAMPLE_CAPTURE", recorder.environment({}))
        finally:
            del os.environ["RAYRAI_EXAMPLE_CAPTURE"]

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

    def test_runtime_joints_is_linked_before_category_selection(self):
        target = (EXAMPLES / "worlds/runtime_joints.md").resolve()
        for page in (DOCS / "sections/Examples.md", EXAMPLES / "overview.md"):
            introduction = page.read_text(encoding="utf-8").split("<ExampleGallery", 1)[0]
            links = re.findall(r"\[Runtime Joints\]\(([^)]+)\)", introduction)
            self.assertTrue(links, f"{page}: Runtime Joints must be visible outside the filtered gallery")
            self.assertIn(target, [(page.parent / link).resolve() for link in links])

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
