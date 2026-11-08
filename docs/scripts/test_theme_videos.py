#!/usr/bin/env python3
"""Checks that the theme's lazily played clips give up their media when they
leave the screen.

A paused <video> keeps its decoder and buffered frames. LazyVideo held on to
every clip it had played, so scrolling through the example gallery (65 clips)
or the home page took Chrome to several gigabytes and crashed the tab. A clip
off screen must remove its src and call load(), which empties the element, and
must do the same when it unmounts (a gallery category switch).

Run with: python3 -m unittest discover -s docs/scripts
"""

import re
import unittest
from pathlib import Path

LAZY_VIDEO = Path(__file__).resolve().parent.parent / ".vitepress" / "theme" / "components" / "LazyVideo.vue"


class LazyVideoReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = LAZY_VIDEO.read_text(encoding="utf-8")

    def test_release_empties_the_element(self):
        release = re.search(r"function release\(element: HTMLVideoElement\): void \{(.*?)\n\}", self.text, re.S)
        self.assertIsNotNone(release, "LazyVideo has no release(element) function")
        body = release.group(1)
        self.assertIn("element.removeAttribute('src')", body)
        self.assertIn("element.load()", body)
        self.assertLess(body.index("removeAttribute('src')"), body.index("element.load()"),
                        "load() must run after the source is removed")

    def test_clips_are_released_off_screen_and_on_unmount(self):
        observer = re.search(r"new IntersectionObserver\((.*?)\n    \{ rootMargin", self.text, re.S)
        self.assertIsNotNone(observer)
        self.assertRegex(observer.group(1), r"else \{\s*release\(element\)")
        unmount = re.search(r"onBeforeUnmount\(\(\) => \{(.*?)\n\}\)", self.text, re.S)
        self.assertIsNotNone(unmount)
        self.assertIn("release(", unmount.group(1))

    def test_the_template_does_not_bind_the_source(self):
        # A reactive :src would set src="" on release, which starts a failing load.
        template = self.text[self.text.index("<template>"):]
        self.assertNotRegex(template, r"(:|v-bind:)src=")


if __name__ == "__main__":
    unittest.main()
