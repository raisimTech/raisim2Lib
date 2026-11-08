#!/usr/bin/env python3
"""Checks the <PlatformTabs> blocks in the Markdown pages.

A block holds one named slot per platform (linux, macos, windows), each with
blank lines around its Markdown so VitePress renders it, and is closed. A slot
with another name would never show, and Markdown without the blank lines would
be printed as raw text.

Run with: python3 -m unittest discover -s docs/scripts
"""

import re
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
PLATFORMS = {"linux", "macos", "windows"}


class PlatformTabsTest(unittest.TestCase):
    def test_blocks_are_well_formed(self):
        used = 0
        for page in (DOCS / "sections").rglob("*.md"):
            text = page.read_text(encoding="utf-8")
            for block in re.findall(r"<PlatformTabs>(.*?)</PlatformTabs>", text, re.S):
                used += 1
                slots = re.findall(r"<template #(\w+)>\n\n.*?\n\n</template>", block, re.S)
                self.assertEqual(len(slots), block.count("<template"), f"{page.name}: a slot lacks blank lines")
                self.assertTrue(slots and set(slots) <= PLATFORMS, f"{page.name}: slots {slots}")
                self.assertEqual(len(slots), len(set(slots)), f"{page.name}: a platform appears twice")
            self.assertEqual(text.count("<PlatformTabs>"), text.count("</PlatformTabs>"), page.name)
        self.assertGreater(used, 0, "no page uses <PlatformTabs>")

    def test_the_component_is_registered(self):
        index = (DOCS / ".vitepress" / "theme" / "index.ts").read_text(encoding="utf-8")
        self.assertIn("app.component('PlatformTabs', PlatformTabs)", index)


if __name__ == "__main__":
    unittest.main()
