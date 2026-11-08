#!/usr/bin/env python3
"""Checks the Hall of Fame data (.vitepress/theme/data/papers.ts) and where it is shown.

Every paper records the evidence that it used RaiSim and links to the paper;
every Science Robotics paper known to use RaiSim stays listed; and the home page
and the Hall of Fame page both show the list.

Run with: python3 -m unittest discover -s docs/scripts
"""

import re
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
PAPERS = DOCS / ".vitepress" / "theme" / "data" / "papers.ts"
# Science Robotics papers whose results were obtained with RaiSim (2026-10-25).
SCIENCE_ROBOTICS = {
    "10.1126/scirobotics.aau5872",  # Hwangbo et al. 2019
    "10.1126/scirobotics.abc5986",  # Lee et al. 2020
    "10.1126/scirobotics.abk2822",  # Miki et al. 2022
    "10.1126/scirobotics.ade2256",  # Choi et al. 2023
    "10.1126/scirobotics.ads6192",  # Kim et al. 2025
}


class HallOfFameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = PAPERS.read_text(encoding="utf-8")
        cls.entries = re.findall(r"^  \{\n(.*?)^  \},", text, re.M | re.S)

    def field(self, entry, name):
        match = re.search(rf"^    {name}: (.+?),?$", entry, re.M)
        return match.group(1) if match else None

    def test_every_paper_is_complete(self):
        self.assertGreaterEqual(len(self.entries), len(SCIENCE_ROBOTICS) + 1)
        for entry in self.entries:
            title = self.field(entry, "title")
            for name in ("title", "authors", "venue", "year", "url", "highlight", "evidence", "kind"):
                self.assertIsNotNone(self.field(entry, name), f"{title}: no {name}")
            self.assertRegex(self.field(entry, "url"), r"^'https://", title)
            self.assertIn(self.field(entry, "kind").strip("'"), ("nature", "science-robotics", "other"), title)
            self.assertGreater(len(self.field(entry, "evidence")), 30, f"{title}: say where RaiSim use is stated")

    def test_every_science_robotics_paper_is_listed(self):
        urls = {self.field(entry, "url").strip("'").removeprefix("https://doi.org/") for entry in self.entries}
        self.assertEqual(sorted(SCIENCE_ROBOTICS - urls), [])

    def test_the_list_is_shown(self):
        theme = DOCS / ".vitepress" / "theme"
        self.assertIn("<HomeHallOfFame />", (theme / "components" / "home" / "HomePage.vue").read_text(encoding="utf-8"))
        self.assertIn("<PaperList />", (DOCS / "sections" / "HallOfFame.md").read_text(encoding="utf-8"))
        self.assertIn("app.component('PaperList', PaperList)", (theme / "index.ts").read_text(encoding="utf-8"))
        self.assertIn("/sections/HallOfFame.md", (DOCS / ".vitepress" / "sidebar.mts").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
