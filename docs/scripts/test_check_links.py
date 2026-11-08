#!/usr/bin/env python3
"""Tests for check_links.py on small hand-written sites.

Run with: python3 -m unittest discover -s docs/scripts
"""

import contextlib
import importlib.util
import io
import ntpath
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parent / "check_links.py"


def load_script():
    spec = importlib.util.spec_from_file_location("check_links", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckLinksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.links = load_script()

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.site = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def page(self, path, body):
        file = self.site / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")

    def failures(self, ignore=()):
        failures, _ = self.links.check(str(self.site), "/", list(ignore))
        return failures

    def test_valid_links_pass(self):
        self.page("index.html", '<a href="/sections/A.html">A</a> <a href="sections/A.html#install">x</a>'
                                ' <a href="/api/">api</a> <a href="#top">top</a> <h1 id="top">t</h1>')
        self.page("sections/A.html", '<h2 id="install">i</h2> <a href="../index.html">home</a>'
                                     ' <a href="B.html#run">b</a>')
        self.page("sections/B.html", '<a name="run"></a>')
        self.page("api/index.html", "api")
        self.assertEqual(self.failures(), [])

    def test_missing_page_without_fragment_fails(self):
        # Links in Vue components (the home page) are not checked by VitePress.
        self.page("index.html", '<a href="/sections/Missing.html">gone</a>')
        failures = self.failures()
        self.assertEqual(len(failures), 1)
        self.assertIn("missing page /sections/Missing.html", failures[0])

    def test_missing_fragment_fails(self):
        self.page("index.html", '<a href="/sections/A.html#gone">x</a> <a href="#nowhere">y</a>')
        self.page("sections/A.html", '<h2 id="there">t</h2>')
        failures = self.failures()
        self.assertEqual(len(failures), 2)
        self.assertIn("no id 'gone' on /sections/A.html", failures[0])
        self.assertIn("no id 'nowhere' on /index.html", failures[1])

    def test_external_links_assets_and_ignored_prefixes_are_skipped(self):
        self.page("index.html",
                  '<a href="https://github.com/raisimTech/raisim2Lib#readme">gh</a>'
                  ' <a href="//cdn.example.com/x.html">cdn</a> <a href="/benchmark_results.json">json</a>'
                  ' <a href="/assets/clip.mp4">mp4</a> <a href="/api/raisim/World.html#gone">api</a>'
                  ' <a href="mailto:raisim@raionrobotics.com">mail</a>')
        self.assertEqual(self.failures(ignore=["/api/"]), [])

    def test_directory_and_extensionless_links_resolve(self):
        self.page("index.html", '<a href="/api/">api</a> <a href="/sections/A">a</a>')
        self.page("api/index.html", "api")
        self.page("sections/A.html", "a")
        self.assertEqual(self.failures(), [])

    def test_dotted_page_names_and_markdown_links_are_pages(self):
        # Only existing non-HTML files and asset extensions are assets: a page
        # named v2.11 and a Markdown link that VitePress left alone are pages.
        self.page("index.html",
                  '<a href="/sections/changelog/v2.11">ok</a>'
                  ' <a href="sections/changelog/v2.11#gone">frag</a>'
                  ' <a href="/sections/changelog/v2.12">missing</a>'
                  ' <a href="sections/Install.md">md</a>'
                  ' <a href="/downloads/LICENSE">file</a> <a href="/img/later.png">asset</a>')
        self.page("sections/changelog/v2.11.html", '<h2 id="fix">f</h2>')
        (self.site / "downloads").mkdir()
        (self.site / "downloads" / "LICENSE").write_text("license", encoding="utf-8")
        self.assertEqual(self.failures(), [
            "/index.html:1: sections/changelog/v2.11#gone -> no id 'gone' on /sections/changelog/v2.11.html",
            "/index.html:1: /sections/changelog/v2.12 -> missing page /sections/changelog/v2.12",
            "/index.html:1: sections/Install.md -> missing page /sections/Install.md",
        ])

    def test_root_relative_links_outside_the_base_are_reported(self):
        self.page("index.html",
                  '<a href="/raisim/sections/A.html#a">in</a> <a href="/raisim">root</a>'
                  ' <a href="/other/B.html">out</a> <a href="/">top</a> <a href="sections/A.html">rel</a>')
        self.page("sections/A.html", '<h2 id="a">a</h2>')
        failures, checked = self.links.check(str(self.site), "/raisim/", [])
        self.assertEqual(failures, [
            "/index.html:1: /other/B.html -> outside the site base /raisim/",
            "/index.html:1: / -> outside the site base /raisim/",
        ])
        self.assertEqual(checked, 5)

    def test_windows_paths_are_matched_by_site_url(self):
        # On Windows os.walk yields backslash paths. Pages must be matched by
        # their site URL, not by a joined file path with mixed separators.
        tree = {
            "C:\\site": (["api", "sections"], ["index.html"]),
            "C:\\site\\api": (["raisim"], []),
            "C:\\site\\api\\raisim": ([], ["World.html"]),
            "C:\\site\\sections": ([], ["A.html", "v2.11.html"]),
        }
        pages = {
            "C:\\site\\index.html":
                '<a href="/api/raisim/World.html#load">w</a> <a href="sections/A.html">a</a>'
                ' <a href="/sections/Gone.html">gone</a>',
            "C:\\site\\api\\raisim\\World.html":
                '<h3 id="load">l</h3> <a href="../../sections/v2.11">v</a>',
            "C:\\site\\sections\\A.html": '<a href="../api/raisim/World.html">w</a>',
            "C:\\site\\sections\\v2.11.html": "v",
        }

        def parse(path):
            parser = self.links._PageParser()
            parser.feed(pages[path])
            return parser

        windows_os = types.SimpleNamespace(
            path=ntpath, sep="\\", walk=lambda top: [(root, *entries) for root, entries in tree.items()])
        with mock.patch.object(self.links, "os", windows_os), mock.patch.object(self.links, "_parse", parse):
            failures, checked = self.links.check("C:\\site", "/", [])
        self.assertEqual(failures, ["/index.html:1: /sections/Gone.html -> missing page /sections/Gone.html"])
        self.assertEqual(checked, 5)

    def test_main_exit_status(self):
        self.page("index.html", '<a href="/sections/Missing.html">gone</a>')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            argv = sys.argv
            try:
                sys.argv = ["check_links.py", str(self.site)]
                self.assertEqual(self.links.main(), 1)
                self.page("sections/Missing.html", "here now")
                self.assertEqual(self.links.main(), 0)
            finally:
                sys.argv = argv


if __name__ == "__main__":
    unittest.main()
