#!/usr/bin/env python3
"""Checks that the theme's styles never style <html> through a page flag by accident.

The site marks <html> with flag classes (rs-js, rs-ready, rs-home-route) from the
inline script in .vitepress/config.mts and from theme code. A flag is meant to be
used as an ancestor, as in `.rs-js .rs-reveal` or `html.rs-home-route .VPNav`. A
rule whose subject carries a flag class also matches <html> itself. That is how
the home page lost scrolling: its wrapper class was also the <html> flag, so the
wrapper's `overflow: hidden` applied to <html>. Styling <html> on purpose must
say so, by writing the selector as `html.<flag>`.

Run with: python3 -m unittest discover -s docs/scripts
"""

import re
import unittest
from pathlib import Path

VITEPRESS = Path(__file__).resolve().parent.parent / ".vitepress"

_FLAG_RE = re.compile(r"""(?:documentElement|\broot)\.classList\.(?:add|toggle)\(\s*['"]([\w-]+)['"]""")
_STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S)
_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)


def flag_classes():
    """Classes that the inline head script or theme code put on <html>."""
    flags = set()
    for path in [VITEPRESS / "config.mts", *VITEPRESS.glob("theme/**/*.ts"), *VITEPRESS.glob("theme/**/*.vue")]:
        flags.update(_FLAG_RE.findall(path.read_text(encoding="utf-8")))
    return flags


def stylesheets():
    """(name, css) for every stylesheet and <style> block of the theme."""
    for path in sorted(VITEPRESS.glob("theme/**/*.css")):
        yield path.relative_to(VITEPRESS).as_posix(), path.read_text(encoding="utf-8")
    for path in sorted(VITEPRESS.glob("theme/**/*.vue")):
        for block in _STYLE_RE.findall(path.read_text(encoding="utf-8")):
            yield path.relative_to(VITEPRESS).as_posix(), block


def selectors(css):
    """Every selector of every style rule, skipping at-rule preludes."""
    css = _COMMENT_RE.sub("", css)
    for prelude in re.findall(r"([^{}]+)\{", css):
        prelude = prelude.strip()
        if not prelude or prelude.startswith("@"):
            continue
        yield from split_top_level(prelude, ",")


def split_top_level(text, separators):
    """Splits text at separator characters that are outside parentheses."""
    parts, depth, current = [], 0, ""
    for char in text:
        depth += char == "("
        depth -= char == ")"
        if depth == 0 and char in separators:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def subject(selector):
    """The last compound selector, the one that names the styled element."""
    return split_top_level(selector, " >+~\n\t")[-1]


def styles_flag_holder(selector, flags):
    compound = subject(selector)
    while re.search(r"\([^()]*\)", compound):
        compound = re.sub(r"\([^()]*\)", "", compound)
    classes = set(re.findall(r"\.([\w-]+)", compound))
    return bool(classes & flags) and not compound.startswith("html")


class ThemeCssTest(unittest.TestCase):
    def test_the_detector_sees_the_flags_and_the_bug_it_guards_against(self):
        self.assertTrue({"rs-js", "rs-ready", "rs-home-route"} <= flag_classes())
        flags = {"rs-home"}
        self.assertTrue(styles_flag_holder(".rs-home", flags))
        self.assertTrue(styles_flag_holder(":root .rs-home:not(.x)", flags))
        self.assertFalse(styles_flag_holder(".rs-home .rs-grad-text", flags))
        self.assertFalse(styles_flag_holder("html.rs-home .VPNav", flags))
        self.assertFalse(styles_flag_holder("html.rs-home", flags))
        self.assertFalse(styles_flag_holder(".x:not(.rs-home)", flags))
        self.assertEqual(list(selectors("@media (max-width: 9px) { .a, .b > .c { x: y } }")), [".a", ".b > .c"])

    def test_no_rule_styles_html_through_a_flag_class(self):
        flags = flag_classes()
        offenders = [f"{name}: {selector}"
                     for name, css in stylesheets()
                     for selector in selectors(css)
                     if styles_flag_holder(selector, flags)]
        self.assertEqual(offenders, [], f"these rules also match <html> (flags: {sorted(flags)})")


if __name__ == "__main__":
    unittest.main()
