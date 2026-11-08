#!/usr/bin/env python3
"""Check the internal links of a built site: pages must exist, fragments must resolve.

VitePress's dead-link check covers links written in Markdown, but not links in
Vue components (the home page) and not the #fragment of any link. Run this on a
built site to catch both:

    npx vitepress build
    python3 scripts/check_links.py .vitepress/dist

For every HTML page under the site directory, each ``href`` that points into the
site (same-page ``#frag``, relative, or root-relative links) must name an existing
page, and a fragment must name an ``id`` (or ``<a name>``) on that page.

A link names a page unless it names an existing non-HTML file of the site or has
an asset extension (ASSET_EXTENSIONS). So a page with a dot in its name
(``/sections/changelog/v2.11``) and a leftover Markdown link (``Install.md``) are
checked as pages. A link without ``.html`` names ``<path>.html`` or
``<path>/index.html``, the files a web server serves for it. Pages are matched
by their site URL, case-sensitively, as on the web server.

External URLs (any scheme, or protocol-relative ``//``) are ignored, as are
fragments that are empty or only ``#``. A root-relative link outside the site
base (``--base``) is reported, because it leaves the site.

Options:
    --ignore PREFIX   skip links whose target path starts with PREFIX
                      (site path, e.g. /api/); may be repeated.
    --base PATH       the site base (VitePress ``base``), default "/".

Exits with status 1 and prints one line per broken link, 0 otherwise. Uses only the Python 3 standard library.
"""
from __future__ import annotations

import argparse
import html.parser
import os
import posixpath
import sys
import urllib.parse

# Links with these extensions name files, not pages, even when the file is missing.
ASSET_EXTENSIONS = frozenset({
    # Images, video and audio.
    ".apng", ".avif", ".bmp", ".gif", ".ico", ".jpeg", ".jpg", ".png", ".svg", ".webp",
    ".m4v", ".mov", ".mp4", ".ogg", ".ogv", ".webm", ".mp3", ".wav",
    # Fonts, scripts and styles.
    ".eot", ".otf", ".ttf", ".woff", ".woff2", ".css", ".js", ".map", ".mjs",
    # Data and downloads.
    ".7z", ".bz2", ".csv", ".gz", ".json", ".pdf", ".tar", ".tgz", ".txt", ".whl",
    ".xml", ".xz", ".zip",
})


class _PageParser(html.parser.HTMLParser):
    """Collects ids, <a name> targets and href values of one page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.hrefs: list[tuple[str, int]] = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value is None:
                continue
            if key == "id":
                self.ids.add(value)
            elif key == "name" and tag == "a":
                self.ids.add(value)
            elif key == "href" and tag in ("a", "area"):
                self.hrefs.append((value, self.getpos()[0]))

    handle_startendtag = handle_starttag


def _parse(path: str) -> _PageParser:
    parser = _PageParser()
    with open(path, encoding="utf-8", errors="replace") as f:
        parser.feed(f.read())
    parser.close()
    return parser


def _site_url(site: str, file_path: str) -> str:
    """The site URL of a file, with ``/`` separators on every platform."""
    return "/" + os.path.relpath(file_path, site).replace(os.sep, "/")


def _read_site(site: str) -> tuple[dict[str, _PageParser], set[str]]:
    """Returns the parsed pages and the URLs of all files, keyed by site URL."""
    pages: dict[str, _PageParser] = {}
    files: set[str] = set()
    for root, _dirs, names in os.walk(site):
        for name in names:
            path = os.path.join(root, name)
            url = _site_url(site, path)
            files.add(url)
            if name.endswith(".html"):
                pages[url] = _parse(path)
    return pages, files


def _link_url(base: str, page_url: str, href_path: str) -> str | None:
    """The site URL that a link path names; None if it leaves the site base."""
    if href_path.startswith("/"):
        if href_path + "/" == base:
            return "/"
        if not href_path.startswith(base):
            return None
        url = "/" + href_path[len(base):]
    else:
        url = posixpath.normpath(posixpath.join(posixpath.dirname(page_url), href_path))
        if href_path.endswith("/") and not url.endswith("/"):
            url += "/"
    return urllib.parse.unquote(url)


def _is_asset(url: str, files: set[str]) -> bool:
    if url.endswith(".html"):
        return False
    return url in files or posixpath.splitext(url)[1].lower() in ASSET_EXTENSIONS


def _page_candidates(url: str) -> list[str]:
    if url.endswith("/"):
        return [url + "index.html"]
    if url.endswith(".html"):
        return [url]
    return [url + ".html", url + "/index.html"]


def check(site: str, base: str, ignore: list[str]) -> tuple[list[str], int]:
    site = os.path.abspath(site)
    pages, files = _read_site(site)

    failures = []
    checked = 0
    for page_url, page in sorted(pages.items()):
        for href, line in page.hrefs:
            parts = urllib.parse.urlsplit(href)
            if parts.scheme or href.startswith("//"):
                continue
            fragment = urllib.parse.unquote(parts.fragment)
            if not fragment and not parts.path:
                continue
            target_url = page_url
            if parts.path:
                url = _link_url(base, page_url, parts.path)
                if url is None:
                    checked += 1
                    failures.append(f"{page_url}:{line}: {href} -> outside the site base {base}")
                    continue
                if _is_asset(url, files) or any(url.startswith(p) for p in ignore):
                    continue
                target_url = next((c for c in _page_candidates(url) if c in pages), url)
            checked += 1
            target_page = pages.get(target_url)
            if target_page is None:
                failures.append(f"{page_url}:{line}: {href} -> missing page {target_url}")
            elif fragment and fragment not in target_page.ids:
                failures.append(f"{page_url}:{line}: {href} -> no id '{fragment}' on {target_url}")
    return failures, checked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("site", help="built site directory, e.g. .vitepress/dist")
    ap.add_argument("--ignore", action="append", default=[], metavar="PREFIX")
    ap.add_argument("--base", default="/")
    args = ap.parse_args()
    base = "/" + args.base.strip("/") + "/" if args.base.strip("/") else "/"
    if not os.path.isdir(args.site):
        print(f"not a directory: {args.site}", file=sys.stderr)
        return 2
    failures, checked = check(args.site, base, args.ignore)
    for f in failures:
        print(f)
    if failures:
        print(f"{len(failures)} broken link(s)", file=sys.stderr)
        return 1
    print(f"all {checked} internal links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
