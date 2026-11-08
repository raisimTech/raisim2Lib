#!/usr/bin/env python3
"""Generate the TCP viewer SVG diagrams in the Read the Docs theme's visual language.

Standalone: needs only python3, Pillow and the Lato faces the theme itself uses
(``fonts-lato`` on Debian/Ubuntu). It is not part of the CMake image pipeline,
which builds the rendered PNGs.

    python3 docs/image_generators/tcp_viewer_diagrams.py

Every box is sized from measured text rather than guessed, so no label can
overflow its frame, and the gaps between boxes are sized from the connector
captions that have to fit in them. Arrowheads are declared with
``markerUnits="userSpaceOnUse"``: the default is stroke-width units, which
multiplies the marker by the line width and is what made the previous
arrowheads large enough to cover the text behind them.

Diagrams are kept narrow on purpose. The documentation site scales a figure to
the content column, so every extra column of boxes shrinks the type the reader
actually sees.
"""
import os
from PIL import ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "rsc", "docs", "image", "rayrai", "tcp_viewer")
LATO = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"
LATO_BOLD = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Light diagram palette (originally the Sphinx theme's colors)
INK = "#404040"        # body text
MUTED = "#666666"      # secondary text
ACCENT = "#2980b9"     # links / primary accent
ACCENT_SOFT = "#6ab0de"  # admonition header
NOTE_BG = "#e7f2fa"    # note admonition body
RULE = "#e1e4e5"       # borders and rules
PAGE = "#fcfcfc"       # content background
PANEL = "#ffffff"
CODE_INK = "#e74c3c"   # inline literal
WARN = "#f0b37e"       # warning admonition border
WARN_BG = "#ffedcc"
GREEN = "#1abc9c"

FONT_STACK = "Lato,'Helvetica Neue',Arial,sans-serif"
MONO_STACK = "SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono',monospace"

_cache = {}


def font(path, size):
    key = (path, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(path, size)
    return _cache[key]


def width(text, size, bold=False, mono=False):
    face = MONO if mono else (LATO_BOLD if bold else LATO)
    return font(face, size).getlength(text)


def esc(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text_el(x, y, content, size=14, bold=False, mono=False, fill=INK, anchor="start"):
    stack = MONO_STACK if mono else FONT_STACK
    weight = ' font-weight="700"' if bold else ""
    anchor_attr = ' text-anchor="%s"' % anchor if anchor != "start" else ""
    return ('<text x="%g" y="%g" font-family="%s" font-size="%g" fill="%s"%s%s>%s</text>'
            % (x, y, stack, size, fill, weight, anchor_attr, esc(content)))


def rounded(x, y, w, h, fill=PANEL, stroke=RULE, sw=1, rx=4):
    return ('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s" stroke="%s" '
            'stroke-width="%g"/>' % (x, y, w, h, rx, fill, stroke, sw))


def markers():
    """Small arrowheads in user space: independent of the line's stroke width."""
    out = ['<defs>']
    for name, colour in (("aAccent", ACCENT), ("aGreen", GREEN), ("aMuted", MUTED),
                         ("aWarn", "#d9822b")):
        out.append('<marker id="%s" markerUnits="userSpaceOnUse" markerWidth="9" '
                   'markerHeight="8" refX="8.5" refY="4" orient="auto">'
                   '<path d="M0,0 L9,4 L0,8 z" fill="%s"/></marker>' % (name, colour))
    out.append('</defs>')
    return "".join(out)


class Box:
    """A titled card whose width is the widest measured line plus padding."""

    PAD_X = 14
    PAD_TOP = 26
    LINE = 19

    def __init__(self, title, lines, accent=ACCENT, chip=None):
        self.title = title
        self.lines = lines
        self.accent = accent
        self.chip = chip
        self.title_size = 15
        self.line_size = 13
        self.chip_size = 12
        needed = [width(title, self.title_size, bold=True)]
        needed += [width("•  " + line, self.line_size) for line in lines]
        if chip:
            needed.append(width(chip, self.chip_size, mono=True) + 16)
        self.w = max(needed) + 2 * self.PAD_X
        self.h = self.PAD_TOP + 10 + self.LINE * len(lines) + (24 if chip else 0) + 12

    def render(self, x, y):
        parts = [rounded(x, y, self.w, self.h)]
        # accent rule along the top edge, the way the theme marks a heading
        parts.append('<path d="M%g %g H%g" stroke="%s" stroke-width="3" stroke-linecap="round"/>'
                     % (x + 1, y + 1.5, x + self.w - 1, self.accent))
        parts.append(text_el(x + self.PAD_X, y + self.PAD_TOP, self.title,
                             size=self.title_size, bold=True))
        cursor = y + self.PAD_TOP + 24
        for line in self.lines:
            parts.append(text_el(x + self.PAD_X, cursor, "•", size=self.line_size, fill=MUTED))
            parts.append(text_el(x + self.PAD_X + 12, cursor, line, size=self.line_size))
            cursor += self.LINE
        if self.chip:
            chip_w = width(self.chip, self.chip_size, mono=True) + 16
            parts.append(rounded(x + self.PAD_X, cursor - 4, chip_w, 20,
                                 fill="#f8f8f8", stroke=RULE, rx=3))
            parts.append(text_el(x + self.PAD_X + 8, cursor + 10, self.chip,
                                 size=self.chip_size, mono=True, fill=CODE_INK))
        return "".join(parts)


def arrow(x1, y1, x2, y2, colour=ACCENT, marker="aAccent", dashed=False, sw=1.6):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return ('<path d="M%g %g L%g %g" fill="none" stroke="%s" stroke-width="%g"%s '
            'marker-end="url(#%s)"/>' % (x1, y1, x2, y2, colour, sw, dash, marker))


def label(cx, y, content, colour=MUTED, size=12, bold=False):
    """A caption on a connector, on a page-coloured plate so the line cannot cross it."""
    w = width(content, size, bold=bold) + 10
    return (rounded(cx - w / 2, y - size + 1, w, size + 7, fill=PAGE, stroke="none", sw=0, rx=2)
            + text_el(cx, y, content, size=size, fill=colour, anchor="middle", bold=bold))


def svg(width_px, height_px, body, title, desc):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%g" height="%g" '
            'viewBox="0 0 %g %g" role="img" aria-labelledby="t d">\n'
            '<title id="t">%s</title>\n<desc id="d">%s</desc>\n'
            '%s\n<rect width="%g" height="%g" fill="%s"/>\n%s\n</svg>\n'
            % (width_px, height_px, width_px, height_px, esc(title), esc(desc),
               markers(), width_px, height_px, PAGE, body))


# ---------------------------------------------------------------- data flow --
def data_flow():
    """Server and viewer only.

    A figure is scaled to the content column, so every extra column shrinks the
    type. The operator's side of the story is told in prose under `Sim control
    workflow`, and leaving it out here buys the two boxes that matter legible
    text at the width the theme actually renders.
    """
    server = Box("RaisimServer", [
        "Owns the World and its clock",
        "Serializes scene and contacts",
        "Applies pause, step, force, pose",
        "Validates returned camera data",
    ], accent=GREEN, chip="binds 127.0.0.1:8080")

    viewer = Box("rayrai TCP viewer", [
        "Renders the remote scene",
        "Connection / Object / Render tabs",
        "Records and replays sessions",
        "Renders RGB/depth cameras",
        "Drives pause, step, force, pose",
    ], accent=ACCENT, chip="rayrai_tcp_viewer")

    # Connector captions live in the gap, so the gap is sized from them.
    captions = ["scene updates · sensor requests", "control requests · camera pixels"]
    gap = max(width(t, 12) for t in captions) + 34
    margin = 18
    top = 20
    total_w = margin * 2 + server.w + viewer.w + gap
    body_h = max(server.h, viewer.h)
    height = top + body_h + 86

    x_server = margin
    x_viewer = x_server + server.w + gap
    y_server = top + (body_h - server.h) / 2
    y_viewer = top + (body_h - viewer.h) / 2

    parts = [server.render(x_server, y_server), viewer.render(x_viewer, y_viewer)]

    mid = top + body_h / 2
    mid_x = (x_server + server.w + x_viewer) / 2
    parts.append(arrow(x_server + server.w + 6, mid - 20, x_viewer - 6, mid - 20))
    parts.append(text_el(mid_x, mid - 28, captions[0], size=12, fill=ACCENT, anchor="middle"))
    parts.append(text_el(mid_x, mid + 4, "one TCP session", size=11, fill=MUTED,
                         anchor="middle"))
    parts.append(arrow(x_viewer - 6, mid + 20, x_server + server.w + 6, mid + 20,
                       colour=GREEN, marker="aGreen"))
    parts.append(text_el(mid_x, mid + 40, captions[1], size=12, fill=GREEN, anchor="middle"))

    # UDP discovery, routed under both cards
    y_udp = top + body_h + 36
    x_from = x_server + server.w / 2
    x_to = x_viewer + viewer.w / 2
    parts.append('<path d="M%g %g V%g H%g V%g" fill="none" stroke="#d9822b" '
                 'stroke-width="1.4" stroke-dasharray="6 5" marker-end="url(#aWarn)"/>'
                 % (x_from, y_server + server.h + 4, y_udp, x_to, y_viewer + viewer.h + 8))
    note = "UDP 59312 discovery beacon — optional"
    note_w = width(note, 12) + 24
    parts.append(rounded(mid_x - note_w / 2, y_udp - 13, note_w, 26,
                         fill=WARN_BG, stroke=WARN, rx=3))
    parts.append(text_el(mid_x, y_udp + 5, note, size=12, fill="#8a5b1b", anchor="middle"))

    return svg(total_w, height, "\n".join(parts),
               "RaiSim TCP viewer data flow",
               "One TCP session carries scene updates, control requests and sensor "
               "pixels between RaisimServer and the rayrai TCP viewer. UDP beacons "
               "only provide optional discovery.")


# --------------------------------------------------------- sensor round trip --
def sensor_round_trip():
    steps = [
        ("1 · Sensor due", "MANUAL, update period elapsed", ACCENT),
        ("2 · Request", "pose · lens · resolution · clip", ACCENT),
        ("3 · rayrai render", "complete RGB or depth pass", ACCENT),
        ("4 · Preview", "texture · timing · depth range", ACCENT),
    ]
    returns = [
        ("5 · Return frame", "BGRA bytes or float depth", "REQUEST_SENSOR_UPDATE", GREEN),
        ("6 · Validate all", "tag · name · type · dimensions", "no partial writes", GREEN),
        ("7 · Publish", "replace buffer + timestamp", "application reads it", GREEN),
    ]

    pad_x, title_size, line_size, small_size = 14, 14, 12, 11
    cell_w = 0
    for title, line, _ in steps:
        cell_w = max(cell_w, width(title, title_size, bold=True), width(line, line_size))
    for title, line, small, _ in returns:
        cell_w = max(cell_w, width(title, title_size, bold=True), width(line, line_size),
                     width(small, small_size))
    cell_w += 2 * pad_x

    gap = 58
    margin = 18
    total_w = margin * 2 + cell_w * 4 + gap * 3
    row1_y, row1_h = 20, 62
    row2_y, row2_h = 150, 78
    height = row2_y + row2_h + 92

    parts = []
    xs = [margin + i * (cell_w + gap) for i in range(4)]

    for (title, line, accent), x in zip(steps, xs):
        parts.append(rounded(x, row1_y, cell_w, row1_h))
        parts.append('<path d="M%g %g H%g" stroke="%s" stroke-width="3" stroke-linecap="round"/>'
                     % (x + 1, row1_y + 1.5, x + cell_w - 1, accent))
        parts.append(text_el(x + cell_w / 2, row1_y + 26, title, size=title_size, bold=True,
                             anchor="middle"))
        parts.append(text_el(x + cell_w / 2, row1_y + 46, line, size=line_size, fill=MUTED,
                             anchor="middle"))

    for i in range(3):
        parts.append(arrow(xs[i] + cell_w + 5, row1_y + row1_h / 2,
                           xs[i + 1] - 5, row1_y + row1_h / 2))

    # the return path runs right to left underneath, so step 5 sits under step 3
    return_xs = [xs[2], xs[1], xs[0]]
    for (title, line, small, accent), x in zip(returns, return_xs):
        parts.append(rounded(x, row2_y, cell_w, row2_h))
        parts.append('<path d="M%g %g H%g" stroke="%s" stroke-width="3" stroke-linecap="round"/>'
                     % (x + 1, row2_y + 1.5, x + cell_w - 1, accent))
        parts.append(text_el(x + cell_w / 2, row2_y + 26, title, size=title_size, bold=True,
                             anchor="middle"))
        parts.append(text_el(x + cell_w / 2, row2_y + 46, line, size=line_size, fill=MUTED,
                             anchor="middle"))
        parts.append(text_el(x + cell_w / 2, row2_y + 64, small, size=small_size, fill=MUTED,
                             anchor="middle", mono=True))

    parts.append(arrow(xs[2] + cell_w / 2, row1_y + row1_h + 5,
                       xs[2] + cell_w / 2, row2_y - 5, colour=GREEN, marker="aGreen"))
    for i in (2, 1):
        parts.append(arrow(xs[i] - 5, row2_y + row2_h / 2,
                           xs[i - 1] + cell_w + 5, row2_y + row2_h / 2,
                           colour=GREEN, marker="aGreen"))

    # closing note, styled like the theme's note admonition
    note_title = "Failure is explicit"
    note_body = ("An incomplete render, a mismatched identity, invalid dimensions or a "
                 "malformed payload is rejected before sensor state changes.")
    note_w = max(width(note_title, 13, bold=True), width(note_body, 12)) + 36
    note_x = (total_w - note_w) / 2
    note_y = row2_y + row2_h + 28
    parts.append(rounded(note_x, note_y, note_w, 50, fill=NOTE_BG, stroke=NOTE_BG, rx=3))
    parts.append('<path d="M%g %g V%g" stroke="%s" stroke-width="4" stroke-linecap="round"/>'
                 % (note_x + 2, note_y + 4, note_y + 46, ACCENT_SOFT))
    parts.append(text_el(note_x + 18, note_y + 20, note_title, size=13, bold=True))
    parts.append(text_el(note_x + 18, note_y + 38, note_body, size=12, fill=INK))

    return svg(total_w, height, "\n".join(parts),
               "TCP viewer RGB and depth sensor round trip",
               "A manual camera request leaves RaiSim, renders in rayrai, and validated "
               "pixels return to the sensor buffer.")


if __name__ == "__main__":
    for name, content in (("tcp_viewer_data_flow.svg", data_flow()),
                          ("tcp_viewer_sensor_round_trip.svg", sensor_round_trip())):
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        print("wrote", path, len(content), "bytes")
