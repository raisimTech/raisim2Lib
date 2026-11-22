#!/usr/bin/env python3
"""The one look of the benchmark pages' charts, and the chart forms they use.

Every benchmark chart is an SVG drawn on white (the site's dark theme inverts it, see
`img.diagram` in .vitepress/theme/styles/docs.css) with the same fonts, colors, header,
legend, bars, grid and footnotes:

- RaiSim on the CPU is blue, Rayrai (RaiSim's GPU renderer) is orange, and the engines
  and libraries it is compared with are gray; a second compared library is dark gray.
- Bars are at most 22 px thick, square at their baseline and rounded at their value.
- Values are printed at the bar's end; the page's tables give every number.

The chart scripts in this folder import this module; scripts/test_benchmark_chart_style.py
checks that every chart on a benchmark page uses it.
"""

import html
import math
from decimal import ROUND_HALF_UP, Decimal

WIDTH = 960
LEFT = 32
RAISIM, RAYRAI, OTHER, OTHER_DARK = "#2a78d6", "#eb6834", "#a9a8a0", "#6f6d66"
INK, MUTED, GRID, AXIS = "#24313d", "#596674", "#e1e4e5", "#a8b4bf"
SERIES = {"raisim": RAISIM, "rayrai": RAYRAI, "other": OTHER, "other-dark": OTHER_DARK}
STYLE = (f"""text {{ font-family: Arial, Helvetica, sans-serif; fill: {INK}; }}
    .title {{ font-size: 30px; font-weight: 700; }}
    .subtitle {{ font-size: 19px; fill: {MUTED}; }}
    .panel {{ font-size: 21px; font-weight: 700; }}
    .label {{ font-size: 19px; }}
    .value {{ font-size: 19px; font-weight: 700; }}
    .note {{ font-size: 18px; fill: {MUTED}; }}
    .tick {{ font-size: 18px; fill: {MUTED}; }}
    .grid {{ stroke: {GRID}; stroke-width: 1; }}
    .axis {{ stroke: {AXIS}; stroke-width: 1; }}
    .reference {{ stroke: {MUTED}; stroke-width: 1.5; stroke-dasharray: 6 4; }}"""
         + "".join(f"\n    .{name} {{ fill: {color}; }}" for name, color in SERIES.items())
         + "\n    .failed { fill: url(#failed); }")
# Hatching for a measurement that failed its validation, in the compared library's gray.
DEFS = (f'<defs><pattern id="failed" width="8" height="8" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><rect width="8" height="8" fill="#e4e3de"/>'
        f'<rect width="3" height="8" fill="{OTHER_DARK}"/></pattern></defs>')


def rounded(value, digits=0):
    """Rounds half up, as the page's numbers are read, unlike Python's round()."""
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def fixed(value, digits=0):
    return f"{rounded(value, digits):,.{digits}f}"


def significant(value, digits=3):
    """A value with about `digits` significant figures, thousands separated."""
    if value == 0:
        return "0"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(value))))
    return fixed(value, decimals)


def recorded(value):
    """A recorded value with the decimals it was recorded with, thousands separated."""
    whole, _, decimals = repr(value).partition(".")
    return f"{int(whole):,}" + (f".{decimals}" if decimals else "")


def ratio(value):
    return f"{fixed(value)}×" if value >= 10 else f"{fixed(value, 1)}×"


def text(x, y, content, css, anchor="start", extra=""):
    anchor_attribute = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'  <text x="{x:.1f}" y="{y:.1f}"{anchor_attribute} class="{css}"{extra}>{html.escape(content)}</text>'


def start(title, description, subtitle, height, legend=()):
    """The chart's header: title, subtitle and, for two or more series, a legend.

    legend: [(css class, label, "box" | "line")].
    """
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        f'  <title id="title">{html.escape(title)}</title>',
        f'  <desc id="desc">{html.escape(description)}</desc>',
        f"  <style>{STYLE}</style>",
        f"  {DEFS}",
        f'  <rect width="{WIDTH}" height="{height}" fill="#fff"/>',
        text(LEFT, 44, title, "title"),
        text(LEFT, 76, subtitle, "subtitle"),
    ]
    x = LEFT
    for css, label, kind in legend:
        if kind == "line":
            color = SERIES[css]
            out.append(f'  <line x1="{x}" y1="112" x2="{x + 24}" y2="112" stroke="{color}" stroke-width="2"/>')
            out.append(f'  <circle cx="{x + 12}" cy="112" r="4.5" fill="{color}" stroke="#fff" stroke-width="2"/>')
        else:
            out.append(f'  <rect x="{x + 2}" y="104" width="18" height="16" rx="3" class="{css}"/>')
        out.append(text(x + 32, 118, label, "label"))
        x += 32 + 10.5 * len(label) + 36
    return out


def finish(out):
    return "\n".join(out + ["</svg>"]) + "\n"


def footnotes(out, lines, height):
    for index, line in enumerate(reversed(lines)):
        out.append(text(LEFT, height - 16 - 26 * index, line, "note"))


def hbar(x0, y, width, height, css, tooltip):
    """A horizontal bar, square at its baseline x0 and rounded at its value."""
    width = max(width, 1.0)
    r = min(4.0, width / 2, height / 2)
    x1 = x0 + width
    return (f'  <path d="M{x0:.1f} {y:.1f}H{x1 - r:.1f}A{r:.1f} {r:.1f} 0 0 1 {x1:.1f} {y + r:.1f}'
            f'V{y + height - r:.1f}A{r:.1f} {r:.1f} 0 0 1 {x1 - r:.1f} {y + height:.1f}H{x0:.1f}Z" class="{css}">'
            f"<title>{html.escape(tooltip)}</title></path>")


def vbar(x, y0, width, height, css, tooltip):
    """A column, square at its baseline y0 and rounded at its value."""
    height = max(height, 1.0)
    r = min(4.0, width / 2, height / 2)
    y1 = y0 - height
    return (f'  <path d="M{x:.1f} {y0:.1f}V{y1 + r:.1f}A{r:.1f} {r:.1f} 0 0 1 {x + r:.1f} {y1:.1f}'
            f'H{x + width - r:.1f}A{r:.1f} {r:.1f} 0 0 1 {x + width:.1f} {y1 + r:.1f}V{y0:.1f}Z" class="{css}">'
            f"<title>{html.escape(tooltip)}</title></path>")


def linear_ticks(top, count=5):
    """Round tick values from zero to at least top."""
    raw = top / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    ticks = [0.0]
    while ticks[-1] < top - 1e-9 * top:
        ticks.append(ticks[-1] + step)
    return ticks


def log_ticks(low, high, base=10):
    """Powers of base spanning [low, high]."""
    first = math.floor(math.log(low, base) + 1e-9)
    last = math.ceil(math.log(high, base) - 1e-9)
    return [base ** exponent for exponent in range(first, last + 1)]


def tick_text(value):
    if value >= 1:
        return fixed(value)
    return significant(value, 1)


class Bar:
    """One bar of a chart: its row or series label, color class, value and printed text."""

    def __init__(self, label, css, value, text, note="", tooltip=""):
        self.label, self.css, self.value, self.text = label, css, value, text
        self.note, self.tooltip = note, tooltip or f"{label}: {text}"


def bar_panels(title, description, subtitle, panels, axis, footnote_lines, legend=(), label_width=200,
               axis_title=""):
    """Horizontal bars in panels (one per scene or case), on one axis from its baseline.

    panels: [(panel title or "", [Bar])]; axis: LinearAxis or LogAxis over the bars' values.
    """
    top_margin = 104 + (40 if legend else 0)
    left, right, pitch, thickness = LEFT + label_width, 760, 36, 22
    body = sum((36 if name else 0) + pitch * len(rows) + 18 for name, rows in panels)
    height = top_margin + body + 70 + 26 * len(footnote_lines)
    out = start(title, description, subtitle, height, legend)
    x = axis.mapper(left, right)
    y = top_margin
    for name, rows in panels:
        if name:
            out.append(text(LEFT, y + 22, name, "panel"))
            y += 36
        bottom = y + pitch * len(rows)
        for tick in axis.ticks:
            out.append(f'  <line x1="{x(tick):.1f}" y1="{y - 4}" x2="{x(tick):.1f}" y2="{bottom}" class="grid"/>')
        if axis.reference is not None:
            out.append(f'  <line x1="{x(axis.reference):.1f}" y1="{y - 4}" x2="{x(axis.reference):.1f}" '
                       f'y2="{bottom}" class="reference"/>')
        out.append(f'  <line x1="{left}" y1="{y - 4}" x2="{left}" y2="{bottom}" class="axis"/>')
        for bar in rows:
            middle = y + pitch / 2
            out.append(text(left - 12, middle + 6, bar.label, "label", "end"))
            out.append(hbar(left, middle - thickness / 2, x(bar.value) - left, thickness, bar.css, bar.tooltip))
            note = (f'<tspan class="note" font-weight="400">  {html.escape(bar.note)}</tspan>'
                    if bar.note else "")
            out.append(f'  <text x="{x(bar.value) + 8:.1f}" y="{middle + 6:.1f}" class="value">'
                       f"{html.escape(bar.text)}{note}</text>")
            y += pitch
        y += 18
    for tick in axis.ticks:
        out.append(text(x(tick), y + 6, axis.label(tick), "tick", "middle"))
    if axis_title:
        out.append(text((left + right) / 2, y + 36, axis_title, "note", "middle"))
    footnotes(out, footnote_lines, height)
    return finish(out)


def grouped_columns(title, description, subtitle, groups, series, axis, footnote_lines, axis_title,
                    group_title, value_text):
    """Columns in groups along the horizontal axis, one column per series.

    groups: [(label, second line or "")]; series: [(css, legend label, [value or None, ...],
    [failed, ...] or None)]. A missing value prints "not measured"; a failed one is hatched.
    """
    top_margin, left, right = 160, 112, WIDTH - 24
    plot_top, plot_bottom = top_margin + 10, top_margin + 380
    height = plot_bottom + 112 + 26 * len(footnote_lines)
    legend = [(css, label, "box") for css, label, _, _ in series]
    if any(failed and any(failed) for _, _, _, failed in series):
        legend.append(("failed", "failed validation", "box"))
    out = start(title, description, subtitle, height, legend)
    y = axis.mapper(plot_bottom, plot_top)
    for tick in axis.ticks:
        out.append(f'  <line x1="{left}" y1="{y(tick):.1f}" x2="{right}" y2="{y(tick):.1f}" class="grid"/>')
        out.append(text(left - 12, y(tick) + 6, axis.label(tick), "tick", "end"))
    out.append(f'  <line x1="{left}" y1="{plot_bottom}" x2="{right}" y2="{plot_bottom}" class="axis"/>')
    out.append(f'  <text x="26" y="{(plot_top + plot_bottom) / 2:.1f}" transform="rotate(-90 26 '
               f'{(plot_top + plot_bottom) / 2:.1f})" text-anchor="middle" class="note">{html.escape(axis_title)}</text>')
    pitch = (right - left) / len(groups)
    width = min(22.0, pitch * 0.4 / len(series))
    gap = min(24.0, pitch * 0.5 / len(series))
    for index, (label, second) in enumerate(groups):
        center = left + (index + 0.5) * pitch
        first = center - (len(series) * width + (len(series) - 1) * gap) / 2
        for number, (css, name, values, failed) in enumerate(series):
            x = first + number * (width + gap)
            value = values[index]
            if value is None:
                out.append(text(x + width / 2, plot_bottom - 8, "n/a", "tick", "middle",
                                ' font-size="13px"'))
                continue
            failure = bool(failed and failed[index])
            out.append(vbar(x, plot_bottom, width, plot_bottom - y(value), "failed" if failure else css,
                            f"{name}, {group_title} {label}: {value_text(value)}"
                            + (" (failed validation)" if failure else "")))
            out.append(f'  <text x="{x + width / 2:.1f}" y="{y(value) - 8:.1f}" text-anchor="middle" class="tick" '
                       f'font-size="14px">{html.escape(value_text(value))}</text>')
        out.append(text(center, plot_bottom + 26, label, "tick", "middle"))
        if second:
            out.append(text(center, plot_bottom + 48, second, "tick", "middle", ' font-size="14px"'))
    out.append(text((left + right) / 2, plot_bottom + 80, group_title[0].upper() + group_title[1:], "note", "middle"))
    footnotes(out, footnote_lines, height)
    return finish(out)


def line_panels(title, description, subtitle, panels, x_axis, y_axis, footnote_lines, x_title, y_title,
                value_text, legend):
    """Small multiples of lines; each panel shares both axes.

    panels: [(panel title, [(css, name, [(x, y), ...], label_points)])], where label_points lists the
    indices of the points whose values are printed.
    """
    top, bottom = 196, 496
    height = bottom + 92 + 26 * len(footnote_lines)
    out = start(title, description, subtitle, height, [(css, label, "line") for css, label in legend])
    y = y_axis.mapper(bottom, top)
    gap, start_x = 40, 92
    width = (WIDTH - start_x - 24 - gap * (len(panels) - 1)) / len(panels)
    for index, (name, lines) in enumerate(panels):
        left = start_x + index * (width + gap)
        x = x_axis.mapper(left + 24, left + width - 24)
        out.append(text(left, top - 26, name, "panel"))
        for tick in y_axis.ticks:
            out.append(f'  <line x1="{left:.1f}" y1="{y(tick):.1f}" x2="{left + width:.1f}" y2="{y(tick):.1f}" class="grid"/>')
            if index == 0:
                out.append(text(left - 10, y(tick) + 6, y_axis.label(tick), "tick", "end"))
        out.append(f'  <line x1="{left:.1f}" y1="{bottom}" x2="{left + width:.1f}" y2="{bottom}" class="axis"/>')
        for tick in x_axis.ticks:
            out.append(text(x(tick), bottom + 26, x_axis.label(tick), "tick", "middle"))
        for css, series_name, points, labelled in lines:
            color = SERIES[css]
            out.append(f'  <polyline points="{" ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in points)}" fill="none" '
                       f'stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            for a, b in points:
                out.append(f'  <circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4.5" fill="{color}" stroke="#fff" '
                           f'stroke-width="2"><title>{html.escape(f"{name}, {series_name}, {x_axis.label(a)}: {value_text(b)}")}'
                           f"</title></circle>")
        # Values at the labelled points: above the higher line, below the lower one, and above
        # either where it lies near the bottom of the plot.
        for point in range(len(lines[0][2])):
            here = [(points[point][1], css) for css, _, points, labelled in lines if point in labelled]
            for b, css in here:
                a = next(points[point][0] for c, _, points, _ in lines if c == css)
                highest = b >= max(value for value, _ in here)
                above = highest or y(b) > bottom - 0.2 * (bottom - top)
                anchor = "start" if point == 0 else "end" if point == len(lines[0][2]) - 1 else "middle"
                dx = -6 if anchor == "start" else 6 if anchor == "end" else 0
                out.append(text(x(a) + dx, y(b) + (-16 if above else 28), value_text(b), "value", anchor))
    out.append(f'  <text x="22" y="{(top + bottom) / 2:.1f}" transform="rotate(-90 22 {(top + bottom) / 2:.1f})" '
               f'text-anchor="middle" class="note">{html.escape(y_title)}</text>')
    out.append(text(start_x + (WIDTH - start_x - 24) / 2, bottom + 60, x_title, "note", "middle"))
    footnotes(out, footnote_lines, height)
    return finish(out)


class LinearAxis:
    def __init__(self, top, ticks=None, label=tick_text, reference=None):
        self.ticks = ticks if ticks is not None else linear_ticks(top)
        self.top = max(top, self.ticks[-1])
        self.label, self.reference = label, reference

    def mapper(self, low, high):
        return lambda value: low + value / self.top * (high - low)


class LogAxis:
    def __init__(self, low, high, base=10, label=tick_text, reference=None):
        self.ticks = log_ticks(low, high, base)
        self.low, self.high = self.ticks[0], self.ticks[-1]
        self.label, self.reference = label, reference

    def mapper(self, low, high):
        span = math.log(self.high) - math.log(self.low)
        return lambda value: low + (math.log(value) - math.log(self.low)) / span * (high - low)
