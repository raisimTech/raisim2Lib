#!/usr/bin/env python3
"""Writes the two charts of the benchmark page (sections/Benchmark.rst).

    python3 benchmark_charts.py [output directory]   (default: rsc/docs/image)

RESULTS holds the median wall-clock seconds of the recorded run, the same numbers as the results
table of the page; update both together after a new measurement.
"""
import math
import os
import sys

MACHINE = "AMD Ryzen 9 3950X"
# (benchmark, RaiSim median s, MuJoCo median s), in the order of the charts
RESULTS = [
    ("Chain20 speed", 0.569, 3.535),
    ("Heightmap ANYmal speed", 1.152, 5.519),
    ("Chain10 speed", 0.283, 1.093),
    ("Strandbeest", 0.178, 0.478),
    ("Primitive speed", 3.376, 8.804),
    ("ANYmal falling", 0.283, 0.732),
    ("ANYmal standing", 5.519, 13.819),
]

STYLE = """    text { font-family: Arial, Helvetica, sans-serif; fill: #24313d; }
    .title { font-size: 24px; font-weight: 700; }
    .subtitle { font-size: 14px; fill: #607080; }
    .axis { stroke: #a8b4bf; stroke-width: 1; }
    .grid { stroke: #d8dee5; stroke-width: 1; }
    .label { font-size: 14px; font-weight: 600; }
    .tick { font-size: 12px; fill: #607080; }"""

LEFT, RIGHT = 230, 890  # the plot's x range


def nice_step(maximum, ticks=4):
    """a round tick spacing that covers maximum in about `ticks` steps"""
    raw = maximum / ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 1.5, 2, 2.5, 3, 4, 5, 10):
        if factor * magnitude >= raw:
            return factor * magnitude
    return 10 * magnitude


def tick_label(value):
    return ("%g" % round(value, 6))


def backend_times(results):
    rows = len(results)
    top, pitch = 78, 60
    bottom = top + pitch * rows
    height = bottom + 90
    step = nice_step(max(max(r, m) for _, r, m in results))
    span = step * 4
    scale = (RIGHT - LEFT) / span
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="%d" viewBox="0 0 960 %d" role="img" '
           'aria-labelledby="title desc">' % (height, height),
           '  <title id="title">RaiSim and MuJoCo benchmark timing comparison</title>',
           '  <desc id="desc">Horizontal bar chart comparing median RaiSim and MuJoCo wall-clock seconds for %d '
           'benchmarks, based on three single-threaded runs. Lower times are better.</desc>' % rows,
           '  <style>', STYLE,
           '    .value { font-size: 12px; fill: #24313d; }',
           '    .legend { font-size: 13px; fill: #405060; }',
           '    .raisim { fill: #14806f; }',
           '    .mujoco { fill: #d47b42; }',
           '  </style>',
           '  <rect width="960" height="%d" fill="#ffffff"/>' % height,
           '  <text x="30" y="36" class="title">Wall-clock time by backend</text>',
           '  <text x="30" y="60" class="subtitle">Median of three single-threaded runs, %s, default benchmark '
           'arguments. Lower is better.</text>' % MACHINE,
           '  <rect x="650" y="32" width="14" height="14" rx="2" class="raisim"/>',
           '  <text x="672" y="44" class="legend">RaiSim</text>',
           '  <rect x="744" y="32" width="14" height="14" rx="2" class="mujoco"/>',
           '  <text x="766" y="44" class="legend">MuJoCo</text>']
    for k in range(5):
        x = LEFT + k * (RIGHT - LEFT) / 4
        out.append('  <line x1="%d" y1="%d" x2="%d" y2="%d" class="grid"/>' % (x, top, x, bottom))
    out.append('  <line x1="%d" y1="%d" x2="%d" y2="%d" class="axis"/>' % (LEFT, bottom, RIGHT, bottom))
    for k in range(5):
        x = LEFT + k * (RIGHT - LEFT) / 4
        out.append('  <text x="%d" y="%d" text-anchor="middle" class="tick">%s%s</text>'
                   % (x, bottom + 22, tick_label(k * step), " s" if k == 4 else ""))
    out.append('  <text x="%d" y="%d" text-anchor="middle" class="subtitle">Wall-clock seconds</text>'
               % ((LEFT + RIGHT) / 2, bottom + 56))
    for i, (name, raisim, mujoco) in enumerate(results):
        y = top + 4 + i * pitch
        wr, wm = max(2, round(raisim * scale)), max(2, round(mujoco * scale))
        out += ['  <text x="30" y="%d" class="label">%s</text>' % (y + 16, name),
                '  <rect x="%d" y="%d" width="%d" height="16" rx="3" class="raisim"/>' % (LEFT, y, wr),
                '  <rect x="%d" y="%d" width="%d" height="16" rx="3" class="mujoco"/>' % (LEFT, y + 21, wm),
                '  <text x="%d" y="%d" class="value">%.3f s</text>' % (LEFT + wr + 8, y + 13, raisim),
                '  <text x="%d" y="%d" class="value">%.3f s</text>' % (LEFT + wm + 8, y + 34, mujoco)]
    out.append('</svg>')
    return "\n".join(out) + "\n"


def speedup(results):
    rows = len(results)
    top, pitch = 82, 50
    bottom = top + pitch * rows + 8
    height = bottom + 80
    ratios = [m / r for _, r, m in results]
    step = nice_step(max(ratios))
    span = step * 4
    right = 900
    scale = (RIGHT - 51 - LEFT) / span  # leave room for the value labels
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="%d" viewBox="0 0 960 %d" role="img" '
           'aria-labelledby="title desc">' % (height, height),
           '  <title id="title">RaiSim speedup over MuJoCo</title>',
           '  <desc id="desc">Horizontal bar chart showing MuJoCo median time divided by RaiSim median time for '
           '%d benchmarks, based on three single-threaded runs. Higher values mean RaiSim is faster.</desc>' % rows,
           '  <style>', STYLE,
           '    .value { font-size: 13px; font-weight: 700; fill: #24313d; }',
           '    .bar { fill: #2f6fb7; }',
           '  </style>',
           '  <rect width="960" height="%d" fill="#ffffff"/>' % height,
           '  <text x="30" y="36" class="title">RaiSim speedup over MuJoCo</text>',
           '  <text x="30" y="60" class="subtitle">MuJoCo median / RaiSim median across three single-threaded runs, '
           '%s. Higher is better.</text>' % MACHINE]
    for k in range(5):
        x = LEFT + k * step * scale
        out.append('  <line x1="%d" y1="%d" x2="%d" y2="%d" class="grid"/>' % (x, top, x, bottom))
    out.append('  <line x1="%d" y1="%d" x2="%d" y2="%d" class="axis"/>' % (LEFT, bottom, right, bottom))
    for k in range(5):
        x = LEFT + k * step * scale
        out.append('  <text x="%d" y="%d" text-anchor="middle" class="tick">%sx</text>'
                   % (x, bottom + 22, tick_label(k * step)))
    out.append('  <text x="%d" y="%d" text-anchor="middle" class="subtitle">MuJoCo time / RaiSim time</text>'
               % ((LEFT + right) / 2, bottom + 56))
    for i, ((name, _, _), ratio) in enumerate(zip(results, ratios)):
        y = top + 12 + i * pitch
        w = max(2, round(ratio * scale))
        out += ['  <text x="30" y="%d" class="label">%s</text>' % (y + 18, name),
                '  <rect x="%d" y="%d" width="%d" height="24" rx="4" class="bar"/>' % (LEFT, y, w),
                '  <text x="%d" y="%d" class="value">%.2fx</text>' % (LEFT + w + 12, y + 17, ratio)]
    out.append('</svg>')
    return "\n".join(out) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "rsc", "docs", "image")
    for name, text in (("benchmark_backend_times.svg", backend_times(RESULTS)),
                       ("benchmark_speedup.svg", speedup(RESULTS))):
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text)
        print("wrote", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
