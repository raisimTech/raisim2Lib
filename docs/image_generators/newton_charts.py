#!/usr/bin/env python3
"""Draw the RaiSim vs Newton performance-per-dollar charts from the recorded results.

public/raisim_newton_random_actions.json and public/raisim_newton_ten_step_value.json hold the
measurements and the component prices. Run without arguments to refresh the checked-in charts in
rsc/docs/image; an output directory instead writes them there, for previewing a change.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_chart_style import LinearAxis, fixed, grouped_columns, significant  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent
PRICES = "265K: $339.99 advertised (out of stock) · RTX 5070: $899.99 Newegg retail · prices checked 15 Nov 2026."
CHARTS = {
    "raisim_newton_anymal_random_fps_per_usd": ("raisim_newton_random_actions.json", "anymal",
        "ANYmal with random actions", "ten physics steps per action",
        ["CPU FP64, GPU FP32: the same actions, different solver constraints and contact trajectories."]),
    "raisim_newton_strandbeest_random_fps_per_usd": ("raisim_newton_random_actions.json", "strandbeest",
        "Strandbeest with random actions", "ten physics steps per action",
        ["CPU FP64, GPU FP32: the same actions, different solver constraints and contact trajectories."]),
    "raisim_newton_anymal_fps_per_usd": ("raisim_newton_ten_step_value.json", "anymal",
        "ANYmal standing", "ten steps per control interval",
        ["Newton from 4,096 worlds: fresh runs on 15 Nov; smaller batches: 14 Nov."]),
    "raisim_newton_strandbeest_fps_per_usd": ("raisim_newton_ten_step_value.json", "strandbeest",
        "Strandbeest", "ten steps per control interval",
        ["Newton from 4,096 worlds: fresh runs on 15 Nov; smaller batches: 14 Nov."]),
}


def value_text(value):
    return significant(value, 3)


def value_chart(rows, cpu_price, gpu_price, title, subtitle, notes, failed=None):
    """RaiSim and Newton physics steps per second per dollar at each world count."""
    groups, cpu, gpu = [], [], []
    for row in rows:
        workers = row.get("workers", row.get("cpu_workers"))
        groups.append((fixed(row["worlds"]), f"{workers} worker{'s' if workers > 1 else ''}" if workers else "GPU only"))
        cpu.append(row["cpu_fps"] / cpu_price if row["cpu_fps"] is not None else None)
        gpu.append(row["gpu_fps"] / gpu_price)
    top = max(value for value in cpu + gpu if value is not None) * 1.1
    measured = [fixed(row["worlds"]) for row in rows if row["cpu_fps"] is None]
    description = (f"{title}: grouped columns of the physics steps per second per dollar of RaiSim on a Core Ultra 7 "
                   f"265K CPU and Newton on an RTX 5070 GPU at {len(rows)} world counts. Higher is better."
                   + (f" RaiSim is not measured at {', '.join(measured)} worlds." if measured else "")
                   + (" Hatched Newton columns failed validation." if failed and any(failed) else ""))
    return grouped_columns(
        title, description, f"Physics steps per second per dollar, higher is better · {subtitle}", groups,
        [("raisim", "RaiSim · Core Ultra 7 265K", cpu, None), ("other", "Newton · RTX 5070", gpu, failed)],
        LinearAxis(top), [PRICES] + notes, "Physics steps per second per dollar", "independent worlds", value_text)


def chart(name, docs=DOCS):
    source, model, title, subtitle, notes = CHARTS[name]
    data = json.loads((docs / "public" / source).read_text(encoding="utf-8"))
    rows = [row for row in data["rows"] if row["model"] == model]
    return value_chart(rows, data["cpu_price_usd"], data["gpu_price_usd"], title, subtitle, notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    folder = args.output or DOCS.parent / "rsc" / "docs" / "image"
    for name in CHARTS:
        path = folder / f"{name}.svg"
        path.write_text(chart(name), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
