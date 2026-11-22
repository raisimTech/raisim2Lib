#!/usr/bin/env python3
"""Generate the voxel ray casting benchmark page's tables and charts from its recorded JSON.

public/voxel_ray_casting_results.json holds the measurements. This script writes
sections/_voxel_ray_casting_results.md, whose regions sections/BenchmarkVoxelRayCasting.md
includes as tables, and the page's charts in rsc/docs/image/voxel_ray_casting_*.svg.

Run without arguments to refresh the checked-in files. An output directory instead
writes them all there, for previewing a change.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_chart_style import Bar, LinearAxis, LogAxis, bar_panels, fixed, line_panels, ratio  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent
DATA = DOCS / "public" / "voxel_ray_casting_results.json"
TABLES = "_voxel_ray_casting_results.md"
CHARTS = ("cpu", "gpu", "scan", "robot", "edits")


def load(path=DATA):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scene_names(data):
    return [scene["name"] for scene in data["scenes"]]


def differ(count, total):
    if count == 0:
        return "0"
    if count / total < 1e-3:
        return f"{count} ray" if count == 1 else f"{count} rays"
    return f"{fixed(100 * count / total, 1)} %"


def differ_note(count, total):
    if count == 0:
        return ""
    if count / total >= 1e-3:
        return f"{differ(count, total)} of rays differ"
    return f"{differ(count, total)} differ" + ("s" if count == 1 else "")


def median(runs, *keys):
    values = []
    for run in runs:
        for key in keys:
            run = run[key]
        values.append(run)
    return statistics.median(values)


class Results:
    """The page's numbers, computed once from the recorded measurements."""

    def __init__(self, data):
        self.data = data
        self.scenes = scene_names(data)
        cpu = data["cpu"]
        self.libraries = cpu["libraries"]
        self.cpu = cpu["results"]
        gpu = data["gpu"]
        self.gpu_keys = ("rayrai", "nanovdb_cuda", "dense_dda_cuda")
        self.gpu_ns = {s: {k: median(gpu["runs"], k, s) for k in self.gpu_keys} for s in self.scenes}
        self.gpu_differ = gpu["rays_that_differ"]
        scan = data["scan"]
        self.scan_cpu_ms = {s: self.cpu[s]["raisim"]["lidar_ns_per_ray"] * scan["rays"] * 1e-6 for s in self.scenes}
        self.scan_gpu_ms = {s: median(scan["runs"], "gpu_ms", s) for s in self.scenes}
        self.scan_wall_ms = {s: median(scan["runs"], "wall_ms", s) for s in self.scenes}
        edits = data["edits"]
        self.edit_scan_ms = {s: median(edits["runs"], "rayrai_scan_after_edit_ms", s) for s in self.scenes}
        self.edit_nanovdb_ms = {s: median(edits["runs"], "nanovdb_rebuild_and_upload_ms", s) for s in self.scenes}

    def cpu_speedups(self, library, rays="lidar"):
        key = f"{rays}_ns_per_ray"
        return [self.cpu[s][library][key] / self.cpu[s]["raisim"][key] for s in self.scenes]

    def headlines(self):
        """Ranges the page's summary and the benchmark overview quote."""
        def span(values, digits, unit=""):
            low, high = fixed(min(values), digits), fixed(max(values), digits)
            return f"{low}–{high}{unit}"

        others = [x for lib in ("nanovdb", "openvdb", "bonxai", "octomap") for x in self.cpu_speedups(lib)]
        scan = [self.scan_cpu_ms[s] / self.scan_wall_ms[s] for s in self.scenes]
        return {
            "cpu_ns": span([self.cpu[s]["raisim"]["lidar_ns_per_ray"] for s in self.scenes], 0, " ns"),
            "cpu_vs_nanovdb": span(self.cpu_speedups("nanovdb"), 1, "×"),
            "cpu_vs_openvdb": span(self.cpu_speedups("openvdb"), 1, "×"),
            "cpu_vs_bonxai": span(self.cpu_speedups("bonxai"), 1, "×"),
            "cpu_vs_octomap": span(self.cpu_speedups("octomap"), 0, "×"),
            "cpu_vs_others": f"{fixed(min(others), 1)}–{fixed(max(others))}×",
            "gpu_ns": span([self.gpu_ns[s]["rayrai"] for s in self.scenes], 1, " ns"),
            "gpu_vs_nanovdb": span([self.gpu_ns[s]["nanovdb_cuda"] / self.gpu_ns[s]["rayrai"] for s in self.scenes], 1,
                                   "×"),
            "scan_cpu_ms": span(self.scan_cpu_ms.values(), 0, " ms"),
            "scan_wall_ms": span(self.scan_wall_ms.values(), 2, " ms"),
            "scan_speedup": span(scan, 0, "×"),
            "edit_scan_ms": span(self.edit_scan_ms.values(), 2, " ms"),
            "edit_nanovdb_ms": span(self.edit_nanovdb_ms.values(), 1, " ms"),
        }


def table(header, align, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def tables(results):
    """The page's tables, each as (region name, Markdown lines)."""
    r, data = results, results.data
    names = [lib["name"] for lib in r.libraries]
    keys = [lib["key"] for lib in r.libraries]
    rows = []
    for s in r.scenes:
        for rays in ("lidar", "random"):
            cells = [s if rays == "lidar" else "", rays]
            for key in keys:
                text = fixed(r.cpu[s][key][f"{rays}_ns_per_ray"])
                cells.append(f"**{text}**" if key == "raisim" else text)
            rows.append(cells)
    cpu_time = table(["Scene", "Rays"] + names, ["---", "---"] + ["---:"] * len(names), rows)

    compared = [lib for lib in r.libraries if lib["key"] != "dense_dda"]
    rows = [[s] + [differ(r.cpu[s][lib["key"]]["lidar_rays_that_differ"], data["cpu"]["lidar_rays"]) for lib in compared]
            for s in r.scenes]
    cpu_differ = table(["Scene"] + [lib["name"] for lib in compared], ["---"] + ["---:"] * len(compared), rows)

    rows = []
    for s in r.scenes:
        rows.append([s, "build (ms)"] + [fixed(r.cpu[s][k]["build_ms"]) for k in keys])
        rows.append(["", "memory (MB)"] + [fixed(r.cpu[s][k]["memory_mb"], 1) for k in keys])
    cpu_build = table(["Scene", ""] + names, ["---", "---"] + ["---:"] * len(names), rows)

    rows = [[s] + [(f"**{fixed(r.gpu_ns[s][k], 2)}**" if k == "rayrai" else fixed(r.gpu_ns[s][k], 2))
                   for k in r.gpu_keys] for s in r.scenes]
    gpu_time = table(["Scene", "rayrai (OpenGL)", "NanoVDB (CUDA)", "dense DDA (CUDA)"], ["---", "---:", "---:", "---:"],
                     rows)
    rows = [[s] + [differ(r.gpu_differ[k][s], data["gpu"]["rays"]) for k in r.gpu_keys] for s in r.scenes]
    gpu_differ = table(["Scene", "rayrai", "NanoVDB", "dense DDA"], ["---", "---:", "---:", "---:"], rows)

    rows = [[s, f"{fixed(r.scan_cpu_ms[s], 1)} ms", f"{fixed(r.scan_gpu_ms[s], 2)} ms", f"{fixed(r.scan_wall_ms[s], 2)} ms",
             ratio(r.scan_cpu_ms[s] / r.scan_wall_ms[s])] for s in r.scenes]
    scan = table(["Scene", "CPU (RaiSim, one thread)", "GPU (rayrai), GPU time", "GPU (rayrai), wall clock", "Speedup"],
                 ["---", "---:", "---:", "---:", "---:"], rows)

    rows = []
    for terrain in data["robot"]["terrains"]:
        for index, rate in enumerate(terrain["updates_per_second"]):
            cpu, gpu = terrain["cpu_us"][index], terrain["gpu_us"][index]
            update = (lambda us: f"{fixed(us / 1000, 2)} ms") if rate < 100 else (lambda us: f"{fixed(us)} µs")
            rows.append([terrain["name"] if index == 0 else "", fixed(rate), update(cpu), update(gpu), ratio(cpu / gpu)])
    robot = table(["Terrain", "Updates per second", "CPU per update", "GPU per update", "GPU speedup"],
                  ["---", "---:", "---:", "---:", "---:"], rows)

    low, high = data["edits"]["rayrai_upload_ms"]
    rows = [[s, f"{fixed(r.edit_scan_ms[s], 2)} ms", f"{fixed(low, 2)}–{fixed(high, 2)} ms",
             f"{fixed(r.edit_nanovdb_ms[s], 1)} ms"] for s in r.scenes]
    edits = table(["Scene", "rayrai: scan after the edit", "rayrai: upload of the edit", "NanoVDB: rebuild and upload"],
                  ["---", "---:", "---:", "---:"], rows)
    return [("cpu-time", cpu_time), ("cpu-differ", cpu_differ), ("cpu-build", cpu_build), ("gpu-time", gpu_time),
            ("gpu-differ", gpu_differ), ("scan", scan), ("robot", robot), ("edits", edits)]


def tables_file(results):
    lines = ["<!-- This file is generated by docs/image_generators/voxel_ray_casting_charts.py from",
             "     public/voxel_ray_casting_results.json. BenchmarkVoxelRayCasting.md includes its regions. -->"]
    for name, rows in tables(results):
        lines += ["", f"<!-- #region {name} -->"] + rows + [f"<!-- #endregion {name} -->"]
    return "\n".join(lines) + "\n"


def panel_title(scene):
    return f"{scene['name'].capitalize()} · {scene['voxel_m']:.2f} m voxels"


def cpu_chart(r):
    total = r.data["cpu"]["lidar_rays"]
    panels = []
    for scene in r.data["scenes"]:
        rows = []
        for lib in r.libraries:
            result = r.cpu[scene["name"]][lib["key"]]
            rate = 1e3 / result["lidar_ns_per_ray"]
            note = "reference" if lib["key"] == "dense_dda" else differ_note(result["lidar_rays_that_differ"], total)
            rows.append(Bar(lib["name"], "raisim" if lib["key"] == "raisim" else "other", rate, fixed(rate, 2), note,
                            f"{lib['name']}, {scene['name']}: {fixed(result['lidar_ns_per_ray'])} ns per ray"))
        panels.append((panel_title(scene), rows))
    return bar_panels(
        "Lidar rays per second on one CPU core",
        "Horizontal bars, one panel per scene, of how many million lidar rays per second RaiSim, NanoVDB, OpenVDB, "
        "Bonxai, OctoMap and a dense grid stepped voxel by voxel cast on one core. RaiSim's bars are the longest "
        "in every scene. Notes give the share of rays whose hit differs from the dense grid's.",
        "Million rays per second, higher is better · median of five runs · AMD Ryzen 9 3950X",
        panels, LinearAxis(6, ticks=list(range(0, 7))),
        ["A ray differs when it misses or adds a hit, or reports another voxel or distance than the dense grid."],
        axis_title="Million lidar rays per second (1,000 / nanoseconds per ray)")


def gpu_chart(r):
    total = r.data["gpu"]["rays"]
    names = {"rayrai": "rayrai (OpenGL)", "nanovdb_cuda": "NanoVDB (CUDA)", "dense_dda_cuda": "dense DDA (CUDA)"}
    panels = []
    for scene in r.data["scenes"]:
        rows = []
        for key in r.gpu_keys:
            ns = r.gpu_ns[scene["name"]][key]
            rows.append(Bar(names[key], "rayrai" if key == "rayrai" else "other", 1 / ns, fixed(1 / ns, 2),
                            differ_note(r.gpu_differ[key][scene["name"]], total),
                            f"{names[key]}, {scene['name']}: {fixed(ns, 2)} ns per ray"))
        panels.append((panel_title(scene), rows))
    return bar_panels(
        "Voxel rays per second on the GPU",
        "Horizontal bars, one panel per scene, of how many billion lidar rays per second rayrai's OpenGL voxel pass, "
        "NanoVDB in CUDA and a dense grid in CUDA cast on an RTX 2070 SUPER. rayrai's bars are the longest in every "
        "scene. Notes give the rays whose hit differs from RaiSim's CPU ray cast.",
        "Billion rays per second, higher is better · median of five runs · NVIDIA GeForce RTX 2070 SUPER",
        panels, LinearAxis(1.2, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], label="{:.1f}".format),
        ["The voxel pass alone, 524,288 rays per scene; a ray differs when its hit differs from RaiSim's CPU ray cast."],
        axis_title="Billion lidar rays per second (1 / nanoseconds per ray)")


def scan_chart(r):
    panels = []
    for scene in r.data["scenes"]:
        s = scene["name"]
        cpu, wall = r.scan_cpu_ms[s], r.scan_wall_ms[s]
        panels.append((panel_title(scene), [
            Bar("RaiSim, one CPU core", "raisim", cpu, f"{fixed(cpu, 1)} ms", "",
                f"RaiSim on one CPU core, {s}: {fixed(cpu, 1)} ms"),
            Bar("rayrai, GPU", "rayrai", wall, f"{fixed(wall, 2)} ms", f"{ratio(cpu / wall)} faster",
                f"rayrai on the GPU, {s}: {fixed(wall, 2)} ms wall clock"),
        ]))
    speedup = r.headlines()["scan_speedup"].replace("–", " to ")[:-1]
    return bar_panels(
        "One 65,536-ray lidar scan",
        "Horizontal bars, one panel per scene, of the milliseconds a 65,536-ray scan takes with RaiSim on one CPU "
        f"core and with rayrai on the GPU. The GPU bars are {speedup} times shorter.",
        "Milliseconds, lower is better · GPU wall clock, including the readback · median of five runs",
        panels, LinearAxis(20, ticks=list(range(0, 21, 5))),
        ["RaiSim's time is its nanoseconds per lidar ray times 65,536."], label_width=230,
        axis_title="Milliseconds per scan")


def edits_chart(r):
    panels = []
    for scene in r.data["scenes"]:
        s = scene["name"]
        rayrai, nanovdb = r.edit_scan_ms[s], r.edit_nanovdb_ms[s]
        panels.append((panel_title(scene), [
            Bar("rayrai: upload and scan", "rayrai", rayrai, f"{fixed(rayrai, 2)} ms", "",
                f"rayrai, {s}: a scan right after the edit takes {fixed(rayrai, 2)} ms"),
            Bar("NanoVDB: rebuild and upload", "other", nanovdb, f"{fixed(nanovdb, 1)} ms", "",
                f"NanoVDB, {s}: rebuilding and uploading the grid takes {fixed(nanovdb, 1)} ms"),
        ]))
    return bar_panels(
        "Ray casting right after an edit",
        "Horizontal bars, one panel per scene, of the milliseconds after carving or filling a 0.5 m sphere: rayrai "
        "uploads the changed bricks and scans; NanoVDB rebuilds its grid and uploads it, before any scan.",
        "Milliseconds after carving or filling a 0.5 m sphere, lower is better · median of five runs",
        panels, LinearAxis(20, ticks=list(range(0, 21, 5))),
        ["rayrai's time includes a whole 65,536-ray scan; NanoVDB's does not include its scan."], label_width=270,
        axis_title="Milliseconds")


def robot_chart(r):
    panels = []
    for terrain in r.data["robot"]["terrains"]:
        rates = terrain["updates_per_second"]
        last = len(rates) - 1
        panels.append((terrain["name"].capitalize(), [
            ("raisim", "CPU", [(rate, us * rate / 1000) for us, rate in zip(terrain["cpu_us"], rates)], (0, last)),
            ("rayrai", "GPU", [(rate, us * rate / 1000) for us, rate in zip(terrain["gpu_us"], rates)], (0, last)),
        ]))
    return line_panels(
        "LiDAR cost per second of simulated time",
        "Three panels, one per terrain, with two lines each: the milliseconds of LiDAR per simulated second on one "
        "CPU core and on the GPU when an Ouster OS1-64's 10 turns per second are cast in 10, 100 or 1,000 updates. "
        "The CPU line stays near 100 ms; the GPU line rises from a few milliseconds and crosses it before 1,000 "
        "updates per second.",
        "An Ouster OS1-64 at 10 turns per second on ANYmal · lower is better · median of three runs",
        panels, LogAxis(10, 1000), LinearAxis(200, ticks=list(range(0, 201, 50))),
        ["Each GPU update waits about 0.15 ms for its ranges, however few rays it casts."],
        "Updates per second (the same 655,360 rays per second, split into more, smaller updates)",
        "LiDAR milliseconds per simulated second",
        lambda value: f"{fixed(value, 1) if value < 10 else fixed(value)} ms",
        [("raisim", "RaiSim, one CPU core"), ("rayrai", "rayrai, GPU")])


def charts(results):
    return {"cpu": cpu_chart(results), "gpu": gpu_chart(results), "scan": scan_chart(results),
            "robot": robot_chart(results), "edits": edits_chart(results)}


def outputs(results, sections, images):
    files = {sections / TABLES: tables_file(results)}
    for name, svg in charts(results).items():
        files[images / f"voxel_ray_casting_{name}.svg"] = svg
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    results = Results(load(args.data))
    sections = args.output or DOCS / "sections"
    images = args.output or DOCS.parent / "rsc" / "docs" / "image"
    for path, content in outputs(results, sections, images).items():
        path.write_text(content, encoding="utf-8")
        print("wrote", path)
    for key, value in results.headlines().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
