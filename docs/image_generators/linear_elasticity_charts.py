#!/usr/bin/env python3
"""Draw the linear elasticity benchmark page's charts from its recorded results.

public/linear_elasticity_results.json holds the numbers of the page's tables. Run without arguments
to refresh rsc/docs/image/linear_elasticity_*.svg; an output directory instead writes them there.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_chart_style import Bar, LogAxis, bar_panels, fixed, ratio, recorded, significant  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent
DATA = DOCS / "public" / "linear_elasticity_results.json"
CHARTS = ("cold", "loads", "multigrid", "meshing")


def load(path=DATA):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ms(value):
    return f"{recorded(value)} ms"


def seconds(value):
    return f"{recorded(value)} s"


def cold_chart(data):
    cold = data["cold_solve"]
    panels = []
    for index, tetrahedra in enumerate(cold["tetrahedra"]):
        values = [("RaiSim", "raisim", cold["raisim_ms"][index]), ("MFEM", "other", cold["mfem_ms"][index]),
                  ("scikit-fem", "other-dark", cold["scikit_fem_ms"][index])]
        panels.append((f"{fixed(tetrahedra)} tetrahedra",
                       [Bar(name, css, value, ms(value)) for name, css, value in values]))
    return bar_panels(
        "Cold linear elastic solve",
        "Horizontal bars, one panel per mesh, of the milliseconds RaiSim, MFEM and scikit-fem take to assemble and "
        "solve a box in tension from scratch. The axis is logarithmic.",
        "Milliseconds, lower is better · box in tension, Poisson ratio 0.3 · AMD Ryzen 9 3950X, one thread",
        panels, LogAxis(1, 10000, label=lambda value: fixed(value)),
        ["RaiSim's time includes stress and strain recovery; MFEM's does not. scikit-fem uses SuperLU."],
        label_width=150, axis_title="Milliseconds (logarithmic)")


def loads_chart(data):
    loads = data["changing_loads"]
    panels = []
    for index, tetrahedra in enumerate(loads["tetrahedra"]):
        raisim, mfem = loads["raisim_direction_change_ms"][index], loads["mfem_direction_change_ms"][index]
        panels.append((f"{fixed(tetrahedra)} tetrahedra", [
            Bar("RaiSim", "raisim", raisim, ms(raisim), f"{ratio(mfem / raisim)} faster"),
            Bar("MFEM", "other", mfem, ms(mfem))]))
    return bar_panels(
        "A load that changes direction",
        "Horizontal bars, one panel per mesh, of the milliseconds RaiSim and MFEM take to solve again after the "
        "load changes direction, in MFEM's changing-nodal-load sweep. The axis is logarithmic.",
        "Milliseconds per solve, lower is better · the same changing-load sweep in both",
        panels, LogAxis(0.1, 1000, label=lambda value: significant(value, 1)),
        ["RaiSim reuses up to eight cached load responses and recovers all fields; MFEM omits stress recovery."],
        label_width=150, axis_title="Milliseconds (logarithmic)")


def multigrid_chart(data):
    panels = []
    for case in data["multigrid"]["cases"]:
        multigrid, iccg = case["multigrid_ms"], case["iccg_ms"]
        panels.append((f"{fixed(case['tetrahedra'])} tetrahedra, Poisson ratio {case['poisson_ratio']}", [
            Bar("Multigrid", "raisim", multigrid, ms(multigrid),
                f"{fixed(case['multigrid_iterations'])} iterations, {ratio(iccg / multigrid)} faster"),
            Bar("ICCG", "other", iccg, ms(iccg), f"{fixed(case['iccg_iterations'])} iterations")]))
    return bar_panels(
        "Cantilever bending: multigrid and ICCG",
        "Horizontal bars, one panel per case, of the milliseconds RaiSim's solve takes with the multigrid "
        "preconditioner (the default) and with incomplete-Cholesky CG. The axis is logarithmic.",
        "Milliseconds per cold solve, lower is better · Intel Core Ultra 7 265K, one core",
        panels, LogAxis(100, 100000, label=lambda value: fixed(value)),
        ["Both stop at a relative residual of 1e-11 and agree to ten digits."],
        label_width=150, axis_title="Milliseconds (logarithmic)")


def meshing_chart(data):
    panels = []
    for case in data["meshing"]["cases"]:
        delaunay, ftetwild = case["delaunay_s"], case["ftetwild_s"]
        panels.append((f"{case['geometry']}, {fixed(case['max_edge_mm'])} mm edges", [
            Bar("RaiSim Delaunay", "raisim", delaunay, seconds(delaunay), f"{ratio(ftetwild / delaunay)} faster"),
            Bar("fTetWild", "other", ftetwild, seconds(ftetwild))]))
    return bar_panels(
        "Tetrahedral meshing of CAD parts",
        "Horizontal bars, one panel per part and element size, of the seconds RaiSim's constrained Delaunay mesher "
        "and fTetWild take to mesh it. The axis is logarithmic.",
        "Seconds, lower is better · median of three runs · Intel Core Ultra 7 265K, one core",
        panels, LogAxis(0.01, 10, label=lambda value: significant(value, 1)),
        ["Times include helper startup, CAD translation, meshing and output."],
        label_width=200, axis_title="Seconds (logarithmic)")


def charts(data):
    return {"cold": cold_chart(data), "loads": loads_chart(data), "multigrid": multigrid_chart(data),
            "meshing": meshing_chart(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    folder = args.output or DOCS.parent / "rsc" / "docs" / "image"
    for name, svg in charts(load()).items():
        path = folder / f"linear_elasticity_{name}.svg"
        path.write_text(svg, encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
