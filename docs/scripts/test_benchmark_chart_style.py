#!/usr/bin/env python3
"""Checks that every benchmark chart shares one look and matches its recorded results.

The charts of the benchmark pages are drawn by the scripts in docs/image_generators with
benchmark_chart_style.py, from the JSON files in docs/public. A chart drawn some other way, or a
checked-in chart that no longer matches its data, fails here.

Run with: python3 -m unittest discover -s docs/scripts
"""

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
GENERATORS = DOCS / "image_generators"
IMAGES = DOCS.parent / "rsc" / "docs" / "image"
PAGES = ["Benchmark.md", "BenchmarkNewton.md", "LinearElasticityBenchmarks.md", "BenchmarkVoxelRayCasting.md",
         "BenchmarkOverview.md"]


def load(name):
    sys.path.insert(0, str(GENERATORS))
    spec = importlib.util.spec_from_file_location(name, GENERATORS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkChartStyleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.style = load("benchmark_chart_style")

    def test_every_chart_on_a_benchmark_page_uses_the_shared_style(self):
        charts = []
        for page in PAGES:
            text = (DOCS / "sections" / page).read_text(encoding="utf-8")
            for source in re.findall(r'<img class="diagram" src="([^"]+\.svg)"', text):
                charts.append((page, (DOCS / "sections" / source).resolve()))
        self.assertEqual(len(charts), 15)
        for page, chart in charts:
            svg = chart.read_text(encoding="utf-8")
            self.assertIn(f"<style>{self.style.STYLE}</style>", svg, f"{chart.name} on {page}")
            self.assertIn('<rect width="960"', svg, f"{chart.name} on {page} is not 960 px wide")

    def test_newton_charts_match_the_recorded_results(self):
        newton = load("newton_charts")
        for name in newton.CHARTS:
            self.assertEqual((IMAGES / f"{name}.svg").read_text(encoding="utf-8"), newton.chart(name),
                             f"{name}: rerun docs/image_generators/newton_charts.py")

    def test_newton_charts_plot_the_tables_of_the_page(self):
        page = (DOCS / "sections" / "BenchmarkNewton.md").read_text(encoding="utf-8")
        data = json.loads((DOCS / "public" / "raisim_newton_ten_step_value.json").read_text(encoding="utf-8"))
        for row in data["rows"]:
            if row["cpu_fps"] is None:
                continue
            cpu = row["cpu_fps"] / data["cpu_price_usd"]
            self.assertIn(f"| {row['worlds']:,} | {row['workers']} | {cpu:,.2f} |", page)

    def test_linear_elasticity_charts_match_the_recorded_results_and_the_page(self):
        charts = load("linear_elasticity_charts")
        data = charts.load()
        for name, svg in charts.charts(data).items():
            self.assertEqual((IMAGES / f"linear_elasticity_{name}.svg").read_text(encoding="utf-8"), svg,
                             f"{name}: rerun docs/image_generators/linear_elasticity_charts.py")
        page = (DOCS / "sections" / "LinearElasticityBenchmarks.md").read_text(encoding="utf-8")
        value = self.style.recorded
        cold = data["cold_solve"]
        for i, tetrahedra in enumerate(cold["tetrahedra"]):
            self.assertIn(f"| {tetrahedra:,} | {value(cold['raisim_ms'][i])} | {value(cold['mfem_ms'][i])} | "
                          f"{value(cold['scikit_fem_ms'][i])} |", page)
        loads = data["changing_loads"]
        for i, tetrahedra in enumerate(loads["tetrahedra"]):
            self.assertIn(f"| {tetrahedra:,} | {value(loads['mfem_scaled_warm_ms'][i])} | "
                          f"{value(loads['mfem_direction_change_ms'][i])} | "
                          f"{value(loads['raisim_direction_change_ms'][i])} |", page)
        for case in data["multigrid"]["cases"]:
            self.assertIn(f"| {case['tetrahedra']:,} | {case['poisson_ratio']} | {case['iccg_ms']:,} | "
                          f"{case['iccg_iterations']:,} | {case['multigrid_ms']:,} | {case['multigrid_iterations']} |",
                          page)
        for case in data["meshing"]["cases"]:
            self.assertIn(f"| {case['geometry']} | {case['max_edge_mm']} | {case['delaunay_s']:.3f} | "
                          f"{case['ftetwild_s']:.3f} |", page)

    def test_recorded_values_keep_their_decimals(self):
        self.assertEqual(self.style.recorded(5738.8), "5,738.8")
        self.assertEqual(self.style.recorded(59.0), "59.0")
        self.assertEqual(self.style.recorded(32070), "32,070")
        self.assertEqual(self.style.fixed(18.45, 1), "18.5")


if __name__ == "__main__":
    unittest.main()
