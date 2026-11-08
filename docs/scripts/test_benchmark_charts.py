#!/usr/bin/env python3
"""Checks that the Benchmark page's generated table and chart match the recorded results.

docs/image_generators/benchmark_charts.py writes sections/_benchmark_results.md
(included by sections/Benchmark.md) and rsc/docs/image/benchmark_speedup.svg from
public/benchmark_results.json. Editing the JSON without rerunning the script, or
editing the outputs by hand, leaves the page out of sync with its own data.

Run with: python3 -m unittest discover -s docs/scripts
"""

import importlib.util
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
SCRIPT = DOCS / "image_generators" / "benchmark_charts.py"


def load_script():
    spec = importlib.util.spec_from_file_location("benchmark_charts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkChartsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.charts = load_script()
        cls.results = cls.charts.load_results(DOCS / "public" / "benchmark_results.json")

    def test_checked_in_outputs_match_the_recorded_results(self):
        table = self.charts.results_table(self.results)
        chart = self.charts.speedup(self.results)
        self.assertEqual(
            (DOCS / "sections" / "_benchmark_results.md").read_text(encoding="utf-8"), table,
            "Rerun docs/image_generators/benchmark_charts.py")
        self.assertEqual(
            (DOCS.parent / "rsc" / "docs" / "image" / "benchmark_speedup.svg").read_text(encoding="utf-8"),
            chart, "Rerun docs/image_generators/benchmark_charts.py")

    def test_table_is_a_markdown_table_with_one_row_per_workload(self):
        table = self.charts.results_table(self.results)
        rows = [line for line in table.splitlines() if line.startswith("| ")]
        self.assertEqual(rows[0], "| Workload | RaiSim (s) | MuJoCo (s) | Speedup |")
        self.assertEqual(rows[1], "| --- | --- | --- | --- |")
        self.assertEqual(len(rows) - 2, len(self.charts.BENCHMARKS))
        for row in rows[2:]:
            self.assertEqual(row.count(" | "), 3, row)

    def test_an_invalid_comparison_has_no_speedup(self):
        table = self.charts.results_table(self.results)
        cube = next(line for line in table.splitlines() if line.startswith("| Deformable cube stack"))
        self.assertIn("| Unstable |", cube)
        self.assertTrue(cube.endswith("| — |"))

    def test_speedups_use_unrounded_medians(self):
        table = self.charts.results_table(self.results)
        chain = next(line for line in table.splitlines() if line.startswith("| Chain, 20 links"))
        _, raisim, mujoco = next(r for r in self.results if r[0] == "Chain, 20 links")
        expected = mujoco["median_seconds"] / raisim["median_seconds"]
        self.assertTrue(chain.endswith(f"| {expected:.2f}× |"), chain)


if __name__ == "__main__":
    unittest.main()
