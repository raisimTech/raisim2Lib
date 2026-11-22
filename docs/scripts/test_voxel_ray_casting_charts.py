#!/usr/bin/env python3
"""Checks that the voxel ray casting benchmark page matches its recorded results.

docs/image_generators/voxel_ray_casting_charts.py writes sections/_voxel_ray_casting_results.md
(whose regions sections/BenchmarkVoxelRayCasting.md includes as tables) and the page's charts,
rsc/docs/image/voxel_ray_casting_*.svg, from public/voxel_ray_casting_results.json. Editing the
JSON without rerunning the script, editing the outputs by hand, or quoting other numbers in the
page's summary leaves the page out of sync with its own data.

Run with: python3 -m unittest discover -s docs/scripts
"""

import importlib.util
import re
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
SCRIPT = DOCS / "image_generators" / "voxel_ray_casting_charts.py"
PAGE = DOCS / "sections" / "BenchmarkVoxelRayCasting.md"
OVERVIEW = DOCS / "sections" / "BenchmarkOverview.md"


def load_script():
    spec = importlib.util.spec_from_file_location("voxel_ray_casting_charts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VoxelRayCastingChartsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.charts = load_script()
        cls.results = cls.charts.Results(cls.charts.load())
        cls.page = PAGE.read_text(encoding="utf-8")

    def test_checked_in_outputs_match_the_recorded_results(self):
        files = self.charts.outputs(self.results, DOCS / "sections", DOCS.parent / "rsc" / "docs" / "image")
        self.assertEqual(len(files), 1 + len(self.charts.CHARTS))
        for path, content in files.items():
            self.assertTrue(path.exists(), path)
            self.assertEqual(path.read_text(encoding="utf-8"), content,
                             f"{path.name}: rerun docs/image_generators/voxel_ray_casting_charts.py")

    def test_the_page_shows_every_table_and_chart(self):
        regions = [name for name, _ in self.charts.tables(self.results)]
        included = re.findall(r"<!--@include: \./_voxel_ray_casting_results\.md#([\w-]+)-->", self.page)
        self.assertEqual(included, regions)
        tables = self.charts.tables_file(self.results)
        for name in regions:
            self.assertEqual(tables.count(f"<!-- #region {name} -->"), 1, name)
            self.assertEqual(tables.count(f"<!-- #endregion {name} -->"), 1, name)
        shown = re.findall(r'<img class="diagram" src="\.\./\.\./rsc/docs/image/voxel_ray_casting_(\w+)\.svg" '
                           r'alt="[^"]{40,}">', self.page)
        self.assertEqual(sorted(shown), sorted(self.charts.CHARTS))

    def test_the_summary_and_the_overview_quote_the_recorded_results(self):
        quoted = self.results.headlines()
        summary = self.page.split("## Setup")[0]
        for key in ("cpu_ns", "cpu_vs_nanovdb", "cpu_vs_openvdb", "cpu_vs_bonxai", "cpu_vs_octomap", "gpu_ns",
                    "gpu_vs_nanovdb", "scan_cpu_ms", "scan_wall_ms", "scan_speedup", "edit_scan_ms", "edit_nanovdb_ms"):
            self.assertIn(f"**{quoted[key]}**", summary, key)
        overview = OVERVIEW.read_text(encoding="utf-8")
        row = next(line for line in overview.splitlines() if "BenchmarkVoxelRayCasting.md" in line)
        for key in ("cpu_vs_others", "gpu_vs_nanovdb", "scan_speedup"):
            self.assertIn(f"**{quoted[key]}**", row, key)

    def test_every_median_is_taken_over_five_runs_of_every_scene(self):
        data = self.results.data
        scenes = set(self.results.scenes)
        for section, keys in (("gpu", self.results.gpu_keys), ("scan", ("gpu_ms", "wall_ms")),
                              ("edits", ("rayrai_scan_after_edit_ms", "nanovdb_rebuild_and_upload_ms"))):
            runs = data[section]["runs"]
            self.assertEqual(len(runs), 5, section)
            for run in runs:
                for key in keys:
                    self.assertEqual(set(run[key]), scenes, (section, key))
                    self.assertTrue(all(value > 0 for value in run[key].values()), (section, key))

    def test_numbers_round_half_up(self):
        self.assertEqual(self.charts.fixed(1764.5), "1,765")
        self.assertEqual(self.charts.fixed(18.45, 1), "18.5")
        self.assertEqual(self.charts.fixed(0.125, 2), "0.13")

    def test_rays_that_differ_read_as_counts_or_shares(self):
        differ = self.charts.differ
        self.assertEqual(differ(0, 524288), "0")
        self.assertEqual(differ(1, 524288), "1 ray")
        self.assertEqual(differ(4, 524288), "4 rays")
        self.assertEqual(differ(487579, 524288), "93.0 %")


if __name__ == "__main__":
    unittest.main()
