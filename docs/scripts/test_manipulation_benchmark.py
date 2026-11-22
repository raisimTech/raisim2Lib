"""Keep the manipulation comparison, trial records, and video aligned."""
from fractions import Fraction
import importlib.util
import xml.etree.ElementTree as ET
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import unittest

DOCS = Path(__file__).resolve().parents[1]
DATA = DOCS / "public/raisim_newton_manipulation.json"
PAGE = DOCS / "sections/BenchmarkNewton.md"


class ManipulationBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DATA.read_text(encoding="utf-8"))
        page = PAGE.read_text(encoding="utf-8")
        cls.section = page.split("## Robotiq manipulation", 1)[1].split(
            "::: details", 1)[0]
        cls.records = {
            (record["backend"], record["worlds"]): record
            for record in cls.data["records"]
        }

    def test_value_chart_and_table_match_recorded_prices(self):
        prices = self.data["value_comparison"]
        self.assertEqual(prices["cpu_price_usd"], 339.99)
        self.assertEqual(prices["gpu_price_usd"], 899.99)
        generator = DOCS / "image_generators/manipulation_value_chart.py"
        spec = importlib.util.spec_from_file_location("value_chart", generator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        chart_path = DOCS.parent / "rsc/docs/image/raisim_newton_manipulation_fps_per_usd.svg"
        chart = chart_path.read_text(encoding="utf-8")
        self.assertEqual(chart, module.chart(self.data))
        tree = ET.fromstring(chart)
        ns = {"s": "http://www.w3.org/2000/svg"}
        failed = [r for r in tree.findall(".//s:path", ns)
                  if r.find("s:title", ns) is not None and
                  "failed validation" in r.find("s:title", ns).text]
        self.assertEqual(len(failed), sum(not r["gpu_quality_pass"]
                                          for r in self.data["rows"]))
        self.assertIn("Historical scenarios", chart)
        self.assertIn("checked 15 Nov 2026", chart)
        self.assertIn("CPU out of stock", chart)
        self.assertTrue(all(r.get("class") == "failed" for r in failed))
        self.assertTrue(all(r.find("s:title", ns).text.startswith("Newton") for r in failed))
        for row in self.data["rows"]:
            cpu = (row["cpu_fps"] / prices["cpu_price_usd"]
                   if row["cpu_fps"] is not None else None)
            gpu = row["gpu_fps"] / prices["gpu_price_usd"]
            ratio = (f"{cpu / gpu:.2f}×" if cpu is not None and
                     row["gpu_quality_pass"] else "—")
            cpu_text = f"{cpu:,.2f}" if cpu is not None else "Not measured"
            dagger = "" if row["gpu_quality_pass"] else "†"
            line = (f"| {row['worlds']:,} | {row['cpu_workers'] or '—'} | "
                    f"{cpu_text} | {gpu:,.2f}{dagger} | {ratio} |")
            self.assertIn(line, self.section)
        self.assertIn("historical price scenarios", self.section)

    def test_measured_counts_and_trial_arithmetic(self):
        counts = {
            "raisim": [1, 16, 64, 256, 1024, 2048, 4048],
            "newton": [1, 16, 64, 256, 1024, 2048, 4048, 8096],
        }
        for backend, expected_counts in counts.items():
            actual = [r for r in self.data["records"]
                      if r["backend"] == backend]
            self.assertEqual([r["worlds"] for r in actual], expected_counts)
            for record in actual:
                self.assertEqual(record["steps"], 30000)
                self.assertEqual(record["dt"], .001)
                self.assertEqual(len(record["seconds"]), 3)
                self.assertTrue(all(math.isfinite(s) and s > 0
                                    for s in record["seconds"]))
                fps = (record["worlds"] * record["steps"] /
                       statistics.median(record["seconds"]))
                self.assertAlmostEqual(record["aggregate_fps"], fps, places=6)
                if backend == "raisim":
                    self.assertEqual(record["threads"],
                                     min(record["worlds"], 30))
                    self.assertEqual(record["requested_threads"], 30)
                    self.assertTrue(record["quality_pass"])

    def test_side_by_side_table_uses_raw_records(self):
        for row in self.data["rows"]:
            gpu_record = self.records[("newton", row["worlds"])]
            cpu_record = self.records.get(("raisim", row["worlds"]))
            self.assertEqual(row["gpu_fps"], gpu_record["aggregate_fps"])
            self.assertEqual(row["gpu_quality_pass"],
                             gpu_record["quality_pass"])
            self.assertEqual(row["cpu_fps"],
                             cpu_record["aggregate_fps"] if cpu_record else None)
            self.assertEqual(row["cpu_workers"],
                             cpu_record["threads"] if cpu_record else None)
            cpu = (f"{row['cpu_fps']:,.0f}" if cpu_record else "Not measured")
            gpu = f"{row['gpu_fps']:,.0f}"
            if not row["gpu_quality_pass"]:
                gpu += "†"
            ratio = (f"{row['cpu_fps'] / row['gpu_fps']:.2f}×"
                     if cpu_record and row["gpu_quality_pass"] else "—")
            workers = str(row["cpu_workers"]) if cpu_record else "—"
            line = (f"| {row['worlds']:,} | {workers} | {cpu} | "
                    f"{gpu} | {ratio} |")
            self.assertIn(line, self.section)
        largest = self.data["rows"][-1]
        self.assertEqual(largest["worlds"], 8096)
        self.assertIsNone(largest["cpu_fps"])
        self.assertIsNone(largest["cpu_workers"])

    def test_failed_checks_and_record_schema_are_explicit(self):
        fields = set(self.data["records"][0])
        failed_counts = []
        for record in self.data["records"]:
            self.assertEqual(set(record), fields)
            valid = (record["finite"] and
                     (record["backend"] == "raisim" or
                      record["overflow_bitmask"] == 0) and
                     record["max_mimic_error_rad"] < .01 and
                     record["bilateral_grasp_samples"] > 0 and
                     record["lifted_samples"] > 0)
            self.assertEqual(record["quality_pass"], valid)
            self.assertEqual(record["validation_status"],
                             "passed" if valid else "failed")
            self.assertEqual(record["process_returncode"], 0 if valid else 1)
            if record["backend"] == "newton":
                label = "Passed" if valid else "Failed"
                line = (f"| {record['worlds']:,} | "
                        f"{record['max_mimic_error_rad']:.6f} | {label} |")
                self.assertIn(line, self.section)
                if not valid:
                    failed_counts.append(f"{record['worlds']:,}")
        summary = ", ".join(failed_counts[:-1]) + " and " + failed_counts[-1]
        self.assertIn(f"failed the mimic check at {summary} worlds.",
                      " ".join(self.section.split()))
        self.assertIn("0.01 rad", self.section)
        self.assertIn("not accepted", self.section)
        self.assertEqual(self.data["mimic_error_limit_rad"], .01)
        self.assertEqual(self.data["measurement_timezone"], "Asia/Seoul")

    def test_shared_inputs_match(self):
        provenance = self.data["provenance"]
        self.assertEqual(provenance["raisim"]["workload"],
                         provenance["newton"]["workload"])
        for row in self.data["rows"][:-1]:
            cpu, gpu = [self.records[(backend, row["worlds"])]
                        for backend in ("raisim", "newton")]
            self.assertEqual(cpu["force_checksum"], gpu["force_checksum"])
            self.assertEqual(cpu["resets"], gpu["resets"])

    def test_video_is_the_recorded_physical_scene(self):
        video = self.data["video"]
        media = DOCS.parent / video["repo_path"]
        self.assertTrue(media.is_file())
        self.assertEqual(hashlib.sha256(media.read_bytes()).hexdigest(),
                         video["sha256"])
        self.assertIn("<video", self.section)
        self.assertIn("controls", self.section)
        self.assertIn(media.name, self.section)
        capture = video["capture"]
        for field in ("bilateral_grasp", "lifted", "carried", "released"):
            self.assertTrue(capture[field])
        self.assertGreater(capture["resets"], 0)
        self.assertEqual(capture["visual_headless_state_max_error"], 0)
        self.assertFalse(capture["rendering_timed"])
        self.assertEqual(capture["playback_speed"], 2)
        self.assertEqual(set(capture["compared_quantities"]), {
            "joint_coordinates", "joint_velocities", "object_position",
            "object_quaternion", "object_linear_velocity",
            "object_angular_velocity",
        })
        expected_step = int(Fraction(capture["frames"] - 1) *
                            Fraction(str(capture["playback_speed"])) /
                            (capture["fps"] * Fraction(str(capture["dt"]))))
        self.assertEqual(capture["last_simulation_step"], expected_step)
        workload = self.data["provenance"]["raisim"]["workload"]
        self.assertEqual(capture["replay_prepared_urdf_sha256"],
                         workload["prepared_urdf_sha256"])

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe unavailable")
    def test_video_encoding_matches_capture(self):
        video = self.data["video"]
        media = DOCS.parent / video["repo_path"]
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,nb_frames:format=duration",
            "-of", "json", str(media),
        ], text=True)
        info = json.loads(raw)
        stream = info["streams"][0]
        capture = video["capture"]
        self.assertEqual(stream["codec_name"], "h264")
        self.assertEqual(stream["pix_fmt"], "yuv420p")
        self.assertEqual((stream["width"], stream["height"]),
                         (capture["width"], capture["height"]))
        self.assertEqual(int(stream["nb_frames"]), capture["frames"])
        self.assertEqual(stream, video["encoding"]["streams"][0])
        expected_duration = capture["frames"] / capture["fps"]
        self.assertAlmostEqual(float(info["format"]["duration"]),
                               expected_duration, places=2)


if __name__ == "__main__":
    unittest.main()
