"""Run the actual example entry points with bounded simulation loops."""
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest


class ExampleTests(unittest.TestCase):
    def test_all_examples(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        for example in sorted(examples.glob("*.py")):
            if example.name.startswith("_"):
                continue
            if example.stem == "rayrai_render" and os.environ.get("RAISIMPY_TEST_RAYRAI") != "1":
                continue  # Requires a working OpenGL context; exercised by the renderer test run.
            with self.subTest(example=example.name):
                steps = 6001 if example.stem == "newtonsCradle" else 200
                if example.stem == "checkpoints":
                    steps = 1200  # Exercise checkpoint restoration as well as capture.
                if example.stem == "buggy":
                    steps = 3000  # Settle the suspension, then drive and steer.
                args = [sys.executable, "-X", "faulthandler", str(example), "--steps", str(steps), "--no-sleep"]
                if example.stem == "buggy":
                    args += ["--throttle", "0.35", "--steer", "0.15"]
                if os.environ.get("RAISIMPY_TEST_SERVER") == "1":
                    args += ["--port", "0"]
                    if example.stem in ("city", "warehouse", "forest", "buggy"):
                        args += ["--renderer", "tcp"]
                else:
                    args += ["--headless"]
                proc = subprocess.run(args, capture_output=True, text=True, timeout=90)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertIn("Simulated ", proc.stdout)
                if example.stem == "buggy":
                    telemetry = re.search(r"Buggy: travel=([0-9.]+) m, steering=([-0-9.]+) rad", proc.stdout)
                    self.assertIsNotNone(telemetry, proc.stdout)
                    self.assertGreater(float(telemetry[1]), 0.5, "Throttle did not move the buggy")
                    self.assertGreater(float(telemetry[2]), 0.1, "Steering did not reach the mechanism")


if __name__ == "__main__":
    unittest.main()
