"""Run with PYTHONPATH pointing to the built module and RAISIM_ACTIVATION_KEY set."""
import os
import unittest
import raisimpy as r


class ServerSliderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        key = os.environ.get("RAISIM_ACTIVATION_KEY")
        if key:
            r.World.setActivationKey(key)

    def test_sliders_start_within_their_range(self):
        world = r.World()
        server = r.RaisimServer(world)
        speed = server.addSlider("Stream speed [m/s]", 0.0, 3.0, 0.8)
        gain = server.addSlider("Gain", -1.0, 1.0, 5.0)
        self.assertEqual((speed, gain), (0, 1))
        self.assertAlmostEqual(server.getSlider(speed), 0.8)
        self.assertAlmostEqual(server.getSlider(gain), 1.0)


if __name__ == "__main__":
    unittest.main()
