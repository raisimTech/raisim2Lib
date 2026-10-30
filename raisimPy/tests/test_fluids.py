"""Run with PYTHONPATH pointing to the built module and RAISIM_ACTIVATION_KEY set."""
import math
import os
import unittest
import numpy as np
import raisimpy as r


class FluidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        key = os.environ.get("RAISIM_ACTIVATION_KEY")
        if key:
            r.World.setActivationKey(key)

    def setUp(self):
        self.world = r.World()
        self.world.setTimeStep(0.001)
        self.world.setGravity(np.array([0.0, 0.0, -9.81]))

    def test_sphere_buoyancy(self):
        pool = r.FluidVolume()
        pool.medium = r.FluidMedium.water()
        pool.surfaceHeight = 0.0
        self.world.addFluidVolume(pool)
        sphere = self.world.addSphere(0.1, 1.0)
        sphere.setPosition(0.0, 0.0, -0.5)
        self.world.integrate1()
        volume = 4.0 / 3.0 * math.pi * 0.1 ** 3
        force = sphere.getFluidForce(0)
        self.assertAlmostEqual(force[2], r.FluidMedium.water().density * 9.81 * volume, places=9)

    def test_current_and_coefficients(self):
        wind = r.FluidMedium.air()
        wind.velocity = np.array([10.0, 0.0, 0.0])
        self.world.setAmbientFluid(wind)
        box = self.world.addBox(0.4, 0.6, 0.2, 100.0)
        box.setPosition(0.0, 0.0, 3.0)
        coefficients = r.FluidCoefficients()
        coefficients.enabled = False
        self.world.setFluidCoefficients(box, coefficients)
        self.world.integrate1()
        self.assertTrue(np.allclose(box.getFluidForce(0), np.zeros(3)))
        coefficients.enabled = True
        self.world.setFluidCoefficients(box, coefficients)
        self.world.integrate()
        self.world.integrate1()
        self.assertGreater(box.getFluidForce(0)[0], 0.0)

    def test_vectors_read_back(self):
        wind = r.FluidMedium.air()
        wind.velocity = np.array([10.0, 0.0, -1.0])
        self.assertTrue(np.array_equal(wind.velocity, [10.0, 0.0, -1.0]))
        pool = r.FluidVolume()
        pool.min = np.array([-1.0, -2.0])
        pool.max = np.array([3.0, 4.0])
        self.assertTrue(np.array_equal(pool.min, [-1.0, -2.0]) and np.array_equal(pool.max, [3.0, 4.0]))
        wave = r.FluidWave(np.array([0.0, 1.0]), 8.0, 0.2)
        self.assertTrue(np.array_equal(wave.direction, [0.0, 1.0]))
        sample = r.FluidSample()
        sample.surfaceNormal = np.array([0.0, 0.0, 1.0])
        sample.velocity = np.array([0.5, 0.0, 0.0])
        sample.acceleration = np.array([0.0, 0.1, 0.0])
        self.assertTrue(np.array_equal(sample.surfaceNormal, [0.0, 0.0, 1.0]))
        self.assertTrue(np.array_equal(sample.velocity, [0.5, 0.0, 0.0]))
        self.assertTrue(np.array_equal(sample.acceleration, [0.0, 0.1, 0.0]))

    def test_sampler(self):
        volume = r.FluidVolume()
        volume.medium = r.FluidMedium.water()

        def sampler(point):
            sample = r.FluidSample()
            sample.surfaceHeight = 0.2
            sample.velocity = np.array([0.5, 0.0, 0.0])
            return sample

        volume.setSampler(sampler)
        self.assertTrue(volume.hasSampler())
        self.world.addFluidVolume(volume)
        ball = self.world.addSphere(0.05, 0.1)
        ball.setPosition(0.0, 0.0, 0.0)
        for _ in range(10):
            self.world.integrate()
        # the current carries the submerged ball along +x
        self.assertGreater(ball.getLinearVelocity()[0], 0.0)

    def test_waves(self):
        volume = r.FluidVolume()
        volume.medium = r.FluidMedium.water()
        volume.waves = [r.FluidWave(np.array([1.0, 0.0]), 20.0, 0.1)]
        self.assertEqual(len(volume.waves), 1)
        self.assertAlmostEqual(volume.waves[0].wavelength, 20.0)
        self.world.addFluidVolume(volume)
        ball = self.world.addSphere(0.05, 300.0 * 4.0 / 3.0 * np.pi * 0.05 ** 3)
        ball.setPosition(0.0, 0.0, 0.02)
        heights = []
        for _ in range(int(4.0 / self.world.getTimeStep())):
            self.world.integrate()
            heights.append(ball.getPosition()[2])
        # a light float rides the wave up and down
        self.assertGreater(max(heights[len(heights) // 2:]) - min(heights[len(heights) // 2:]), 0.15)


if __name__ == "__main__":
    unittest.main()
