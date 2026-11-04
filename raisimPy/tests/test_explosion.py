"""Run with PYTHONPATH pointing to the built module and RAISIM_ACTIVATION_KEY set."""
import os
import unittest
import numpy as np
import raisimpy as r


class ExplosionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        key = os.environ.get("RAISIM_ACTIVATION_KEY")
        if key:
            r.World.setActivationKey(key)

    def setUp(self):
        self.world = r.World()
        self.world.setTimeStep(0.001)
        self.world.setGravity(np.array([0.0, 0.0, 0.0]))

    def test_blast_wave(self):
        wave = r.blast.wave(5.0, 1.0)
        self.assertGreater(wave.overpressure, 0.0)
        self.assertGreater(wave.reflectedOverpressure, wave.overpressure)
        self.assertAlmostEqual(wave.overpressureAt(0.0), wave.overpressure)
        self.assertEqual(wave.overpressureAt(-1.0), 0.0)
        self.assertAlmostEqual(r.blast.reach(8.0), 2.0 * r.blast.reach(1.0))
        with self.assertRaises(ValueError):
            r.blast.wave(1.0, 0.0)

    def test_explosion_pushes_a_box(self):
        box = self.world.addBox(0.5, 0.5, 0.5, 20.0)
        box.setPosition(4.0, 0.0, 0.0)
        properties = r.ExplosionProperties()
        properties.chargeMass = 2.0
        identifier = self.world.addExplosion(np.zeros(3), properties)
        for _ in range(100):
            self.world.integrate()
        self.assertGreater(box.getLinearVelocity()[0], 0.0)
        explosions = self.world.getExplosions()
        self.assertEqual(len(explosions), 1)
        self.assertEqual(explosions[0].id, identifier)
        self.assertTrue(np.allclose(explosions[0].position, np.zeros(3)))
        self.assertEqual(explosions[0].properties.chargeMass, 2.0)

    def test_projectile_hit_and_detonation(self):
        plate = self.world.addBox(0.02, 1.0, 1.0, 5.0)
        plate.setPosition(10.0, 0.0, 0.0)
        shell = r.ProjectileProperties()
        shell.explosion.chargeMass = 0.5
        self.world.addProjectile(np.zeros(3), np.array([500.0, 0.0, 0.0]), shell)
        self.assertEqual(len(self.world.getProjectiles()), 1)
        for _ in range(50):
            self.world.integrate()
        self.assertEqual(len(self.world.getProjectiles()), 0)
        hits = self.world.getProjectileHits()
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].objectIndex, plate.getIndexInWorld())
        self.assertTrue(hits[0].detonated)
        self.assertAlmostEqual(hits[0].normal[0], -1.0)
        self.assertEqual(len(self.world.getExplosions()), 1)
        self.assertGreater(plate.getLinearVelocity()[0], 0.0)

    def test_wind_carries_projectiles(self):
        air = r.FluidMedium.air()
        air.velocity = np.array([0.0, 15.0, 0.0])
        self.world.setAmbientFluid(air)
        ball = r.ProjectileProperties()
        ball.explosion.chargeMass = 0.0
        ball.mass = 0.05
        ball.radius = 0.05
        self.world.addProjectile(np.zeros(3), np.array([100.0, 0.0, 0.0]), ball)
        for _ in range(500):
            self.world.integrate()
        projectile = self.world.getProjectiles()[0]
        self.assertGreater(projectile.velocity[1], 1.0)
        self.assertTrue(np.allclose(projectile.launchPosition, np.zeros(3)))
        self.assertTrue(np.allclose(projectile.launchVelocity, np.array([100.0, 0.0, 0.0])))
        self.assertLess(projectile.velocity[0], 100.0)

    def test_ray_normal(self):
        self.world.addGround()
        hits = self.world.rayTest(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 10.0)
        self.assertEqual(hits.size(), 1)
        self.assertAlmostEqual(hits[0].getNormal()[2], 1.0)


if __name__ == "__main__":
    unittest.main()
