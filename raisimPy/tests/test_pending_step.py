"""Contact sampling can cancel a prepared step without integrating."""
import os
import unittest

import raisimpy as r


class PendingStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        key = os.environ.get("RAISIM_ACTIVATION_KEY")
        if key:
            r.World.setActivationKey(key)

    def test_collision_sampling_allows_scene_edits_after_discard(self):
        world = r.World()
        ground = world.addGround()
        a = world.addSphere(0.1, 1.0)
        b = world.addSphere(0.1, 1.0)
        a.setPosition(0.0, 0.0, 1.0)
        b.setPosition(0.15, 0.0, 1.0)
        world.integrate1()
        self.assertGreater(len(a.getContacts()), 0)
        with self.assertRaises(RuntimeError):
            world.removeObject(ground)
        world.discardPendingStep()
        self.assertEqual(world.getWorldTime(), 0.0)
        self.assertEqual(len(a.getContacts()), 0)
        world.removeObject(ground)
        self.assertIsNotNone(world.addBox(1.0, 1.0, 1.0, 1.0))
        world.integrate()
        self.assertAlmostEqual(world.getWorldTime(), world.getTimeStep())

    def test_empty_and_no_detection_preparation(self):
        world = r.World()
        world.discardPendingStep()
        body = world.addSphere(0.1, 1.0)
        world.integrateNoContactDetection()
        world.discardPendingStep()
        world.discardPendingStep()
        world.removeObject(body)
        self.assertEqual(world.getWorldTime(), 0.0)
