import os
import unittest

import numpy as np
import raisimpy as r


@unittest.skipUnless(hasattr(r, "VoxelMap"), "requires voxel-map support")
class VoxelMapTests(unittest.TestCase):
    def setUp(self):
        key = os.environ.get("RAISIM_ACTIVATION_KEY")
        if key:
            r.World.setActivationKey(key)

    def test_edits_rays_and_contacts(self):
        world = r.World()
        world.setTimeStep(0.002)
        voxels = world.addVoxelMap(0.1, [-20, -20, -10], [40, 40, 20])
        self.assertEqual(voxels.getObjectType(), r.ObjectType.VOXELMAP)
        self.assertEqual(voxels.fillBox([-20, -20, -10], [19, 19, -1], True), 40 * 40 * 10)
        hit = world.rayTest(np.array([0.05, 0.05, 2.0]), np.array([0.0, 0.0, -1.0]), 5.0)
        self.assertEqual(hit.size(), 1)
        self.assertAlmostEqual(hit[0].getPosition()[2], 0.0, places=12)

        ball = world.addSphere(0.1, 1.0)
        ball.setPosition(0.3, 0.2, 0.5)
        for _ in range(1000):
            world.integrate()
        self.assertAlmostEqual(ball.getPosition()[2], 0.1, delta=2e-3)

        keys = np.array([[0, 0, 0], [1, 0, 0], [50, 0, 0]], dtype=np.int32)
        self.assertEqual(voxels.setVoxels(keys, True), 2)
        self.assertTrue(voxels.isOccupied(1, 0, 0))
        occupied = voxels.getOccupiedKeys()
        self.assertEqual(occupied.shape, (40 * 40 * 10 + 2, 3))
        self.assertEqual(voxels.getKeyAtWorldPosition(np.array([0.15, 0.05, 0.05])), [1, 0, 0])

    def test_dense_occupancy_and_shared_resources(self):
        occupancy = np.zeros((4, 5, 6), dtype=np.uint8)
        occupancy[1, 2, 3] = 1
        resource = r.VoxelMapResource.fromOccupancy(0.2, [0, 0, 0], [6, 5, 4], occupancy)
        first, second = r.World(), r.World()
        a = first.addVoxelMap(resource)
        b = second.addVoxelMap(resource)
        self.assertTrue(a.isOccupied(3, 2, 1))
        a.setVoxel(0, 0, 0, True)
        self.assertEqual(a.getOccupiedCount(), 2)
        self.assertEqual(b.getOccupiedCount(), 1)
        b.setOccupancy(np.ones((4, 5, 6), dtype=np.uint8))
        self.assertEqual(b.getOccupiedCount(), 120)

    def test_arrays_are_read_without_narrowing(self):
        world = r.World()
        voxels = world.addVoxelMap(1.0, [0, 0, 0], [2, 3, 1])
        far = np.array([[2**32, 0, 0], [-(2**40), 1, 0]], dtype=np.int64)
        self.assertEqual(voxels.setVoxels(far, True), 0)
        self.assertEqual(voxels.getOccupiedCount(), 0)
        self.assertEqual(voxels.setVoxels(np.array([[1, 2, 0]], dtype=np.int64), True), 1)
        self.assertEqual(voxels.setVoxels(np.array([[0, 1, 0]], dtype=np.uint16), True), 1)
        with self.assertRaises(ValueError):
            voxels.setVoxels(np.array([[0.5, 0, 0]]), True)
        with self.assertRaises(ValueError):
            voxels.setVoxels(np.zeros((0, 3)), True)
        self.assertEqual(voxels.setVoxels(np.zeros((0, 3), dtype=np.int32), True), 0)
        counts = np.zeros((1, 3, 2), dtype=np.int64)
        counts[0, 0, 1] = 256
        voxels.setOccupancy(counts)
        self.assertEqual(voxels.getOccupiedCount(), 1)
        self.assertTrue(voxels.isOccupied(1, 0, 0))
        with self.assertRaises(ValueError):
            voxels.setOccupancy(np.ones((1, 2, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            r.VoxelMapResource.fromOccupancy(1.0, [0, 0, 0], [2, 3, 1], np.ones((3, 2, 1), dtype=bool))


if __name__ == "__main__":
    unittest.main()
