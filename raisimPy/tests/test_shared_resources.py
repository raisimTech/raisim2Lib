import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import raisimpy as r


@unittest.skipUnless(hasattr(r, "MeshResource"), "requires shared-resource support")
class SharedResourceTests(unittest.TestCase):
    def test_resources_outlive_inputs_and_keep_world_state_separate(self):
        key = os.environ.get("RAISIM_ACTIVATION_KEY")
        if key:
            r.World.setActivationKey(key)

        robot = r.ArticulatedSystemResource.load(
            '<robot name="shared"><link name="base"><inertial><mass value="2"/>'
            '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
            '</inertial></link></robot>'
        )
        heights = [1.0] * 4
        terrain = r.HeightMapResource.fromSamples(2, 2, 4.0, 4.0, 0.0, 0.0, heights)
        heights[0] = 5.0
        with TemporaryDirectory() as directory:
            source = Path(directory) / "tetra.obj"
            source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                              "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
            mesh = r.MeshResource.load(str(source))

        first, second = r.World(), r.World()
        robot_a = first.addArticulatedSystem(robot)
        robot_b = second.addArticulatedSystem(robot)
        first.addHeightMap(terrain).setPosition(0.0, 0.0, 10.0)
        terrain_b = second.addHeightMap(terrain)
        mesh_a = first.addMesh(mesh, 1.0)
        mesh_b = second.addMesh(mesh, 3.0, np.eye(3), np.zeros(3))
        del robot, mesh, terrain
        self.assertEqual(robot_a.getDOF(), robot_b.getDOF())
        self.assertEqual(terrain_b.getHeightMap(), [1.0] * 4)
        self.assertEqual(mesh_a.getMass(0), 1.0)
        self.assertEqual(mesh_b.getMass(0), 3.0)
