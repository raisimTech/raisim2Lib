import os
import unittest

import numpy as np
import raisimpy as r


DRONE = """<robot name="drone">
  <link name="base">
    <inertial><mass value="2"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.04"/></inertial>
  </link>
  <thruster name="lift" link="base" torque_coefficient="0.02">
    <origin xyz="0.05 0 0"/><axis xyz="0 0 1"/><limit lower="0" upper="10"/>
  </thruster>
</robot>"""

TOOL = """<robot name="tool">
  <link name="world"/>
  <joint name="mount" type="fixed"><parent link="world"/><child link="tool"/></joint>
  <link name="tool">
    <inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><geometry><cylinder radius="0.03" length="0.02"/></geometry></collision>
  </link>
  <gripper name="magnet" link="tool" type="magnetic" max_force="50" range="0.01">
    <origin xyz="0 0 -0.01"/><axis xyz="0 0 -1"/>
  </gripper>
</robot>"""


class ForceDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        key = os.environ.get("RAISIM_ACTIVATION_KEY")
        if key:
            r.World.setActivationKey(key)

    def setUp(self):
        self.world = r.World()
        self.world.setTimeStep(0.001)
        self.world.setGravity(np.zeros(3))

    def test_thruster_pushes_and_turns_its_link(self):
        drone = self.world.addArticulatedSystem(DRONE)
        drone.setGeneralizedCoordinate(np.array([0., 0., 1., 1., 0., 0., 0.]))
        self.assertEqual(drone.getThrusterIndex("lift"), 0)
        drone.setThrust("lift", 20.)
        self.assertEqual(drone.getThrust(0), 10.)
        drone.setThrusts(np.array([4.]))
        self.assertEqual(drone.getThrusts(), [4.])
        position, axis = drone.getThrusterPose(0)
        np.testing.assert_allclose(position, [0.05, 0., 1.])
        np.testing.assert_allclose(axis, [0., 0., 1.])
        self.world.integrate()
        u = drone.getGeneralizedVelocity()
        np.testing.assert_allclose(u[2], 4. / 2. * 0.001, rtol=1e-9)
        torque = np.cross([0.05, 0., 0.], [0., 0., 4.]) + [0., 0., 0.02 * 4.]
        np.testing.assert_allclose(u[3:6], torque / np.array([0.02, 0.03, 0.04]) * 0.001, atol=1e-12)

        extra = r.ThrusterDefinition()
        extra.name, extra.link = "side", "base"
        extra.axis = np.array([2., 0., 0.])
        drone.setThrusters(drone.getThrusters() + [extra])
        self.assertEqual(len(drone.getThrusters()), 2)
        np.testing.assert_allclose(drone.getThrusters()[1].axis, [1., 0., 0.])

    def test_leaky_suction_uses_the_configured_radius_and_coefficient(self):
        tool = self.world.addArticulatedSystem(TOOL)
        cup = tool.getGrippers()[0]
        cup.name = "cup"
        cup.type = r.GripperDefinition.Type.SUCTION
        cup.maxForce = 20.
        cup.range = 0.004
        cup.cupRadius = 0.02
        cup.leakCoefficient = 2.
        tool.setGrippers([cup])
        self.assertEqual(tool.getGrippers()[0].cupRadius, 0.02)
        self.assertEqual(tool.getGrippers()[0].leakCoefficient, 2.)
        box = self.world.addBox(0.1, 0.1, 0.1, 1.)
        box.setPosition(np.array([0., 0., -0.061]))
        tool.setGripperActive("cup", True)
        self.world.integrate()
        state = tool.getGripperState(0)
        self.assertTrue(state.engaged)
        self.assertAlmostEqual(state.force, 10.)
        self.assertAlmostEqual(state.gap, 0.001)

    def test_magnet_pulls_only_magnetic_materials(self):
        tool = self.world.addArticulatedSystem(TOOL)
        self.assertEqual(tool.getGrippers()[0].type, r.GripperDefinition.Type.MAGNETIC)
        box = self.world.addBox(0.1, 0.1, 0.1, 2., "steel")
        box.setPosition(np.array([0., 0., -0.01 - 0.003 - 0.05]))
        tool.setGripperActive("magnet", True)
        self.assertTrue(tool.isGripperActive(0))
        self.world.integrate()
        self.assertFalse(tool.getGripperState(0).engaged)

        self.world.setMagneticMaterial("steel")
        self.assertTrue(self.world.isMagneticMaterial("steel"))
        self.world.integrate()
        state = tool.getGripperState(0)
        self.assertTrue(state.engaged)
        self.assertEqual(state.object.getName(), box.getName())
        self.assertAlmostEqual(state.force, 50. * 0.7 ** 2)
        self.assertAlmostEqual(state.gap, 0.003)
        np.testing.assert_allclose(state.position, [0., 0., -0.013], atol=1e-12)
        position, axis = tool.getGripperPose(0)
        np.testing.assert_allclose(axis, [0., 0., -1.])


if __name__ == "__main__":
    unittest.main()
