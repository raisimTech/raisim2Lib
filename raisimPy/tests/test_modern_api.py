"""Runtime regressions for the packaged RaiSim API and Python ownership."""
import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref

import numpy as np
import raisimpy as r


SLIDER = """<robot name="slider">
  <link name="world"/>
  <link name="tip">
    <inertial><mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="world"/><child link="tip"/><axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="100" velocity="10"/>
  </joint>
</robot>"""


class ModernApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        key = os.environ.get("RAISIM_ACTIVATION_KEY")
        if key:
            r.World.setActivationKey(key)

    def setUp(self):
        self.world = r.World()
        self.world.setTimeStep(0.001)

    def test_object_list_borrows_objects_and_retains_world(self):
        # Isolate a regression that otherwise aborts the whole test process.
        code = """
import gc, os
import raisimpy as r
key = os.environ.get('RAISIM_ACTIVATION_KEY')
if key: r.World.setActivationKey(key)
w = r.World()
w.addGround()
w.addSphere(0.1, 1.).setPosition(0., 0., 2.)
for _ in range(3):
    objects = w.getObjList()
    del objects
    gc.collect()
    w.integrate()
objects = w.getObjList()
del w
gc.collect()
assert objects[1].getPosition()[2] > 1.
del objects
gc.collect()
"""
        proc = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_server_retains_world(self):
        class Marker:
            pass
        world = r.World()
        world.marker = Marker()
        marker = weakref.ref(world.marker)
        server = r.RaisimServer(world)
        del world
        gc.collect()
        self.assertIsNotNone(marker())
        server.integrateWorldThreadSafe()
        del server
        gc.collect()
        self.assertIsNone(marker())

    def test_owned_numpy_properties_can_be_read_and_are_copies(self):
        transform = r.Transformation()
        transform.pos = np.array([1., 2., 3.])
        transform.rot = np.eye(3)
        position = transform.pos
        position[:] = 99.
        np.testing.assert_array_equal(transform.pos, [1., 2., 3.])
        np.testing.assert_array_equal(transform.rot, np.eye(3))
        site = r.Tendon.Site(position=[1., 2., 3.])
        np.testing.assert_array_equal(site.position, [1., 2., 3.])
        path = r.Tendon.PathElement.cylinder(site, 0.2)
        np.testing.assert_array_equal(path.axis, [0., 0., 1.])
        properties = r.Tendon.Properties()
        properties.color = [0.1, 0.2, 0.3, 1.]
        np.testing.assert_array_equal(properties.color, [0.1, 0.2, 0.3, 1.])
        self.assertEqual(r.hasWrenchSensors, hasattr(r, "ForceSensor"))
        self.assertEqual(r.hasWrenchSensors, hasattr(r, "TactileSensor"))

    def test_server_pause_step_resume_without_a_socket(self):
        server = r.RaisimServer(self.world)
        server.pauseSimulation()
        server.integrateWorldThreadSafe()
        self.assertEqual(self.world.getWorldTime(), 0.)
        server.stepSimulation(2)
        for _ in range(3):
            server.integrateWorldThreadSafe()
        self.assertAlmostEqual(self.world.getWorldTime(), 0.002)
        self.assertTrue(server.isSimulationPaused())
        server.resumeSimulation()
        server.integrateWorldThreadSafe()
        self.assertAlmostEqual(self.world.getWorldTime(), 0.003)

    def test_checkpoint_replays_exactly_and_rejects_invalid_worlds(self):
        ball = self.world.addSphere(0.1, 1.)
        ball.setPosition(0., 0., 2.)
        checkpoint = self.world.captureCheckpoint()
        self.assertTrue(checkpoint)
        self.assertFalse(r.World.Checkpoint())
        for _ in range(20):
            self.world.integrate()
        position, velocity = ball.getPosition(), ball.getLinearVelocity()
        self.assertTrue(self.world.restoreCheckpoint(checkpoint))
        self.assertEqual(self.world.getWorldTime(), checkpoint.getWorldTime())
        for _ in range(20):
            self.world.integrate()
        np.testing.assert_array_equal(ball.getPosition(), position)
        np.testing.assert_array_equal(ball.getLinearVelocity(), velocity)
        other = r.World()
        self.assertFalse(other.restoreCheckpoint(checkpoint))
        with self.assertRaises(ValueError):
            other.restoreCheckpoint(checkpoint, raiseOnError=True)
        self.assertEqual(other.getWorldTime(), 0.)
        self.world.addGround()
        self.assertFalse(self.world.restoreCheckpoint(checkpoint))
        np.testing.assert_array_equal(ball.getPosition(), position)

    def test_checkpoint_rejects_split_step(self):
        self.world.addSphere(0.1, 1.)
        self.world.integrate1()
        with self.assertRaises(RuntimeError):
            self.world.captureCheckpoint()
        self.world.integrate2()
        self.assertTrue(self.world.captureCheckpoint())

    def test_snapshot_clone_and_stable_ids(self):
        ball = self.world.addSphere(0.1, 1.)
        ball.setPosition(1., 2., 3.)
        snapshot = self.world.captureSingleBodySnapshot(ball)
        clone = self.world.cloneSingleBodyObject(ball, "clone")
        self.assertIs(self.world.getObjectById(clone.getId()), clone)
        self.assertNotEqual(ball.getId(), clone.getId())
        ball.setPosition(4., 5., 6.)
        self.assertTrue(self.world.restoreSingleBodySnapshot(ball.getId(), snapshot))
        np.testing.assert_array_equal(ball.getPosition(), [1., 2., 3.])
        self.assertTrue(self.world.setObjectCollisionFilter(ball, 4, 8))
        self.assertTrue(self.world.setObjectMaterial(ball, "test-material"))
        self.assertTrue(self.world.restoreSingleBodySnapshot(ball, snapshot))
        identifier = clone.getId()
        self.world.removeObject(clone)
        self.assertFalse(self.world.isObjectAlive(identifier))
        self.assertIsNone(self.world.getObjectById(identifier))

    def test_ik_overloads_preserve_state_and_return_independent_results(self):
        robot = self.world.addArticulatedSystem(SLIDER)
        original = robot.getGeneralizedCoordinate()
        target = np.array([0.5, 0., 0.])
        options = r.IKOptions()
        options.applySolution = False
        options.seed = [-0.2]
        result = robot.solveIK("slide", target, options)
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.solution[0], 0.5, delta=1e-4)
        np.testing.assert_array_equal(robot.getGeneralizedCoordinate(), original)
        options.applySolution = True
        frame = robot.getFrameIdxByName("slide")
        for selector in (frame, "slide"):
            self.assertTrue(robot.solveIK(selector, target, options).success)
            result = robot.solveIK(selector, target, np.eye(3), options)
            self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(robot.getFramePosition("slide"), target, atol=1e-4)
        pose = result.bestFramePosition
        pose[:] = 100.
        self.assertLess(result.bestFramePosition[0], 1.)
        before_failure = robot.getGeneralizedCoordinate()
        result = robot.solveIK("slide", np.array([3., 0., 0.]))
        self.assertFalse(result.success)
        np.testing.assert_array_equal(robot.getGeneralizedCoordinate(), before_failure)

    def test_actuator_limits_affect_dynamics_and_return_copies(self):
        robot = self.world.addArticulatedSystem(SLIDER)
        motor = r.DcMotorParameters()
        motor.resistance = 1.
        motor.torqueConstant = 0.1
        motor.velocityConstant = 10.
        motor.busVoltage = 24.
        motor.peakTorque = 2.
        self.assertTrue(motor.isValid())
        self.assertEqual(r.clipMotorTorque(motor, 0., 100.), 2.)
        self.assertEqual(r.motorTorqueBounds(motor, 0.).upper, 2.)
        self.assertEqual(r.motorOperatingRegionViolation(motor, 0., 100.), 98.)
        actuator = r.ActuatorDefinition()
        actuator.name, actuator.joint = "drive", "slide"
        actuator.motor = motor
        robot.setActuators([actuator])
        self.assertEqual(robot.getActuatorNames(), ["drive"])
        self.assertEqual(robot.getActuatorIndex("drive"), 0)
        robot.setActuatorTorque("drive", 100.)
        robot.setActuatorTorque(0, 100.)
        robot.setActuatorTorques(np.array([100.]))
        self.world.integrate()
        self.assertGreater(robot.getGeneralizedVelocity()[0], 0.)
        state = robot.getActuatorStates()[0]
        self.assertEqual(state.saturation, r.MotorSaturation.PEAK_TORQUE)
        self.assertAlmostEqual(state.motorTorque, 2.)
        lower, upper = robot.getActuatorTorqueBounds()
        self.assertEqual(lower.shape, (1,))
        self.assertLessEqual(upper[0], 2.)
        definitions = robot.getActuators()
        definitions[0].motor.busVoltage = 12.
        self.assertEqual(robot.getActuators()[0].motor.busVoltage, 24.)
        robot.setBusVoltage(20.)
        robot.setBusVoltage("drive", 18.)
        self.assertEqual(robot.getActuators()[0].motor.busVoltage, 18.)
        robot.setMotorOperatingRegionEnforced(False)
        self.assertFalse(robot.isMotorOperatingRegionEnforced())

    def test_cloth_pins_and_copied_particle_arrays(self):
        vertices = np.array([[0., 0., 1.], [0.2, 0., 1.], [0., 0.2, 1.]])
        material = r.DeformableObject.Material()
        material.substeps = 2
        cloth = self.world.addDeformableCloth(vertices, [[0, 1, 2]], [0], material)
        self.assertIsInstance(cloth, r.DeformableObject)
        self.assertEqual(cloth.getObjectType(), r.ObjectType.DEFORMABLE)
        positions = cloth.getPositions()
        positions[:] = 99.
        for _ in range(10):
            self.world.integrate()
        np.testing.assert_array_equal(cloth.getPositions()[0], vertices[0])
        self.assertLess(cloth.getPositions()[1, 2], 1.)
        self.assertEqual(cloth.getVelocities().shape, (3, 3))
        self.assertEqual(len(cloth.getTriangles()), 1)
        self.assertEqual(cloth.getDistanceConstraintCount(), 3)
        cloth.unpinVertex(0, 0.1)
        cloth.pinVertex(1)
        self.world.integrate()
        self.assertLess(cloth.getPositions()[0, 2], 1.)

    def test_volume_and_obj_factories(self):
        vertices = np.array([[0., 0., 1.], [0.2, 0., 1.], [0., 0.2, 1.], [0., 0., 1.2]])
        material = r.DeformableObject.Material()
        material.youngsModulus = 1e3
        volume = self.world.addDeformableVolume(vertices, [[0, 1, 2, 3]], [0], material)
        self.assertEqual(len(volume.getTetrahedra()), 1)
        with tempfile.TemporaryDirectory() as folder:
            mesh = Path(folder, "sheet.obj")
            mesh.write_text("v 2 0 1\nv 2.2 0 1\nv 2 0.2 1\nf 1 2 3\n")
            cloth = self.world.addDeformableCloth(str(mesh))
            options = r.DeformableObject.MeshBuildOptions()
            other = self.world.addDeformableObject(str(mesh), r.DeformableObject.Material(), options)
            other.setPositionOffset(np.array([1., 0., 0.]))
            self.world.integrate()
            self.assertTrue(np.isfinite(volume.getPositions()).all())
            self.assertEqual(cloth.getNumParticles(), 3)
            self.assertGreaterEqual(other.getNumParticles(), 3)

    def test_granular_mutation_and_state_roundtrip(self):
        grains = self.world.addGranularParticles(np.array([[0., 0., 1.], [1., 0., 1.]]), [0.02, 0.03])
        self.assertEqual(grains.getObjectType(), r.ObjectType.GRANULAR)
        original = grains.getPositions()
        grains.reserveParticles(16)
        fixed = grains.addParticle([2., 0., 1.], 0.02, fixed=True)
        grains.setVelocity(0, [1., 0., 0.])
        grains.setAngularVelocity(0, [0., 1., 0.])
        for _ in range(10):
            self.world.integrate()
        self.assertGreater(grains.getPositions()[0, 0], 0.)
        self.assertLess(grains.getPositions()[0, 2], 1.)
        np.testing.assert_array_equal(grains.getPositions()[fixed], [2., 0., 1.])
        np.testing.assert_array_equal(original, [[0., 0., 1.], [1., 0., 1.]])
        self.assertEqual(grains.getAngularVelocities().shape, (3, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder, "grains.bin"))
            saved = grains.getPositions()
            grains.saveGranularState(path)
            grains.removeParticle(0)
            self.assertEqual(grains.getNumParticles(), 2)
            grains.loadGranularState(path)
            np.testing.assert_array_equal(grains.getPositions(), saved)
        self.assertEqual(grains.getLastStepStats().velocityClamps, 0)

    def test_granular_box_and_seeded_emission(self):
        options = r.GranularSystem.BoxOptions()
        options.minCorner = np.array([0., 0., 1.])
        options.maxCorner = np.array([0.2, 0.2, 1.2])
        options.maxParticles = 4
        grains = self.world.addGranularBox(options)
        self.assertEqual(grains.getNumParticles(), 4)
        grains.setGroundPlane(0.)
        grains.setContainerBox([-2., -2., 0.], [2., 2., 3.])
        grains.disableContainerBox()
        grains.setPeriodicBoundary([-2., -2., 0.], [2., 2., 3.], True, True, False)
        grains.disablePeriodicBoundary()
        count = grains.emitBoxRandom([1., 0., 1.], [1.2, 0.2, 1.2], 0.01, 0.02, 0.05, 3, 123)
        self.assertEqual(count, 3)
        self.assertEqual(grains.removeParticlesInBox([0.5, -1., 0.], [2., 1., 2.]), 3)
        self.assertEqual(grains.getNumParticles(), 4)

    def test_soil_state_is_copied_and_restorable(self):
        soil = self.world.addDeformableTerrain(3, 3, 2., 2., 0., 0., [0.] * 9)
        self.assertIsInstance(soil, r.HeightMap)
        self.assertGreater(soil.pressure(0.01, 0., 0.1), 0.)
        state = soil.getSoilState()
        state.sinkage = [0.001] * 9
        self.assertEqual(soil.getSoilState().sinkage, [0.] * 9)
        soil.setSoilState(state)
        self.assertAlmostEqual(soil.getHeightMap()[4], -0.001)
        soil.resetSoil()
        self.assertEqual(soil.getHeightMap(), [0.] * 9)

    def test_unsupported_soil_particle_combinations_raise_without_terminating(self):
        self.world.addDeformableTerrain(3, 3, 2., 2., 0., 0., [0.] * 9)
        vertices = np.array([[0., 0., 1.], [0.2, 0., 1.], [0., 0.2, 1.]])
        with self.assertRaisesRegex(ValueError, "cannot share a world"):
            self.world.addDeformableCloth(vertices, [[0, 1, 2]])
        with self.assertRaises(ValueError):
            self.world.addGranularBox(r.GranularSystem.BoxOptions())
        self.assertEqual(self.world.getObjectCount(), 1)
        other = r.World()
        other.addGranularParticles(np.array([[0., 0., 1.]]), [0.02])
        with self.assertRaises(ValueError):
            other.addDeformableTerrain(3, 3, 2., 2., 0., 0., [0.] * 9)
        self.assertEqual(other.getObjectCount(), 1)

    def test_usd_articulation_import_and_integration(self):
        with tempfile.TemporaryDirectory() as folder:
            usd = Path(folder, "robot.usda")
            usd.write_text('''#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
    def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)
        point3f physics:centerOfMass = (0, 0, 0)
        def Cube "Shape" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
            double size = 0.2
        }
    }
}
''')
            robot = self.world.addUsdArticulatedSystem(str(usd))
            self.assertEqual(robot.getDOF(), 6)
            self.world.integrate()
            self.assertTrue(np.isfinite(robot.getGeneralizedCoordinate()).all())
            self.assertLess(robot.getGeneralizedCoordinate()[2], 0.)

    def test_contact_settings_are_copied_and_vector_shapes_are_checked(self):
        settings = self.world.getContactSettings()
        settings.sweptCcdEnabled = True
        self.assertFalse(self.world.getContactSettings().sweptCcdEnabled)
        self.world.setContactSettings(settings)
        self.assertTrue(self.world.getContactSettings().sweptCcdEnabled)
        broadphase = r.BroadphaseSettings()
        broadphase.type = r.BroadphaseType.MultiBoxPrune
        broadphase.mbpWorldMin = np.array([-5., -5., -5.])
        self.world.setBroadphaseSettings(broadphase)
        self.assertEqual(self.world.getBroadphaseSettings().type, r.BroadphaseType.MultiBoxPrune)
        self.world.setSleepingParameters(0.01, 0.02, 8)
        self.assertEqual(self.world.getSleepingParameters().quietSteps, 8)
        self.world.setSleepingEnabled(False)
        self.assertFalse(self.world.isSleepingEnabled())
        self.world.setMonolithicDeformables(False)
        self.assertFalse(self.world.isMonolithicDeformables())
        self.world.setCollisionCountersEnabled(True)
        self.world.integrate()
        self.assertEqual(self.world.getLastCollisionCounters().sweptCcdContacts, 0)
        with self.assertRaises(ValueError):
            self.world.setGravity(np.ones((1, 3)))
        with self.assertRaises(ValueError):
            self.world.addGranularParticles(np.zeros((2, 2)), [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
