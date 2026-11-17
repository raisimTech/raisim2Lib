"""Join objects into articulated systems while the simulation runs.

A gantry lowers its hook onto a box, joins it with a fixed joint, carries it, and releases it with
World.detachJoint(). Two carriages on fixed rails dock through a fixed joint; both are already fixed to
the world, so the dock closes a kinematic loop. A force on one carriage moves both until they undock.
"""
from pathlib import Path
import sys

import numpy as np
import raisimpy as r


GANTRY = """<robot name="gantry">
<link name="world"/>
<link name="frame">
  <inertial><mass value="5"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  <visual><origin xyz="0 0 0"/><geometry><box size="1.46 0.06 0.06"/></geometry>
    <material name="steel"><color rgba="0.28 0.34 0.42 1"/></material></visual>
</link>
<link name="carriage">
  <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  <visual><origin xyz="0 0 -0.08"/><geometry><box size="0.16 0.12 0.1"/></geometry>
    <material name="blue"><color rgba="0.12 0.45 0.88 1"/></material></visual>
</link>
<link name="hook">
  <inertial><mass value="0.5"/><inertia ixx="0.005" iyy="0.005" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial>
  <visual><origin xyz="0 0 0.2"/><geometry><box size="0.03 0.03 0.6"/></geometry>
    <material name="grey"><color rgba="0.6 0.6 0.6 1"/></material></visual>
  <collision><origin xyz="0 0 -0.12"/><geometry><box size="0.12 0.12 0.04"/></geometry></collision>
</link>
<joint name="mount" type="fixed"><parent link="world"/><child link="frame"/><origin xyz="0 0 1.4"/></joint>
<joint name="travel" type="prismatic"><parent link="frame"/><child link="carriage"/><axis xyz="1 0 0"/>
  <limit lower="-0.6" upper="0.6" effort="1000" velocity="10"/></joint>
<joint name="lift" type="prismatic"><parent link="carriage"/><child link="hook"/><origin xyz="0 0 -0.4"/>
  <axis xyz="0 0 1"/><limit lower="-0.8" upper="0.2" effort="1000" velocity="10"/></joint>
</robot>"""

CARRIAGE = """<robot name="carriage">
<link name="world"/>
<link name="rail">
  <inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
</link>
<link name="cart">
  <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  <visual><geometry><box size="0.3 0.2 0.14"/></geometry></visual>
</link>
<joint name="rail_mount" type="fixed"><parent link="world"/><child link="rail"/></joint>
<joint name="slide" type="prismatic"><parent link="rail"/><child link="cart"/><axis xyz="1 0 0"/>
  <limit lower="-0.2" upper="2.6" effort="100" velocity="10"/></joint>
</robot>"""


def point_in_body(system, body, point):
    position, rotation = system.getBodyPose(body)
    return rotation.T @ (np.asarray(point) - position)


def smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class Scene:
    """The gantry and the two carriages, scripted step by step."""

    def __init__(self, world):
        self.world = world
        self.gantry = world.addArticulatedSystem(GANTRY)
        self.gantry.setName("gantry")
        self.gantry.setGeneralizedCoordinate(np.array([-0.4, 0.0]))
        self.gantry.setPdGains(np.array([800.0, 3000.0]), np.array([80.0, 200.0]))
        self.tool = world.addBox(0.12, 0.12, 0.16, 0.4)
        self.tool.setName("tool")
        self.tool.setPosition(np.array([-0.4, 0.0, 0.08]))
        self.tool.setAppearance("1.0, 0.45, 0.08, 1.0")

        driven = world.addArticulatedSystem(CARRIAGE)
        passive = world.addArticulatedSystem(CARRIAGE)
        driven.setName("driven")
        passive.setName("passive")
        driven.setBasePos(np.array([1.0, -0.2, 0.25]))
        passive.setBasePos(np.array([1.0, 0.2, 0.25]))
        options = r.JoinOptions()
        options.childPrefix = "passive/"
        options.name = "docked_pair"
        dock = r.Joint(r.JointType.FIXED, "dock", position=[0.0, 0.2, 0.0])
        self.pair = world.joinObjects(driven, driven.getBodyIdx("cart"), passive, passive.getBodyIdx("cart"),
                                      dock, options)
        self.loop_joints = self.pair.getLoopJointNames()
        self.driven_gv = self.pair.getMappingFromBodyIndexToGeneralizedVelocityIndex()[self.pair.getBodyIdx("cart")]
        self.carried_height = 0.0
        self.release_height = 0.0
        self.docked_error = 0.0

    def before_step(self, step):
        world = self.world
        t = step * world.getTimeStep()
        lift = -0.698 * smooth(t) if t < 1.0 else -0.698 * (1.0 - smooth(t - 1.0))
        travel = -0.4 + 0.8 * smooth((t - 1.8) / 1.4)
        target = np.zeros(self.gantry.getGeneralizedCoordinateDim())
        target[self.gantry.getJoint("travel").getIdxInGeneralizedCoordinate()] = travel
        target[self.gantry.getJoint("lift").getIdxInGeneralizedCoordinate()] = lift
        self.gantry.setPdTarget(target, np.zeros(self.gantry.getDOF()))
        if step == 1000:
            hook = self.gantry.getBodyIdx("hook")
            grip = self.tool.getPosition() + np.array([0.0, 0.0, 0.08])
            mount = r.Joint(r.JointType.FIXED, "tool_mount", position=point_in_body(self.gantry, hook, grip))
            self.gantry = world.joinObjects(self.gantry, hook, self.tool, 0, mount)
            self.tool = None
            print("Joined the tool to the hook")
        if step == 3600:
            self.carried_height = self.gantry.getFramePosition("tool_mount")[2] - 0.08
            self.gantry, self.tool = world.detachJoint(self.gantry, "tool_mount")
            self.release_height = self.tool.getBodyCOM_W()[0][2]
            print(f"Released the tool at a height of {self.carried_height:.3f} m")

        force = np.zeros(self.pair.getDOF())
        force[self.driven_gv] = 0.3 if step < 3000 else -0.3
        self.pair.setGeneralizedForce(force)
        if step == 3000:
            self.pair = world.detachJoint(self.pair, "dock")[0]
            print("Undocked the carriages")
        if 0 < step < 3000:
            velocity = self.pair.getGeneralizedVelocity()
            self.docked_error = max(self.docked_error, abs(velocity[0] - velocity[1]))

    def metrics(self):
        return {
            "carried_tool_height": float(self.carried_height),
            "released_tool_drop": float(self.release_height - self.tool.getBodyCOM_W()[0][2]),
            "dock_loop_joints": list(self.loop_joints),
            "docked_velocity_error": float(self.docked_error),
            "undocked_velocities": [float(v) for v in self.pair.getGeneralizedVelocity()],
        }


def make_world():
    world = r.World()
    world.setTimeStep(0.001)
    world.setSleepingEnabled(False)
    world.addGround()
    return world


def run_demo(activation_key=None, steps=5000):
    """Runs the whole script without a viewer and returns what it measured."""
    if activation_key is not None:
        r.World.setActivationKey(str(activation_key))
    world = make_world()
    scene = Scene(world)
    for step in range(steps):
        scene.before_step(step)
        world.integrate()
    return scene.metrics()


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_utils import setup, simulation_steps, pace, start_server, finish_example, locked_server
    args = setup(r)
    world = make_world()
    scene = Scene(world)
    server = r.RaisimServer(world)
    start_server(server, args)
    for step in simulation_steps(args):
        with locked_server(server):
            scene.before_step(step)
        server.integrateWorldThreadSafe()
        pace(args, world.getTimeStep())
    finish_example(server, world)


if __name__ == "__main__":
    main()
