"""Dock two independent robot carriages, transfer force, and release the dock."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import raisimpy as r


MODEL = """<robot name="carriage_robot">
<link name="world"/>
<link name="base">
  <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
</link>
<link name="carriage">
  <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
</link>
<joint name="mount" type="fixed"><parent link="world"/><child link="base"/></joint>
<joint name="slide" type="prismatic">
  <parent link="base"/><child link="carriage"/><axis xyz="1 0 0"/>
  <limit lower="-1000" upper="1000" effort="100" velocity="100"/>
</joint>
</robot>"""


def run_demo(activation_key=None, steps=1000):
    if steps < 1:
        raise ValueError("steps must be positive")
    if activation_key is not None:
        r.World.setActivationKey(str(activation_key))
    world = r.World()
    world.setTimeStep(0.001)
    world.setGravity(np.zeros(3))
    world.setSleepingEnabled(False)
    first = world.addArticulatedSystem(MODEL)
    second = world.addArticulatedSystem(MODEL)
    first.setName("first_robot")
    second.setName("second_robot")
    dock = world.addRuntimeJoint(
        "dock", r.RuntimeJoint.Type.Fixed,
        r.RuntimeJoint.Attachment(first, first.getBodyIdx("carriage")),
        r.RuntimeJoint.Attachment(second, second.getBodyIdx("carriage")),
    )
    first.setGeneralizedForce(np.array([2.0]))
    for _ in range(steps):
        world.integrate()
    connected_error = float(abs(first.getGeneralizedVelocity()[0] -
                                second.getGeneralizedVelocity()[0]))
    reaction = dock.getForce().tolist()
    world.removeRuntimeJoint(dock)
    del dock
    for _ in range(steps):
        world.integrate()
    return {
        "connected_velocity_error": connected_error,
        "reaction_force_on_first_robot": reaction,
        "detached_velocity_difference": float(first.getGeneralizedVelocity()[0] -
                                              second.getGeneralizedVelocity()[0]),
        "remaining_joints": len(world.getRuntimeJoints()),
        "simulated_seconds": world.getWorldTime(),
    }


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_utils import setup
    args = setup(r, argparse.ArgumentParser(
        description=__doc__ + " --steps sets the steps before the release and again after it (1000 by default)."))
    result = run_demo(steps=1000 if args.steps is None else args.steps)
    print(json.dumps(result, indent=2))
    print(f"Simulated {result['simulated_seconds']:.6f} seconds")


if __name__ == "__main__":
    main()
