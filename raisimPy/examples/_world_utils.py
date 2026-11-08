"""Load the same scene assets and standing ANYmal posture as the native examples."""
import argparse
from pathlib import Path

import numpy as np

from _example_utils import (
    RESOURCE_DIR, setup, simulation_steps, pace, start_server, stop_server, finish_example,
)


def require_resource(relative):
    path = RESOURCE_DIR / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing example asset: {path}. See examples/tools for asset preparation.")
    return path


def setup_scene(raisim, name):
    parser = argparse.ArgumentParser(description=f"The RaiSim {name} scene in Python")
    parser.add_argument("--scene", type=Path, default=RESOURCE_DIR / name / f"rayrai_{name}.rscene",
                        help="scene file to load instead of the bundled scene")
    add_renderer_option(parser, raisim)
    args = setup(raisim, parser)
    args.scene = args.scene.expanduser().resolve()
    if not args.scene.is_file():
        parser.error(f"scene does not exist: {args.scene}")
    return args


def add_standing_anymal(raisim, world, x, y, yaw):
    options = raisim.ArticulatedSystemOption()
    options.convexify_collision_meshes = True
    robot = world.addArticulatedSystem(str(require_resource("anymal_c/urdf/anymal.urdf")), options=options)
    robot.setName("anymal")
    pose = np.array([x, y, 0.56, np.cos(yaw / 2), 0., 0., np.sin(yaw / 2),
                     0.03, 0.4, -0.8, -0.03, 0.4, -0.8,
                     0.03, -0.4, 0.8, -0.03, -0.4, 0.8])
    velocity = np.zeros(robot.getDOF())
    p_gain = np.zeros(robot.getDOF())
    d_gain = np.zeros(robot.getDOF())
    p_gain[6:], d_gain[6:] = 200., 10.
    robot.setState(pose, velocity)
    robot.setControlMode(raisim.ControlMode.PD_PLUS_FEEDFORWARD_TORQUE)
    robot.setPdGains(p_gain, d_gain)
    robot.setPdTarget(pose, velocity)
    robot.setGeneralizedForce(np.zeros(robot.getDOF()))
    return robot


def run_scene(raisim, world, args):
    server = create_scene_renderer(raisim, world, args)
    start_server(server, args)
    try:
        for _ in simulation_steps(args):
            server.integrateWorldThreadSafe()
            pace(args, world.getTimeStep())
    finally:
        stop_server(server)
    finish_example(server, world)


def add_renderer_option(parser, raisim):
    parser.add_argument("--renderer", choices=("rayrai", "tcp"),
                        default="rayrai" if raisim.hasRayrai else "tcp",
                        help="in-process Rayrai window or TCP viewer")


class _RayraiExampleViewer:
    """Pace rendering independently from the scene's physics control loop."""

    def __init__(self, raisim, world):
        self.viewer = raisim.rayrai.RayraiWindow(world, 1280, 720)
        self.world = world
        self.next_frame = 0.

    def __getattr__(self, name):
        return getattr(self.viewer, name)

    def launchServer(self, _port):
        self.viewer.render()

    def killServer(self):
        self.viewer.close()

    def integrateWorldThreadSafe(self):
        self.viewer.integrateWorldThreadSafe()
        if self.world.getWorldTime() >= self.next_frame:
            if not self.viewer.render():
                raise SystemExit(0)
            self.next_frame = self.world.getWorldTime() + 1. / 60.


def create_scene_renderer(raisim, world, args):
    if not args.headless and args.renderer == "rayrai":
        return _RayraiExampleViewer(raisim, world)
    return raisim.RaisimServer(world)
