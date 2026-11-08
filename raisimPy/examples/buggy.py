"""Drive the native off-road buggy URDF on its race-track heightmap.

Use the viewer's Throttle, Steering, and Brake sliders, or set their
initial values on the command line. Suspension and steering linkages are
simulated from the URDF's springs, dampers, pin/equality/mimic constraints.
"""
import argparse
import math

import numpy as np
import raisimpy as raisim

from _example_utils import (
    setup, simulation_steps, pace, start_server, stop_server, finish_example, locked_server,
)
from _world_utils import require_resource, add_renderer_option, create_scene_renderer


def control_value(lower, upper):
    def parse(value):
        number = float(value)
        if not math.isfinite(number) or not lower <= number <= upper:
            raise argparse.ArgumentTypeError(f"expected a number between {lower} and {upper}")
        return number
    return parse


def load_terrain(world):
    # The native example uses the raw 16-bit PNG values with this same scale.
    fields = {}
    for line in require_resource("offroad_buggy/terrain/terrain.txt").read_text().splitlines():
        words = line.split()
        if words and words[0] in ("extent", "height", "start"):
            fields[words[0]] = [float(value) for value in words[1:]]
    extent, = fields["extent"]
    minimum, maximum = fields["height"]
    terrain = world.addHeightMap(
        str(require_resource("offroad_buggy/terrain/height.png")),
        0., 0., extent, extent, (maximum - minimum) / 65535., minimum, "terrain",
    )
    terrain.setName("terrain")
    terrain.setAppearance("0.42,0.33,0.20,1")
    return terrain, fields["start"]


class Buggy:
    """Rear drive with torque/power limits, differential, brakes, and power steering."""

    def __init__(self, world, terrain, start):
        options = raisim.ArticulatedSystemOption()
        options.convexify_collision_meshes = True
        self.system = world.addArticulatedSystem(
            str(require_resource("offroad_buggy/buggy.urdf")), options=options,
        )
        self.system.setName("offroad_buggy")
        corners = ("fl", "fr", "rl", "rr")
        self.wheel_gc = [self.system.getJoint(f"{corner}_wheel").getIdxInGeneralizedCoordinate()
                         for corner in corners]
        self.wheel_gv = [self.system.getGeneralizedVelocityIndex(f"{corner}_wheel") for corner in corners]
        self.steering_gc = self.system.getJoint("steering_wheel").getIdxInGeneralizedCoordinate()
        self.steering_gv = self.system.getGeneralizedVelocityIndex("steering_wheel")
        radius = max(self.system.getCollisionBody(f"{corner}_wheel/0").shapeParam[0] for corner in corners)
        for corner in corners:
            self.system.ignoreCollisionBetween(0, self.system.getBodyIdx(f"{corner}_wheel"))
        self.system.updateSelfCollisionCache(world)

        x, y, heading = start
        cosine, sine = math.cos(heading), math.sin(heading)
        height = max(terrain.getHeight(x + cosine * dx - sine * dy, y + sine * dx + cosine * dy)
                     for dx in (-1.4, 1.4) for dy in (-1.1, 1.1))
        pose = self.system.getGeneralizedCoordinate()
        pose[:7] = [x, y, height + radius + 0.015, math.cos(heading / 2), 0., 0., math.sin(heading / 2)]
        self.system.setState(pose, np.zeros(self.system.getDOF()))
        self.initial_position = pose[:3].copy()
        self.p_target = pose.copy()
        self.v_target = np.zeros(self.system.getDOF())
        self.p_gain = np.zeros(self.system.getDOF())
        self.d_gain = np.zeros(self.system.getDOF())
        self.force = np.zeros(self.system.getDOF())
        self.held = [False] * 4
        self.hold_angles = pose[self.wheel_gc].copy()
        self.steering_angle = 0.
        self.system.setControlMode(raisim.ControlMode.PD_PLUS_FEEDFORWARD_TORQUE)

    def apply_controls(self, throttle, steering, brake, reverse, dt):
        pose = self.system.getGeneralizedCoordinate()
        velocity = self.system.getGeneralizedVelocity()
        forward = self.system.getOrientation(0)[:, 0]
        speed = float(forward @ velocity[:3])

        # Match the native vehicle's speed-sensitive steering lock and slew rate.
        lock = 5.8 / (1. + speed * speed / 500.)
        self.steering_angle += float(np.clip(steering * lock - self.steering_angle, -7. * dt, 7. * dt))
        self.p_target[self.steering_gc] = self.steering_angle
        self.p_gain.fill(0.)
        self.d_gain.fill(0.)
        self.p_gain[self.steering_gv] = 500.
        self.d_gain[self.steering_gv] = 25.

        left, right = velocity[self.wheel_gv[2:]]
        wheel_speed = 0.5 * (left + right)
        direction = -1. if reverse else 1.
        available = min(600. if reverse else 1500., 75000. / max(abs(wheel_speed), 1.))
        if direction * wheel_speed > 0.:
            available *= float(np.clip((78. - abs(wheel_speed)) / 8., 0., 1.))
        axle = direction * throttle * available
        limit = 0.5 * abs(axle) + 120.
        bias = float(np.clip(60. * (left - right), -limit, limit))
        self.force.fill(0.)
        self.force[self.wheel_gv[2]] = 0.5 * axle - bias
        self.force[self.wheel_gv[3]] = 0.5 * axle + bias

        hold = abs(speed) < 0.6 and (brake > 0. or throttle == 0.)
        for index, (gc, gv) in enumerate(zip(self.wheel_gc, self.wheel_gv)):
            if hold:
                if not self.held[index]:
                    self.hold_angles[index] = pose[gc]
                self.p_gain[gv], self.d_gain[gv] = 6000., 400.
                self.p_target[gc] = self.hold_angles[index]
            else:
                # A zero velocity PD target supplies implicit viscous braking
                # while preserving all joint damping specified by the URDF.
                self.d_gain[gv] = brake * (420. if index < 2 else 280.)
            self.held[index] = hold

        self.system.setPdGains(self.p_gain, self.d_gain)
        self.system.setPdTarget(self.p_target, self.v_target)
        self.system.setGeneralizedForce(self.force)
        drag = -0.5 * 1.2 * 0.95 * np.linalg.norm(velocity[:3]) * velocity[:3]
        self.system.setExternalForce(0, np.array([-0.14, 0., 0.10]), drag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throttle", type=control_value(0., 1.), default=0., help="initial throttle, from 0 to 1")
    parser.add_argument("--steer", type=control_value(-1., 1.), default=0., help="initial steering; positive turns left")
    parser.add_argument("--brake", type=control_value(0., 1.), default=0., help="initial brake, from 0 to 1")
    parser.add_argument("--reverse", action="store_true", help="drive backwards when throttle is applied")
    add_renderer_option(parser, raisim)
    args = setup(raisim, parser)

    world = raisim.World()
    world.setTimeStep(0.001)
    world.setMaterialPairProp("tire", "terrain", 0.85, 0., 0.)
    world.setMaterialPairProp("body", "terrain", 0.5, 0., 0.)
    terrain, start = load_terrain(world)
    buggy = Buggy(world, terrain, start)
    server = create_scene_renderer(raisim, world, args)
    throttle = server.addSlider("Throttle", 0., 1., args.throttle)
    steering = server.addSlider("Steering", -1., 1., args.steer)
    brake = server.addSlider("Brake", 0., 1., args.brake)
    server.focusOn(buggy.system)
    start_server(server, args)
    try:
        for _ in simulation_steps(args):
            with locked_server(server):
                # Let the suspension settle for one simulated second first.
                if world.getWorldTime() < 1.:
                    buggy.apply_controls(0., 0., 1., args.reverse, world.getTimeStep())
                else:
                    buggy.apply_controls(server.getSlider(throttle), server.getSlider(steering),
                                         server.getSlider(brake), args.reverse, world.getTimeStep())
            server.integrateWorldThreadSafe()
            pace(args, world.getTimeStep())
    finally:
        stop_server(server)
    pose = buggy.system.getGeneralizedCoordinate()
    distance = np.linalg.norm(pose[:2] - buggy.initial_position[:2])
    print(f"Buggy: travel={distance:.3f} m, steering={pose[buggy.steering_gc]:.3f} rad")
    finish_example(server, world)


if __name__ == "__main__":
    main()
