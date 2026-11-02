from _example_utils import (
    setup, simulation_steps, pace, start_server, finish_example, locked_server
)
"""Replay a bouncing ball from the same complete physics checkpoint."""
import raisimpy as raisim



args = setup(raisim)
world = raisim.World()
world.setTimeStep(0.002)
world.addGround()
ball = world.addSphere(0.15, 1.)
ball.setPosition(0., 0., 2.)
checkpoint = world.captureCheckpoint()

server = raisim.RaisimServer(world)
start_server(server, args)
for step in simulation_steps(args):
    if step and step % 1000 == 0:
        with locked_server(server):
            world.restoreCheckpoint(checkpoint, raiseOnError=True)
    server.integrateWorldThreadSafe()
    pace(args, world.getTimeStep())
finish_example(server, world)
