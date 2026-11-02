"""A rigid load settles into a deformable soil surface."""
import raisimpy as raisim

from _example_utils import (
    setup, simulation_steps, pace, start_server, finish_example
)


args = setup(raisim)
world = raisim.World()
world.setTimeStep(0.001)
soil = world.addDeformableTerrain(21, 21, 2., 2., 0., 0., [0.] * (21 * 21))
soil.setName("soil")
load = world.addBox(0.25, 0.25, 0.2, 5.)
load.setPosition(0., 0., 0.3)
server = raisim.RaisimServer(world)
start_server(server, args)
for _ in simulation_steps(args):
    server.integrateWorldThreadSafe()
    pace(args, world.getTimeStep())
finish_example(server, world)
