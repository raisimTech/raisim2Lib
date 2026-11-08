from _example_utils import (
    setup, simulation_steps, pace, start_server, finish_example, locked_server
)
import math

import numpy as np
import raisimpy as raisim


args = setup(raisim)

world = raisim.World()
world.setTimeStep(0.002)
world.addGround()

server = raisim.RaisimServer(world)

point_cloud = server.addPointCloud("scan")
point_cloud.pointSize = 0.03

count = 200
positions = np.zeros((count, 3), dtype=np.float64)
colors = np.zeros((count, 4), dtype=np.float64)
point_cloud.resize(count)

start_server(server, args)

t = 0.0
for _ in simulation_steps(args):
    t += world.getTimeStep()
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        radius = 1.5 + 0.3 * math.sin(t * 2.0 + angle)
        positions[i, 0] = radius * math.cos(angle)
        positions[i, 1] = radius * math.sin(angle)
        positions[i, 2] = 0.4 + 0.2 * math.sin(t + angle * 3.0)
        colors[i, 0] = 0.5 + 0.5 * math.sin(angle)
        colors[i, 1] = 0.5 + 0.5 * math.sin(angle + 2.1)
        colors[i, 2] = 0.6
        colors[i, 3] = 1.0

    with locked_server(server):
        point_cloud.position = positions
        point_cloud.color = colors
    server.integrateWorldThreadSafe()
    pace(args, world.getTimeStep())

finish_example(server, world)
