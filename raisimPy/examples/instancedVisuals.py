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

count = 60
size = np.array([0.15, 0.15, 0.15], dtype=np.float64)
color1 = np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float64)
color2 = np.array([1.0, 0.2, 0.4, 1.0], dtype=np.float64)

inst = server.addInstancedVisuals("inst_boxes", raisim.ShapeType.Box, size, color1, color2)
inst.resize(count)

for i in range(count):
    angle = 2.0 * math.pi * i / count
    pos = np.array([2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.6], dtype=np.float64)
    inst.setPosition(i, pos)
    inst.setOrientation(i, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    inst.setScale(i, np.array([1.0, 1.0, 1.0], dtype=np.float64))
    inst.setColorWeight(i, i / float(count - 1))

start_server(server, args)

t = 0.0
for _ in simulation_steps(args):
    with locked_server(server):
        t += world.getTimeStep()
        for i in range(count):
            angle = 2.0 * math.pi * i / count + t
            height = 0.6 + 0.2 * math.sin(angle * 2.0)
            pos = np.array([2.0 * math.cos(angle), 2.0 * math.sin(angle), height], dtype=np.float64)
            inst.setPosition(i, pos)
            inst.setColorWeight(i, 0.5 + 0.5 * math.sin(angle))
    server.integrateWorldThreadSafe()
    pace(args, world.getTimeStep())

finish_example(server, world)
