from _example_utils import (
    setup, simulation_steps, pace, start_server, finish_example, locked_server
)
import math

import numpy as np
import raisimpy as raisim


args = setup(raisim)

world = raisim.World()
world.setTimeStep(0.01)
world.addGround()

server = raisim.RaisimServer(world)

# Charts are currently rendered only in RaisimUnreal.
series = server.addTimeSeriesGraph("Signals", ["sin", "cos"], "time (s)", "value")
bars = server.addBarChart("Bars", ["A", "B", "C", "D"])

start_server(server, args)

for _ in simulation_steps(args):
    with locked_server(server):
        t = world.getWorldTime()
        series.addDataPoints(t, np.array([math.sin(t), math.cos(t)], dtype=np.float64))

        bars.setData([
            abs(math.sin(t)),
            abs(math.sin(t + 0.7)),
            abs(math.sin(t + 1.4)),
            abs(math.sin(t + 2.1)),
        ])
    server.integrateWorldThreadSafe()
    pace(args, world.getTimeStep())

finish_example(server, world)
