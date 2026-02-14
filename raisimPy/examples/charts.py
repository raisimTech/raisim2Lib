import os
import math
import time

import numpy as np
import raisimpy as raisim


rsc_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/"
raisim.World.setLicenseFile(rsc_dir + "activation.raisim")

world = raisim.World()
world.setTimeStep(0.01)
world.addGround()

server = raisim.RaisimServer(world)
server.launchServer(8080)

# Charts are currently rendered only in RaisimUnreal.
series = server.addTimeSeriesGraph("Signals", ["sin", "cos"], "time (s)", "value")
bars = server.addBarChart("Bars", ["A", "B", "C", "D"])

while True:
    t = world.getWorldTime()
    series.addDataPoints(t, np.array([math.sin(t), math.cos(t)], dtype=np.float64))

    bars.setData([
        abs(math.sin(t)),
        abs(math.sin(t + 0.7)),
        abs(math.sin(t + 1.4)),
        abs(math.sin(t + 2.1)),
    ])

    server.integrateWorldThreadSafe()
    time.sleep(world.getTimeStep())
