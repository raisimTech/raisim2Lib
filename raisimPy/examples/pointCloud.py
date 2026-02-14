import os
import math
import time

import numpy as np
import raisimpy as raisim


rsc_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/"
raisim.World.setLicenseFile(rsc_dir + "activation.raisim")

world = raisim.World()
world.setTimeStep(0.002)
world.addGround()

server = raisim.RaisimServer(world)
server.launchServer(8080)

point_cloud = server.addPointCloud("scan")
point_cloud.pointSize = 0.03

count = 200
positions = np.zeros((count, 3), dtype=np.float64)
colors = np.zeros((count, 4), dtype=np.float64)
point_cloud.resize(count)

t = 0.0
while True:
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

    point_cloud.position = positions
    point_cloud.color = colors

    server.integrateWorldThreadSafe()
    time.sleep(world.getTimeStep())
