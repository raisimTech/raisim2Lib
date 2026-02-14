import os
import math
import time

import raisimpy as raisim


rsc_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/"
raisim.World.setLicenseFile(rsc_dir + "activation.raisim")

world = raisim.World()
world.setTimeStep(0.01)
world.addGround()

server = raisim.RaisimServer(world)
server.launchServer(8080)

x_samples = 40
y_samples = 40
x_size = 12.0
y_size = 12.0
center_x = 0.0
center_y = 0.0


def build_heights(phase):
    heights = []
    for y in range(y_samples):
        for x in range(x_samples):
            x_pos = (x / float(x_samples - 1) - 0.5) * x_size
            y_pos = (y / float(y_samples - 1) - 0.5) * y_size
            height = 0.6 * math.sin(0.5 * x_pos + phase) * math.cos(0.5 * y_pos + phase)
            heights.append(height)
    return heights


height_map = server.addVisualHeightMap(
    "hm_visual",
    x_samples,
    y_samples,
    x_size,
    y_size,
    center_x,
    center_y,
    build_heights(0.0),
    0.2,
    0.6,
    0.2,
    1.0,
)
height_map.setColor(0.2, 0.6, 0.2, 1.0)

phase = 0.0
while True:
    phase += 0.05
    height_map.update(center_x, center_y, x_size, y_size, build_heights(phase))

    server.integrateWorldThreadSafe()
    time.sleep(world.getTimeStep())
