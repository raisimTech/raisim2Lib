"""A pinned cloth and a box of native granular particles."""
import numpy as np
import raisimpy as raisim

from _example_utils import (
    setup, simulation_steps, pace, start_server, finish_example
)


args = setup(raisim)
world = raisim.World()
world.setTimeStep(0.001)
world.addGround(-0.5)

size = 6
vertices = np.array([[-2. + x / (size - 1), y / (size - 1) - 0.5, 1.]
                     for y in range(size) for x in range(size)])
triangles = []
for y in range(size - 1):
    for x in range(size - 1):
        i = y * size + x
        triangles.extend([[i, i + 1, i + size], [i + 1, i + size + 1, i + size]])
cloth_material = raisim.DeformableObject.Material()
cloth_material.distanceStiffness = 1e3
cloth_material.bendStiffness = 0.01
cloth = world.addDeformableCloth(vertices, triangles, [0, size - 1], cloth_material)
cloth.setName("cloth")

box = raisim.GranularSystem.BoxOptions()
box.minCorner = np.array([-0.25, -0.25, 0.1])
box.maxCorner = np.array([0.25, 0.25, 0.6])
box.radius, box.spacing, box.maxParticles = 0.04, 0.09, 96
grains = world.addGranularBox(box)
grains.setName("grains")
grains.setGroundPlane(-0.5)

server = raisim.RaisimServer(world)
start_server(server, args)
for _ in simulation_steps(args):
    server.integrateWorldThreadSafe()
    pace(args, world.getTimeStep())
finish_example(server, world)
