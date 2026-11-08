from _example_utils import (
    setup, simulation_steps, pace, start_server, finish_example
)
import os
import raisimpy as raisim


args = setup(raisim)
world = raisim.World()
world.setTimeStep(0.001)

cartpole_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/springDamper/cartpole.urdf"
chain_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/springDamper/chainSpringed.urdf"

server = raisim.RaisimServer(world)
world.addGround()

options = raisim.ArticulatedSystemOption()
options.convexify_collision_meshes = True
rev_and_pris_spring_and_damper = world.addArticulatedSystem(cartpole_urdf_file, options=options)
ball_spring_and_damper = world.addArticulatedSystem(chain_urdf_file, options=options)

rev_and_pris_spring_and_damper.setName("rev_pris_joint")
ball_spring_and_damper.setName("ball_joint")

start_server(server, args)

for i in simulation_steps(args, 500000):
    pace(args, 0.001)
    server.integrateWorldThreadSafe()

finish_example(server, world)
