from _example_utils import (
    setup, simulation_steps, pace, start_server, finish_example
)
import os
import numpy as np
import raisimpy as raisim


args = setup(raisim)
anymal_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/anymal/urdf/anymal.urdf"
laikago_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/laikago/laikago.urdf"
atlas_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/atlas/robot.urdf"
monkey_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/monkey/monkey.obj"
dummy_inertia = np.zeros([3, 3])
np.fill_diagonal(dummy_inertia, 0.1)

world = raisim.World()
world.setTimeStep(0.001)
server = raisim.RaisimServer(world)
ground = world.addGround()

options = raisim.ArticulatedSystemOption()
options.convexify_collision_meshes = True
anymal = world.addArticulatedSystem(anymal_urdf_file, options=options)
anymal.setName("anymal")
anymal_nominal_joint_config = np.array([0, -1.5, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8,
                                        -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8])
anymal.setGeneralizedCoordinate(anymal_nominal_joint_config)
anymal.setPdGains(200*np.ones([18]), np.ones([18]))
anymal.setPdTarget(anymal_nominal_joint_config, np.zeros([18]))

laikago = world.addArticulatedSystem(laikago_urdf_file, options=options)
laikago.setName("laikago")
laikago_nominal_joint_config = np.array([0, 1.5, 0.48, 1, 0.0, 0.0, 0.0, 0.0, 0.5, -1, 0, 0.5, -1,
                                         0.00, 0.5, -1, 0, 0.5, -0.7])
laikago.setGeneralizedCoordinate(laikago_nominal_joint_config)
laikago.setPdGains(200*np.ones([18]), np.ones([18]))
laikago.setPdTarget(laikago_nominal_joint_config, np.zeros([18]))

atlas = world.addArticulatedSystem(atlas_urdf_file, options=options)
atlas.setName("atlas")
atlas_nominal_joint_config = np.zeros(atlas.getGeneralizedCoordinateDim())
atlas_nominal_joint_config[2] = 1.5
atlas_nominal_joint_config[3] = 1
atlas.setGeneralizedCoordinate(atlas_nominal_joint_config)


for i in range(5):
    for j in range(5):
        object_type = (i + j*6) % 5

        if object_type == 0:
            obj = world.addMesh(monkey_file, 5.0, dummy_inertia, np.array([0., 0., 0.]), 0.3, collision_mode=raisim.MeshCollisionMode.CONVEX_HULL)
            obj.setAppearance("blue")
        elif object_type == 1:
            obj = world.addCylinder(0.2, 0.3, 2.0)
            obj.setAppearance("red")
        elif object_type == 2:
            obj = world.addCapsule(0.2, 0.3, 2.0)
            obj.setAppearance("green")
        elif object_type == 3:
            obj = world.addBox(0.4, 0.4, 0.4, 2.0)
            obj.setAppearance("purple")
        else:
            obj = world.addSphere(0.3, 2.0)
            obj.setAppearance("orange")

        obj.setPosition(i-2.5, j-2.5, 5)

world.integrate1()

### get dynamic properties
# mass matrix
mass_matrix = anymal.getMassMatrix()
# non-linear term (gravity+coriolis)
gravity = np.array([0.0, 0.0, -9.81])
non_linearities = anymal.getNonlinearities(gravity)
# Jacobians
jaco_foot_lh_linear = anymal.getDenseFrameJacobian("LF_ADAPTER_TO_FOOT")
jaco_foot_lh_angular = anymal.getDenseFrameRotationalJacobian("LF_ADAPTER_TO_FOOT")

world.integrate2()
start_server(server, args)

for i in simulation_steps(args, 500000):
    server.integrateWorldThreadSafe()
    pace(args, world.getTimeStep())

finish_example(server, world)
