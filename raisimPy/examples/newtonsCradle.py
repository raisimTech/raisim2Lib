from _example_utils import (
    setup, simulation_steps, pace, start_server, finish_example, locked_server
)
import os
import numpy as np
import raisimpy as raisim
import math


args = setup(raisim)
world = raisim.World()
ground = world.addGround()
world.setTimeStep(0.001)

world.setMaterialPairProp("steel", "steel", 0.1, 1.0, 0.0)

pin1 = world.addSphere(0.1, 0.8)
pin1.setAppearance("1,0,0,0.3")
pin1.setPosition(0.0, 0.0, 3.0)
pin1.setBodyType(raisim.BodyType.STATIC)

pin2 = world.addSphere(0.1, 0.8)
pin2.setAppearance("0,1,0,0.3")
pin2.setPosition(0.3, 0.0, 3.0)
pin2.setBodyType(raisim.BodyType.STATIC)

pin3 = world.addSphere(0.1, 0.8)
pin3.setAppearance("0,0,1,0.3")
pin3.setPosition(0.6, 0.0, 3.0)
pin3.setBodyType(raisim.BodyType.STATIC)

pin4 = world.addSphere(0.1, 0.8)
pin4.setAppearance("1,0,0,0.3")
pin4.setPosition(0.9, 0.0, 3.0)
pin4.setBodyType(raisim.BodyType.STATIC)

pin5 = world.addSphere(0.1, 0.8)
pin5.setPosition(0.9, 0.0, 6.0)
pin5.setBodyType(raisim.BodyType.STATIC)

pin6 = world.addSphere(0.1, 0.8)
pin6.setPosition(-3., 0.0, 7.0)
pin6.setBodyType(raisim.BodyType.STATIC)

pin7 = world.addSphere(0.1, 0.8)
pin7.setPosition(-4., 0.0, 7.0)
pin7.setBodyType(raisim.BodyType.STATIC)

anymalB_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/anymal/urdf/anymal.urdf"
anymalC_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/anymal_c/urdf/anymal.urdf"

options = raisim.ArticulatedSystemOption()
options.convexify_collision_meshes = True
anymalC = world.addArticulatedSystem(anymalC_urdf_file, options=options)
anymalB = world.addArticulatedSystem(anymalB_urdf_file, options=options)

jointNominalConfig = np.array([-3, 0, 4.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8])
jointVelocityTarget = np.zeros([anymalC.getDOF()])
jointPgain = np.ones(anymalC.getDOF()) * 100.0
jointDgain = np.ones(anymalC.getDOF()) * 1.0

anymalC.setGeneralizedCoordinate(jointNominalConfig)
anymalC.setPdGains(jointPgain, jointDgain)
anymalC.setPdTarget(jointNominalConfig, jointVelocityTarget)
anymalC.setName("anymalC")

jointNominalConfig[0] = -4
anymalB.setGeneralizedCoordinate(jointNominalConfig)
anymalB.setPdGains(jointPgain, jointDgain)
anymalB.setPdTarget(jointNominalConfig, jointVelocityTarget)
anymalB.setName("anymalB")

ball1 = world.addSphere(0.1498, 0.8, "steel")
ball1.setPosition(0, 0.0, 1.0)

ball2 = world.addSphere(0.1499, 0.8, "steel")
ball2.setPosition(0.3, 0.0, 1.0)

ball3 = world.addSphere(0.1499, 0.8, "steel")
ball3.setPosition(0.6, 0.0, 1.0)

ball4 = world.addSphere(0.1499, 0.8, "steel")
ball4.setPosition(2.9, 0.0, 3.0)

box = world.addBox(.1, .1, .1, 1)
box.setPosition(0.9, 0.0, 4.2)

Path = raisim.Tendon.PathElement
Site = raisim.Tendon.Site
cable = raisim.Tendon.Properties()
cable.upperLimit = 2.0
cable.width = 0.02
for index, (pin, ball) in enumerate([(pin1, ball1), (pin2, ball2), (pin3, ball3), (pin4, ball4)]):
    world.addSpatialTendon(f"cradle_{index}", [Path.via(Site(pin)), Path.via(Site(ball))], cable)

spring = raisim.Tendon.Properties()
spring.springLower = spring.springUpper = 2.0
spring.stiffness = 200.0
spring.width = 0.02
world.addSpatialTendon("box_spring", [Path.via(Site(pin5)), Path.via(Site(box))], spring)
spring.stiffness = 1000.0
world.addSpatialTendon("robot_spring", [Path.via(Site(pin6)), Path.via(Site(anymalC))], spring)
actuator = world.addSpatialTendon("robot_lift", [Path.via(Site(pin7)), Path.via(Site(anymalB))])
actuator.setTension(310.0)

server = raisim.RaisimServer(world)
start_server(server, args)

for i in simulation_steps(args, 500000):
    pace(args, 0.001)
    server.integrateWorldThreadSafe()
    if i == 5000:
        with locked_server(server):
            world.removeTendon(actuator)

finish_example(server, world)
