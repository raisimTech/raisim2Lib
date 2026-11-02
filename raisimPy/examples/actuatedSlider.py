from _example_utils import (
    setup, simulation_steps, pace, start_server, finish_example, locked_server
)
"""An IK target drives a slider through a torque-limited DC actuator."""
import math
import numpy as np
import raisimpy as raisim



args = setup(raisim)
world = raisim.World()
world.setTimeStep(0.001)
world.addGround()
robot = world.addArticulatedSystem("""<robot name="actuated_slider">
  <link name="world"/>
  <link name="carriage">
    <inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="world"/><child link="carriage"/><origin xyz="0 0 0.5"/>
    <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="100" velocity="10"/>
  </joint>
</robot>""")

motor = raisim.DcMotorParameters()
motor.resistance = 1.
motor.torqueConstant = 0.5
motor.velocityConstant = 2.
motor.busVoltage = 24.
motor.peakTorque = 10.
drive = raisim.ActuatorDefinition()
drive.name, drive.joint, drive.motor = "slide_motor", "slide", motor
robot.setActuators([drive])
options = raisim.IKOptions()
options.applySolution = False

server = raisim.RaisimServer(world)
target = server.addVisualSphere("target", 0.04, 1., 0.2, 0.2, 1.)
start_server(server, args)
for _ in simulation_steps(args):
    with locked_server(server):
        position = np.array([0.7 * math.sin(world.getWorldTime()), 0., 0.5])
        result = robot.solveIK("slide", position, options)
        if not result.success:
            raise RuntimeError(result.message)
        command = 40. * (result.solution[0] - robot.getGeneralizedCoordinate()[0])
        command -= 8. * robot.getGeneralizedVelocity()[0]
        robot.setActuatorTorque("slide_motor", command)
        target.setPosition(position)
    server.integrateWorldThreadSafe()
    pace(args, world.getTimeStep())
finish_example(server, world)
