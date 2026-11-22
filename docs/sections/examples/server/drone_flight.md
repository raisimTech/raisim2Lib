# Server Example: Drone Flight

A 1.5 kg quadrotor drone takes off from its pad, flies two figure eights around
two posts at 1.5 m, always facing the way it flies, and lands on the pad again.
Two seconds later it takes off for the next flight. Four URDF thrusters, one
under each propeller, fly it.

<video class="doc-video" src="../../../../rsc/docs/video/examples/drone_flight.mp4" poster="../../../../rsc/docs/video/examples/drone_flight.jpg" controls loop muted playsinline preload="metadata" aria-label="A quadrotor drone flying a figure eight around two red posts above its landing pad"></video>

## The drone

The drone is a single body with four `<thruster>`s in the plus layout, 0.2 m
from its center: front (the orange propeller), back, left and right. Each
pushes along the body's vertical axis with up to 15 N:

```xml
<thruster name="front" link="base" torque_coefficient="0.016">
    <origin xyz="0.2 0 0.03"/>
    <limit lower="0" upper="15"/>
</thruster>
<thruster name="left" link="base" torque_coefficient="-0.016">
    <origin xyz="0 0.2 0.03"/>
    <limit lower="0" upper="15"/>
</thruster>
<!-- back like front, right like left -->
```

A spinning propeller drags on the air, so the air turns the drone the other
way: `torque_coefficient` gives each thruster a torque of 0.016 N m per newton
of thrust about its axis. The front and back propellers spin one way and the
left and right ones the other, so in a hover, with equal thrusts, the torques
cancel. Making the front and back propellers push harder than the left and
right ones, by the same total, turns the drone about its vertical axis without
lifting it; pushing harder on one side than the other tilts it.

```cpp
drone->setThrusts(thrusts);   // front, back, left, right [N]; held until set again
```

## Flying

The drone follows the figure eight $x = 2.5\sin\alpha$, $y = \sin 2\alpha$ at
1.5 m, one loop every 14 s. Its speed rises smoothly from rest over the pad and
falls back to rest at the end, with up to 1.4 m/s where the two loops cross.

- **Thrust.** The controller wants the path's acceleration plus a correction
  toward the path. The total force that gives it, $m(\boldsymbol{a} -
  \boldsymbol{g})$, sets where the drone's vertical axis should point and,
  projected on that axis, the total thrust.
- **Attitude.** Torques about the body axes turn the drone toward that
  attitude, with its front arm toward the direction of the path. The four
  thrusts are the ones that give the total thrust and the three torques.
- **Landing.** The drone sinks onto the pad over 4 s, touches down at about
  0.15 m/s and stops its propellers once the skids are on the ground.

The drone follows the path to within 3 cm and its heading to within 0.03 rad.

## Build and run

CMake target: `drone_flight`. Start `rayrai_tcp_viewer`, then the example:

```bash
./build-examples/examples/drone_flight
```

On Windows, run `build-examples\bin\drone_flight.exe`. The viewer follows the
drone. The scene and the controller are in `examples/include/drone_flight.hpp`;
the example adds the pad and the path as visual objects.

## Details

- [Rocket Hop](./rocket_hop.md) steers a rocket by tilting its engine, and
  [Magnetic Wall Climbing](./magnetic_wall_climbing.md) and
  [Suction Cup Picking](./suction_cup_picking.md) use the grippers.
- See [Thrusters and Grippers](../../articulated_system/ThrustersAndGrippers.md)
  for the thruster model.
