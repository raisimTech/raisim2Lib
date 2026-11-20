# Server Example: Rocket Hop

A 627 kg rocket hops between two landing pads 20 m apart. It lifts off, climbs
25 m, flies across, descends, sets down on its legs at 0.35 m/s and cuts its
engine. Three seconds later it flies back. It steers as launchers do, by
tilting its engine on a gimbal, and four small thrusters at the top hold its
roll.

<video class="doc-video" src="../../../../rsc/docs/video/examples/rocket_hop.mp4" poster="../../../../rsc/docs/video/examples/rocket_hop.jpg" controls loop muted playsinline preload="metadata" aria-label="A rocket descending on its engine's flame and landing on its legs on a pad"></video>

## The rocket

`rsc/rocket/rocket.urdf` describes a 5 m tank with a nose cone, four fins and
four landing legs. The engine hangs from a two-axis gimbal at the bottom of the
tank: the revolute joints `gimbal_x` and `gimbal_y` tilt it by up to 0.15 rad.
The thrusters are URDF `<thruster>` elements:

```xml
<thruster name="main_engine" link="engine" torque_coefficient="0.002">
    <origin xyz="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="12000"/>
</thruster>
<thruster name="rcs_roll_left_0" link="body">
    <origin xyz="0.36 0 3.95"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="60"/>
</thruster>
<!-- three more roll thrusters around the tank -->
```

The main engine sits on the engine link at the gimbal point and pushes along
the nozzle, so tilting the engine tilts the thrust. Its `torque_coefficient`
gives the rocket a roll torque of 0.002 N m per newton of thrust, as the swirl
of an engine's exhaust does: about 12 N m in a hover. The roll thrusters push
sideways at the edge of the tank, two for each direction of roll; each pair
pushes in opposite directions on opposite sides, so it turns the rocket
without moving it.

```cpp
rocket->setThrust("main_engine", 6200.0);   // newtons; held until set again
rocket->setThrust(rollLeft, 17.0);
raisim::Vec<3> gimbal, axis;
rocket->getThrusterPose(rocket->getThrusterIndex("main_engine"), gimbal, axis);
```

## Flying

Each flight follows four smooth segments for the rocket's center of mass:
25 m straight up in 9 s, across to the other pad in 9 s, and down in 9 s until
it is 1 m above where it rests on the pad. It flies the last metre at a steady
0.35 m/s, as landers do, and cuts its engine as soon as the legs touch.

- **Thrust.** The controller wants the acceleration of the segment plus a
  correction toward it. The thrust vector that gives it, $m(\boldsymbol{a} -
  \boldsymbol{g})$, sets where the rocket's axis should point (at most 15° from
  the vertical) and, projected on the axis, the engine's thrust. The engine
  cannot throttle below 40 %, 4.8 kN, so it is either off or between 4.8 and
  12 kN; the rocket weighs 6.15 kN, so at the lowest throttle it descends.
- **Pitch and yaw.** The engine thrust acts at the gimbal, 1.6 m below the
  center of mass. Tilting the engine by a small angle $\delta$ turns the rocket
  with a torque of about $T\,l\,\delta$, with the thrust $T$ and the distance
  $l$. The controller tilts the engine to turn the rocket's axis toward the
  direction it wants. Servos hold the gimbal joints at the commanded angles.
- **Roll.** The controller knows the engine's swirl from its
  `torque_coefficient` and the thrust, and has the roll thrusters cancel it and
  keep the fins on the world axes. They can only push, so one pair fires at a
  time.

The rocket tilts up to 0.23 rad when it starts across, and lands within 8 cm of
the pad's center.

## Build and run

CMake target: `rocket_hop`. Start `rayrai_tcp_viewer`, then the example:

```bash
./build-examples/examples/rocket_hop
```

On Windows, run `build-examples\bin\rocket_hop.exe`. The viewer follows the
rocket. The flame under the nozzle grows with the thrust, and a puff of gas
shows each roll thruster that fires; both are visual objects of the example.
The scene and the flight controller are in `examples/include/rocket_hop.hpp`.

## Details

- The rocket's mass is constant: it does not burn propellant.
- The tank, its bands, fins and legs are URDF primitives; the nose cone and the
  bell nozzle are surfaces of revolution generated for this example. The four
  foot pads and the tank are the collision bodies.
- The pads are static cylinders 5 m across and 0.2 m high.
- See [Thrusters and Grippers](../../articulated_system/ThrustersAndGrippers.md)
  for the thruster model.
