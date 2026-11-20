# Server Example: Suction Cup Picking

A Kinova arm with a vacuum suction cup stacks four parcels and a can into a
tower, then puts each back where it was, and starts over. The cup seals on the
top face of each object, carries it hanging from the cup, and lets go when the
vacuum is switched off.

<video class="doc-video" src="../../../../rsc/docs/video/examples/suction_cup_picking.mp4" poster="../../../../rsc/docs/video/examples/suction_cup_picking.jpg" controls loop muted playsinline preload="metadata" aria-label="A Kinova arm with a suction cup lifting a parcel off a table and stacking it on a tower"></video>

A ring around the cup shows the vacuum: green while the cup holds an object,
orange while the vacuum is on but the cup has not sealed.

## The suction cup

`rsc/kinova_suction/kinova_suction.urdf` is the Kinova arm of the
[Robotiq examples](./robotiq_gripper_mimic.md) with a vacuum tool on its
wrist: an ejector on a short shaft and a bellows cup 50 mm across. A URDF
`<gripper>` makes the cup grip:

```xml
<gripper name="suction_cup" link="suction_tool" type="suction" max_force="80"
         range="0.005" seal_angle="0.35">
    <origin xyz="0 0 0.145"/>
    <axis xyz="0 0 1"/>
</gripper>
```

The cup seals on a surface within 5 mm of its lip that faces it within 0.35 rad
(20°), and pulls it with 80 N, about what a 50 mm cup gives at 40 kPa below
atmospheric pressure. Where the object touches the lip, the lip pushes back, so
the pull presses the object onto the cup. The cup holds a parcel against a load
of up to 80 N away from it, and friction at the lip holds it against a sideways
load of up to the friction coefficient times the force that presses it onto the
lip: the pull, less the weight of an object that hangs from the cup. The parcels weigh 0.15 to
0.3 kg and the can 0.25 kg, so the arm can carry them briskly. A cup that pulls
with only 1 N leaves the 0.3 kg parcel on the table.

```cpp
robot->setGripperActive("suction_cup", true);   // vacuum on
const auto& state = robot->getGripperState(robot->getGripperIndex("suction_cup"));
if (state.engaged) { /* state.object is held with state.force newtons */ }
```

## Picking and placing

For each object the arm:

1. moves the cup above the object's top face,
2. switches the vacuum on and comes down until the cup is 4 mm into the top
   face, so its lip seals against it,
3. waits 0.25 s, lifts the object clear of everything on the table and carries
   it over its place,
4. lowers it until it is 3 mm above where it will rest, switches the vacuum off
   and backs away; the object drops onto the table or the tower.

The tower is built with the largest parcel at the bottom, and taken apart from
the top. The arm reads where each object is before it picks it, so an object
that was nudged is still picked at its center.

The cup always points straight down. Every step,
`ArticulatedSystem::solveIK()` turns the cup's target pose into joint angles,
starting from the previous solution so the arm stays in one configuration, and
the joints' PD controllers follow them. The arm stands on the table: a Kinova
arm pointing its tool straight down reaches little more than 0.45 m from its
base at the height of a tower on a low bench.

## Build and run

CMake target: `suction_cup_picking`. Start `rayrai_tcp_viewer`, then the
example:

```bash
./build-examples/examples/suction_cup_picking
```

On Windows, run `build-examples\bin\suction_cup_picking.exe`. The scene and
the controller are in `examples/include/suction_picking.hpp`.

## Details

- Building the tower and taking it apart again takes about 80 s.
- From the seal to the release, the arm checks that the cup holds the object it
  picked. If the cup does not seal or the object tears off, the arm backs off
  and tries again. It picks up an object lying on any of its faces, from the
  face on top, but not one that rests tilted: a cup that points straight down
  seals only on a face within about 10° of level.
- The arm's collision bodies are the primitive envelopes of the Kinova/Robotiq
  model; the tool collides as three cylinders.
- See [Thrusters and Grippers](../../articulated_system/ThrustersAndGrippers.md)
  for the gripper model.
