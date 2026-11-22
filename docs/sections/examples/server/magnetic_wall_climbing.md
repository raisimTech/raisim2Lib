# Server Example: Magnetic Wall Climbing

A Go1 quadruped with magnetic feet trots up a 4 m steel wall at 0.5 m/s,
rests at the top, climbs back down and starts over. Each foot carries a
magnet that is switched off just before the foot lifts and on again as it
lands, so the feet on the wall always hold the robot. The feet are modeled on
those of MARVEL, the climbing quadruped of KAIST's Dynamic Robot Control and
Design Laboratory (Hong et al., "Agile and versatile climbing on ferromagnetic
surfaces with a quadrupedal robot", *Science Robotics* 7, eadd1017, 2022).

<video class="doc-video" src="../../../../rsc/docs/video/examples/magnetic_wall_climbing.mp4" poster="../../../../rsc/docs/video/examples/magnetic_wall_climbing.jpg" controls loop muted playsinline preload="metadata" aria-label="A Go1 quadruped trotting up a steel wall, a red disc on each foot that holds the wall"></video>

The disc on each sole shows its magnet: red while it holds the wall, orange
while it is on but its foot has not reached the wall yet, grey while it is off.

## Magnetic feet

MARVEL's feet are electropermanent magnets with soles of magnetorheological
elastomer, a rubber filled with iron particles. A short current pulse switches
such a magnet on or off, and it holds without power in between. Each foot
holds to steel with 530 N, and its sole resists 440 N of sliding. The example
gives the Go1 the same feet: four magnetic grippers, set in code on the foot
links of the unmodified `go1.urdf`.

```cpp
world.setMagneticMaterial("steel");
auto* wall = world.addBox(0.3, 3.0, 4.0, 1.0, "steel");
wall->setBodyType(raisim::BodyType::STATIC);

raisim::GripperDefinition magnet;
magnet.name = "FR_magnet";
magnet.link = "FR_foot";
magnet.type = raisim::GripperDefinition::Type::MAGNETIC;
magnet.maxForce = 530.0;  // N, at contact
magnet.range = 0.01;      // m, where the pull vanishes
// ... axis and position, one per foot
robot->setGrippers(magnets);
robot->setGripperActive("FR_magnet", true);
```

A magnet pulls its foot toward the wall, and the wall pushes back where the
foot touches it. The pull therefore presses the foot onto the steel: pressed on
by the full pull, a foot holds against a load of up to 530 N away from the wall
and, with the sole's friction coefficient of 440/530 = 0.83 against steel, a
load of up to 440 N along it. A gap or a tilted pad weakens the pull, and the
grip with it. The robot weighs 13.1 kg, about 129 N, so two feet carry it with a
wide margin. Without its magnets the same controller loses the wall at once:
the robot slides off and falls within a second.

The pad of each magnet sits on the surface of the foot's sphere and faces the
wall when the leg stands in its nominal pose, so it is tilted against the
shank by the shank's pitch in that pose.

## Walking with magnets

The robot trots: diagonal legs move together, each pair on the wall for 70 %
of a 0.3 s period, so all four feet are on the wall for a short time at the
end of each step. In that time a magnet switches off, 20 ms before its foot
lifts, so the leg does not have to tear the foot off the wall. A swinging foot
switches its magnet back on for the last quarter of its swing; the magnet
reaches 1 cm in front of the pad and pulls the foot onto the wall as it lands.

The steps are short for a reason. Two diagonal feet hold the robot along the
line between them, and the robot's weight acts about 0.2 m out from the wall,
so gravity turns the robot about that line, away from the wall at its free
corners. The feet are points: forces on that line cannot stop the turn. Short
swings give the turn little time, and the four-foot phase stops it, so the base
tilts no more than 0.16 rad. MARVEL climbs the same way, with fast steps.

## The controller

The controller works in the frame of the wall, where the robot walks as if on
a floor, with gravity pulling backward along it. The base follows a height on
the wall that rises at up to 0.5 m/s, starting and stopping smoothly.

- A foot on the wall stays where it landed. Its joint targets come from the
  leg's inverse kinematics, between that foothold and where the base should
  be, so the stance legs carry the base along as it moves.
- A swinging foot follows a smooth arc from where it lifted to a foothold
  ahead of the hip, 4 cm off the wall at mid-swing. It aims from where the base
  actually is, so it lands on the wall even while the base is tilted.
- Every foot that a magnet holds pushes down along the wall with its share of
  the robot's weight, through the joint torques $\boldsymbol{\tau} =
  \boldsymbol{J}^\top m\boldsymbol{g}/n$, where $\boldsymbol{J}$ is the
  Jacobian of the foot and $n$ is the number of feet that hold.

The feet are Go1's spheres, so a foot on the wall rolls as its shank turns
during a step; the point of the sole on the wall does not slide.

## Build and run

CMake target: `magnetic_wall_climbing`. Start `rayrai_tcp_viewer`, then the
example:

```bash
./build-examples/examples/magnetic_wall_climbing
```

On Windows, run `build-examples\bin\magnetic_wall_climbing.exe`. The viewer
looks at the face of the wall the robot climbs, the one the viewer's default
sun lights. The scene and the controller are in
`examples/include/magnetic_wall_climber.hpp`; the example adds the seams of
the wall's plates and the magnet discs as visual objects.

## Details

- The wall is a static 0.3 m × 3 m × 4 m box of the magnetic material `steel`.
  The robot climbs its face toward −x, moving its base from 0.75 m to 3.1 m
  above the ground, and rests 1.5 s at each end.
- The magnet's pull falls off as $F_\mathrm{max}(1 - d/r)^2$ with the gap $d$
  and the range $r = 1$ cm.
- The joints are PD-controlled with gains of 120 N m/rad and 3 N m s/rad, and
  their torques are limited by Go1's motors (23.7 N m at the hips, 35.55 N m at
  the knees).
- See [Thrusters and Grippers](../../articulated_system/ThrustersAndGrippers.md)
  for the gripper model.
