# Thrusters and Grippers

An articulated system can carry two kinds of devices that push and pull with forces of their own,
fixed to its links:

- a **thruster** pushes its link along an axis: a propeller, a jet or a water thruster;
- a **gripper** pulls an object toward a pad on its link: a **suction** cup or an **electromagnet**.

Both are written in the URDF as children of `<robot>`, or set in code. Their forces act on the
links like forces set with `setExternalForce()` before every step, so they move the robot through
its joints, show up in the viewers as external forces, and work with every other feature of the
system.

## Thrusters

```xml
<thruster name="rotor_front" link="base" torque_coefficient="0.016">
    <origin xyz="0.2 0 0.02" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="8"/>
</thruster>
```

| Element | Meaning |
| --- | --- |
| `name` | Thruster name (unique in the robot). |
| `link` | The link the thruster is fixed to. A link fixed to its parent is merged into the parent as usual. |
| `torque_coefficient` | The reaction torque about the axis per newton of thrust [N m/N] (optional, default 0). A propeller that spins one way turns the link the other way; propellers that spin in opposite directions have opposite signs. |
| `<origin>` | Where the thruster sits on the link, as a joint origin (optional, default: the link origin). |
| `<axis xyz>` | The direction of the thrust in the frame of `<origin>`, like a joint axis (optional, default `0 0 1`). |
| `<limit lower upper>` | The smallest and largest thrust [N] (optional, default: unlimited). |

With the thrust $T$, the unit axis $\boldsymbol{a}$ and the reaction-torque coefficient $c$,
the thruster applies the force and the torque

$$
\boldsymbol{f} = T\,\boldsymbol{a}, \qquad \boldsymbol{\tau} = c\,T\,\boldsymbol{a}
$$

to its link, the force at the thruster's position. The thrust is a command: it holds until it is
set again, clamped to the limits.

```cpp
auto* quad = world.addArticulatedSystem("quadrotor.urdf");
quad->setThrust("rotor_front", 3.7);                  // by name
quad->setThrusts(Eigen::VectorXd::Constant(4, 3.7));  // all, in the order of getThrusters()
double applied = quad->getThrust(0);                  // the command after clamping
raisim::Vec<3> position, axis;
quad->getThrusterPose(0, position, axis);             // in the world frame
```

A quadrotor hovers with four thrusts that add up to its weight. Two propellers spin one way and two
the other, so their reaction torques cancel; shifting thrust from one pair to the other, at the
same total, turns the quadrotor about its vertical axis without lifting or tilting it.

`setThrusters()` replaces the thrusters at run time and `getThrusters()` returns them, with
positions and unit axes in the link frame.

## Grippers

```xml
<gripper name="cup" link="tool" type="suction" max_force="60" range="0.005"
         seal_angle="0.35" cup_radius="0.025" leak_coefficient="4">
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
</gripper>
<gripper name="magnet" link="tool" type="magnetic" max_force="200" range="0.01">
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
</gripper>
```

| Element | Meaning |
| --- | --- |
| `name` | Gripper name (unique in the robot). |
| `link` | The link the gripper is fixed to. |
| `type` | `suction` or `magnetic`. |
| `max_force` | Suction: the pull of a sealed cup [N], the vacuum times the cup area. Magnetic: the pull on a magnetic body that touches the pad [N]. |
| `range` | Maximum axial gap [m] (default 0.005). With rim sampling, a rim sample at this gap counts as fully open. |
| `seal_angle` | Suction only. Maximum angle between the surface normal and the reversed cup axis [rad] (default 0.35, about 20 degrees). |
| `cup_radius` | Suction only. A positive radius [m] enables rim sampling. The default of zero uses the center ray alone. |
| `leak_coefficient` | Suction only. Non-negative, dimensionless leak sensitivity (default 4). |
| `<origin>` | The center of the pad on the link. |
| `<axis xyz>` | The direction the pad faces, in the frame of `<origin>` (default `0 0 1`). |

The pad should lie on the surface of a collision body of the link: that body is what a held object
rests against.

### How a gripper pulls

While a gripper is on, every step it looks along its axis for the first surface in front of the pad.
If the gap $d$ between the pad and that surface is within the range $r$, it pulls the surface
toward the pad with the force

$$
F = \begin{cases}
F_\mathrm{max} & \text{suction, center ray alone},\\
F_\mathrm{max}\,(1 - d/r)^2 & \text{magnetic, on a magnetic material},
\end{cases}
$$

The gripper pulls its own link toward the surface with the opposite force, both along the axis.
The suction cup requires a center hit that faces it within `seal_angle`. A magnet pulls only
bodies whose collision material is magnetic (see below).

With a positive `cup_radius`, the cup tests eight evenly spaced points around the rim. Each
sample contributes its axial gap divided by `range`, clamped to zero through one. A sample
counts as fully open if it misses, hits another body, or hits a surface outside the angle limit.
If $o$ is the mean opening and $k$ is `leak_coefficient`, the resulting pull is

$$
F = F_\mathrm{max} \frac{1-o}{1+k\,o}.
$$

A closed rim gives full force. A fully open rim gives zero force. Use `cupRadius` and
`leakCoefficient` to configure the same model in C++ and Python.

The search starts 1 cm behind the pad, so a surface the pad touches or has pressed slightly into is
found. A solid whose surface lies behind the pad counts only if it reaches the pad, and the face
where the search leaves a solid is never taken for the surface in front: a solid around the pad that
ends in front of it blocks the search instead of letting it pull a body farther away. Triangle meshes
have no inside, so their surfaces count from both sides, but only up to 2 mm behind the pad.

Where the pad touches the object, the contacts push back, so the pull presses the object onto the
pad. This gives a gripper the strength of a real one:

- it holds an object against a load of up to $F$ away from the pad (a box heavier than the pull
  stays on the table);
- friction holds it against a sideways load of up to $\mu F$, with the friction coefficient $\mu$
  of the two materials;
- the rim of the pad holds it against tilting, up to $F$ times the distance from the pad center to
  the rim.

A load beyond these tears the object off: it leaves the range (or tilts past the seal angle) and the
gripper lets go. A gripper also lets go when it is switched off.

A gripper pulls single bodies and the links of articulated systems. A static object (the ground, a
wall, a heightmap) cannot move, so the gripper pulls its own link toward it instead: a robot with
suction cups or magnets anchors itself to the floor or climbs a wall. A gripper does not pull
deformable objects or particles, or its own robot.

```cpp
robot->setGripperActive("cup", true);
world.integrate();
const auto& state = robot->getGripperState(robot->getGripperIndex("cup"));
if (state.engaged)  // state.object, state.localIdx, state.force [N], state.gap [m], state.position
  std::cout << state.object->getName() << " pulled with " << state.force << " N\n";
```

`getGripperState()` reports what the gripper did in the last step; nothing while it is off or its
robot sleeps. A gripper that reaches a sleeping object wakes it, and an object that comes into
reach of a sleeping robot's gripper wakes the robot. A robot whose thrusters or grippers act does not
fall asleep, nor does what it holds.

### Magnetic materials

Magnetic grippers pull only magnetic materials. A material is magnetic once it is marked so:

```cpp
world.setMagneticMaterial("steel");                  // in code
auto* part = world.addBox(0.1, 0.1, 0.02, 0.8, "steel");
```

```xml
<raisim>
    <material>
        <magnetic name="steel"/>
    </material>
</raisim>
```

### Limitations

- A gripper sees only the first surface along its axis: a thin non-magnetic plate in front of a
  steel part blocks a magnet.
- The pull acts at the point of the surface on the axis, so a cup sealed on the edge of an object
  pulls there.
- Rim sampling estimates geometric leaks at eight points. The model updates the pull immediately
  and assumes an impermeable workpiece.
- A gripper ignores collision groups: it pulls what is in front of its pad even if its link does
  not collide with it, and such an object is pulled into the link.

## Examples

- [Drone Flight](../examples/server/drone_flight.md): a quadrotor flown by four
  thrusters takes off, flies figure eights around two posts facing the way it
  flies, and lands; the reaction torques of its propellers turn it.
- [Magnetic Wall Climbing](../examples/server/magnetic_wall_climbing.md): a
  quadruped with magnetic feet trots up a steel wall, switching each magnet off
  just before its foot lifts.
- [Rocket Hop](../examples/server/rocket_hop.md): a rocket steers with a
  thruster on a gimballed engine, holds its roll with four small thrusters, and
  lands on its legs.
- [Suction Cup Picking](../examples/server/suction_cup_picking.md): an arm with
  a suction cup stacks parcels into a tower and takes it apart again.

## Saving, loading and joining

A world saved with `World::exportToXml()` keeps the thrusters and grippers of each articulated
system, with their thrusts and whether each gripper is on, and the magnetic materials. On loading,
these replace the ones of the robot description, so thrusters and grippers set with
`setThrusters()` and `setGrippers()` are kept too. Checkpoints (`World::captureCheckpoint()`) also
keep them. `World::joinObjects()` and `World::detachJoint()` keep the thrusters and grippers on
their links, with their thrusts and whether each gripper is on; the child's names get
`JoinOptions::childPrefix`.
