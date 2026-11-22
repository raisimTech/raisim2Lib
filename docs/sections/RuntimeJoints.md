# Runtime joints

Runtime joints connect rigid objects and articulated links, including links from
different robots. Create or remove them between simulation steps for tool
changes, docking, detachable mechanisms, and temporary attachments. Their
impulses are solved together with contacts, tendons, and joint limits.

Runtime joints keep the connected objects separate. To merge objects into one
articulated system whose joint is part of its generalized coordinates, exact
and as fast as a robot loaded from a file, use
[`World::joinObjects()`](./ArticulatedJoins.md); that page compares the two.

## Joint types and attachment frames

Each `RuntimeJoint::Attachment` contains an object, a local body index, a position,
and an orientation. Positions and orientations are expressed in that body's
frame. Use body index `0` for a single rigid body, or
`robot->getBodyIdx("link_name")` for an articulated link. A null object denotes
an attachment frame fixed in world coordinates; its local index must be zero.
Use a proper rotation matrix for the orientation. Fixed links merged by the
robot loader must be expressed in their surviving body's frame.

| Type | Constrained motion | Free motion |
| --- | --- | --- |
| `Fixed` | Relative position and orientation | None |
| `Ball` | Origins coincide | All relative rotation |
| `Hinge` | Origins coincide and local X axes align | Rotation about local X |
| `Slider` | Relative orientation and displacement perpendicular to frame A's X axis | Translation along frame A's X axis |

The two joint frames should initially satisfy these conditions. Small errors
are corrected by `positionCorrection`, the fraction corrected per physics
step (default `0.2`, allowed range `[0, 0.3]`). Set it to zero for velocity-only
enforcement. The slider guide follows frame A when that frame rotates.

Creation requires at least one dynamic endpoint. Existing joints survive later
changes of body type, and XML import preserves them. Static and kinematic bodies
receive no constraint impulses; a kinematic body follows its prescribed motion.
Deformable and granular particles are not rigid joint endpoints; use tendons for
those attachments.

## C++: attach and release a tool

```cpp
raisim::World world;
auto* wrist = world.addBox(0.2, 0.2, 0.2, 1.);
wrist->setBodyType(raisim::BodyType::KINEMATIC);
wrist->setPosition(0., 0., 1.);
auto* tool = world.addBox(0.2, 0.2, 0.2, 1.);
tool->setPosition(0., 0., 0.8);

using Joint = raisim::RuntimeJoint;
Joint::Properties properties;
properties.breakForce = 200.;
properties.breakTorque = 20.;
auto* mount = world.addRuntimeJoint("tool_mount", Joint::Type::Fixed,
    {wrist, 0, {0., 0., -0.1}}, {tool, 0, {0., 0., 0.1}}, properties);

world.integrate();
const auto force = mount->getForce();
const auto torque = mount->getTorque();
world.removeRuntimeJoint(mount);
```

For a robot link, replace the wrist attachment with
`{robot, robot->getBodyIdx("wrist"), localPosition, localOrientation}`.
Both endpoints may belong to articulated systems, or to different links of the
same system. Internal pin and mimic constraints remain enforced.

The `runtime_joints` example in `examples/src/worlds` includes tool attachment
and release and docking two independent robot carriages. Its dedicated
[example page](./examples/worlds/runtime_joints.md) has recorded videos, figures,
and viewer commands. It serves the world to `rayrai_tcp_viewer`.
From the `raisim2Lib` root:

```bash
cmake --build build-examples --target runtime_joints --parallel 12
./build-examples/examples/runtime_joints
```

On Windows, add `--config Release` and run
`build-examples\bin\runtime_joints.exe`. Pass `--activation-key FILE`
to select an activation key explicitly. The CTest checks `testRuntimeJointExample`
and `testRuntimeJointDockingExample` step both scenes without a viewer and check
their physics.

## Python

```python
import raisimpy as r

world = r.World()
load = world.addSphere(0.1, 1.0)
joint = world.addRuntimeJoint(
    "hinge", r.RuntimeJoint.Type.Hinge,
    r.RuntimeJoint.Attachment(),
    r.RuntimeJoint.Attachment(load),
)
world.integrate()
force = joint.getForce()
properties = joint.getProperties()
properties.breakTorque = 10.0
joint.setProperties(properties)
joint.setEnabled(False)
world.removeRuntimeJoint(joint)
```

`getProperties()` and the attachment getters return copies in Python. Update
properties through `setProperties()` so validation and waking take place.

The runnable `raisimPy/examples/runtimeJoints.py` example connects the carriages
of two independent articulated robots. A force applied to the first robot
accelerates both while docked; removing the joint lets their velocities diverge.
It prints the shared-velocity error, reaction force, and velocity difference
after release. From the `raisim2Lib` root with `raisimpy` installed:

```bash
python raisimPy/examples/runtimeJoints.py --activation-key rsc/activation.raisim --steps 1000
```

## Loads, breaking, collision, and lifetime

`getForce()` and `getTorque()` report the force and pure couple on endpoint A,
in world coordinates. Opposite loads act on endpoint B. The force acts at each
frame's origin for fixed, ball, and hinge joints. For a slider, it acts at
frame B's origin on both bodies, allowing a rotating guide to carry an extended
carriage. A moment caused by the force's lever arm is separate from the reported
pure couple. Force sensors include joint loads when their body selection can
resolve the attachment; ambiguous loads across a merged fixed-joint sensor cut
are rejected, as with tendon and pin loads.

Break thresholds default to infinity. A joint breaks when the magnitude of its
solved force or pure couple exceeds the corresponding threshold. Breakage is
latched: `isBroken()` stays true until `resetBreakage()`. Breaking is detected
after the solve, so the breaking physics step still applies its solved impulse;
subsequent steps leave the joint inactive. In a substepped world, threshold
checks occur per substep and returned loads are averaged over the complete
`integrate()` call. Explicit `integrate1()` / `integrate2()` calls report that
pair's step. `isEnabled()` is independent of breakage; resetting breakage does
not enable a disabled joint.

By default, collisions between the two attached local bodies are suppressed
while the joint is enabled and unbroken. Set `collideConnected = true` to allow
them. Other links retain their collision filters. A null world attachment does
not suppress collisions with ground objects.

Connected dynamic bodies form a sleep island. Moving an anchor, applying a load,
changing joint properties, enabling/disabling a joint, or removing one wakes
affected bodies. A sleeping joint retains its last support load.

The world owns each joint. `getRuntimeJoint(name)` looks it up and
`getRuntimeJoints()` enumerates the joints in creation order. Joint pointers
remain valid until explicit removal, removal of either attached object, or
world destruction. Discard Python handles after removal. Adding/removing a
joint invalidates existing whole-world checkpoints; checkpoints preserve
properties, enabled/broken state, and measured loads for existing joints.
World XML export/import preserves joint types, frames, properties, and
enabled/broken state. It follows the normal world-export state contract.

Adding/removing joints or changing their properties, enabled state, or breakage
between `integrate1()` and `integrate2()` throws `std::logic_error` (`RuntimeError`
in Python). Finish the step or call `discardPendingStep()` first.

## Solver work and accuracy

The solver reuses working memory between steps and resolves body indices in
constant time. Geometry, Jacobians, inverse-inertia responses, and impulses are
still refreshed every step, including after body mass, fluid, activation, and
topology changes. Checkpoint restoration invalidates the working state.
Working buffers keep consistent alignment across fresh and restored runs;
regressions require exact replay for coupled contacts and changing active
joint/body layouts.

An exactly diagonal normalized constraint matrix uses its diagonal spectral
inverse. Any nonzero off-diagonal entry uses the full eigensolver, including
very small angular coupling. The rank cutoffs, position correction, iteration
budget, and convergence tolerance are unchanged. These optimizations neither
approximate the inertia nor reuse a previous step's dynamics.

## Validation and benchmark

`testRuntimeJoint` checks free and constrained motion, momentum balance,
rotating sliders, rigid/articulated attachments, contacts and tendons, sensors,
sleeping, breakage, checkpoint replay, and XML reload. Regressions also cover
alternating active joint/body types, pin and fluid changes, high-DOF and
same-articulation attachments, remapped object indices, mass updates, and
transitions between small angular coupling and exact alignment. The companion
Python integration test is `testPythonRuntimeJoints`; it runs the public two-robot
docking example and checks force transfer and independent motion after release.
Run the registered native, example, and Python checks with:

```bash
ctest --test-dir /path/to/test-build -j12 --output-on-failure \
  -R '^test(RuntimeJoint|PythonRuntimeJoints)'
```

The serial `runtime_joint_benchmark` target reports median step time for each
joint type, an unconstrained reference, coupled fixed joints with offset
attachments, and pairs of independent one-DOF articulated robots joined by
fixed docks. It validates attachment positions,
shared velocities, and transmitted forces. Configure the engine benchmark build
with `-DRAISIM_BENCHMARK=ON`, then run:

```bash
cmake --build /path/to/benchmark-build --target runtime_joint_benchmark --parallel 12
/path/to/benchmark-build/benchmark/runtime_joint_benchmark --copies 16 --steps 10000
```

See [Runtime joint workloads](./Benchmark.md#runtime-joint-workloads) for scene and timing details.
