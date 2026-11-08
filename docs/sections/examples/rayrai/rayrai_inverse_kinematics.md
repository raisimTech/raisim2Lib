# Rayrai Example: Inverse Kinematics

<video class="doc-video" src="../../../../rsc/docs/video/examples/rayrai_inverse_kinematics.mp4" loop muted playsinline preload="metadata" aria-label="A Kinova arm with a Robotiq gripper following a target that moves in a figure eight above a workbench with three cubes"></video>

## Overview

A Kinova arm with a Robotiq 2F-85 gripper reaches for a target that you move
with the mouse. Every frame,
[`ArticulatedSystem::solveIK()`](/api/raisim/ArticulatedSystem.md#solveik)
computes the joint angles that put the gripper's grasp frame on the target,
in position and, if you choose, in orientation.

The arm is simulated, not posed. Its joint PD controllers drive it towards the
IK solution at a limited joint speed, so it pushes the cubes on the workbench
aside, and with the gripper closed it can pick them up.

The example renders in-process with `raisin::RayraiWindow`; it needs no TCP
viewer.

## Target

CMake target: `rayrai_inverse_kinematics`.

## Run

Run the build-tree executable:

```bash
./build-examples/examples/rayrai_inverse_kinematics
```

On Windows, run `rayrai_inverse_kinematics.exe` instead. The example takes no
arguments. `--tour` moves the target along the figure eight of the clip above
until you grab it, and `--frames=N` exits after N frames.

## Controls

| Input | Effect |
| --- | --- |
| Left-drag the orange ball | Move the target in the plane that faces the camera |
| Mouse wheel over the ball | Push the target away from the camera or pull it closer |
| Right-drag | Turn the target, while *Follow the target's orientation* is on |
| Left-drag elsewhere, mouse wheel | Orbit and zoom the camera |
| `G` | Close or open the gripper |
| `R` | Reset the arm, the cubes and the target |

The panel switches the orientation task on and off and selects the IK method:
damped least squares (with its damping), pseudo-inverse, SVD or Jacobian
transpose. It shows whether the last solve converged and how far the gripper
is from the target. The ball turns red when the solver cannot reach the
target.

## How it works

The scene and the controller are in `include/arm_ik_demo.hpp`. Each frame the
example does three things:

1. **The mouse moves the target.** The ray through the pointer is intersected
   with the plane through the target that faces the camera. The grabbed point
   stays under the pointer, so the ball does not jump to the cursor when you
   grab it off-centre.
2. **IK solves for the target.** `Scene::SolveIk()` starts from the previous
   solution and leaves the simulated arm alone:

   ```cpp
   ik_options_.seed.assign(ik_solution_.data(), ik_solution_.data() + ik_solution_.size());
   ik_options_.applySolution = false;
   ik_options_.useOrientation = track_orientation_;
   ik_result_ = robot_->solveIK(ik_frame_, position, orientation, ik_options_);
   const auto& solution = ik_result_.success ? ik_result_.solution : ik_result_.bestSolution;
   ```

   - **Seed.** Starting from the previous solution keeps the arm on one IK
     branch. The joint angles change continuously as the target moves; a fresh
     solve from an arbitrary pose could flip the elbow or the wrist.
   - **`applySolution = false`.** The solver restores the arm's state when it
     returns, so only the physics moves the arm.
   - **`bestSolution`.** Out of reach, it is the closest pose the solver found,
     and the arm stretches towards the target.
   - **`orientationWeight = 0.3`.** When the target cannot be reached in both
     position and orientation, the solver trades 1 rad of hand orientation for
     0.3 m of position (see
     [`IKOptions`](/api/raisim/ArticulatedSystem/IKOptions.md)).
3. **The arm follows.** `Scene::Step()` moves the joint PD targets towards the
   solution with a time constant of 0.08 s. All joints are scaled together so
   that none turns faster than 2 rad/s, and they arrive at the same time. The
   velocity targets are the rates of those position targets. Then the world
   integrates one 1 ms step. The gripper's knuckle ramps the same way; its five
   mimic joints follow passively (see [Mimic Joints](../../articulated_system/MimicJoints.md)).

## What to look for

- **Singular poses.** With the gripper pointing down, the arm nears singular
  configurations when the target is high above the bench or far out. There,
  small turns of the hand need large joint motions or cannot be made at all:
  tilt the target and the ball turns red. The home pose is low over the bench,
  away from them.
- **Physical tracking.** While the target moves, the gripper trails it by about
  the approach time constant. Once the target stops, the gripper settles within
  a few millimetres; the panel shows the distance.
- **Grasping.** Lower the target onto a cube with the fingers square to it,
  press `G`, and lift: friction at the finger pads holds the 100 g cube.
- **Jacobian transpose.** It takes many small steps. With 50 iterations per
  frame it needs about a second and a half to converge on a target 17 cm away,
  which the other methods reach in the first frame.

## Full source

<<< @/../examples/src/rayrai/robots/rayrai_inverse_kinematics.cpp{cpp:line-numbers}

<<< @/../examples/include/arm_ik_demo.hpp{cpp:line-numbers}

## Related

- [Robotiq Gripper Mimic](../server/robotiq_gripper_mimic.md): the same arm and
  gripper in a scripted pick and place.
- [Dynamics and Control](../../articulated_system/DynamicsAndControl.md): joint
  PD control and its torque limits.
- [`ArticulatedSystem::IKResult`](/api/raisim/ArticulatedSystem/IKResult.md):
  the residuals and solutions the panel reports.
