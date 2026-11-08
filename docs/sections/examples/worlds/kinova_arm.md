# Server Example: Kinova Arm

![kinova_arm example](../../../../rsc/docs/image/kinova_arm.png)

## Overview

Loads the Kinova arm, applies joint PD gains and targets, and runs it on flat ground. This example focuses on manipulator setup and joint-level control.

## Target

CMake target: `kinova_arm`.

## Run

Run the build-tree executable:

```bash
./build-examples/examples/kinova_arm
```

On Windows, run `kinova_arm.exe` instead.
This example uses RaisimServer. Start `rayrai_tcp_viewer` and connect to port 8080.

## Details

- Loads the Kinova arm URDF on flat ground and applies joint PD gains/targets.
- Uses a simple scene for fast rendering.
- Focuses the camera on the arm for a fixed-base articulation demo.
