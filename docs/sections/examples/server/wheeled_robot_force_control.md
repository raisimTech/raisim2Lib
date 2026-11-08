# Server Example: Wheeled Robot Force Control

## Overview

Loads a wheeled robot (SMB/megabot) and applies wheel forces in force-control mode. Use it to see a basic wheeled robot setup.

## Screenshot

![wheeled_robot_force_control example](../../../../rsc/docs/image/wheeled_robot_force_control.png)

## Target

CMake target: `wheeled_robot_force_control`.

## Run

Run the build-tree executable:

```bash
./build-examples/examples/wheeled_robot_force_control
```

On Windows, run `wheeled_robot_force_control.exe` instead.
This example uses RaisimServer. Start `rayrai_tcp_viewer` and connect to port 8080.

## Details

- Loads the SMB wheeled robot and applies constant wheel torques.
- Uses `FORCE_AND_TORQUE` control mode with semi-implicit integration.
- Focuses the camera on the robot for rollout.
