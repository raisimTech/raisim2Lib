# Server Example: Minitaur PD

## Overview

Runs Minitaur with PD control, including per-joint gains that zero out unactuated joints. It is a compact reference for legged robot PD setup.

## Screenshot

![minitaur_pd example](../../../../rsc/docs/image/minitaur_pd.png)

## Target

CMake target: `minitaur_pd`.

## Run

Run the build-tree executable:

```bash
./build-examples/examples/minitaur_pd
```

On Windows, run `minitaur_pd.exe` instead.
This example uses RaisimServer. Start `rayrai_tcp_viewer` and connect to port 8080.

## Details

- Loads the Minitaur URDF (closed-loop) and applies PD to actuated joints.
- Sets gains to zero for unactuated joints in the closed-loop model.
- Focuses the camera on the robot to show posture stability.
