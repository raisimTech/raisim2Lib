# Server Example: Dzhanibekov Effect

## Overview

Demonstrates the Dzhanibekov effect by spinning a box in zero gravity and visualizing the unstable rotation.

## Screenshot

![dzhanibekov_effect example](../../../../rsc/docs/image/dzhanibekov_effect.png)

## Target

CMake target: `dzhanibekov_effect`.

## Run

Run the build-tree executable:

```bash
./build-examples/examples/dzhanibekov_effect
```

On Windows, run `dzhanibekov_effect.exe` instead.
This example uses RaisimServer. Start `rayrai_tcp_viewer` and connect to port 8080.

## Details

- Simulates a freely rotating asymmetric box in zero gravity.
- Shows the intermediate-axis (Dzhanibekov) flipping behavior.
- Focuses the camera on the box for clear visualization.
