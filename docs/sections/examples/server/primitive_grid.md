# Server Example: Primitive Grid

## Overview

Spawns a grid of boxes, spheres, capsules, and cylinders to show the basic primitive creation APIs. It is a quick visual check for simple shapes and placements.

## Screenshot

![primitive_grid example](../../../../rsc/docs/image/primitive_grid.png)

## Target

CMake target: `primitive_grid`.

## Run

Run the build-tree executable:

```bash
./build-examples/examples/primitive_grid
```

On Windows, run `primitive_grid.exe` instead.
This example uses RaisimServer. Start `rayrai_tcp_viewer` and connect to port 8080.

## Details

- Creates a 3D lattice of boxes, spheres, capsules, and cylinders.
- Assigns per-shape colors and positions them in a grid.
- Useful for collision/rendering sanity checks.
