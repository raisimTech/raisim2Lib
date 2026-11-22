# Server Example: Voxel Map Collision

## Overview

Drops spheres, boxes, capsules, cylinders, convex meshes and triangle meshes onto a voxel
playground with stairs down into a pit, a wall with windows, pillars and rubble, while new
craters are carved into the floor. Use `rayrai_tcp_viewer` for supported visualization.

## Screenshot

![voxel_map_collision example](../../../../rsc/docs/image/voxel_map_collision.png)

## Target

CMake target: `voxel_map_collision`.

## Run

Run the build-tree executable:

```bash
./build-examples/examples/voxel_map_collision
```

On Windows, run `voxel_map_collision.exe` instead.
This example uses RaisimServer. Start `rayrai_tcp_viewer` and connect to port 8080.

## Details

- Builds a 12 m x 12 m map of 5 cm voxels on 3 m of ground, with a low rim around the edge, and
  shapes it with `VoxelMap::fillBox()`. The ground is deeper than a crater, since the map is empty
  below its domain.
- Adds a new body every 0.3 s, cycling through every collision shape, until 120 bodies are in
  the scene. The convex meshes are the convex hull of the bundled `monkey/monkey.obj`; the
  triangle meshes are a U-shaped block and an arch (`rayrai/coacd`), collided face by face.
- Every 8 s, `VoxelMap::fillSphere()` carves a new crater of radius 1 m into the floor, and the
  bodies on it fall in. Sleeping bodies wake when the voxels around them change.
- See [Voxel Map](../../VoxelMap.md).
