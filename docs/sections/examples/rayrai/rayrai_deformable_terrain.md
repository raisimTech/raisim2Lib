# Rayrai Example: Deformable Terrain

![The off-road buggy driving across deformable soil](../../../../rsc/docs/image/rayrai/deformable_terrain_car.png)

*The buggy and its dampers are licensed under CC BY 4.0 and credited in
[Asset Credits](../../AssetCredits.md#off-road-buggy-and-race-venue).*

## Overview

An off-road buggy drives automatically across a 16 m by 6 m soil bed, leaving
permanent wheel ruts. The deformable height mesh responds to the wheel loads
and slip, while the buggy's double-wishbone suspension follows the changing
surface. The camera follows the car.

The example renders in process with `raisin::RayraiWindow`; it needs no TCP
viewer. See [Deformable terrain](../../DeformableTerrain.md) for the soil model
and [Off-Road Buggy](./rayrai_offroad_buggy.md) for the vehicle.

## Run

```bash
./build-examples/examples/rayrai_deformable_terrain
```

On Windows, run `build-examples/bin/rayrai_deformable_terrain.exe`.

CMake target: `rayrai_deformable_terrain`. Source:
`examples/src/rayrai/physics/rayrai_deformable_terrain.cpp`.
The example reads the buggy's URDF and meshes from `rsc/offroad_buggy`,
which CMake copies next to the executables.

To run without a window for a fixed number of physics steps:

```bash
./build-examples/examples/rayrai_deformable_terrain --headless --steps=3000
```

Options: `--hidden` renders in a hidden window; `--steps=N` limits the number
of physics steps; `--export=DIR` saves the world as `world.xml` and its assets;
`--activation-key=FILE` selects a RaiSim activation key.

## Details

- `World::addDeformableTerrain` creates a 201 by 76 sample soil mesh.
- Physics runs at a 1 ms time step, with 16 steps per rendered frame.
- The driver maintains about 0.8 m/s and brakes near the end of the bed.
- On exit, the example reports the car's speed, wheel loads, suspension loop
  error, terrain contact rows and maximum permanent rut depth, and checks
  that the car and soil state remain finite.
