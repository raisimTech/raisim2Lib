# Server Example: Fluid Pool

## Overview

Bodies of different densities fall into a pool 4 m by 2.4 m, filled with water to 0.7 m under air
([Fluids](../../Fluids.md)). The water holds each one up with the weight of the water it displaces,
at the center of that volume.
- The ice block floats with 8 % of its height above the surface.
- The half-dense wooden cube turns an edge up: a cube of density between 0.28 and 0.72 times that of
  water does not float on a face.
- The raft and the log float flat.
- The hollow ball rides high. It also feels the buoyancy of the air, 6 % of its weight.
- The steel ball and the brick sink to the floor.

The waves the bodies make as they bob carry their energy away, so they settle within a few periods.
Every fifteen seconds the world starts over, and the example prints how each body came to rest:

| Body | Expected | Example |
|---|---|---|
| Ice above the water | 8.1 % | 7.6 % (still settling) |
| Wooden cube tilt | 45° | 42 to 45° |
| Raft draft | 4.0 cm | 4.0 cm |
| Steel ball over the floor | 6 cm | 5.9 cm |

## Target

CMake target: `fluid_pool`.

## Run

```bash
./build-examples/examples/fluid_pool
```

Connect `rayrai_tcp_viewer` to port 8080. On Windows, run `build-examples/bin/fluid_pool.exe`.

## Details

- The pool is a `FluidVolume` bounded by its walls, from the floor up to its surface, and the air
  fills the rest of the world (`World::setAmbientFluid()`).
- The TCP viewer draws the water of the pool: the server sends it the world's volumes of liquid.
- The scene lives in `include/fluid_pool.hpp`.
