# Server Example: Cloth on a Slope

## Overview

A 1 kg box stands on a cloth on each of two ramps, with a friction coefficient
of 0.5 everywhere. The ground holds the cloth under the box with the friction of
the box's weight, so on the 18 degree ramp, below the friction angle
(tan 18° = 0.32 < 0.5), the cloth and the box stay where they landed. On the
30 degree ramp (tan 30° = 0.58) they slide down together. The cloth is solved
together with the box and the ramps
([`World::setMonolithicDeformables()`](../../DeformableObject.md#solving-deformables-together-with-the-bodies)).
Every four seconds the world starts over, and the example prints how far each
box moved between 0.5 s and 1.5 s.

| Ramp | Default coupling | Solved together |
|---|---|---|
| 18 degrees: box travel | 74 mm | 0.001 mm |
| 30 degrees: box travel (Coulomb friction: 0.66 m) | 1.26 m | 1.12 m |

In the default coupling the ground holds each particle with the friction of its
own weight only, so the box drags the cloth down the gentle ramp.

## Target

CMake target: `cloth_on_slope`.

## Run

```bash
./build-examples/examples/cloth_on_slope
```

Connect `rayrai_tcp_viewer` to port 8080. On Windows, run
`build-examples/bin/cloth_on_slope.exe`.

## Details

- The ramps are static boxes tilted about the y axis; each cloth is a 0.48 m
  square of 25 x 25 particles weighing 0.15 kg, laid on the upper part of its
  ramp, with the box on its middle.
- On the steep ramp the box and the cloth slide faster than Coulomb friction
  allows; see the known limitations in
  [Deformable Objects](../../DeformableObject.md#solving-deformables-together-with-the-bodies).
- The scene lives in `include/cloth_on_slope.hpp`.
