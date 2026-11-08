# Server Example: Heavy Objects on Cloth

## Overview

Three loads rest on cloth: a 10 kg ball on a cloth on the ground, a 5 kg box on
two stacked cloths, and a 5 kg blanket draped over a 1 kg box. The cloth is
solved together with the bodies
([`World::setMonolithicDeformables()`](../../DeformableObject.md#solving-deformables-together-with-the-bodies)),
so each load rests on what carries the cloth: the ball and the box on the
ground through the cloth, and the blanket on the box, which carries the
blanket's weight to the ground. Every five seconds the world starts over, and
the example prints how far each load pressed into what carries it.

| Load | Default coupling | Solved together |
|---|---|---|
| Cloth under the 10 kg ball, pressed into the ground | 10.8 mm | < 0.001 mm |
| Upper cloth under the 5 kg box, pressed into the lower one | 18 mm | 0.4 mm |
| 1 kg box under the 5 kg blanket, pressed into the ground | 30 mm | 0.03 mm |

Solved together, the scene also steps faster (3.3 ms instead of 4.8 ms per
1 ms step on a Ryzen 9 3950X), since the contacts under the loads settle in
fewer iterations.

## Target

CMake target: `heavy_objects_on_cloth`.

## Run

Run the build-tree executable and connect `rayrai_tcp_viewer` to port 8080:

```bash
./build-examples/examples/heavy_objects_on_cloth
```

On Windows, run `build-examples/bin/heavy_objects_on_cloth.exe` instead.

## Details

- The cloths are particle meshes with nearly inextensible edges and a bending
  modulus of 1e-4 N m; their particles collide as 8 mm spheres. The cloth under
  the ball is 0.64 m square and weighs 0.2 kg; the stacked cloths are 0.32 m
  square and weigh 0.1 kg each, the blanket is 0.5 m square.
- The ball and the box are dropped from a few centimetres; the blanket falls
  20 cm onto the box and drapes over its edges to the ground.
- The scene lives in `include/heavy_objects_on_cloth.hpp`, so the same scene can
  run without a viewer (`Scene::Step()`), as the RaiSim test of the example does.
