# Rayrai Example: Water Stream

<video class="doc-video" src="../../../../rsc/docs/video/examples/rayrai_water_stream.mp4" loop muted playsinline preload="metadata" aria-label="A creek runs down a grassy hillside between mossy rocks into a pool; wooden balls float down it"></video>

## Overview

A creek runs down a hillside between grassy banks, over two small drops and
around mossy rocks, into a pool and out the far side. The water is rayrai's
[flowing water](../../rayrai/Water.md#flowing-water): a shallow-water
simulation on the GPU that carries the spring's water down the terrain, parts it
around the rocks and draws its ripples and foam drifting with the current.

Five wooden balls are launched at the spring one after another. Every physics
step applies the buoyancy and drag of the simulated water to them, so they ride
the current, bump into the rocks, slow down in the pool and float there before
the water carries them out. The example renders in process with
`raisin::RayraiWindow` and does not need `rayrai_tcp_viewer`.

## Target

CMake target: `rayrai_water_stream` (C++20).

## Run

Run the build-tree executable:

```bash
./build-examples/examples/rayrai_water_stream
```

On Windows, run `rayrai_water_stream.exe` instead. The example takes no
arguments. It reads the plants, rocks and ground textures of `rsc/forest` from
the `rsc` copy next to the executables. Drag to look around and use WASD and
Space to move.

After the assets load, the example runs the water for 60 s of water time before
it shows the scene, with the spring six times stronger for the first 24 s, so the
pool is full and the creek running from the first frame. This takes a few
seconds.

## The creek

The terrain is a 40 m × 28 m heightmap. The creek bed falls 7.5 cm per metre
along +x and drops 30 cm and 25 cm at two short steps, 3.4 m in all. Its
channel winds across the slope between banks that rise about 1 m on each
side. At x = 12 m it opens into a round pool 3.2 m in radius and 0.8 m deeper
than the bed around it, which spills over its lip and runs on to the east edge.

The water is one simulated rectangle, 38 m × 12 m, that covers the creek and the
pool with 8 cm cells (475 × 150). It starts dry; a spring at its upper end adds
0.18 m³/s:

```cpp
raisin::WaterBody water = raisin::defaultWaterBody(raisin::WaterPreset::CalmLake);
water.shape = raisin::WaterBodyShape::Rectangle;
water.center = {0, 0, -3.2};          // below the whole terrain: no water at the start
water.halfExtents = {19.f, 6.f};
water.waves.clear();                  // the flow makes the surface
water.flow.enabled = true;
water.flow.cellSize = .08f;
water.flow.bedCeiling = 6;            // the terrain lies between center.z and center.z + 6 m
water.flow.manningRoughness = .035f;  // a natural stream bed
water.flow.foamStrength = .7f;
raisin::WaterFlowSource spring;
spring.position = {kSourceX, creekY(kSourceX)};
spring.radius = .5f;
spring.flowRate = .18f;               // m^3/s
spring.velocity = {.8f, 0};
water.flow.sources.push_back(spring);
viewer->setWaterBody("creek", water);
```

The water runs down the channel about 1.5 m wide, up to 16 cm deep and at about
1.8 m/s, where gravity down the slope balances the friction of the bed. Foam
forms where it runs fast and shallow, it piles up and parts around the rocks in
the riffles, and it slows down as it spreads into the pool. The whole flow of
the spring passes every point of the creek: measured across the channel, depth
times velocity adds up to 0.18 m³/s from the spring to the pool.

The rocks are instanced visuals, so the water sees them in its bed. The ten in
the creek also have hidden convex-hull colliders from the same models, so the
balls bounce off them instead of passing through.

## The floating balls

The balls are RaiSim spheres 12 cm in radius with a density of 650 kg/m³. Before
every frame the example asks the water under each ball for its surface and
velocity, then applies at every physics step (`stream::floatingForce()`):

- **Buoyancy:** the weight of the water the ball displaces, from the height of
  the spherical cap below the surface.
- **Drag:** ½ ρ C<sub>d</sub> A |**v**<sub>rel</sub>| **v**<sub>rel</sub>
  against the velocity of the ball relative to the water, with C<sub>d</sub> =
  0.8 and A the cross-section below the surface.
- **Rolling damping:** a small torque against spinning while the ball is wet.

`sampleWaterFlow()` reports the water around a ball rather than over it, since
the simulation lets the ball push the water aside. In the pool a ball floats
with its center a few centimetres below the surface, close to the 2.4 cm of a
ball 65 % as dense as water. In the creek the water is about as deep as the
14 cm a floating ball sinks to, so the balls float and roll along the bed by
turns as the current carries them down. A ball that leaves the water body or
has floated for 90 s is launched at the spring again.

## Scene setup

The scene is built in `src/rayrai/worlds/stream_scene.hpp` (terrain, plant and
rock placement, the water body, the rock colliders and the force on a floating
ball) and `stream_viewer.hpp` (the viewer, the balls, the sky, the sun and the
camera, the warm-up and the physics step), which the GPU test of the example
also uses.

- 60 trees and about 14,000 grasses, ferns and flowers grow on the banks and
  thin out away from the creek. No plant grows in the creek or the pool, and
  the trees keep out of the camera's view down the creek.
- 30 mossy rocks: ten in the riffles and at the drops, the rest on the banks.
- The sky is the weather's procedural sky with volumetric clouds; the sun
  stands 27° above the horizon.
- The camera stands on the bank beside the pool's outlet and looks up the
  creek.
