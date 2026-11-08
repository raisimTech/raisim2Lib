# Rayrai Example: Water Pond

<video class="doc-video" src="../../../../rsc/docs/video/examples/rayrai_water_pond.mp4" loop muted playsinline preload="metadata" aria-label="A pond in a meadow with trees and a wooden jetty on its banks; ripples move over the water and the sun glitters on it"></video>

## Overview

A pond in a meadow, drawn with rayrai [water](../../rayrai/Water.md). Grass,
ferns, flowers, trees, mossy rocks and a wooden jetty stand on its banks.
Light ripples move over the surface and the afternoon sun, low in front of the
camera, glitters on it. Near the banks the water is shallow and clear enough
to show the muddy floor; further out the floor fades into green water.

The scene is visual: the water does not act on bodies. The example renders in
process with `raisin::RayraiWindow` and does not need `rayrai_tcp_viewer`.

## Target

CMake target: `rayrai_water_pond` (C++20).

## Run

Run the build-tree executable:

```bash
./build-examples/examples/rayrai_water_pond
```

On Windows, run `rayrai_water_pond.exe` instead. The example takes no
arguments. It reads the plants and rocks of `rsc/forest` and the timber texture
of `rsc/city` from the `rsc` copy next to the executables. Drag to look around
and use WASD and Space to move.

## The pond

The terrain is a 90 m × 90 m heightmap with an oval hollow, 30 m by 18 m and
1.7 m deep, with a wavy shoreline. The banks rise about 0.85 m above the water
and blend into rolling hills.

The water is one rectangular water body, turned with the pond and 30 % larger
than it. Its outline runs under the banks everywhere, so what shows is the
water inside the hollow, and the shoreline is wherever the ground rises out of
it.

```cpp
raisin::WaterBody water = raisin::defaultWaterBody(raisin::WaterPreset::CalmLake);
water.shape = raisin::WaterBodyShape::Rectangle;
water.center = {0, 10, 0};                 // the mean surface is at z = 0
water.yawRadians = .22f;
water.halfExtents = {19.5f, 11.7f};        // 1.3 times the pond's half axes
water.depth = 2.2f;                        // below the deepest point of the floor
water.waves = raisin::generateGerstnerWaves(wind);  // 12 waves, 0.3 to 4 m long
water.absorption = {.55f, .30f, .95f};     // per metre: blue is absorbed first
water.scattering = {.02f, .035f, .015f};
water.roughness = .02f;
water.rippleStrength = .22f;
water.shorelineFoamDepth = 0;              // a calm pond has no surf
viewer->setWaterBody("pond", water);
```

The waves are wind ripples a few centimetres high, between 0.3 m and 4 m
long, spread over about 100° around the wind direction. The baked ripple
detail rides on them. The example advances the water clock by the real time
between frames (`advanceWaterTime()`), so the ripples move at their natural
speed whatever the frame rate.

The water absorbs blue light about three times faster than green, and red in
between, so the light that comes back from the floor turns green with depth.
Where the floor is 0.5 m down, sunlight from overhead crosses 1 m of water on
its way down and back up, and keeps about 70 % of its green and 40 % of its
blue. Over the deepest part, 1.7 m down, little of the floor is left. Near the banks the sunlit floor shows
caustics: the ripples focus the sunlight into a moving net of bright lines.

## The sun on the water

The sun stands 28° above the horizon, ahead and to the right of the camera.
A perfectly flat pond would show one mirror image of the sun, 28° below the
horizon. Here that point falls on the bank just in front of the camera. The
ripples tilt the surface a little everywhere, and every facet that happens to
face halfway between the sun and the camera sends sunlight into the lens.
Those facets lie along the sun's azimuth and form the glitter that stretches
from the near bank to the far one. Facets steep enough to do this are rarer
far away, where the water is seen at a grazing angle, but each pixel there
covers more of them; the renderer adds the slope of the ripples a pixel
averages to its roughness, so the distant glitter turns into a broad sheen
instead of flickering.

Away from the glitter, the water reflects the sky and the banks. At the
grazing angles toward the far bank, Fresnel reflection makes it reflect about
half of the light. Near the camera, where the view is about 30° below the
horizon, it reflects about 5 % and the floor shows through.

## Scene setup

The scene is built in `src/rayrai/worlds/pond_scene.hpp` (terrain, plant and
rock placement, the jetty layout and the water body) and `pond_viewer.hpp` (the
viewer, the jetty's wood, the weather sky, the sun and the camera), which the
GPU test of the example also uses.

- 70 trees, about 16,000 grasses, ferns and flowers, and 36 rocks are
  scattered over the dry ground. No plant grows below the water line.
  The trees keep clear of the strip between the camera and the sun, so their
  shadows do not cover the glitter.
- 22 of the rocks lie on the waterline, partly submerged.
- The jetty is viewer visuals (boxes and cylinders) with the Poly Haven timber
  texture of `rsc/city`, its grain along each board. Its posts stand on the pond
  floor. It is decoration: no physics body collides with it.
- The sky is the weather's procedural sky. At the High quality preset its
  clouds are volumetric: cumulus at 1.5 km, 700 m thick, lit by the sun and
  covering about a third of the sky. The water reflects them.
