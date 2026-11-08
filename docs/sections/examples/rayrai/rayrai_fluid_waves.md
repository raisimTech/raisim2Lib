# Rayrai Example: Floating on Waves

<video class="doc-video" src="../../../../rsc/docs/video/examples/rayrai_fluid_waves.mp4" loop muted playsinline preload="metadata" aria-label="Wooden crates, a raft, blue barrels and beach balls ride the swell of the sea"></video>

## Overview

Wooden crates, a raft, barrels and beach balls ride the swell of the sea. The sea of the RaiSim
world ([Fluids](../../Fluids.md)) samples the same Gerstner waves that rayrai draws, at the physics
time, through `FluidVolume::sampler`. The water holds every body up where the drawn surface is: they
heave on the crests and roll and pitch with the slope of the waves. The sampler also returns how fast
the surface rises and falls, so the bodies move with the water instead of only being pushed by it.
The example renders in process with `raisin::RayraiWindow` and does not need `rayrai_tcp_viewer`.

## Target

CMake target: `rayrai_fluid_waves` (C++20).

## Run

```bash
./build-examples/examples/rayrai_fluid_waves
```

On Windows, run `rayrai_fluid_waves.exe` instead. The example takes no arguments. Drag to look
around and use WASD and Space to move.

## Details

- The sea is rayrai's ocean with twelve Gerstner waves from 1.2 m to 16 m long
  (`raisin::generateGerstnerWaves()`). `raisin::sampleWaterSurface()` evaluates the same waves on the
  CPU, so the physics does not wait for the renderer.
- The crates are 600 kg/m³, the raft 450 kg/m³, the barrels 500 kg/m³. The beach balls weigh 0.4 kg
  for 0.11 m³, so they ride on top of the waves and are blown about by the slopes.
- The sea is sea water under air (`World::setAmbientFluid()`), and the bodies make waves of their
  own as they bob, which damps them.
- The scene lives in `src/rayrai/physics/fluid_waves_scene.hpp`.
