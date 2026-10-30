# Server Example: Fluid Stream

## Overview

A stream flows down a channel 20 m long and 0.8 m deep, past a post, under air
([Fluids](../../Fluids.md)). Its current is the velocity of the water's `FluidVolume`.
- Crates, logs and beach balls float off with the stream. The water's drag carries them toward its
  speed, and the still air over the stream holds back what they lift above it: in a stream of
  0.8 m/s, after 15 s a beach ball, riding mostly above the water, drifts at about 0.65 m/s and a
  crate at about 0.77 m/s. A log that meets the post wraps against it.
- Bricks on the stony bed stay put in a gentle stream. Past about 2.1 m/s the water's push beats the
  friction of the bed on what the water leaves of their weight, and they slide along.

What reaches the end of the channel drops back in at its start, and the example prints how many bodies
the stream has carried down.

## Controlling the stream

The example adds a slider, `Stream speed [m/s]`, from 0 to 3 m/s
(`RaisimServer::addSlider()`). The rayrai TCP viewer shows it in its **Controls** window, at the lower
right of the view. Drag it to change the current. Each step, the example reads it with
`RaisimServer::getSlider()` and sets the water's velocity with `World::setFluidVolume()`.

## Target

CMake target: `fluid_stream`.

## Run

```bash
./build-examples/examples/fluid_stream
```

Connect `rayrai_tcp_viewer` to port 8080. On Windows, run `build-examples/bin/fluid_stream.exe`.

## Details

- The channel is a `FluidVolume` bounded by its banks, from the bed up to its surface, with the
  current along x. The bed's friction is 1.2, that of a stony bed.
- The TCP viewer draws the water of the channel: the server sends it the world's volumes of liquid.
- The scene lives in `include/fluid_stream.hpp`, which it shares with
  [Fluid Stream with ANYmal](./fluid_stream_anymal.md).
