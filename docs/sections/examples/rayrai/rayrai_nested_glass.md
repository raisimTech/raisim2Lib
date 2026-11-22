# Rayrai Example: Nested Glass

## Overview

Traces refraction through a water-filled glass with an enclosed air bubble, an
empty hollow cup, a concave amber torus, and two overlapping tinted glass
solids. The media have explicit indices of refraction and dielectric
priorities, so overlapping interfaces resolve in a defined order. The default
backend uses Vulkan hardware ray queries when available and otherwise selects
the portable OpenGL tracer. It starts with 10 paths per frame and at most 10
bounces.

## Screenshot

![Nested water, glass, and air with overlapping transparent solids](../../../../rsc/docs/image/rayrai_nested_glass_showcase.png)

## Target

CMake target: `rayrai_nested_glass`.

## Run

Run the build-tree executable:

```bash
./build-examples/examples/rayrai_nested_glass
```

On Windows, run `rayrai_nested_glass.exe` instead.
This example renders in process with rayrai and does not need `rayrai_tcp_viewer`.
Drag to look, use WASD and Space to move, and click an object to orbit it. The
**Geometry refraction** panel controls the backend, samples per frame, and
bounce count.

To save the view as a PNG and exit, set `RAYRAI_EXAMPLE_CAPTURE`:

```bash
RAYRAI_EXAMPLE_CAPTURE=nested_glass.png ./build-examples/examples/rayrai_nested_glass
```
