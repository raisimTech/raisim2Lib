# Rayrai Example: Complete Showcase

## Overview

Comprehensive rayrai showcase with a sensored ANYmal (spinning LiDAR and front RGB/depth camera), heightmap terrain, YCB objects, custom visuals, instanced geometry, and a live LiDAR point cloud. Use it as an end-to-end check for rayrai and sensor rendering.

## Screenshot

![rayrai_complete_showcase example](../../../../rsc/docs/image/rayrai_complete_showcase.png)

*The YCB objects are licensed under CC BY 4.0 and credited in [Asset Credits](../../AssetCredits.md#ycb-objects).*

## Target

CMake target: `rayrai_complete_showcase`.

## Run

Run the build-tree executable:

```bash
./build-examples/examples/rayrai_complete_showcase
```

On Windows, run `rayrai_complete_showcase.exe` instead.
This example renders in process with rayrai and does not need `rayrai_tcp_viewer`.

## Details

- Loads a sensored ANYmal, heightmap terrain, primitives, and YCB objects.
- Updates LiDAR scans into a point cloud and renders RGB/depth sensors.
- Draws the camera frustums in the scene and shows the RGB image, the depth
  image, and the raw sensor buffers in ImGui windows.

## Frame capture

The documentation GIF is recorded with the capture that the rayrai examples
share (see `rayrai/example_common.hpp`). `RAYRAI_EXAMPLE_CAPTURE_WINDOW=1`
saves the whole window, sensor panels included. The loop advances one physics
step per rendered frame, so every 16th frame of the default 5 ms time step puts
80 ms of simulation in each captured frame, real time at 12.5 fps:

```bash
RAYRAI_EXAMPLE_CAPTURE=/tmp/showcase_frames RAYRAI_EXAMPLE_CAPTURE_WINDOW=1 \
RAYRAI_EXAMPLE_CAPTURE_FRAMES=75 RAYRAI_EXAMPLE_CAPTURE_START=120 \
RAYRAI_EXAMPLE_CAPTURE_EVERY=16 \
  ./build-examples/examples/rayrai_complete_showcase

ffmpeg -framerate 12.5 -i /tmp/showcase_frames/frame_%03d.png \
  -filter_complex "[0:v] scale=954:540:flags=lanczos,split [a][b];\
                   [a] palettegen=stats_mode=diff [p];\
                   [b][p] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  -loop 0 rsc/docs/image/rayrai_complete_showcase.gif
```
