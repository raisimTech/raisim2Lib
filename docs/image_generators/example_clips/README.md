# Example gallery clips

`record_example_clips.py` records the clips and posters of the example gallery
(`rsc/docs/video/examples`, shown by `<ExampleGallery>` on the Examples pages).
Each example runs headless on the GPU with SDL's offscreen video driver, and
`framecap.c`, compiled into an `LD_PRELOAD` library, reads back every finished
frame and pipes it to ffmpeg. rayrai examples are recorded from their own
window; RaisimServer examples from `rayrai_tcp_viewer`, which starts first and
connects as soon as the server is up. Each raw clip is trimmed, cropped to
16:10 and encoded as a 960x600 H.264 tile of about 1 Mbit/s with a poster from
its middle.

`clips.json` holds every setting, one example per line; its keys are described
at the top of the script. Examples that only print to the console have no clip:
they are listed under `console` in `docs/sections/examples/gallery.json`.

Requirements: Linux with a GPU, a C compiler, ffmpeg and ffprobe, and the
examples built from `raisim2Lib/examples` against the RaiSim and rayrai packages
in this repository:

```bash
cmake -S examples -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DRAISIM_PREFIX=$PWD/raisim -DRAYRAI_PREFIX=$PWD/rayrai
cmake --build <build> -j6
python3 docs/image_generators/example_clips/record_example_clips.py --build <build> [name ...]
```

`rayrai_tank_survival` runs the Tank Survival game, which is built from `raisim2Lib/wasm`
(see its page); record it on its own with `--build` set to that build.

Without names it records all of them, which takes about 20 minutes. It writes into
`rsc/docs/video/examples` (`--out` to change), and keeps the raw clips and the
program logs only with `--work <dir>`. Server examples use port 8080, so run one
recording at a time.

## Adding an example

1. Add the page to a category in `docs/sections/examples/gallery.json` (the
   order there is the order of the tiles), and to its `console` map with the
   command to show if the example has no window.
2. Add a line to `clips.json` and record it. A server example usually needs
   `"viewer_args": ["--auto-frame"]` or a fixed
   `["--camera-lookat", "px,py,pz,tx,ty,tz", "--force-camera-lookat"]`, and a
   `start` that skips loading; check the poster, then adjust `crop`. A category's
   tiles are seen together, so frame a clip that looks like another one there
   differently, as `rayrai_city_from_xml` does with a closer crop.
3. `python3 -m unittest discover -s docs/scripts` checks that every example page
   has a tile and a clip, and that every clip stays under 2 MB.
