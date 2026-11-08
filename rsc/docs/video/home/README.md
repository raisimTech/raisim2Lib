# Home-page videos

Clips and images for the documentation site's home page
(`docs/.vitepress/theme/media.ts`). Each clip is a RaiSim simulation. The `hero_*`
and most `tile_*` clips are rayrai renders cut from the 4K promo videos (the
`raisim-promo` project); `tile_gym.mp4` is an older raisimGymTorch recording.

| Files | Size | Use |
|---|---|---|
| `hero_*.mp4` | 1920 px wide, about 4.5 Mbit/s | Hero reel on wide screens |
| `hero_*-960.mp4` | 960 px wide, about 1.2 Mbit/s | Hero reel on phones and with data saver; rendering tiles |
| `tile_*.mp4` | 960 px wide, about 1.2 Mbit/s | Lazy-loaded section tiles |
| `*.jpg` posters | 960 or 1280 px wide, frame at the cut point | Shown until the clip plays |
| `tcp_viewer.jpg` | 1280 px wide | TCP viewer card (a JPEG of `rsc/docs/image/rayrai/tcp_viewer/tcp_viewer_rscene_forest.png`) |
| `tile_tracks.mp4`, `tile_tracks.jpg` | 960 px wide, about 1.2 Mbit/s | Tracked-vehicle tile, rendered by a docs image generator (below) |

Cut points in the promo renders (start and length in seconds):

| Clip | Source | Start | Length |
|---|---|---|---|
| `hero_offroad` | `2.9.0/clips/offroad.mp4` | 0 | 7.5 |
| `hero_forest` | `clips/forest.mp4` | 0 | 8 |
| `hero_city` | `clips/city_street.mp4` | 0 | 9 |
| `hero_rain` | `clips/city_rain.mp4` | 0 | 7 |
| `tile_offroad_drive` | `2.9.0/clips/offroad.mp4` | 8 | 8 |
| `tile_suspension` | `2.9.0/clips/offroad.mp4` | 20 | 8 |
| `tile_soil` | `2.9.0/clips/terrain.mp4` | 14 | 8 |
| `tile_glass` | `clips/glass.mp4` | 0 | 5 |
| `tile_strandbeest` | `clips/beest_wide.mp4` | 0 | 6 |
| `tile_lidar` | `clips/sensors_main.mp4` | 0 | 8 |
| `tile_depth` | `clips/sensors_depth.mp4` | 0 | 8 |
| `tile_rgb` | `clips/sensors_rgb.mp4` | 0 | 8 |
| `tile_hand` | `clips/shadow_hand.mp4` | 0 | 6 |
| `tile_usd` | `clips/usd_robots.mp4` | 0 | 7 |
| `tile_army` | `clips/anymal_army.mp4` | 0 | 9 |

`tile_tracks.mp4` is not cut from the promo renders. The docs image generator
`docs/image_generators/doc_image_home_tracks.cpp` records the tracked robot of the
`templated_tracked_robot` example as it drives over loose concrete blocks and climbs a
three-step stair: 7.5 s simulated in real time, rendered at 1920 × 1200 and 30 fps. The
scene uses only CC0 assets (Poly Haven asphalt, concrete and sky from `rsc/city`). A docs
build with `-DRAISIM_DOCS_BUILD_RAYRAI_IMAGES=ON` regenerates the tile and its poster
(frame 150, at 5 s) with the tile settings below, scaled to 960 px. To render it alone,
build the `doc_image_home_tracks` target, then run

```bash
RAISIM_DOC_GALLERY_FRAMES_DIR=<frames> doc_image_home_tracks <any output dir> <raisim2Lib>/rsc
ffmpeg -framerate 30 -i <frames>/frame_%03d.png -vf "scale=960:-2:flags=lanczos" -an -c:v libx264 \
  -preset slow -crf 26 -maxrate 1100k -bufsize 2200k -pix_fmt yuv420p -profile:v high \
  -movflags +faststart tile_tracks.mp4
ffmpeg -i <frames>/frame_150.png -vf "scale=960:-2:flags=lanczos" -q:v 4 tile_tracks.jpg
```

`tile_gym.mp4` (872 px wide, 20 fps) is cropped from the raisimGymTorch GIF that
`rsc/docs/image/raisimGymTorch_trainedPolicy2.mp4` was converted from:
436 × 332 pixels at x = 164, scaled 2× with Lanczos, with slightly raised
brightness and contrast.

All clips are silent H.264 (High profile, yuv420p) with `+faststart`, so they
play on every browser and start before they finish downloading. Always cut from
the promo renders, not from these files, so a clip is compressed only once. The
small and tile encodes use

```bash
ffmpeg -ss 8 -t 8 -i offroad.mp4 -an -vf "scale=960:-2:flags=lanczos" -r 30 \
  -c:v libx264 -preset slow -crf 26 -maxrate 1100k -bufsize 2200k -pix_fmt yuv420p \
  -profile:v high -movflags +faststart tile_offroad_drive.mp4
```

and a poster is the frame at the cut point:

```bash
ffmpeg -ss 8 -i offroad.mp4 -frames:v 1 -vf "scale=960:-2:flags=lanczos" -q:v 4 tile_offroad_drive.jpg
```

The wide hero encodes use `scale=1920:-2`, `-crf 25 -maxrate 4500k -bufsize 8M`,
and their posters `scale=1280:-2`. A phone that scrolls through the whole home
page downloads about 16.5 MB of video.

The physics cards also reuse `examples/underwater_rov.mp4` and
`examples/deformable_objects.mp4`, with their existing posters. `tile_tendons.mp4`
is a silent H.264 encode of `rsc/docs/image/rayrai/tendon_pulleys.gif`, preserving
its 640 × 400 pixels, 15 fps and four-second loop. Its poster is taken at 1.5 s.

```bash
ffmpeg -i tendon_pulleys.gif -an -c:v libx264 -threads 1 -preset slow -crf 23 \
  -pix_fmt yuv420p -profile:v high -movflags +faststart tile_tendons.mp4
ffmpeg -ss 1.5 -i tile_tendons.mp4 -frames:v 1 -q:v 3 tile_tendons.jpg
```
