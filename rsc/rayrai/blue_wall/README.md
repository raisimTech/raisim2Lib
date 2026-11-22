# Blue Wall scene

Source: [Poly Haven Blue Wall scene](https://blog.polyhaven.com/blue-wall-scene-file/)
([download](https://dl.polyhaven.org/file/ph-assets/Scenes/blue_wall.zip)). License: CC0.

`blue_wall.rscene` is the Engine scene description. It records the initial
camera, environment, visual-only mesh, and imported lights. The example reads
this file, then loads its `blue_wall.gltf` mesh and `blue_wall.bin` buffer.
The glTF payload contains the original geometry, 133 embedded images, and PBR
material extensions without texture recompression. The
`blue_wall.gltf.rayrai_lights.json` sidecar supplies the imported area light,
and `hamburg_canal_1k.hdr` supplies environment lighting.

Regenerate the package from the original Poly Haven GLB with:

```sh
python3 examples/tools/convert_blue_wall_to_rscene.py \
  /path/to/blue_wall.glb rsc/rayrai/blue_wall
```

Build the `rayrai_blue_wall_scene` target, then run it to explore the room.
The initial camera matches `09_blue_wall_scene.png` from the RayRai feature
showcase. Setting `RAYRAI_EXAMPLE_CAPTURE=output.png` saves that view and exits.
`blue_wall_glb_reference.png` is the original GLB rendered from the same
camera; the Raisim test `rayrai_blue_wall_camera` compares the example's view
with it and checks the wall, painting, and dresser colors.
