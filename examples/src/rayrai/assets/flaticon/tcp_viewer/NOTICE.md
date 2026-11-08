# Flaticon assets for RayRai TCP Viewer

These icons are generated as PNG assets from Flaticon UIcons, solid rounded style (fi-sr).

Source: https://www.flaticon.com/uicons
Package: @flaticon/flaticon-uicons 3.3.1 from https://www.npmjs.com/package/@flaticon/flaticon-uicons
Attribution: UIcons by Flaticon

Every glyph must come from this one family and style, so the panel reads as a single
icon set. To add one, take it from the same package rather than downloading a
one-off icon from flaticon.com, whose free icons are drawn by many different
authors in mismatched styles.

Generation: the package ships webfonts rather than SVGs, so a glyph is rasterized
from `css/uicons-solid-rounded-*.woff` (converted to TTF with fontTools) at
256x256 RGBA, pure white with the coverage in the alpha channel -- the viewer
applies its own per-icon tint at draw time, so a coloured PNG would tint twice.
The glyph is centred on its ink bounding box and scaled so the longest ink side
is 176px, which matches the optical weight of the icons already here.

Every file here is therefore 256x256, pure white, with a 168-184px ink box. The
sim-transport and inspector icons (play, pause, step, step-fast, rotate-left,
robot, sign-out-alt) were originally 64x64 with a near-black glyph, which the
viewer's multiplicative tint rendered almost invisible; they have been
regenerated to this standard, as has settings-sliders, which was drawn larger
than the rest. Keep new icons inside that envelope.

Glyphs used:
- fi-sr-plug-connection -> connect_uicons_sr_plug_connection.png
- fi-sr-link-horizontal-slash -> disconnect_uicons_sr_link_horizontal_slash.png
- fi-sr-refresh -> refresh_uicons_sr_refresh.png
- fi-sr-disk -> save_uicons_sr_disk.png
- fi-sr-house-signal -> home_uicons_sr_house_signal.png
- fi-sr-target -> focus_uicons_sr_target.png
- fi-sr-camera-viewfinder -> camera_uicons_sr_camera_viewfinder.png
- fi-sr-folder-open -> folder_uicons_sr_folder_open.png
- fi-sr-file-export -> export_uicons_sr_file_export.png
- fi-sr-settings-sliders -> options_uicons_sr_settings_sliders.png
- fi-sr-robot -> robot_uicons_sr_robot.png            (AS inspector header)
- fi-sr-rotate-left -> reset_uicons_sr_rotate_left.png (Reset pose button)
- fi-sr-sign-out-alt -> exit_uicons_sr_sign_out_alt.png (Close inspector button)
- fi-sr-pause -> pause_uicons_sr_pause.png              (Pause sim button)
- fi-sr-play -> play_uicons_sr_play.png                 (Resume sim button)
- fi-sr-stop -> stop_uicons_sr_stop.png                 (Stop recording / stop session log)
- fi-sr-video-camera -> video_uicons_sr_video_camera.png (Start video recording)
- fi-sr-bolt -> force_uicons_sr_bolt.png                (Apply Force button)
- fi-sr-rotate-right -> torque_uicons_sr_rotate_right.png (Apply Torque button)

File/folder picker rows (see fileBrowserIconKind()):

- fi-sr-folder -> folder_closed_uicons_sr_folder.png    (directory row)
- fi-sr-level-up-alt -> folder_up_uicons_sr_level_up_alt.png (parent-directory button)
- fi-sr-file -> file_uicons_sr_file.png                 (file of unrecognized type)
- fi-sr-file-code -> file_code_uicons_sr_file_code.png  (xml, json, yaml, mjcf, sdf, scripts)
- fi-sr-picture -> file_image_uicons_sr_picture.png     (png, jpg, hdr, exr, pgm)
- fi-sr-file-csv -> file_csv_uicons_sr_file_csv.png     (csv, tsv)
- fi-sr-file-zipper -> file_archive_uicons_sr_file_zipper.png (zip, tar, gz, 7z)

The picker also reuses icons listed above: fi-sr-robot for .urdf,
fi-sr-vector-polygon for mesh formats, fi-sr-video-camera for video files, and
fi-sr-disk for .rrtcs session logs.
- fi-sr-square-plus -> add_uicons_sr_square_plus.png    (Spawn object button)
- fi-sr-trash -> delete_uicons_sr_trash.png             (Delete selected object)
- fi-sr-palette -> render_uicons_sr_palette.png         (Render tab)
- fi-sr-chart-histogram -> diagnostics_uicons_sr_chart_histogram.png (Diagnostics tab)
- fi-sr-layers -> objects_uicons_sr_layers.png          (Objects tab)
- fi-sr-keyboard -> help_uicons_sr_keyboard.png          (Help tab)
- fi-sr-angle-double-small-left -> collapse_uicons_sr_angle_double_small_left.png (Collapse the left panel)

Options > Camera view buttons. Arrows for the four side views; front and back
follow the drafting convention of a dot for the axis coming toward the viewer and
an X for the axis going away.

- fi-sr-arrow-up -> view_top_uicons_sr_arrow_up.png              (Top view)
- fi-sr-arrow-down -> view_bottom_uicons_sr_arrow_down.png       (Bottom view)
- fi-sr-arrow-left -> view_left_uicons_sr_arrow_left.png         (Left view)
- fi-sr-arrow-right -> view_right_uicons_sr_arrow_right.png      (Right view)
- fi-sr-bullet -> view_front_uicons_sr_bullet.png                (Front view)
- fi-sr-cross -> view_back_uicons_sr_cross.png                   (Back view)
- fi-sr-panorama -> view_perspective_uicons_sr_panorama.png      (Perspective toggle)
- fi-sr-step-forward -> step_uicons_sr_step_forward.png (Step 1 button)
- fi-sr-forward-fast -> step_fast_uicons_sr_forward_fast.png (Step 10 button)
- fi-sr-scanner-image -> depth_uicons_sr_scanner_image.png (depth camera)
- fi-sr-compass-alt -> imu_uicons_sr_compass_alt.png (IMU)
- fi-sr-radar -> lidar_uicons_sr_radar.png (spinning lidar)
- fi-sr-sensor -> sensor_uicons_sr_sensor.png (unknown sensor fallback)
- fi-sr-transformation-shapes -> visual_uicons_sr_transformation_shapes.png (visual object)
- fi-sr-sphere -> sphere_uicons_sr_sphere.png (sphere object)
- fi-sr-cube -> box_uicons_sr_cube.png (box object)
- fi-sr-database -> cylinder_uicons_sr_database.png (cylinder object)
- fi-sr-capsules -> capsule_uicons_sr_capsules.png (capsule object)
- fi-sr-vector-polygon -> mesh_uicons_sr_vector_polygon.png (mesh object)
- fi-sr-land-layers -> ground_uicons_sr_land_layers.png (halfspace object)
- fi-sr-mountain -> heightmap_uicons_sr_mountain.png (heightmap object)
- fi-sr-cubes -> compound_uicons_sr_cubes.png (compound object)
- fi-sr-wave-square -> deformable_uicons_sr_wave_square.png (deformable object)
- fi-sr-braille -> granular_uicons_sr_braille.png (granular object)
