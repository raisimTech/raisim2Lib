# Tank Survival assets

The tank survival example (`raisim2Lib/wasm`) draws these models and textures. They were downloaded with
`examples/tools/download_tank_assets.py` and converted with `examples/tools/prepare_tank_assets.py`
(vehicles split into hull, turret and gun; textures reduced to 1K). `sources.json` lists every
downloaded file with its URL and SHA-256.

## Vehicles (CC BY 4.0)

These models are licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
They were modified: split into parts, rescaled, re-oriented and re-textured at 1K.

| Folder | Model | Author |
| --- | --- | --- |
| `vehicles/leopard_2a4` | [Leopard 2A4M Main Battle Tank](https://sketchfab.com/3d-models/80b589fc9c0b4720888b40d01d6e5153) | [MirzaArrafiERV_45](https://sketchfab.com/MirzaArrafiERV_45) |
| `vehicles/t72b3` | [T-72B3 Main Battle Tank](https://sketchfab.com/3d-models/dfe130dfce724c30958cec51f2bd86b7) | [MirzaArrafiERV_45](https://sketchfab.com/MirzaArrafiERV_45) |
| `vehicles/t90am` | [Uralvagonzavod T-90AM Main Battle Tank](https://sketchfab.com/3d-models/619de9f0908e4c8395fe036ffe6e3884) | [MirzaArrafiERV_45](https://sketchfab.com/MirzaArrafiERV_45) |
| `vehicles/bmd2` | [Russian BMD-2M](https://sketchfab.com/3d-models/16c4113b6592474baa897902e8fbc637) | [42manako](https://sketchfab.com/42manako) |
| `vehicles/sprut_sd` | [2S25 Sprut-SD (Armored Warfare)](https://sketchfab.com/3d-models/bca9950218c545ecbd70b15704a35bf6) | [KojfDiscord](https://sketchfab.com/KojfDiscord) |
| `vehicles/msta_s` | [2S19 MSTA-S Self-Propelled Artillery](https://sketchfab.com/3d-models/c12432ada9a84859bc74c3d5ef7046f8) | [MirzaArrafiERV_45](https://sketchfab.com/MirzaArrafiERV_45) |
| `vehicles/t14_armata` | [T-14 Armata](https://sketchfab.com/3d-models/24c9a06973a64651a8b0baedee29fd57) | [42manako](https://sketchfab.com/42manako) |

## Props and ground textures (CC0)

From [Poly Haven](https://polyhaven.com), under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/):

- Props (`props/`): barrel_03, wooden_military_crate, old_military_crate, ammo_box, metal_jerrycan_green,
  concrete_road_barrier, old_tyre, shrub_02, shrub_04, wild_rooibos_bush, tree_stump_01,
  namaqualand_boulder_04 and namaqualand_boulder_05.
- Ground textures (`ground/`): sparse_grass, brown_mud_rocks_01, aerial_sand, snow_02, burned_ground_01,
  rubble, cracked_concrete and concrete_wall_006.

The game also uses the CC0 plants, trees and rocks of `rsc/forest` and the CC0 terrain textures of
`rsc/offroad_buggy/terrain` (see their own attribution files).
