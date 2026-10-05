# City asset licenses and attribution

The third-party models, textures and sky image in this directory are free
assets under two licenses:

* **CC0 1.0 Universal (CC0-1.0)** for everything except the cars: copying,
  modification, commercial use and redistribution are allowed without
  attribution. The legal text is in [LICENSE-CC0-1.0.txt](LICENSE-CC0-1.0.txt).
* **CC BY 4.0** for the four cars in `cars/`: the same freedoms, provided the
  author is credited as below
  (https://creativecommons.org/licenses/by/4.0/). Each car folder keeps the
  `license.txt` that Sketchfab ships with the model.

## Poly Haven (CC0)

Provider's license statement: https://polyhaven.com/license. Powered by Poly Haven.

| Asset | Author | Used as |
| --- | --- | --- |
| [Modular Urban Apartments Facade](https://polyhaven.com/a/modular_urban_apartments_facade) | James Ray Cock | `apartments/` building modules |
| [Modular Factory Facade](https://polyhaven.com/a/modular_factory_facade) | James Ray Cock | `factory/` building modules |
| [Street Lamp 01](https://polyhaven.com/a/street_lamp_01) | Josh Dean | `props/street_lamp_01` |
| [Fire Hydrant](https://polyhaven.com/a/fire_hydrant) | Gonçalo Felício | `props/fire_hydrant` |
| [Metal Trash Can](https://polyhaven.com/a/metal_trash_can) | GurJas Studios | `props/metal_trash_can` |
| [Modular Street Seating](https://polyhaven.com/a/modular_street_seating) | Stuart Attenborrow | `props/modular_street_seating` |
| [Water Manhole Cover](https://polyhaven.com/a/water_manhole_cover) | Raunox | `props/water_manhole_cover` |
| [Utility Box 02](https://polyhaven.com/a/utility_box_02) | James Ray Cock | `props/utility_box_02` |
| [Concrete Road Barrier](https://polyhaven.com/a/concrete_road_barrier) | Amal Kumar | `props/concrete_road_barrier` |
| [Asphalt 02](https://polyhaven.com/a/asphalt_02) | Rob Tuytel | road surface |
| [Concrete Pavement 02](https://polyhaven.com/a/concrete_pavement_02) | Charlotte Baglioni | sidewalks |
| [Concrete Floor 01](https://polyhaven.com/a/concrete_floor_01) | Rob Tuytel | curbs |
| [Bitumen](https://polyhaven.com/a/bitumen) | Rob Tuytel | roofs |
| [Kloofendal 48d Partly Cloudy (Pure Sky)](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky) | Greg Zaal, Jarod Guest | `sky/` HDR environment |

The street trees reuse the forest example's
[Tree Small 02](https://polyhaven.com/a/tree_small_02) from `../forest`.

## BlendKit (CC0)

[Traffic Cone (Photoscanned)](https://www.blendkit.com/asset-gallery-detail/4aec6ae7-3b6f-4888-90ec-2e77f890c223/)
by Nik Kottmann, licensed CC0 on BlendKit, is `traffic_cone/`. Only BlendKit
assets marked `cc_zero` are redistributable this way; its royalty-free license
does not allow sharing the asset files themselves.

## Sketchfab (CC BY 4.0)

The parked cars are fictional-brand models by Daniel Zhabotinsky. Credits, as
the author asks for them:

* `cars/fairheaven_lt80`: This work is based on "Fairheaven LT '80 - Low poly model" (https://sketchfab.com/3d-models/fairheaven-lt-80-low-poly-model-e2678da920cc4be68dbc193727919ffb) by Daniel Zhabotinsky (https://sketchfab.com/DanielZhabotinsky) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)
* `cars/fairheaven_sw84`: This work is based on "Fairheaven SW '84 - Low poly model" (https://sketchfab.com/3d-models/fairheaven-sw-84-low-poly-model-aa0becb6e854422596cac6b21bf79787) by Daniel Zhabotinsky (https://sketchfab.com/DanielZhabotinsky) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)
* `cars/kiri86`: This work is based on "Kiri '86 - Low poly model" (https://sketchfab.com/3d-models/kiri-86-low-poly-model-5c17c582ef9549798f694b660feea42a) by Daniel Zhabotinsky (https://sketchfab.com/DanielZhabotinsky) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)
* `cars/canyon75_taxi`: This work is based on "Canyon '75 Taxi - Low poly model" (https://sketchfab.com/3d-models/canyon-75-taxi-low-poly-model-9e8f92a215784f3d8aaaaeab1bef54c4) by Daniel Zhabotinsky (https://sketchfab.com/DanielZhabotinsky) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

Changes made: the node transforms are baked into one Z-up mesh per car with
one primitive per material, textures are reduced to 1K (JPEG unless blended),
the clear-coated metallic body paint is replaced by a plain car paint, and two
extra paint colours are added to three of the cars.

## Local preparation

`examples/tools/download_city_assets.py` downloads the sources and records
their URLs and SHA-256 hashes in [sources.json](sources.json); the Sketchfab
cars need the API token of a free Sketchfab account.
`examples/tools/prepare_city_assets.py` splits the facade kits into Z-up
modules that share the kits' buffers and textures (with their flat, densely
gridded walls rebuilt from a few rectangles), keeps one variant of each prop
lineup, prepares the cars as described above, converts the cone's WebP
textures to JPEG with its rotation baked into the vertices, and clamps and
turns the sky HDR (the scene's directional light provides the sun).
[manifest.json](manifest.json) records the hashes of the prepared files.
`ground/` holds meshes generated for the streets and building shells; they use
the textures in `textures/`.

This notice covers the third-party assets in this directory. It does not
change the licenses of RaiSim, rayrai or the example code.
