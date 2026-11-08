# Server Example: Underwater ROV

## Overview

A remotely operated vehicle holds its station 3 m under the sea ([Fluids](../../Fluids.md)) with six
thrusters. The current runs at 0.5 m/s and turns by a quarter every eight seconds.
- The sea holds the vehicle up a little more than its weight, so the vertical thrusters push it down.
- The sea drags the vehicle with the current and moves the water around it with it (its added mass).
- Its foam block and skids put its center of mass below its center of buoyancy, so it stays level by
  itself.
- A proportional-integral-derivative controller keeps its position and heading.

Each time the current turns, it pushes the vehicle up to 9 cm off its station, and the vehicle comes
back to within 7 mm before the current turns again. The cyan arrow shows the current. The ROV hovers above a rocky survey site with textured sand,
seagrass, sea fans, shells and an acoustic lander. Its detailed shell includes six thruster ducts,
a camera port, work lamps and protective rails; an orange tether follows it toward the surface.

## Target

CMake target: `underwater_rov`.

## Run

```bash
./build-examples/examples/underwater_rov
```

Connect `rayrai_tcp_viewer` to port 8080. On Windows, run `build-examples/bin/underwater_rov.exe`.

## Details

- The vehicle is a compound of a hull, a foam block and two skids. It displaces 90.8 kg of sea water
  and weighs 88 kg.
- The thrusters push at fixed points of the vehicle (`setExternalForce()` with a position). Four
  vectored ones in the horizontal plane give the force along x and y and the torque about z, and two
  vertical ones the force along z. A pseudo-inverse of their effects allocates the thrust, which is
  limited to 60 N each.
- The current is the velocity of the `FluidVolume`, changed with `World::setFluidVolume()`.
- The physics scene lives in `include/fluid_rov.hpp`. The environment and lighting come from
  `rsc/underwater_rov/underwater_rov.rscene`, which the server shares with the TCP viewer. The
  decorative meshes add no collision or buoyancy geometry. The complete asset dependency set is
  about 12 MB, below the 200 MB budget; source and license details are in the asset directory.
- `underwater_rov_scene_benchmark` in the source repository compares scene loading and physics step
  time with and without the environment.
