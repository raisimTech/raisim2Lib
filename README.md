# RaiSim2

RaiSim is a physics engine for robotics and artificial intelligence research that provides efficient and accurate simulations for robotic systems. We specialize in running rigid-body simulations while having an accessible, easy to use C++ library.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/CN0ah5-OWik/0.jpg)](https://www.youtube.com/watch?v=CN0ah5-OWik)

## News
Closed-loop system simulation is now available! Check out the [minitaur example](https://github.com/raisimTech/raisimLib/tree/master/examples/src/server/minitaur_pd.cpp)

## Dependencies

RaiSim ships with prebuilt packages under `raisim/<OS>` and can be built on Linux/macOS/Windows (see the Build section for platform-specific notes).

### - Eigen3
Vendored under `thirdParty/Eigen3` if you prefer using the bundled copy.
##### Ubuntu
```bash
sudo apt install libeigen3-dev libsdl2-dev
```

#### Windows
You can use the bundled copy at `thirdParty/Eigen3`, or install Eigen via vcpkg.

##### Windows (vcpkg)
```powershell
# from your vcpkg checkout
.\vcpkg.exe install eigen3:x64-windows sdl2:x64-windows
```

Recommended: configure with the vcpkg toolchain so `find_package(Eigen3 ...)` resolves automatically:
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\\scripts\\buildsystems\\vcpkg.cmake"
```

If you do not want to use the toolchain file, you can point CMake directly at Eigen's package config:
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DEigen3_DIR="$env:VCPKG_ROOT\\installed\\x64-windows\\share\\eigen3\\cmake"
```

Note: this repo already adds `thirdParty/Eigen3` to `CMAKE_PREFIX_PATH` on Windows. If you want to force the vcpkg copy, pass `-DEigen3_DIR=...` as above.

### - CMake
Install version >= 3.11 [here](https://cmake.org/download/).

### - Visual Studio 2019+
Install [here](https://visualstudio.microsoft.com/vs/older-downloads/), make sure to install the C++ workload during setup.

### - Optional (RaisimPy)
To build the Python wrapper (`RAISIM_PY=ON`), install a matching Python interpreter and its development headers.
##### Ubuntu
```bash
sudo apt install python3-dev
```

### - Optional (Docs / Sphinx)
If you build the documentation (`RAISIM_DOC=ON`), install Sphinx and its extensions, preferably in a local venv.

##### Ubuntu/Debian (recommended venv)
```bash
sudo apt install python3-venv
python3 -m venv docs/.venv
. docs/.venv/bin/activate
python -m pip install -r docs/requirements.txt
```
When building docs via CMake, the build will use `docs/.venv` if present and
fall back to a system `sphinx-build` otherwise.

##### Direct pip (not recommended on Ubuntu/Debian system Python)
```bash
python -m pip install sphinx sphinx-rtd-theme breathe sphinx-tabs docutils
```

## Documentation

Further documentation available on the [RaiSim Tech website](http://raisim.com).

## Build

RaiSim and RayRai ship with prebuilt packages under `raisim/<OS>` and `rayrai/<OS>`. To build examples or optional wrappers, use the top-level CMake project.

### Quick build (Linux/macOS)
```bash
cd /path/to/raisimLib
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$LOCAL_INSTALL
cmake --build build -j
cmake --install build
```

### Quick build (Windows, PowerShell)
```powershell
cd C:\path\to\raisim2Lib
# Assumes the prebuilt packages are present in the current directory:
#   .\raisim\<OS> and .\rayrai\<OS>
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Build examples (Windows, PowerShell)

On Windows, the C++ examples are built against the prebuilt RaiSim and RayRai packages:
- RaiSim: `raisim/win32` (headers: `raisim/win32/include`, import libs: `raisim/win32/lib`, DLLs: `raisim/win32/bin`)
- RayRai: `rayrai/win32` (headers: `rayrai/win32/include`, import libs: `rayrai/win32/lib`, DLLs: `rayrai/win32/bin`)

Make sure CMake can find both packages (each should contain `lib/cmake/<package>/<package>Config.cmake`).

Important: the commands below assume your current directory is the repo root (the directory that contains the `raisim/` and `rayrai/` folders). In other words, RaiSim is installed into `.\raisim` and RayRai is installed into `.\rayrai`.

Configure and build with a multi-config generator (Visual Studio):
```powershell
cd C:\path\to\raisim2Lib
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DRAISIM_EXAMPLE=ON
cmake --build build --config Release
```

If you use vcpkg (common on Windows): install the vcpkg packages and configure with a fresh build directory (the toolchain file is only applied on first configure):
```powershell
vcpkg install eigen3:x64-windows sdl2:x64-windows
cmake --fresh -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\\scripts\\buildsystems\\vcpkg.cmake" `
  -DRAISIM_EXAMPLE=ON
cmake --build build --config Release
```

If CMake still cannot find SDL2, point it at the vcpkg package config explicitly:
```powershell
cmake --fresh -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\\scripts\\buildsystems\\vcpkg.cmake" `
  -DSDL2_DIR="$env:VCPKG_ROOT\\installed\\x64-windows\\share\\sdl2" `
  -DRAISIM_EXAMPLE=ON
```

If you have RaiSim/RayRai installed separately (still under this repo root), you can be explicit:
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_PREFIX_PATH="$PWD\\raisim\\win32;$PWD\\rayrai\\win32" `
  -DRAISIM_EXAMPLE=ON
cmake --build build --config Release
```

Where the executables end up:
- Windows builds place the example `.exe` files into `build\bin\` (see `examples/CMakeLists.txt` which sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY_*`).
- The `rsc/` folder is copied next to the executables on Windows so examples can find their resources.

Run an example (PowerShell):
```powershell
cd C:\path\to\raisim2Lib
.\raisim_env.ps1   # adds required DLL folders to PATH (raisim/rayrai, and vcpkg if present)
.\build\bin\primitive_grid.exe
```

Notes:
- Use `--config Debug` to build debug binaries; the shipped package includes both `*d.dll` (Debug) and non-suffixed DLLs (Release).
- If you use a different generator, make sure it targets 64-bit (the prebuilt `win32` package is x64).
- If you see an error like `rayrai_FOUND is FALSE ... missing: stb::stb` while using the vcpkg toolchain, make sure CMake is finding `stb` in config mode (from your RayRai package or from vcpkg). You can force it with: `-Dstb_DIR="C:\path\to\rayrai\win32\lib\cmake\stb"`.
- If you see `Cannot open include file: 'glm/glm.hpp'`, check your RayRai install layout. The expected file is `rayrai\win32\include\glm\glm.hpp`. If you instead have `rayrai\win32\include\glm\include\glm\glm.hpp`, then GLM was installed with an extra `include/` level; reinstall GLM into the `rayrai\win32` prefix (not into a nested `...\include\glm` prefix), or use a vcpkg GLM (`vcpkg install glm:x64-windows`) and point CMake at it with `-Dglm_DIR="%VCPKG_ROOT%\installed\x64-windows\share\glm"`.

### Build options
- `RAISIM_EXAMPLE` (default ON): build C++ examples in `examples/`.
- `RAISIM_PY` (default OFF): build the RaisimPy wrapper. Use the Python you want to target and pass `-DPYTHON_EXECUTABLE=<path>`.
- `RAISIM_MATLAB` (default OFF): build the Matlab wrapper (requires MATLAB).
- `RAISIM_DOC` (default OFF): build documentation.
- `RAISIM_ALL` (default OFF): enables all of the above.

Example with options:
```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=$LOCAL_INSTALL \
  -DRAISIM_EXAMPLE=ON \
  -DRAISIM_PY=ON \
  -DPYTHON_EXECUTABLE=$(which python)
cmake --build build -j
cmake --install build
```

## Examples
Before running examples, load the environment setup script for your shell:
- Linux/macOS: `source /path/to/raisimLib/raisim_env.sh`
- Windows (PowerShell): `.\raisim_env.ps1`
- Windows (cmd.exe): `raisim_env.bat`

## Features
- Supports free camera movement
- Allows the recording of screenshots and recordings
- Contact and collision masks
- Materials system to simulate different textures
- Height maps to create different sytles of terrain
- Ray Test to create collision checkers 

## Troubleshooting
- Ensure that all versions of dependencies fit the documentation. etc.(Visual Studio 2019, CMake verson > 3.10)
- If run into problem with executing into the raisimUnity.x86_64 file, ensure that your graphics card driver is compatible with current graphics card. (Cannot use the default open-source graphics card driver nouveau)
- Make sure to use raisimUnity natively and not on a docker.
- If using Linux, install minizip, ffmpeg, and vulkan.
- If drivers don't support vulkan, use raisimUnityOpengl instead of raisimUnisty. Found in raisimUnityOpengl directory.
- Make sure to set environment variable to $LOCAL_INSTALL when installing raisim.

## License

You should get a valid license and an activation key from the [RaiSim Tech website](http://raisim.com) to use RaiSim.
Post issues to this github repo for questions. 
Send an email to info.raisim@gmail.com for any special inquiry.

## Supported OS

MAC (including m1), Linux, Windows.
