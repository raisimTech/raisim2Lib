# Build, Test, and Benchmark

This page describes the public `raisim2Lib` package workflow. The workspace
downloads RaiSim and rayrai as binary packages, then builds examples and
optional wrappers against those packages. It does not contain the RaiSim engine
source tree. For installation and activation, see [Installation](./Installation.md). For
repository layout and build output paths, see [Project Layout](./ProjectLayout.md).

## Common CMake options

| Option | Meaning |
| --- | --- |
| `RAISIM_EXAMPLE` | Build C++ examples. Enabled by default. |
| `RAISIM_PY` | Build the Python wrapper (`raisimPy`). Disabled by default. |
| `RAISIM_DOC` | Build documentation through CMake. Disabled by default. |
| `RAISIM_ALL` | Shortcut that enables `RAISIM_EXAMPLE`, `RAISIM_PY`, and `RAISIM_DOC`. |
| `RAISIM_MATLAB` | Reserved compatibility option. The current public workspace does not add a MATLAB wrapper subdirectory or build target. |
| `RAISIM_EXAMPLE_DESKTOP_LAUNCHER` | Linux only. After each Release build of `rayrai_tcp_viewer`, install a desktop launcher for it and pin it to the GNOME / Ubuntu dock. Disabled by default; see [Rayrai TCP Viewer](./RayraiTcpViewer.md). |
| `RAISIM_FOREST_COLLISION_BENCHMARK` | Also build the `forest_collision_benchmark` example tool. Disabled by default. |
| `RAISIM_DEBUG_TOOLS` | Build manual regression and debug tools with the examples project (`RAISIM_EXAMPLE=ON`). Disabled by default; see [Regression and Debug Tools](./DebugTools.md). |

The RaiSim version is pinned by the checkout (`RAISIM_VERSION` in the
top-level `CMakeLists.txt`). Configuring with a different
`-DRAISIM_VERSION` is an error; update the checkout instead.

## Build examples

Build the package workspace with examples enabled:

```bash
cd /path/to/raisim2Lib
cmake -S . -B build-examples \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAISIM_EXAMPLE=ON
cmake --build build-examples --parallel 12
```

You can also build only the example CMake project against an installed RaiSim
and rayrai package:

```bash
cmake -S examples -B /tmp/raisim2lib-examples \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAISIM_PREFIX=/path/to/raisim \
  -DRAYRAI_PREFIX=/path/to/rayrai
cmake --build /tmp/raisim2lib-examples --parallel 12
```

Run source-built examples from the build directory. They are not installed by
the `raisim2Lib` install target. A top-level build places them under
`<build-dir>/examples`; a standalone build of `examples` places them
directly in its build directory:

```bash
./build-examples/examples/rayrai_tcp_viewer
./build-examples/examples/primitive_grid
/tmp/raisim2lib-examples/rayrai_tcp_viewer
/tmp/raisim2lib-examples/primitive_grid
```

The release package does not include a prebuilt viewer. The
`rayrai_tcp_viewer` target is built from the viewer sources owned by this
repository and links against the packaged RaiSim and rayrai libraries.

## Debug and Release package variants

The current RaiSim package can contain Debug and Release libraries in one
prefix. Its exported targets map a Debug consumer to the Debug library and
Release, RelWithDebInfo, and MinSizeRel consumers to a non-Debug library.
`RSDEBUG` follows the selected binary automatically. If the prefix contains
only one variant, every consumer configuration maps to that available variant;
building a downstream project in Debug does not manufacture RaiSim debug
assertions when only the Release library is installed.

## Timing examples

`raisim2Lib` includes standalone timing-oriented example executables. The
closed-source engine benchmark runner used for RaiSim release validation is not
shipped in this public binary-package workspace. Run package timing examples on
one thread and repeat the same command several times when comparing package or
scene changes:

```bash
cmake --build build-examples \
  --target anymal_standing_benchmark articulated_system_benchmark --parallel 12
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./build-examples/examples/anymal_standing_benchmark --fast
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./build-examples/examples/articulated_system_benchmark
```

Use [Performance](./Performance.md) for scene-level tuning guidance and for choosing a
representative package example before changing solver settings or sensor
workloads in a downstream application.

## Tests

For manual object lifecycle and rendering regression checks, see
[Regression and Debug Tools](./DebugTools.md).

The engine's own test suite is not shipped. Building with `RAISIM_PY=ON`
and `BUILD_TESTING=ON` registers `raisimpy_runtime`, which runs the Python
API regressions and all Python examples with bounded headless loops:

```bash
RAISIM_ACTIVATION_KEY="$HOME/.raisim/activation.raisim" \
  ctest --test-dir build-py -R raisimpy_runtime --output-on-failure
```

For direct `unittest` use, point `PYTHONPATH` to the built extension:

```bash
PYTHONPATH=build/raisimPy RAISIM_ACTIVATION_KEY=$HOME/.raisim/activation.raisim \
  python3 -m unittest discover -s raisimPy/tests -v
```

Set `RAISIMPY_TEST_SERVER=1` to run the example smoke tests with ephemeral
loopback servers. See [RaisimPy](./RaiSimPy.md) for the APIs and example options.

## Build the documentation

The documentation is a [VitePress](https://vitepress.dev) site built from the
Markdown pages in `docs`. Building it needs Node.js 18 or newer with npm,
Python 3, and Doxygen. The `docs_site` CMake target runs the whole pipeline:
Doxygen XML from the distributed headers, the API reference pages generated
from it, and the site:

```bash
cmake -S /path/to/raisim2Lib/docs -B /tmp/raisim2lib-docs \
  -DCMAKE_PREFIX_PATH="/path/to/raisim2Lib/raisim;/path/to/raisim2Lib/rayrai" \
  -DRAISIM_DOCS_RAISIM_INCLUDE_DIR=/path/to/raisim2Lib/raisim/include/raisim \
  -DRAISIM_DOCS_RAYRAI_INCLUDE_DIR=/path/to/raisim2Lib/rayrai/include/rayrai
cmake --build /tmp/raisim2lib-docs --target docs_site --parallel 12
python3 -m http.server -d /tmp/raisim2lib-docs/docs/site
```

The build fails on a link to a missing page, and it then checks that every
internal link and `#anchor` resolves (`docs/scripts/check_links.py`). The
build writes nothing into the source tree: it copies `docs` fresh into the
build directory (`site-src`) on every run, installs the Node packages there,
and generates the API pages into that copy. Two build directories therefore
never share any output.

The build reuses the checked-in documentation images. To regenerate those
images, configure with `-DRAISIM_DOCS_BUILD_RAYRAI_IMAGES=ON`. The image generators need a working
OpenGL context and overwrite the images in `rsc/docs/image/rayrai`. The option
is cached, so set it back to `OFF` to stop regenerating them.

While you edit pages, the development server shows each change as you save it:

```bash
cd /path/to/raisim2Lib/docs
npm ci
npm run docs:dev
```

Links into the API reference (`/api/`) need the generated API pages. For the
development server, generate them into `docs/api` from the Doxygen XML of a
documentation build:

```bash
cd /path/to/raisim2Lib/docs
python3 scripts/generate_api_docs.py --xml /tmp/raisim2lib-docs/docs/doxygen/xml --out api
```
