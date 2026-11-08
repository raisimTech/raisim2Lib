# Rayrai TCP viewer

The browser version of `rayrai_tcp_viewer` runs in your browser with WebGPU. It is the desktop viewer compiled to WebAssembly, with the same panels, and it does two things:

- **Runs the server examples.** Every RaiSim example built with `RaisimServer` is compiled to WebAssembly from its unchanged source. Pick one under *Server examples next to this viewer*, or use the **Run on web** button on its documentation page. Each example runs in a worker of its own, like a separate process, and the viewer connects to it.
- **Connects to your servers.** A `RaisimServer` also accepts browser viewers on its TCP port, over WebSocket. Type the host and port, or pick a server the local web server has heard on the network.

The documented rayrai examples run in the browser too, each on a page of its own (`web-examples/example.html?example=rayrai_offroad_buggy`) and from its unchanged source: their `ExampleApp` draws on the page's canvas. Use the **Run on web** button on the example's documentation page. An example that needs models or textures asks before downloading them and names their size; one declined fetches nothing. The same dialog offers to keep the files in the browser (checked by default). Kept files are checked against their SHA-256 before each use, and the example then starts without downloading. The browser tools also keep the shader programs they compile, for the same simulator build, so a later visit skips compiling them. Web pages cannot see when the computer restarts, so both stay until a visitor removes them with the footer's **Remove files saved in this browser** button, the browser clears its site data, or (for programs) a new simulator build replaces them.

<BrowserViewer />

No installation, account, or activation key is required. The browser tools are
subject to the [WebAssembly license](./License.md#webassembly-edition): eligible
academic research is free of charge; commercial evaluation is limited to two
hours per individual user per calendar week. Users are responsible for compliance,
without technical time tracking. Production use requires a separate commercial
license. To build your own scene, open the [Browser Scene Editor](./BrowserSceneEditor.md).

## Reaching servers from the browser

A browser cannot listen for the UDP beacons that `RaisimServer` sends, and a page served over HTTPS (such as raisim.com) may only open unencrypted connections to this computer. So:

| Viewer opened from | Servers it can reach | Server list |
| --- | --- | --- |
| raisim.com | `localhost` and `127.0.0.1` | examples in the page |
| A local copy (`python3 web/serve.py`) | this computer and the local network | examples in the page and the beacons `serve.py` hears |

A server accepts pages from `https://raisim.com` and from this computer. To accept a page served elsewhere, call `RaisimServer::allowWebViewerOrigin("http://host:port")` before `launchServer()`. Pages from other origins are refused, so that a web site you visit cannot drive a simulation on your computer.

## Local browser cache

The first run downloads the WASM and saves verified bytes in this browser's CacheStorage. Later visits revalidate the small `bundle-manifest.json`; when its WASM SHA-256 matches a saved entry, the binary is read locally without a WASM download. The JavaScript glue URL carries its own hash so normal HTTP caches keep a separate entry for each build. Content hashes identify builds even while the release version number remains unchanged.

Cached bytes are hashed before use. Corrupt entries are discarded and fetched again; failed integrity checks on downloads stop loading and are not saved. Downloads also use the browser's Subresource Integrity check. A new binary hash downloads the new bytes and removes older entries for this exact deployment path. Other applications and deployment paths keep their caches.

The simulator requires HTTPS or a trusted local origin such as `localhost`, with the cross-origin isolation headers described below. When storage is denied or full, it downloads the binary without saving it. Clearing site data, private browsing policies, or browser storage eviction can require another download. No storage-permission prompt is displayed. CacheStorage belongs to a browser profile and site, rather than the user's Downloads folder. The UI reports when it uses or saves the local simulator.

The manifest is checked online on each fresh load; this is not a complete offline application. A progress bar shows the simulator download in megabytes, then the compile and scene-building stages. Serve the manifest and bootstrap files (`index.html`, `viewer.mjs`, `viewer-streams.mjs`, `viewer-examples.mjs`, `server-example-worker.mjs`, `server-examples.tsv`, `example.html`, `example.mjs`, `rayrai-examples.tsv`, `editor.html`, `editor.mjs`, `scene-model.mjs`, `tool-ui.mjs`, `touch-input.mjs`, `wasm-cache.mjs`) with `Cache-Control: no-cache`, and publish the bundle together to avoid deployment skew.

## Browser profile

The browser build compiles the RaiSim physics sources, including joint force/torque and tactile sensors, the original OpenUSD scene and mesh importers, and Assimp's enabled importers. OpenUSD and oneTBB are built for WebAssembly; USD schemas and plugin metadata are embedded in the module. Physics integration remains single threaded. Renderer asset workers use shared memory and WasmFS.

Rayrai's browser target uses the same renderer source list as desktop, including `RayraiWindow`, its camera, materials, instancing, shadows, HDR targets and rendering passes. A browser adapter owns a real WebGPU device, forwards input, captures physics snapshots and presents Rayrai's final texture. A browser-only dispatch layer maps the renderer's operations to WebGPU resources and command encoders. The actual GLSL shaders compile through glslang, SPIR-V Tools and Tint to WGSL. No WebGL context is created.

The browser reports up to 32 texture units, the number desktop OpenGL drivers report, when the GPU lets a shader stage sample that many textures (Chrome reports 48 for an NVIDIA RTX GPU, for example). Rayrai then uses the same shader profile as on Linux and Windows desktops, including planar ground reflections, height and parallax maps, subsurface maps and the full projected-decal pool. WebGPU allows only 16 samplers per stage, so textures whose samplers are identical share one: each program keeps one assignment of its textures to sampler slots and assigns them again only when a draw gives two textures of one slot different samplers. GPUs that sample fewer textures keep Rayrai's 16-unit profile, as macOS desktops do. The renderer names the browser's GPU adapter (for example `Rayrai WebGPU nvidia blackwell`) as its OpenGL renderer string, so the TCP viewer picks the same automatic render quality it picks for that GPU natively. A current Chrome or Edge browser with WebGPU, WebAssembly JavaScript Promise Integration (JSPI), exception handling and shared memory is required, on HTTPS or localhost. Unsupported browsers display an error before downloading the binaries. Native desktop rendering keeps its existing OpenGL path and has no browser dispatch or shader compiler overhead.

The browser backend supports the graphics and compute operations exercised by these tools, including HDR targets, shadows, instancing, compute storage buffers, mipmaps, anisotropic filtering up to 16× and color/depth readback. Native OpenGL-version gates remain disabled: the shared renderer uses its CPU culling path in these tools, while separate tests exercise compute and indirect draws through the real C++ dispatch layer. WebGPU supports four-sample antialiasing here; requests for two samples use four. Nonzero texture LOD bias, sized point sprites, GPU lidar's buffer textures, geometry/tessellation shaders, wide lines, partial masked or scissored clears, border texture wrapping, nonidentity texture swizzles, unpack byte swapping, partial multisample resolves, and Vulkan ray queries remain unsupported and report errors. Unsupported enabled draw capabilities also report errors when used. Cube-map sampling is always seamless in WebGPU. Float32 linear filtering requires the adapter's corresponding feature. These boundaries prevent a desktop feature-parity claim; there is no WebGL fallback.

On touchscreens, the examples and the TCP viewer receive one finger as the left mouse button (drag to orbit, tap to select or press a panel control) and a two-finger pinch as the mouse wheel; spreading the fingers moves the free camera forward. A finger presses the button only once it moves or lifts, so starting a pinch never selects. Backing stores are sized for at most twice the CSS pixel size, as in the scene editor, so phones with device pixel ratios of 3 do not render nine times the pixels.

JavaScript calls into C++ are serialized while JSPI may suspend. Uniform slices and staging uploads remain valid until GPU completion, with at most three submissions in flight. The shader compiler moves the uniforms Rayrai sets for each object (its transform, color and material) into a uniform block of their own, so a draw copies that block, about 1 KB for the PBR shader, instead of the whole 2–8 KB default block; large uniform arrays keep blocks of their own as before. A readback waits once for the GPU work submitted before it. Chrome's GPU process otherwise looks for finished work only every few milliseconds, so while it waits the layer asks the queue for completion every 0.2 ms; a 1 MB readback then takes about 1 ms instead of 3. Readbacks in the stored format of their source (RGBA8 color, R32F or depth floats, half floats) are plain copies: `glReadPixels` into client memory copies the mapped rows straight into WebAssembly memory, and `glReadPixels` into a pixel-pack buffer whose rows are 256-byte aligned copies on the GPU. BGRA camera images upload to RGBA textures one 32-bit word per texel. A buffer whose `glBufferData` usage is `GL_STREAM_READ`, `GL_STATIC_READ` or `GL_DYNAMIC_READ` gets a mappable copy with each submission that writes it, so mapping it after a fence or a ring of buffers has completed, as Rayrai's water and camera readbacks do, does not wait for GPU work queued since. A map that finds no current copy waits as any readback does. Device-loss errors stop rendering. Run example or Retry recreates the graphics device using the loaded WASM module; the editor retains the authored draft.

The server examples, the documented rayrai examples and the scene editor are the browser entry points; undocumented examples and the native TCP viewer executables are not browser applications. The scene editor exposes primitive scene authoring. The libraries include mesh and USD importers, while these self-contained scenes embed their URDFs and download no model assets.

Scene replacements use Rayrai's `swapWorld()` to retain the camera and reusable assets. Canvas backing dimensions are controlled by the browser UI; SDL does not apply a second DPI scale. Paused tools skip repeated draws and unchanged readings, and redraw after camera movement, resizing or stepping. Solver settings, precision and the 2 ms physics step remain unchanged.

## Build and preview

Install [Emscripten SDK 4.0.0](https://emscripten.org/docs/getting_started/downloads.html), which supplies Clang 20, CMake 3.22 or newer, Java for Closure Compiler, an Eigen 3.4 checkout including `COPYING.MPL2`, and Python with Jinja2 and MarkupSafe (`python3 -m pip install jinja2 markupsafe`). The normal documentation build also needs its existing Doxygen, Node.js, npm, and native RaiSim dependencies.

```bash
export EMSDK=/path/to/emsdk
export RAISIM_WEB_EIGEN_INCLUDE_DIR=/path/to/eigen-3.4.0
./update_doc.sh --no-publish
```

The first build downloads SHA-256-pinned OpenUSD 26.03, oneTBB 2021.12.0, Dawn/Tint, glslang, SPIR-V Headers/Tools and Abseil sources and builds their required C++ libraries with six jobs. The Dawn checkout supplies the shader translator; browser rendering uses the browser WebGPU API. No external graphics service receives scenes or shaders. Later builds reuse the dependency cache. `RAISIM_WEB_DEPENDENCIES_DIR` selects the USD/TBB cache and `RAISIM_WEBGPU_DEPENDENCIES_DIR` selects the shader dependency cache; `RAISIM_WEB_OPENUSD_ROOT` can select an existing matching installation containing `RAISIM_WEB_DEPENDENCY_NOTICES.txt`. The documentation and browser examples also use six build jobs. It prints a local HTTP preview command. Open the printed address: `file://` cannot load the ES modules and WebAssembly. `--no-publish` only disables publishing; the browser binary has no host or machine checks in either mode. The normal publishing path builds the same unrestricted binary before copying generated public assets.

For a local SDK installation, `linux_install.sh` builds the complete documentation website and installs it in `../raisim2Lib/web` (or `<RAISIM_LOCAL_INSTALL_ROOT>/web`), with the audited browser bundle in its `web-examples` directory. Configure Emscripten and Eigen before running the installer and answer **N** at the release prompt. Documentation, browser toolchain and destination checks run before native cleanup. The local path uploads nothing. Do not run it concurrently with `update_doc.sh` against the same build caches.

```bash
python3 web/serve.py --directory ../raisim2Lib/web --port 8003
```

Open `http://localhost:8003/` for the home page; the **Web** menu leads to the examples and the scene editor. To rebuild and install only the website and browser tools into an existing local SDK, run `./update_doc.sh --no-publish --install-root ../raisim2Lib`. An existing managed website is replaced only after copying and auditing succeed; `install-manifest.json` lists its files, and unmanaged files are refused. The build caches and the installed `web` directory must not overlap.

The build also packs the scene editor's robot library into `web-examples/assets`: every robot in `web/robots/catalog.json` with the files its URDF needs and the preview saved next to it. Each browser example gets a pack of the files its source opens, found from its string literals, `exampleRscPath()` folders, path joins and the models those files reference. Meshes above 40,000 triangles are simplified and textures above 1024 pixels downscaled; collision geometry and 16-bit height images are kept. `assets/manifest.json` lists each file with its size and SHA-256 hash, and the bundle audit checks it. The packer is a small native tool that `web/build_examples.py` builds against the installed native RaiSim (`--native-prefix`, default `../raisim2Lib/raisim`). After adding a robot or changing its model, render the previews on a machine with OpenGL:

```bash
cmake -S web/robots -B /tmp/robot-previews -DCMAKE_PREFIX_PATH=../raisim2Lib/raisim
cmake --build /tmp/robot-previews
python3 web/robots/generate_previews.py --tool /tmp/robot-previews/raisim_robot_previews --rsc ../raisim2Lib/rsc
```

To compile and stage the examples separately:

```bash
python3 -B web/build_examples.py \
  --build-dir /tmp/raisim-wasm \
  --site-dir /tmp/raisim-browser-site
python3 web/serve.py --directory /tmp/raisim-browser-site
```

Then open `http://localhost:8000/web-examples/`. `RAISIM_WEB_BUILD_DIR` selects the documentation build's WASM cache directory. `RAISIM_WEB_BUNDLE_DIR` can point to an existing compiled bundle; the same artifact audit runs before it is staged.

The browser build enters a dedicated CMake path only under Emscripten. Native compiler flags, license handling, renderer, solver and hot loops do not gain web runtime checks.

## Release archive

The Linux release workflow creates a separate `webassembly-<version>.zip` alongside the native archives. It is shared by all CPU architectures and contains a README, a localhost preview server and the audited `web-examples/` directory, including these examples and the scene editor. The documentation release uses the same compiled bundle. Source, native libraries, debug files and test modules are excluded.

An existing compiled bundle can also be packaged locally without uploading:

```bash
python3 -B web/package_release.py \
  --bundle-dir /tmp/raisim-wasm/web-bundle \
  --output-dir /tmp/raisim-archives --version 2.12.0
```

Extract the archive, enter its `webassembly-<version>` directory, and run `python3 serve.py`. Open `http://localhost:8000/web-examples/index.html` or `http://localhost:8000/web-examples/editor.html`.

## Hosting and source confidentiality

Serve only the audited `web-bundle` directory, or the generated documentation site containing its `web-examples` subdirectory. Serve `.wasm` as `application/wasm` and `.mjs` as JavaScript. The output manifest records the exact public file set and SHA-256 hashes. Do not serve the CMake build tree or source checkout.

Release builds use full link-time optimization, hidden visibility, JavaScript Closure minification, paired WASM-export/JSPI-name minification, and narrow C interfaces. C++ runtime type information needed for safe casts and exception handling is retained. They do not include source maps, native activation keys, the native license verifier or private source paths. Packaging fails on unexpected files or detected private material. Third-party notices accompany the binaries.

These measures remove useful reverse-engineering information; downloaded executable code remains inspectable, copyable and patchable. The original C++ source files stay in the repository. The binaries contain shader text and schema data required at runtime, which can also be extracted. The compiled algorithms cannot be guaranteed secret in a browser. The audit is a packaging check, not a proof that arbitrary binary content contains no secrets. No browser-side encryption or key can change that limitation.

For the standalone example directory, a suitable Content Security Policy is:

```text
default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; style-src 'self'; font-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'
```

Use HTTPS for public hosting and send these response headers:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin
X-Content-Type-Options: nosniff
```

The documentation pages, the full-screen tools and worker responses must have the appropriate isolation headers. A static host that cannot configure them needs a hosting layer that can. The included localhost server supplies them. The binaries impose no domain or machine restriction.

## Validation and benchmarks

Set `-DRAISIM_WEB_DOCS_SITE=/path/to/built/site` in a browser-test build to include the Web menu, tool links, light/dark controls, preserved scene state and responsive embeds in its browser checks. These checks also record serial theme-switch timings as informational browser measurements.

Browser validation additionally requires Node.js 22 or newer with a global `WebSocket`, plus local Chrome or Chromium. The documentation site itself retains its existing Node.js 18+ requirement.

```bash
python3 -B web/test_audit_bundle.py
python3 -B web/test_build_examples.py
node web/test_wasm_cache.mjs
node web/test_browser.mjs /tmp/raisim-wasm/web-bundle /tmp/raisim-browser-results
```

To register this browser suite with CTest, configure the Emscripten build with `-DRAISIM_WEB_BROWSER_TESTS=ON`, build `raisim_web_examples`, then run `ctest --test-dir <wasm-build> -j1 --output-on-failure`.

The browser suite uses Chrome and local HTTP fixtures. Cache unit tests cover versions, corruption, download integrity, storage failures, timeouts and deployment isolation. A separate C++ renderer test module checks context ownership, cleanup, primitive and compound rendering, appearance-only edits, and pixel equality across static redraws, scene swaps and camera changes. Compound checks cover all four primitive types, child and body rotations, nonzero centers of mass, hidden children, and immutable geometry reuse. DPR 2/3 tests check stable backing dimensions and the absence of repeated canvas storage resets. It also checks direct pose edits without a world-clock change and removed objects. A shared native/WASM math test checks independent aligned allocations and pendulum joint/body motion against a separate small-step Runge–Kutta reference. Additional tests exercise actual USDA/USDC mesh import, USD articulation dynamics, worker filesystem reads, real Rayrai WebGPU pixels and dispatch errors, float texture conversion, buffer mapping and pixel-pack offsets, RGB row-padding guards, combined depth/stencil readback, asymmetric winding/viewport/scissor coordinates, per-attachment blend/color-mask rendering and camera-state restoration after device loss, concurrent frame reservation and device-loss cleanup. Shader conformance tests compile actual PBR, instanced, sky, transmittance, shadow and culling shaders and verify real compute results. Dispatch-layer unit tests cover readback snapshots and their pool bound, readback rows handed over while mapped, completion queries while waiting and their rate, BGRA uploads at aligned and unaligned offsets, sampler sharing and its convergence across materials, sampler-array units, generic vertex attribute values, and when cached pipeline keys and resolved bindings are reused. In Chrome, a BGRA camera image is uploaded and read back into strided client memory. Shader conformance tests check which uniforms move into the per-object block, their offsets, the bindings of both stages and the actual PBR program's block. Touch tests check the finger-to-mouse translation, pinch direction and the pixel-ratio cap, and in Chrome a touch drag, a pinch started during a drag and a two-finger orbit in the editor, and finger and pinch delivery to the TCP viewer. Source provenance checks require the physics and shared renderer sources and actual renderer archive symbols. Test modules are not staged as public assets.

The native/WebGPU reference test shares a fixed 128 × 128 PBR scene and camera with desktop Rayrai. It explicitly selects the same native 16-sampler shader profile, pins shadow resolution, disables temporal effects and multisampling, and uses anisotropy 1 with LOD bias 0. Comparison checks whole-frame and eroded object-mask RGB error, colored-object centroids and coverage, and shadow contribution relative to native. Four separate browser tests deliberately corrupt view-matrix transposition, projection, shadow uniforms and material shading and require the reference checks to reject each defect. The test harness also waits for GPU completion and event delivery and rejects uncaptured validation errors; a negative test proves it catches a late invalid queue submission.

Additional regressions cover inactive VAO/FBO retention after resource deletion, mapped allocation cleanup and failed submission cleanup, shader-source snapshots, draw-buffer switches, planar vertex attributes, negative scissor origins, scissored blits, normalized and integer pixel conversions, PBO offset-zero uploads, texture storage with a bound PBO, RGB opacity, volume packing, and the native file-texture loader. Native and browser tests both exercise the auto-exposure PBO/fence ring. A standalone shader benchmark compares the former full translation at compile time with parse-only validation for actual PBR and sky shaders; full translation still occurs at link time.

To regenerate the checked-in native reference after an intentional rendering change, build the native `rayrai_webgpu_reference_test` target with Clang 20 and run it with an output path in `/tmp` using `cmake -E env RAYRAI_FORCE_LIMITED_PBR_SAMPLER_FALLBACK=1 <native-test-executable> /tmp/native-golden.json`. The generator requires this flag and records the profile in the fixture. Review the resulting image comparison before replacing `web/fixtures/native-golden.json`. Keep all generated shader fixtures and reference modules outside the public bundle.


The shader compiler also has a standalone Clang 20 conformance target. After the browser build has fetched its dependencies, compile it with six jobs, emit fixtures, and register their GPU checks:

```bash
cmake -S visualizer/rayrai/webgpu/tests -B /tmp/raisim-shader-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_CXX_COMPILER=clang++-20 -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DRAISIM_WEBGPU_DEPENDENCIES_DIR=/tmp/raisim-webgpu-dependencies
cmake --build /tmp/raisim-shader-tests --parallel 6
ctest --test-dir /tmp/raisim-shader-tests -j1 --output-on-failure
/tmp/raisim-shader-tests/rayrai_webgpu_shader_tests --benchmark
/tmp/raisim-shader-tests/rayrai_webgpu_shader_tests /tmp/raisim-shader-fixtures
emcmake cmake -S . -B /tmp/raisim-wasm \
  -DRAISIM_WEBGPU_SHADER_FIXTURES=/tmp/raisim-shader-fixtures
```

The fixtures contain shader metadata and WGSL used only by tests; they are excluded from public bundles. The browser shader check renders distinct instanced colors, verifies immutable uniforms and ordered vertex updates, and reads back real compute-culling output. Runtime bit-pattern tests verify scalar and vector float32 NaN/infinity classification, including signed NaN payloads, subnormals and finite limits. Classification uses integer bit tests so the translation does not depend on floating-point comparison shortcuts.

An incremental-build test verifies that the manifest updates after an asset edit without changing WASM. The browser suite checks all four scenes, pause/reset and invalid inputs, settled force/tactile readings, and execution on three different hosts, including an arbitrary test domain. Repeat-navigation tests block the WASM endpoint to prove that matching cached bytes are actually used, then exercise version updates, corrupted storage, denied storage and failed-request retries. Paused-view checks verify zero repeated draw, layout-read and taxel-paint calls, identical displayed pixels, and correct wake-up after orbit, resize, DPI changes, stepping and 2D-context restoration.

For an additional serial render benchmark:

```bash
RAISIM_WEB_PERF=1 node web/test_browser.mjs \
  /tmp/raisim-wasm/web-bundle /tmp/raisim-browser-performance
```

This records twelve cases: all four scenes with static state, camera motion, and eight physics steps per rendered frame. Each case warms up before five batches of eight frames. Awaiting the WebGPU queue includes GPU completion in each measured frame. Reports include raw timing samples, draw/submission/allocation/pipeline counts, framebuffer hashes and raw sensor metrics for comparing builds. The normal suite also records cold/warm loading and step/render-submission timings. Reports identify the actual WebGPU adapter selected by Chrome; command-line flags do not prove that software rendering is in use. Software-adapter results are not hardware GPU throughput measurements, and completion timings are not interactive frame rates. Set `CHROME` to another local Chrome/Chromium binary if needed.

An optional second-browser check is available with installed Firefox and geckodriver:

```bash
node web/test_firefox_cache.mjs /tmp/raisim-wasm/web-bundle /tmp/raisim-firefox-results
```

`GECKODRIVER` and `FIREFOX_BINARY` select custom executable paths. This checks persistent-cache reuse with the WASM network endpoint blocked and, when both WebGPU and JSPI are available, the sensing scene's analytical loads and visible articulated pendulum motion. The Chrome suite also covers loading while hidden, resuming on visibility, and preserving a manual pause.
