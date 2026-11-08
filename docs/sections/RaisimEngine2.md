# RaiSim Engine 2

## Availability

RaiSim Engine 2 is an internal, closed-source world-authoring tool. It builds
RaiSim worlds, for example worlds to train in with [RaisimGymTorch](./RaisimGymTorch.md), but it
does not train policies itself. It is not part of the public `raisim2Lib`
distribution. The public repository does not ship Engine 2 executables,
headers, libraries, CMake targets, examples, or tests, so users cannot build or
invoke it from `raisim2Lib`.

The public package boundary is the downloaded RaiSim and rayrai binaries and
headers together with the example, wrapper, resource, and documentation sources
in `raisim2Lib`. Engine 2 implementation paths and internal build options are
therefore intentionally not documented as user workflows.

## Public Alternatives

Use the supported public interfaces instead:

- construct and modify a `raisim::World` through the C++ or RaisimPy API;
- load RaiSim XML, MJCF, or OpenUSD scenes as documented in
  [World Configuration File](./WorldConfigurationFile.md) and [OpenUSD Loading](./OpenUSD.md);
- use [rayrai Visualizer](./Rayrai.md) or [Rayrai TCP Viewer](./RayraiTcpViewer.md) for visualization; and
- use [RaisimGymTorch](./RaisimGymTorch.md) for reinforcement-learning training.

`.re2` is an internal Engine 2 format. `.rscene` is the scene file Engine 2
writes. RaiSim and rayrai read it through `raisim::World(path)` and a
`raisin::RayraiWindow` on that world, available from v2.8.2 (not yet released; v2.8.0
and v2.8.1 read the file and show it with `raisin::applyRscene`), and show what Engine 2's viewport
shows, except articulated systems, compounds, deformables, granular media,
decals, irradiance volumes and point clouds. [.rscene File](./RsceneFile.md) documents the
format and the supported records, and
[Rayrai Example: Forest From .rscene](./examples/rayrai/rayrai_forest_from_rscene.md) loads a scene file shipped in
`rsc/forest`.
