# Benchmarks

Each benchmark below runs RaiSim and another engine or library on the same machine and the same
workload. The pages list the machine, compiler, versions and number of runs, what is checked, and
how to repeat the measurements.

| Area | Benchmark | Compared with | Result |
| --- | --- | --- | --- |
| Physics engines | [RaiSim vs MuJoCo](./Benchmark.md) | MuJoCo | Ten single-thread workloads (articulated dynamics, contacts, closed loops, ray queries, deformables): RaiSim **2.4–147×** faster. |
| | [RaiSim vs Newton](./BenchmarkNewton.md) | Newton (MuJoCo Warp on a GPU) | CPU worlds against GPU worlds: **2× or more** physics throughput per dollar at every scale measured, **9.9–254×** with 1–64 worlds. |
| Deformable solids | [Linear elasticity](./LinearElasticityBenchmarks.md) | MFEM, scikit-fem | The same tetrahedral meshes: a cold solve of 41,472 tetrahedra takes 337 ms in RaiSim, 456 ms in MFEM and 5.7 s in scikit-fem. |
| Ray casting | [Voxel ray casting](./BenchmarkVoxelRayCasting.md) | OpenVDB, NanoVDB, Bonxai, OctoMap; on the CPU and the GPU | Exact lidar rays **1.7–34×** faster on one CPU core and **1.3–1.8×** faster than NanoVDB in CUDA on the GPU; a GPU scan of 65,536 rays is **27–34×** faster than on one CPU core. |

## How the measurements are made

- **Same work.** Both sides load the same models and scenes, with the same time step, step count,
  rays or meshes. Where the engines model something differently (contact, deformable models), the
  page says how, and which results are left out.
- **Checked results.** The comparisons check what both sides can compute alike: ray hits and their
  distances, residuals against reference solutions, or finite and settled states. Runs that fail,
  such as a simulation that becomes unstable, are left out of the speedups.
- **CPU timings** use one thread pinned to one core, with the BLAS and OpenMP thread counts set to
  one, except in the comparison of parallel worlds with Newton. **GPU timings** use GPU timer
  queries or CUDA events, and wall-clock time where the CPU waits for the result.
- **Repeats.** Each page states how many runs it took and whether it reports the median or the
  best.

Timings depend on the machine, compiler, driver and scene. To time your own workload with the
binary package, see [Build, Test, and Benchmark](./BuildAndTest.md#timing-examples) and the
[benchmark examples](./examples/benchmark/anymal_standing_benchmark.md). [Performance](./Performance.md)
explains the settings that change RaiSim's speed the most, such as convex collision meshes.
