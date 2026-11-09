# Linear elasticity validation and benchmarks

The new solver is written for RaiSim; MFEM and scikit-fem are used independently
for validation. The scripts and raw measurements are in `benchmark/` and
`benchmark/results/fem/` in the source repository.

## Setup

Measurements used an AMD Ryzen 9 3950X, Linux x86-64, Clang 20 Release (`-O3`),
with one solver thread and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`. Build with 12 jobs. This is a shared development machine,
not an isolated performance lab; repeat the measurements on your target system.

Each mesh is the same 1 × 0.25 × 0.25 m box, split into linear tetrahedra. Supports
allow Poisson contraction while removing rigid modes; end traction is 1 MPa.
The reference sweep includes Poisson ratios 0, 0.3, 0.45 and homogeneous or
layered Young's modulus (70/210 GPa), at 1,536, 12,288 and 41,472 tetrahedra.
MFEM receives exactly the same nodes and tetrahedra. The scikit-fem driver
constructs the same topology and uses SciPy SuperLU. All runs check numerical
agreement or analytical axial extension and independent equilibrium residuals.

Dependencies tested: Gmsh 4.15.2 (fixture generation only), MFEM 4.8 (`a01719101027383954b69af1777dc828bf795d62`),
scikit-fem 12.0.2, SciPy 1.18.1, NumPy 2.5.4. MFEM is built serially, without
MPI, METIS, OpenMP or external AMG; its comparison uses CG with symmetric
Gauss-Seidel at relative tolerance 1e-11. This benchmark does not measure all
MFEM backends or make a claim against AMG, PETSc, parallel or high-order FEM.

## Representative measurements

Values below are milliseconds, homogeneous material at Poisson ratio 0.3.
Cold time includes assembly, solver setup and solve. RaiSim includes stress,
strain, principal stress, von Mises, reactions, energy and residual recovery.
MFEM times include assembly/elimination and the displacement solve, but not
stress recovery. scikit-fem times include fields and residual recovery.

| Tetrahedra | RaiSim cold | MFEM cold | scikit-fem/SuperLU cold |
| ---: | ---: | ---: | ---: |
| 1,536 | 7.7 | 7.8 | 59.0 |
| 12,288 | 89.6 | 83.8 | 1050.1 |
| 41,472 | 336.7 | 455.6 | 5738.8 |

RaiSim uses LDLT on the smallest mesh and ICCG on the other two. With 41,472
tetrahedra its stored lower matrix has 520,782 entries; ICCG avoids the millions
of entries in an exact factor. The original direct-only prototype spent seconds
factorizing that mesh, which motivated the iterative backend.

| Tetrahedra | Proportional edit | Direction edit before review | Direction edit now | Density-only edit |
| ---: | ---: | ---: | ---: | ---: |
| 1,536 | 0.47 | 0.54 | 0.63 | 0.75 |
| 12,288 | 3.07 | 54.47 | 10.09 | 4.69 |
| 41,472 | 11.04 | 268.29 | 49.47 | 14.75 |

The response cache retains up to eight energy-orthonormal load directions.
The direction sweep contains ten sinusoidally varied combinations of three
independent tractions: the first new directions require iterations, then later
combinations reuse their responses. It does not imply that arbitrary new loads
solve without iterations. The final direction edit in these runs needed zero
CG iterations. New directions and later combinations have separate columns in
the raw CSV; their averages are:

| Tetrahedra | First two new directions | Later cached combinations |
| ---: | ---: | ---: |
| 1,536 | 0.63 | 0.63 |
| 12,288 | 33.07 | 4.34 |
| 41,472 | 184.12 | 15.81 |

Density edits change gravity loads and leave stiffness unchanged;
the benchmark checks that no assembly or factorization is repeated.

The ordering experiment compared Eigen IC with AMD and natural ordering.
Natural ordering reduced setup and solve costs on this structured mesh sweep.
An initial shift of 0.01 reduced measured IC setup time at high Poisson ratio;
the final code uses natural ordering and that initial shift. Raw before/after
and tuning measurements are retained alongside the final results. Lower-triangle
sparse storage is retained for memory efficiency. A faster analytical principal
stress eigenvalue routine was rejected after near-repeated eigenvalues lost
accuracy; recovery still uses the robust eigensolver.

MFEM's identical changing-nodal-load sweep, in milliseconds:

| Tetrahedra | MFEM scaled warm solve | MFEM direction change | RaiSim direction change |
| ---: | ---: | ---: | ---: |
| 1,536 | 0.07 | 2.21 | 0.53 |
| 12,288 | 0.51 | 29.89 | 9.68 |
| 41,472 | 1.82 | 156.79 | 44.62 |

MFEM warm solve times omit stress recovery. Its changing-load sweep uses an
absolute residual tolerance of 1e-13 times the RHS norm to resolve small
transverse loads on top of the larger axial load. RaiSim includes all field
recovery. The scikit-fem and RaiSim transverse sweeps have different small load
amplitudes, so their changing-load timings are not identical-workload speed
ratios. Direct solvers can be faster for many unrelated right-hand sides,
despite higher setup cost; use `SparseLDLT` explicitly for that workload.


## Imported mesh ordering check

A second fixture is a STEP solid with a cylindrical bore. Gmsh 4.15.2 imports
the STEP geometry and generates 11,002 unstructured tetrahedra with 2,814 nodes
and 8,094 free DOFs. The benchmark preserves Gmsh's node numbering, clamps the
minimum-X end, and applies the same axial/transverse traction at the maximum-X
end. Each row starts from a fresh solver. Setup plus solve/recovery, in ms:

| Poisson ratio | AMD, shift 0.01 | Natural, shift 0.001 | Natural, shift 0.01 |
| ---: | ---: | ---: | ---: |
| 0.3 | 136.8 | 101.6 | 100.2 |
| 0.45 | 281.7 | 249.2 | 224.8 |
| 0.49 | 469.1 | 431.3 | 402.4 |

All cases converged; the largest load-relative force residual was below 2e-11.
These results support keeping natural ordering and shift 0.01 across this
structured/unstructured sample. They do not establish universal optimality.
Node ordering, geometry, material contrast, and host load can change the result;
repeat on representative production meshes. The initial shift is used only if
the unshifted IC attempt fails; it changes the preconditioner, not the physical
stiffness. Raw CSVs also retain iteration counts and separate setup/solve costs.

## Bending convergence

`linear_fem_bending` measures a 1 × 0.1 × 0.1 m clamped cantilever with E=70 GPa
and a uniformly distributed 1 kN transverse end load. It compares area-averaged
tip displacement with Timoshenko beam theory, including
[Cowper's rectangular shear correction](https://pmc.ncbi.nlm.nih.gov/articles/PMC6030651/)
`10(1+nu)/(12+11nu)`. Each longitudinal cell is approximately
as long as a cross-section cell is wide (20/40/80 cells along the length).
Beam theory is an engineering reference; it does not resolve the clamp and end
load boundary layers of three-dimensional elasticity.

| Elements through thickness | Deflection error, nu=0.3 | Deflection error, nu=0.49 |
| ---: | ---: | ---: |
| 2 | 49.6% | 89.6% |
| 4 | 21.3% | 71.2% |
| 8 | 7.0% | 41.9% |

All six cases underpredict deflection. Refinement improves both materials, but
linear tetrahedra remain much too stiff near incompressibility. Agreement with
another T4 solver establishes implementation consistency, not continuum
accuracy. Check mesh convergence for bending and local stress peaks. The
workbench warns above Poisson ratio 0.45. Quadratic elements and mixed
incompressible formulations require additional development.

## Reproduce

Build `linear_fem_benchmark` and run it with no argument for all three sizes,
or pass one subdivision count (4, 8, 12). The MFEM comparison is optional:

```bash
cmake -S . -B /tmp/fem-build -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_BUILD_TYPE=Release -DRAISIM_BENCHMARK=ON -DRAISIM_FEM_MFEM_BENCHMARK=ON \
  -DMFEM_DIR=/path/to/mfem-build
cmake --build /tmp/fem-build --target linear_fem_benchmark linear_fem_bending linear_fem_mesh_benchmark linear_fem_mfem -j12
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/fem-build/benchmark/linear_fem_mfem 12
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/fem-build/benchmark/linear_fem_bending
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 benchmark/linear_fem_scikit.py --sizes 4 8 12
# Gmsh's Python module is needed only to generate this benchmark fixture.
python3 benchmark/linear_fem_cad_fixture.py /tmp/fem-cad
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/fem-build/benchmark/linear_fem_mesh_benchmark /tmp/fem-cad/perforated.msh
```

## Correctness checks

Unit tests cover affine patches, uniaxial extension, engineering shear,
hydrostatic pressure, reaction forces and moments, gravity, heterogeneous
materials, exact prescribed and multipoint constraints, cache changes and failed
solve recovery, invalid/degenerate/unsupported meshes and mechanisms, import
formats, overflow and locale independence. Fixture tests include axial,
torsional and bending bolt response, nominal preload, connector rigid-motion
invariance, bearing center restraints and free roll, remote motion, elastic
foundations, nonmatching ties and RFEM round trips.

Fifteen independent scikit-fem/SciPy comparisons exercise distorted and rotated
meshes, 1e-3/1e3 geometry scaling, negative/positive Poisson ratios, heterogeneous
materials, prescribed motion, body and surface loads, and KKT reference solutions
for multipoint constraints. They compare displacement, reaction, engineering
strain, stress, principal stresses, von Mises and total strain energy.
The MFEM sweep checks 18 additional material/mesh combinations; initial
relative displacement differences were below 3e-12.

Regression tests also cover large unloaded/loaded slider mechanisms, edge and
vertex hinges, weak or incomplete elastic supports, nonzero spring-rest motion,
axis-equation fast paths, skew ICCG with gravity, independent response recycling,
density-only cache edits, bolt stiffness changes and weighted nonrigid patches,
sharp roller corners, connected/smooth picking, exact restored regions and legacy
project compatibility.

The desktop and real Chrome/WebGPU tests additionally exercise file drops,
project loading and solving, tetrahedral volume import, surface meshing, STEP
translation, invalid-file errors and rendering. Browser workers are isolated
from the rendering thread and can be cancelled.

## Native meshing

### Deferred meshing and enforced resolution

The current workbench displays a surface first and tetrahedralizes on **Solve**.
Native and browser builds share the source-built serial kernel. **Max element edge**
is enforced by conforming refinement, independently of surface approximation.
Automatic surface error is `min(0.01 * edge, 0.001 * geometry_diagonal)`;
CAD tessellation receives one quarter of that budget. This replaces the earlier
fixed envelope tolerance and ideal-size hint, so the resulting meshes differ
from the historical measurements below.

Three-run serial medians on the same shared Ryzen 9 3950X host, Clang 20 Release,
with the benchmark pinned to one logical CPU. Times include helper startup,
translation/meshing and output; quality calculations are outside the timed interval.
Raw data: `benchmark/results/fem/native-mesher-resolution.csv`.

| Geometry | Maximum edge (mm) | Preview (ms) | Optimized (s) | Original sampler (s) | Speedup | Tetrahedra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-part assembly | 20 | 42.5 | 0.978 | 2.707 | 2.77× | 1,517 |
| Two-part assembly | 10 | 38.0 | 0.472 | 1.989 | 4.22× | 3,075 |
| Two-part assembly | 5 | 45.6 | 1.786 | 8.804 | 4.93× | 26,970 |
| Hollow cylinder | 15 | 22.3 | 11.706 | 12.302 | 1.05× | 10,776 |
| Sphere | 15 | 47.3 | 10.492 | 9.644 | 0.92× | 14,306 |

Every optimized/original-sampler pair produced byte-identical deterministic meshes
in all three runs. All native maximum-edge ratios were at most one (within
roundoff). Volume error was at most 0.00234% for the assembly, 0.102% for the hollow
cylinder and 0.272% for the sphere. Minimum mean ratios were 0.342–0.496 for the
assembly, 0.293 for the cylinder and 0.360 for the sphere.

The planar certificate helps the assembly substantially. It gives only a small
cylinder gain, and the sphere's certificate attempts add about 9% overhead in
this run. Curved meshing still takes around 10–12 seconds on these cases; opening
them requires only the 22–47 ms surface preparation, followed by display rendering.
Surface preview always contains zero tetrahedra.

Raw Gmsh times were 35–117 ms for the assembly, 40 ms for the cylinder and 32 ms
for the sphere. Its actual maximum edges were 1.41–2.24 times the requested size,
and the sphere's volume error was 2.93%, marked `volume_error`. These are different
meshes and geometric accuracies. The CSV reports actual edge lengths, counts and
quality; these raw Gmsh timings exclude the workbench's maximum-edge refinement.

Provided volume meshes use the independent refiner, retaining original vertices
and region interfaces. Refining a 1 m cube from six supplied tetrahedra to maximum
edges of 0.2, 0.1 and 0.05 m took 20.4, 50.0 and 291.1 ms, producing 6,144, 49,152
and 393,216 tetrahedra. Volume was preserved to reporting precision and minimum
mean ratio was 0.762 in all cases.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 benchmark/fem_mesher_benchmark.py --mesher /path/to/raisim_fem_mesher \
  --geometry all --preview --compare-sampling --repetitions 3 \
  --gmsh /optional/path/to/gmsh
```

### Historical fixed-tolerance configuration

Measurements on 2026-11-09: AMD Ryzen 9 3950X, Linux x86-64, Clang 20.1.8 Release,
fTetWild at the pinned browser revision, Open CASCADE 7.7.2 and Gmsh 4.15.2.
Each completed result is the median of three serial runs with one computational
thread. Timings include process startup, CAD translation, tetrahedralization and
file output. Python quality calculations are outside the timed interval. The host
is shared. Raw results are in `benchmark/results/fem/native-mesher.csv` in the
RaiSim source checkout.

| Geometry | Edge (mm) | Optimized (s) | Original sampler (s) | Speedup | Gmsh (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two-part assembly | 20 | 1.453 | 47.477 | 32.68× | 0.034 |
| Two-part assembly | 10 | 1.229 | 50.346 | 40.97× | 0.047 |
| Two-part assembly | 5 | 1.548 | 62.849 | 40.60× | 0.115 |
| Hollow cylinder | 15 | 95.417 | 147.172 | 1.54× | 0.040 |
| Sphere | 15 | 19.862 | 24.444 | 1.23× | 0.031 |

All optimized native meshes were byte-identical to the original sampler outputs
across all three repetitions. The benchmark fixes the surface and Delaunay input
order with `--deterministic`; ordinary imports retain randomized ordering and can
have different timings. The acceleration preserves the original tolerances and
energy threshold of that earlier configuration. These measurements predate the
shared source-built browser kernel and strict maximum-edge enforcement.

Equal requested edge lengths produce different meshes across backends. Native
volume errors were at most 0.000056% for the assembly, 0.110% for the hollow
cylinder and 0.291% for the sphere. The sphere has 12,309 native tetrahedra and
865 Gmsh tetrahedra; Gmsh has 2.926% volume error there. Its CSV status is
`volume_error`, which flags a completed result exceeding the 1% benchmark target.
Those timings therefore represent different geometric accuracy. The raw CSV also
records node counts, minimum tetrahedron mean ratio and the fraction of envelope
queries certified directly. Curved imports remain expensive: the hollow cylinder
improves by 1.54× and the sphere by 1.23×, compared with 33–41× on the flat assembly.

Reproduce from the RaiSim source checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 benchmark/fem_mesher_benchmark.py --mesher /path/to/raisim_fem_mesher \
  --compare-sampling --geometry all --repetitions 3 --gmsh /optional/path/to/gmsh
```

The benchmark reports timeouts explicitly and retains accuracy-target misses.
The native geometry regression suite independently checks stricter volume limits,
positive tetrahedra, CAD units, separate solid regions and preservation of holes.
Benchmark reporting tests cover accuracy misses, partial timeouts, empty output
and changes to deterministic mesh output.
