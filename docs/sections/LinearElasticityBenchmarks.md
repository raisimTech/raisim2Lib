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
