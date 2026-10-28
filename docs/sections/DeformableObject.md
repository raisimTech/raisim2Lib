# Deformable Objects

`raisim::DeformableObject` is RaiSim's XPBD/PBD deformable-body object.
It is intended for fast simulation and data-generation workloads that need
cloth, soft shells, or soft solids without leaving RaiSim's single-threaded
stepping model.

The object is a set of particles connected by elastic distance constraints
(the mesh edges plus optional extra edges) and, optionally, by bending
constraints between adjacent triangles. With a Young's modulus, a triangle mesh
is instead a continuous elastic membrane and a tetrahedral mesh a continuous
elastic solid (see [Elastic Modulus](#elastic-modulus) and [Elastic Solids](#elastic-solids)). Each particle
collides as a sphere. A particle that diverged (a NaN or infinite position or
velocity) makes no contact and does not affect the contacts of the others, self
contacts included.
Collision is handled by one aggregate deformable object in the broad phase, with
the particle spheres used internally for contact generation. This keeps
deformable objects cheap to register in the world while still allowing
particle-level contacts against the ground, rigid bodies, articulated systems,
and other deformable objects. Contacts are resolved together with the elastic
forces in every substep (see [Contacts](#contacts)), so a body resting on a cloth is
carried by the cloth's tension, and the elastic response cannot pull particles
through a body. The
`deformable_objects` example ([Server Example: Deformable Objects](./examples/server/deformable_objects.md))
drapes a cloth over a sphere and stacks mesh-based soft cubes.

## API Surface

The public constructors are exposed through `raisim::World`:

- `World::addDeformableCloth(vertices, triangles, pinnedVertices, material, contactMaterial, collisionGroup, collisionMask)`
  creates a deformable cloth or shell directly from world-space particle
  positions and triangle indices.
- `World::addDeformableCloth(meshFileInObjFormat, scale, pinnedVertices, material, contactMaterial, collisionGroup, collisionMask)`
  loads an OBJ mesh as a surface cloth/shell and uses its vertices directly as
  particles.
- `World::addDeformableObject(meshFileInObjFormat, material, options, pinnedVertices, contactMaterial, collisionGroup, collisionMask)`
  loads an OBJ mesh with `MeshBuildOptions` for resampled surface particles,
  filled particles, and optional internal struts.
- `World::addDeformableVolume(vertices, tetrahedra, pinnedVertices, material, contactMaterial, collisionGroup, collisionMask)`
  creates a solid from world-space particle positions and tetrahedron indices
  (see [Elastic Solids](#elastic-solids)).

Trailing arguments have defaults: scale `1`, no pinned vertices, a
default-constructed `Material`, contact material `"default"`, collision
group `1`, and a collision mask that accepts every group.
`addDeformableObject` always requires its material and build options. All
creation paths register one `DeformableObject` in the world.
Particle-level operations use local particle indices. Pinned particles report
`BodyType::STATIC` through `getBodyType(localIdx)`; all other particles
report `BodyType::DYNAMIC`.

## Data Model

The simulated state is particle based:

- `getNumParticles()` returns the number of particles.
- `getPositions()` and `getVelocities()` return world-space particle
  positions and velocities.
- `getMass(idx)` returns a particle's mass (infinite for a pinned particle).
  `totalMass` is lumped onto the particles by area: each particle carries one
  third of the area of every triangle it belongs to, so a densely sampled
  region does not become heavier than a coarse one. A particle on no triangle
  receives the average share; an object without triangles splits the mass
  evenly; a filled mesh or a tetrahedral volume lumps it by volume.
- `getTriangles()` returns the visual/surface triangle topology.
- `getTetrahedra()` returns the tetrahedra of a volume (empty for a surface).
- `getVisualPosition(idx)` returns the particle position plus its visual
  offset. For a closed triangle mesh, construction moves each surface particle
  inward so that its sphere lies exactly `collisionRadius` behind every face
  it belongs to, including along edges and at corners; the collision envelope
  then matches the input surface, and the visual offset undoes the shift. For an
  open mesh the offset is zero.
- `getDistanceConstraintCount()` and `getBendingConstraintCount()` expose
  the generated constraint counts for diagnostics.
- `getCollisionBodyCount()` is normally `1` because one aggregate collision
  body represents the deformable object in the broad phase.
- `setPositionOffset(offset)` translates all particles, including the
  positions that pinned particles are held at.
  `applyRigidTransform(translation, rotation)` rotates all particles about the
  world origin and then translates them; velocities and pending forces are
  rotated as well.

The collision proxy radius is not the visual mesh thickness. It controls the
spherical particle contacts used by the solver. If visual geometry appears to
touch but contacts are weak or delayed, check `collisionRadius` and particle
spacing before increasing stiffness.

## Creating A Deformable Object

### Explicit vertices

Use `World::addDeformableCloth` when you already have particle positions and
surface triangle indices. The vertices are in world coordinates.

```cpp
std::vector<raisim::Vec<3>> vertices = {
    {-0.5, -0.5, 1.0},
    { 0.5, -0.5, 1.0},
    { 0.5,  0.5, 1.0},
    {-0.5,  0.5, 1.0}};

std::vector<raisim::DeformableObject::Triangle> triangles = {
    {0, 1, 2},
    {0, 2, 3}};

raisim::DeformableObject::Material material;
material.totalMass = 1.0;
material.distanceCompliance = 1.0e-5;
material.bendCompliance = 1.0e4;   // bending modulus 1e-4 N m, a light fabric
material.damping = 0.001;          // s
material.airDrag = 0.5;            // 1/s
material.collisionRadius = 0.01;
material.iterations = 6;

auto* cloth = world.addDeformableCloth(vertices, triangles, {}, material);
```

### Mesh surface particles

Use `World::addDeformableObject` with `MeshParticleOptions::Mode::Surface`
to resample an OBJ surface. Every triangle is subdivided so that none of its
edges is longer than `spacing` (default 0.05 m), and the resulting vertices
become the particles; triangles that are already finer keep their original
vertices. To use the OBJ vertices directly without resampling, call
`World::addDeformableCloth(meshFile, scale, ...)` instead.

```cpp
raisim::DeformableObject::MeshBuildOptions build;
build.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Surface;
build.particles.scale = 1.0;

auto* shell = world.addDeformableObject("soft_shell.obj", material, build);
shell->setPositionOffset({0.0, 0.0, 1.0});
```

OBJ polygon faces are triangulated as a fan. Texture and normal indices are
ignored. This mode is suitable for cloth and shell meshes.

`addDeformableObject` raises `Material::collisionRadius` to at least
`0.58 * spacing` in both Surface and Filled modes so that neighboring
particle spheres overlap. Pinned-vertex indices passed to
`addDeformableObject` refer to the generated particle list, not to the
original OBJ vertex numbering.

### Filled mesh particles

Use `MeshParticleOptions::Mode::Filled` for a closed OBJ triangle mesh that
should receive interior particles. RaiSim resamples the surface as in Surface
mode and fills the inside. With a positive `youngsModulus` the object becomes
an elastic solid: interior particles are placed on a body-centered cubic
lattice with the particle density of a `spacing` grid, and all particles are
cut into tetrahedra that fill the volume (see [Elastic Solids](#elastic-solids)).

Without `youngsModulus`, the interior particles lie on a regular grid with the
same `spacing`, and every interior particle is connected by distance
constraints to all particles within $\sqrt{3}$ spacings: its lattice
neighbors, including the face, edge, and corner diagonals, and the surface
particles around it. The mass is lumped by volume, one lattice cell per interior
particle and a half-cell layer under the surface share of each surface particle.
Such a lattice of springs has no well-defined elastic modulus and no Poisson
effect; use an elastic solid when the material response matters.

```cpp
raisim::DeformableObject::MeshBuildOptions build;
build.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Filled;
build.particles.scale = 1.0;
build.particles.spacing = 0.05;
build.particles.maxFillParticles = 50000;

auto* softBody = world.addDeformableObject("closed_cube.obj", material, build);
softBody->setPositionOffset({0.0, 0.0, 1.0});
```

Filled mode requires a closed (watertight) triangle mesh with a meaningful
inside/outside volume: interior particles are selected by inside/outside tests
against that surface, which are not meaningful for an open mesh.

When using filled mode, tune `spacing` and `maxFillParticles` (default
50000\) together. Smaller spacing increases particle count and usually improves
shape support, but it also increases contact and constraint cost.
`maxFillParticles` limits the number of interior particles; if the requested
spacing would exceed it, use a larger spacing or a simpler mesh rather than
raising the limit for an unexpectedly large model.

### Internal struts

Internal struts are additional distance constraints used to preserve a rest
shape and provide bounce/recovery for soft shells or filled objects.
`PairsWithinRadius` connects every particle pair whose rest distance is at
most `radius`; pairs that already share a constraint are skipped. A large
radius creates many constraints that each cost solver time.

```cpp
raisim::DeformableObject::MeshBuildOptions build;
build.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Filled;
build.particles.spacing = 0.05;
build.internalStruts.mode =
    raisim::DeformableObject::InternalStrutOptions::Mode::PairsWithinRadius;
build.internalStruts.radius = 0.09;

auto* softBody = world.addDeformableObject("closed_cube.obj", material, build);
```

You can also add constraints after construction:

```cpp
softBody->addDistanceConstraint(0, 7);
softBody->addDistanceConstraints({{0, 6}, {1, 7}});
softBody->addInternalStruts(build.internalStruts);
```

Manual constraints are useful for adding diagonal support to coarse shells.
Their rest length is the current distance between the two particles, so add them
immediately after construction or after intentionally placing the object in its
desired rest pose. `addDistanceConstraints` skips pairs that already have a
constraint; `addDistanceConstraint` does not check for duplicates.

## Material Parameters

`DeformableObject::Material` controls mass, solver behavior, collision proxy
size, and elastic response. Defaults are shown in parentheses:

- `totalMass` (1 kg): total mass, lumped onto the particles by area (see
  [Data Model](#data-model)).
- `distanceCompliance` (-1): XPBD distance compliance. `0` is rigid; larger
  values are softer. Negative values preserve legacy behavior by using
  `1 / distanceStiffness`, so the default compliance is `1e-4`.
- `distanceStiffness` (1e4): legacy stiffness parameter. Prefer
  `distanceCompliance` for new code.
- `bendCompliance` (-1): inverse of the bending modulus $D$, in
  1/(N m) (see [Bending](#bending)). `0` is rigid; larger values fold more easily.
  Negative values use `1 / bendStiffness` when `bendStiffness` is positive
  and otherwise disable bending, so bending is off by default.
- `bendStiffness` (0): the bending modulus $D$ in N m, used when
  `bendCompliance` is negative.
- `youngsModulus` (-1, unused): Young's modulus $E$ in Pa. If positive,
  the triangles form an elastic membrane (see [Elastic Modulus](#elastic-modulus)), or the
  tetrahedra of a volume an elastic solid (see [Elastic Solids](#elastic-solids)), and
  `distanceCompliance` no longer applies to the mesh edges.
- `poissonRatio` (0.3): Poisson's ratio $\nu$ of the membrane or solid,
  between -1 and 0.5.
- `thickness` (0.01 m): membrane thickness $t$.
- `crossSectionArea` (-1, unused): cross-sectional area of the extra distance
  constraints (struts and manual constraints) when `youngsModulus` is
  positive; `thickness * L` when unset.
- `damping` (0 s): stiffness-proportional damping time (see [Damping](#damping)).
- `airDrag` (0 1/s): air-drag rate, which damps the absolute particle
  velocities (see [Damping](#damping)).
- `selfCollision` (false): collide the object's particles with each other
  (see [Contacts](#contacts)).
- `collisionRadius` (0.01 m): radius of each internal particle collision
  proxy.
- `iterations` (6): number of constraint projection iterations per substep.
  Within each iteration, a run of distance constraints that continue each other
  in a nearly straight line (a row, column or diagonal of a grid cloth, a rope;
  each edge turns by less than 60° from the previous one) is solved as a whole,
  so a stiff cloth or rope holds its rest lengths and a pull reaches its far end
  within one step: a 0.2 kg cloth of 33 × 33 particles on the ground pulled at
  one edge with 20 N stretches by less than 0.01 % at the default 6 iterations.
  Where the edges of a mesh turn more sharply, the stiffness spreads by one edge
  per iteration, and such a mesh needs more iterations for the same stretch.
- `substeps` (1): number of deformable substeps per RaiSim world step.
  Deformables whose particles touch step together with the largest value
  among them.
- `solverMode` (`XPBD`): `XPBD` or `PBD`. `PBD` projects every
  constraint rigidly and ignores the compliance magnitudes, so its effective
  stiffness depends on `iterations` and the time step. Whether bending
  constraints exist is still decided by `bendCompliance`/`bendStiffness`.

## Elastic Modulus

For engineering-style material input, set `youngsModulus` instead of
`distanceCompliance`:

```cpp
raisim::DeformableObject::Material material;
material.totalMass = 1.0;
material.youngsModulus = 5.0e4;
material.poissonRatio = 0.3;
material.thickness = 0.02;
material.collisionRadius = 0.01;
```

When `youngsModulus > 0`, each triangle of the mesh is an element of a thin,
isotropic elastic sheet in plane stress. Its in-plane strain
$\boldsymbol{\varepsilon} = (\varepsilon_{xx}, \varepsilon_{yy}, \gamma_{xy})$
is measured after removing the triangle's rotation, so rigid motion does not
strain it, and it stores the energy

$$
\begin{aligned}
E_m = \frac{A t}{2} \boldsymbol{\varepsilon}^\top \boldsymbol{C} \boldsymbol{\varepsilon},
\qquad
\boldsymbol{C} = \frac{E}{1 - \nu^2}
\begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1 - \nu}{2} \end{bmatrix},
\end{aligned}
$$

where $A$ is the rest area of the triangle. A strip of width $W$
pulled with a force $F$ along its length therefore stretches by the strain
$F / (E t W)$ and narrows by $\nu$ times that, independently of
how the mesh is triangulated. The mesh edges are not separate springs in this
mode. Extra distance constraints (struts, `addDistanceConstraint`) remain
springs of compliance $L / (E A_s)$, with the rest length $L$ and
the area $A_s$ = `crossSectionArea`, or `thickness * L` when it is
unset. With the default area the compliance is $1 / (E t)$ for every
length, including a weld of coincident particles ($L = 0$, a mesh seam);
with `crossSectionArea` set, such a weld is rigid.

A membrane that is stiff for its particle masses spreads a load over the whole
sheet, which sweeps over the triangles reach one triangle at a time: a hanging
sheet of 64 × 64 particles, 0.2 kg/m², $E t$ = 1000 N/m stretched 2.8 times as
far as plane stress predicts at the default 6 iterations, and 23 times as far
with $E t$ = 10⁴ N/m. When the inverse mass weight of a triangle's strains
exceeds 4 times its softest compliance $/\,h^2$, the triangles are solved
together in every iteration by conjugate gradients on their multipliers,
preconditioned with a sparse Cholesky factorization of a membrane whose
stiffness bounds that of every triangle in every orientation (computed again
only when the substep length or a mass changes). The number of conjugate
gradient iterations grows with the square root of that stiffness ratio, from 2
to at most 8. Both sheets above then stretch within 3 % of plane stress at 6
iterations (the pinned edge keeps the sheet from narrowing, which stiffens it a
little), at 6.7 and 15.8 ms per step instead of 0.7 ms; the sweeps needed 200
iterations (21 ms per step) to come as close for the softer sheet and were still
17 % off for the stiffer one. The damping of a
membrane is still swept: its compliance, scaled by the damping time, is softer,
and the sweeps converge for it.

Without `youngsModulus`, the mesh edges are springs with
`distanceCompliance`. Such a spring network has no Poisson effect, and its
stiffness depends on the triangulation.

## Elastic Solids

An object made of tetrahedra is an elastic solid when `youngsModulus` is
positive. Such an object comes from a filled mesh (see
[Filled mesh particles](#filled-mesh-particles)) or from a tetrahedral mesh of your own, for example
one generated by a meshing tool:

```cpp
std::vector<raisim::Vec<3>> vertices = {
    {0.0, 0.0, 0.5}, {0.2, 0.0, 0.5}, {0.0, 0.2, 0.5}, {0.0, 0.0, 0.7}};
std::vector<raisim::DeformableObject::Tetrahedron> tetrahedra = {{0, 1, 2, 3}};

raisim::DeformableObject::Material material;
material.totalMass = 0.5;
material.youngsModulus = 2.0e5;   // a soft rubber
material.poissonRatio = 0.45;
material.collisionRadius = 0.005;

auto* solid = world.addDeformableVolume(vertices, tetrahedra, {}, material);
```

Each tetrahedron is an element of an isotropic elastic solid. Its strain
$\boldsymbol{\varepsilon} = (\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \gamma_{xy}, \gamma_{yz}, \gamma_{zx})$
is measured after removing the element's rotation, so rigid motion does not
strain it, and it stores the energy

$$
\begin{aligned}
E_s = \frac{V}{2} \boldsymbol{\varepsilon}^\top \boldsymbol{C} \boldsymbol{\varepsilon},
\qquad
\boldsymbol{C} =
\begin{bmatrix}
\lambda + 2\mu & \lambda & \lambda & & & \\
\lambda & \lambda + 2\mu & \lambda & & & \\
\lambda & \lambda & \lambda + 2\mu & & & \\
& & & \mu & & \\
& & & & \mu & \\
& & & & & \mu
\end{bmatrix},
\end{aligned}
$$

where $V$ is the rest volume of the element and
$\mu = E / (2 (1 + \nu))$ and $\lambda = E \nu / ((1 + \nu)(1 - 2\nu))$
are the Lamé parameters. A bar of cross section $A$ pulled with a force
$F$ therefore stretches by the strain $F / (E A)$ and narrows by
$\nu$ times that, however its volume is cut into tetrahedra; close to
$\nu = 0.5$ the solid is nearly incompressible.

<figure>
<img src="../../rsc/docs/image/rayrai/deformable_solid_poisson.png" alt="two soft cubes squeezed by plates, one bulging sideways">
<figcaption>

Two filled cubes of soft rubber ($E$ = 20 kPa) under the same 15 kg
plate; the wire frames show their rest shape. With $\nu = 0$ (left) the
cube only shortens. With $\nu = 0.45$ (right) it bulges sideways as it
shortens, keeping its volume.

</figcaption>
</figure>

A spinning solid keeps its shape, and a solid crushed until some elements turn
inside out recovers its rest shape once released. Each particle carries a
quarter of the volume of every tetrahedron it belongs to.

The solid has the shape of its mesh. Its surface particles sit
`collisionRadius` inside the surface, like those of any closed mesh, so that
their collision spheres reproduce it, but the elastic body is the full volume of
the mesh: a cube of side $L$ resting on the ground shortens under its own
weight by $\rho g L^2 / (2E)$, whatever its particle spacing and collision
radius. Volume changes are resisted over the neighborhood of every particle
rather than tetrahedron by tetrahedron, so a nearly incompressible solid still
bends: a rubber beam with $\nu = 0.49$ deflects about as far as one with
$\nu = 0.3$. A coarse mesh of tetrahedra is still stiffer in bending than
the material it approximates; with two elements across a beam it deflects about
two thirds of the beam-theory value, with four about nine tenths, so use several
elements across thin parts. The faces that belong to a single tetrahedron are the
surface that `getTriangles()` returns for a volume created with
`addDeformableVolume`. Without `youngsModulus`, the edges of the tetrahedra
are distance constraints with `distanceCompliance`.

## Bending

With bending enabled, every pair of triangles sharing an edge resists the change
of its curvature from the rest shape. Summed over the surface, the bending
energy approximates that of a thin plate,

$$
E_b = \frac{D}{2} \int (2H - 2H_0)^2 \, dA,
$$

where $H$ is the mean curvature, $H_0$ its rest value, and
$D$ the bending modulus set by `bendCompliance` $= 1/D$. The
energy is quadratic in the curvature change, so the bending moment is linear in
it: a cantilevered strip under eight times the load sags eight times as far
while the deflection stays small. For an isotropic sheet of thickness
$t$,

$$
D = \frac{E t^3}{12 (1 - \nu^2)}.
$$

Woven cloth bends far more easily than this formula predicts from its in-plane
stiffness, because its fibers slide against each other, so cloth is usually
given a much smaller $D$. The rest curvature $H_0$ comes from the
mesh as it is created, so closed shells and curved panels keep their shape;
only a flat rest mesh is flat at rest.

A bending modulus that is large for the particle masses, as for paper, plastic
film, cardboard or sheet metal, couples the whole surface: a sweep of the
iterations over the hinges carries a bending moment only one hinge further, so
such a sheet sagged like cloth. When the inverse mass weight of a hinge's
curvature exceeds 4 times its compliance $/\,h^2$ for the substep length $h$,
all hinges are solved together in every iteration, through a sparse Cholesky
factorization of their stiffness that is computed again only when the substep
length or a mass changes (pinning or unpinning a particle). For a flat rest
shape the hinges are linear in the positions and one solve settles them. The
rest curvature of a curved hinge turns with the hinge; its exact derivative
enters the solve, which then takes two conjugate gradient iterations
preconditioned with the same factorization, and its forces carry no spurious
torque. Closed shells, curved panels and sheets whose mesh file rounded their
coordinates out of the plane are solved this way.

A 0.3 m strip clamped at one end then sags 1.2 to 1.3 times as far as beam
theory with 2 iterations as with the default 6, where the sweeps let it sag 16
to 19 times as far; the remaining difference is the discretization of the
bending, which shrinks as the mesh is refined. A stiff half cylinder clamped
along its edges sagged 6 to 8 times its converged deflection at 6 iterations
and now settles at it with 3. This costs about three times as much per step
for a 64 × 64 flat sheet, and about four times as much per iteration for a
curved one. In this solve a hinge is at most $10^{10}$ times stiffer than that
inverse mass weight, beyond which double precision loses the particle masses;
such a hinge is rigid for every practical purpose. Fabrics stay well below the
threshold and keep the sweeps.

## Damping

Two damping terms are available. Together they form Rayleigh damping:

- `airDrag` $c$ (1/s) acts on the absolute velocity of every particle,
  like a linear air drag. A free particle under gravity approaches the terminal
  speed $\|\boldsymbol{g}\| / c$, and the decay
  $e^{-c t}$ is independent of `Δt` and `substeps`. It damps rigid
  motion too, for example a cloth swinging from its pins.
- `damping` $\beta$ (s) acts on the deformation only. Every distance
  and bending constraint, and every strain of a membrane or solid element,
  resists its rate of change with a viscous force equal
  to $\beta$ times its elastic stiffness, so a vibration mode of angular
  frequency $\omega$ decays with the damping ratio
  $\zeta = \beta \omega / 2$: stiff, fast modes are damped strongly and
  slow, large-scale motion only weakly. Translation and rotation of the whole
  object are not damped. A rigid constraint (zero compliance) loses its whole
  rate of change. `damping` has no effect in the `PBD` solver mode.

Both default to zero; the remaining dissipation comes from the implicit time
integration and from contact friction.

::: info Note
Earlier versions interpreted `damping` as the fraction of every particle's
velocity removed per world step, which slowed rigid motion as well and
changed with the time step (it capped free fall at about 0.5 m/s at
`Δt = 1` ms with the old default of 0.02). To reproduce a former value
$d$ used at time step $\Delta t$, set
`airDrag` $= -\ln(1 - d) / \Delta t$ and `damping` to zero.
:::

## Contacts

A deformable particle collides as a sphere of radius `collisionRadius`.
Contacts with rigid bodies, articulated systems, and terrain are resolved in
every substep together with the elastic constraints, as one-sided contacts
between the particle and the body surface:

- The contact force acts on the body as well, through its apparent inertia at
  the contact point, so momentum is exchanged in both directions and a body
  resting on a cloth is carried by the cloth's tension.
- The elastic response cannot push or pull a particle through a body.
- Contacts are inelastic and use Coulomb friction with the friction coefficient
  of the contact materials.
- With `selfCollision` enabled, the particles of one object collide with each
  other too, so that a folded cloth rests on itself. Particle pairs that already
  overlap in the rest shape (neighbors of a coarse surface) are left to the
  elastic constraints. The friction coefficient is that of the object's contact
  material with itself.
- Contacts that a particle can reach within the step are created in advance, so
  a fast particle stops at the surface in the step it arrives instead of
  penetrating. These contacts report a positive `Contact::getDepth()` (a gap)
  and zero impulse until the particle touches the body. A pinned particle only
  takes part in contacts once it touches.

Contacts between particles of different deformable objects are resolved the
same way, in a substep loop that the touching objects share, so stacked soft
bodies support each other. Touching deformables step with the largest
`substeps` among them.

`Contact::getImpulse()` reports the full impulse exchanged in the step, on
both sides of the contact. Contacts between particles of the same object
(`selfCollision`) are not listed in `getContacts()`.

A heavy object on a light cloth, such as a 2 kg ball on a 0.2 kg cloth, needs
enough `iterations` and `substeps` for the cloth to carry the load: with too
few, the cloth stretches far beyond its compliance under the load, and a stretch
large enough to open the gaps between the particle spheres lets the object
through.

### Solving deformables together with the bodies

By default the elastic constraints of the deformables are solved together with
every contact, joint limit and pin of the world
(`World::setMonolithicDeformables(true)`). With
`World::setMonolithicDeformables(false)`, the coupling of v2.10 and before, a
deformable meets its contacts with moving bodies after the world has solved the
contacts and joints of those bodies, so the impulses a deformable hands to a
body arrive after the body's own support was settled for the step. Three things
follow:

- A body that carries a deformable is pushed into its own support every step: a
  5 kg cloth on a 1 kg box presses the box 19 mm into the ground, and a joint
  that holds a soft load overshoots its limit by an amount proportional to the
  load.
- A heavy body resting on a deformable presses its particles into whatever lies
  under them: a 10 kg ball on a 0.2 kg cloth pushes particles 12 mm into the
  ground and spreads the cloth to several times its size.
- The ground holds each particle of a cloth with the friction of the particle's
  own weight only, so a box on a cloth on a gentle slope slides down together
  with the cloth.

Solved together, the weight of a deformable reaches the ground through the
bodies under it within the step, and a particle pressed by a body passes the
load on to what it rests on: the ground, a static or kinematic object, another moving body, or the
particles of another deformable that rest on one of these. A heavy body on a
cloth, or on several layers of cloth, rests on what supports the cloth, and the
cloth under it is held with the friction of the body's weight. A body sliding
over a cloth meets the friction of whichever contact grips less: over a cloth
that the ground holds it slides with the friction between body and cloth, and
on slippery ground it carries the cloth along and slides with the ground's
friction. A body on a cloth on a slope steeper than the friction angle slides
down with the Coulomb acceleration, also with the same friction above and below
the cloth (1.7 times as fast with `setMonolithicDeformables(false)`):

| Scene | `setMonolithicDeformables(false)` | Default |
|---|---|---|
| 10 kg ball on a cloth on the ground: particles in the ground | 12.3 mm, 1.19 ms/step | < 0.001 mm, 0.99 ms/step |
| 5 kg cloth on a 1 kg box: box in the ground | 19.4 mm, 0.83 ms/step | 0.04 mm, 2.13 ms/step |
| 1 kg ball on a cloth on a 5 kg box: particles in the box | 2.9 mm, 0.88 ms/step | < 0.001 mm, 1.03 ms/step |
| 5 kg box on two stacked cloths: upper cloth pressed into the lower one | 10.0 mm, 5.51 ms/step | 0.41 mm, 0.99 ms/step |
| 1 kg box on a cloth on a slope of 0.45 (μ = 0.5): sliding speed | 213 mm/s, 0.77 ms/step | < 0.001 mm/s, 0.75 ms/step |
| 1 kg box on a cloth sliding down a slope of 0.6 (μ = 0.5 above and below the cloth): error of the distance the box travels in a second | 17.7 %, 0.77 ms/step | 0.15 %, 1.70 ms/step |
| Cloth pulled out from under a 1 kg box with 20 N (μ = 0.5): error of the box's acceleration while the cloth slides under it | 2.2 %, 1.35 ms/step | < 0.01 %, 2.93 ms/step |
| The same cloth: largest stretch of an edge beside the box | 3.9 %, 1.39 ms/step | 3.1 %, 3.13 ms/step |
| 0.2 kg cloth on the ground pulled at one edge with 20 N: largest stretch of an edge | 0.009 %, 1.15 ms/step | 0.009 %, 1.26 ms/step |
| Two-finger gripper holding a hanging cloth: fingers in the particles | 0.10 mm, 0.59 ms/step | 0.04 mm, 0.73 ms/step |
| Lever holding a soft cube at its joint limit: overshoot | 17.6 mrad, 0.29 ms/step | 0.02 mrad, 0.29 ms/step |
| 1 kg box pushed with 3 N on a cloth on ground with μ = 0.1: error of the distance both travel | 12.9 %, 0.77 ms/step | 0.35 %, 1.24 ms/step |
| 5 kg box sliding at 1 m/s over a cloth that the ground holds (μ = 0.5 above the cloth, 0.8 below): error of the deceleration | 24.4 %, 1.90 ms/step | 0.001 %, 1.94 ms/step |
| 5 kg box sliding at 1 m/s over a cloth on ground with μ = 0.3, carrying it: deceleration over μ g | 18.6 % off, 1.84 ms/step | 4.5 % more, 4.03 ms/step (the cloth outside the box drags; fully converged 3.5 % more) |

(the `deformable_rigid_coupling` benchmark, single thread, Ryzen 9 3950X).

A heavy body resting on a flat sheet whose hinges are solved together (see
[Bending](#bending)) presses particles that the sheet's bending stiffness holds,
not a support under them. While the elastic constraints are projected, the
contact rows of such a body, and those of the sheet's own supports, see each
particle move with the sheet: the mobility between the contact particles comes
from the factorization of the hinges, and an impulse moves all of them at once,
while each elastic sweep passes the impulses on to the rest of the sheet. A box
on a stiff plate resting on two rods used to ride the plate on a soft, slow
spring (a swing of 16 mm with a period of about 2 s at 6 iterations, where the
plate's own period is 0.26 s); it now settles within a few periods, and in the
`deformable_stiff_sheet` benchmark the plate under a 0.5 kg box sags 2 % more
than beam theory instead of 18 %. Up to 256 contact particles are coupled this
way; a sheet lying on a support (more of them) is held by the support as
described above.

In the joint solve `iterations` counts the sweeps in which the elastic
constraints are projected; the contacts keep converging after them. A
deformable with `substeps` greater than one makes the whole world step in that
many substeps, also while it sleeps (an impact that wakes it is taken in its
substeps), and the forces set for the step act in each of them. A cloth that
touches only static objects costs up to a tenth more per step than with
`setMonolithicDeformables(false)`: the joint solve also holds it with static
friction and meets its self contacts and contacts once more at the end of the
step. A scene where bodies press or carry a deformable costs up to a third more,
some up to 2.6 times as much (see the table).

With `setMonolithicDeformables(false)`, a body on a cloth that is pulled out
from under it at slip speeds of metres per second receives up to three times the
Coulomb friction: the particles that pass under its edges kick it up and along.
The joint solve takes such a particle's contact where the particle will be at
the end of the step, so it meets the face of the body rather than its edge, and
the body receives the Coulomb friction at any slip speed.

Known limitations of the joint solve:

- A cloth pinched between two parts of one body, such as the two pads of a
  gripper, is held by the pads' friction more firmly than its own tension can
  pull it: the edges at the pinch stretch by up to 20 % while the gripper moves
  it at the default `iterations`. (A cloth pressed by a body onto a support, by
  contrast, stays within 3 %.)
- Under a heavy body, the particles of a cloth layer that settle into the gaps
  between those of the layer below are still pressed down by several
  millimetres (a 5 kg box on two layers offset by half the particle spacing:
  7 mm), and the step costs about ten times as much.
- A body sliding with high friction over a particle that a stiff edge ties to a
  pinned or held part of the cloth is braked less than by its Coulomb friction
  within a step; the edge holds, and the following steps keep braking the body
  (μ = 30: a 1 kg box sliding at 1 m/s is slowed by 0.13 m/s in its first step
  instead of 0.29 m/s).
- A particle pressed onto a support from which one of its edges rises steeply
  (a cloth pressed at the foot of a box it is draped over) can start the next
  step still moving towards the support, slowly (1.3 mm/s after a push of 1 m/s
  along the support with the edge at 45°); the support stops it then.
- A stiff strip of cloth jammed between a body that drags one end of it and an
  obstacle at the other end (a wall, another part of the cloth) does not stop
  the body: the strip is crushed (a 10 kg box dragging a particle at 2 m/s
  towards a wall along a rigid 2 cm edge shrinks the edge to a third), and
  when the body lets go the strip springs back, which can push the far particle
  into the obstacle (1.6 mm for particles of 5 mm radius).
- Where a body drives a stiff edge of a cloth folded more than twice into the
  stack, the layers beyond the second can end the step overlapping by a fraction
  of a millimetre (a third of a millimetre for particles of 5 mm radius pushed
  at 2 m/s); the next step separates them.

## Sleeping

With sleeping enabled, a deformable object sleeps only when every particle has
been slower than the linear sleep threshold for the required number of steps.
Applying a force, moving or transforming the object, pinning or unpinning a
particle, or adding constraints wakes it up.

## Pinned Vertices And Forces

Pinned vertices are fixed in world coordinates, have infinite mass, and are
exposed to collision as static proxies. They can be specified at construction
time or changed later; `pinVertex` holds the particle at its current
position:

```cpp
cloth->pinVertex(0);
cloth->unpinVertex(0, 0.01);
```

`unpinVertex(idx, mass)` restores a finite, positive mass for that particle.
Use a mass consistent with the surrounding particles (`getMass()` of a
neighbor). Constraint corrections are distributed in proportion to
inverse mass: a much lighter particle absorbs most of each correction and
reacts strongly to forces, while a much heavier one barely moves and drags its
neighbors as if it were nearly pinned.

External forces can be applied per particle with the standard `Object`
interface. They act on the next step only, and calls on pinned particles are
ignored. Particles have no rotational state, so `setExternalTorque` has no
effect:

```cpp
softBody->setExternalForce(3, {0.0, 0.0, 2.0});
softBody->clearExternalForcesAndTorques();
```

## Solver Tuning

Start with the largest stable time step your application needs, then tune in
this order:

1. Choose a particle spacing that resolves the shape at the level your task
   actually observes.
2. Set `collisionRadius` so neighboring particle proxies cover the surface
   without making the visual mesh appear inflated.
3. Increase `iterations` until distance constraints converge enough for the
   task.
4. Increase `substeps` when heavy objects stretch the material too much or
   contacts are jittery. Several substeps with few iterations each usually
   converge better than one substep with many iterations.
5. Reduce `distanceCompliance` or increase `youngsModulus` only after the
   discretization and solver budget are reasonable.

Stiff bending and stiff membranes (see [Bending](#bending) and
[Elastic Modulus](#elastic-modulus)) are solved over the whole surface in every
iteration, so their stiffness no longer depends on `iterations`; their cost per
iteration is several times that of a sweep, so they converge with fewer
iterations than a soft material needs.

`XPBD` is the recommended default because its compliance gives a material
stiffness that depends far less on the iteration count and time step than
`PBD`, where stiffness is only a by-product of how many rigid projections are
performed. Use `PBD` mainly for backward-compatible behavior or controlled
comparisons.

Common failure modes:

- Exploding or jittering contacts usually mean the time step is too large for
  the stiffness/contact radius combination.
- A cloth that stretches too much needs lower `distanceCompliance` or more
  iterations.
- A cloth that folds too easily needs bending constraints (a nonnegative
  `bendCompliance`; bending is off by default) and then a lower
  `bendCompliance`.
- A cloth that keeps flapping or swinging needs `airDrag`; vibrations of a
  stiff material need `damping`.
- Filled objects that collapse need a `youngsModulus` (an elastic solid), or
  more internal struts, smaller spacing, or higher stiffness.

## Rayrai Visualization

A local `RayraiWindow` draws deformable objects in its world automatically.
`RaisimServer` serializes them as dynamic `Shape::Mesh` visuals for the
Rayrai TCP viewer. The first visualizer packet sends the object name, triangle
topology, and current vertex positions. Later packets stream updated vertex
positions and resend the triangle list only if the topology changes. Rayrai
rebuilds the custom OpenGL mesh, recomputes normals from the triangle list, and
renders the deformable surface in world coordinates. Both paths draw the
vertices at `getVisualPosition()`.

The particle collision proxies are an internal physics representation and are
not rendered as one sphere per vertex. For an open cloth, the drawn surface
passes through the particle centers, so the collision envelope extends about
`collisionRadius` to either side of it. For a closed mesh, the inward particle
shift and the visual offset make the drawn surface approximately coincide with
the collision envelope; the offsets turn with the object's overall rotation, so
the drawn surface stays outside the particles of a tumbling soft body. For stacked soft cubes or other closed shapes, choose
particle spacing and `collisionRadius` together: neighboring particle spheres
should cover the surface without leaving gaps, but the radius should stay small
relative to the object's features.

## World XML

`World::exportToXml` writes each deformable object as a `<deformable>`
object with its material, contact material, collision group and mask, every
particle's position, rest position, velocity, visual offset, inverse mass, and
pinned flag, the triangle list, the tetrahedra of a volume, and the distance
constraints. Loading the file with `raisim::World(path)` recreates the object;
bending constraints, membrane and solid elements are rebuilt from the triangles,
tetrahedra, rest positions, and material, and a solid continues in the pose it
was saved in. The element layout is intended for
this export/import round trip; the loader requires the attributes that the
exporter writes.

## Limitations

This is still an experimental implementation. It does not include topology
changes such as tearing or cutting.
Granular systems do not collide with deformable objects. A rigid edge or corner
can still pass between particle spheres when `collisionRadius` is small
compared with the particle spacing.
Membranes more than about 2300 times stiffer than their particle inertia over a
substep (see [Elastic Modulus](#elastic-modulus)) are not fully converged by the 8
conjugate gradient iterations of a projection. The contact coupling of heavy
bodies with stiff sheets applies to sheets with a flat rest shape in the joint solve only, not to curved
shells or to `setMonolithicDeformables(false)`.
A long rope or cloth with many very shallow wrinkles (bends under 1°) that are
all pushed flat within one step lengthens all at once, and its free end leaves
faster than the push alone would make it: a 3 m rope of 300 particles with
wrinkles of 0.9° gains about as much energy again as the push gave it.

## API Reference

- [`raisim::DeformableObject`](/api/raisim/DeformableObject.md)
