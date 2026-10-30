# Fluids

Bodies in RaiSim can float, sink, swim and fly through water and air. A world can hold volumes of
liquid under a free surface, such as a pool, a lake or the sea, and an ambient fluid, such as air,
everywhere else. Every dynamic body is held up by the weight of the fluid it displaces. The fluid
drags it, lifts it, and carries part of its own mass along with it. Fluids act on single bodies,
compounds, meshes and every link of an articulated system, through their collision shapes.

![Bodies at rest in a pool, each at its own depth: a half-dense cube with an edge up, an ice block mostly under water, a plank, a log and a light ball](../../rsc/docs/image/rayrai/fluid_buoyancy.png)

Fluids are off by default. A world without them steps exactly as before. The equations of the forces
and of the motion they produce are written out in [Dynamics of Fluid Forces](./fluids/Dynamics.md).

## Adding water and air

```cpp
raisim::World world;

// air everywhere outside the water
world.setAmbientFluid(raisim::FluidMedium::air());

// a pool 4 m by 2.4 m, filled to 0.7 m
raisim::FluidVolume pool;
pool.medium = raisim::FluidMedium::water();
pool.surfaceHeight = 0.7;
pool.bottomHeight = 0.0;
pool.min = {-2.0, -1.2};
pool.max = {2.0, 1.2};
const size_t index = world.addFluidVolume(pool);
```

`FluidMedium` holds a fluid's density $\rho$, its dynamic viscosity $\mu$ and its velocity
$\boldsymbol{v}_f$, which is a current or the wind. `water()`, `seawater()` and `air()` hold their
values at 20 °C, 15 °C and 20 °C.

A `FluidVolume` holds the collision shapes whose centroids lie within its horizontal bounds and
above `bottomHeight`, and that reach below `surfaceHeight`; by default it reaches the horizon. The
surface cuts a shape where it crosses it. The walls and the bottom are meant to be solid, like the walls of a
pool: a shape that reaches through them is still wholly in the liquid up to the surface. Where two
volumes overlap, the first one added holds a shape.

To change the current, the water level or the bounds, call `World::setFluidVolume(index, volume)`.
Every call that changes a fluid wakes the sleeping bodies.

```cpp
auto river = world.getFluidVolumes()[index];
river.medium.velocity = {0.5, 0.0, 0.0};
world.setFluidVolume(index, river);
```

### Waves

A volume's `waves` move its surface and the liquid under it. Each `FluidWave` is a linear wave on
deep water: it travels along its `direction` with its `wavelength` $\lambda$, and raises the surface
by

$$
\eta = a \cos\theta,\qquad \theta = k\,\boldsymbol{d}\cdot\boldsymbol{x} - \Omega t + \varphi,\qquad
k = \frac{2\pi}{\lambda},\qquad \Omega = \sqrt{g k},
$$

with its `amplitude` $a$ and `phase` $\varphi$, at the world time $t$. The waves of a volume add up.
A body sees the surface averaged over its extent: a plane with the height and tilt of the waves
averaged over the body's horizontal spread, each wave weakened the more the longer the body is
along it. A float heaves and rolls with the waves. A raft half a wave long feels the crest under one
end and the trough under the other, with a buoyancy and moment within about 1 % of those of the water
over its bottom in RaiSim's tests. A body small against the waves sees the plane tangent to them at
its centroid. Under waves low against their length, the liquid moves in circles that shrink as
$e^{-k z}$ with the depth $z$ below the mean surface: forward under a crest, back under a trough. This motion adds to the current of
the volume. A body is dragged along by it, and the liquid that accelerates pushes the body as it pushes
the liquid the body displaces, and carries the body's added mass:

$$
\boldsymbol{f}_a = (\rho V_s + \boldsymbol{M}_A)\,\boldsymbol{a}_w,
$$

where $\boldsymbol{a}_w$ is the acceleration of the liquid as it moves, which includes its change as it is
carried from place to place. A particle as dense as the water therefore moves on the circle the water
moves on, and over many waves it keeps its depth and drifts forward with them at the Stokes drift of
low waves, $a^2 \Omega k\, e^{-2 k z}$. A body within the volume's horizontal bounds never falls
asleep. rayrai and the TCP viewer draw the waves where the physics moves them, assuming
$g = 9.81\ \text{m/s}^2$.

![A raft half a wave long tilted from a trough to a crest, with a row of crates riding the wave at different heights](../../rsc/docs/image/rayrai/fluid_raft_waves.png)

```cpp
raisim::FluidVolume sea;
sea.medium = raisim::FluidMedium::seawater();
sea.waves.push_back({{1.0, 0.0}, 20.0, 0.3, 0.0});  // direction, wavelength, amplitude, phase
sea.waves.push_back({{0.6, 0.8}, 7.0, 0.1, 1.0});
world.addFluidVolume(sea);
```

### Rivers and other surfaces

A volume's `sampler` gives it a surface and a current that vary from place to place in any other
way, and takes the place of its waves. It receives the
centroid of a collision body and returns the height and normal of the surface over that point, and
the velocity and acceleration of the liquid there. The body sees the surface as the plane through the sample, tilted
by its normal, so it rides the slope of a wave. A wave field or a river simulated elsewhere plugs in
this way. For example, rayrai's `RayraiWindow::sampleWaterSurface()` gives the waves it draws:

```cpp
volume.sampler = [&viewer](const raisim::Vec<3>& point) {
  const auto surface = viewer.sampleWaterSurface({point[0], point[1]});
  raisim::FluidSample sample;
  sample.surfaceHeight = surface.height;
  sample.surfaceNormal = {surface.normal.x, surface.normal.y, surface.normal.z};
  return sample;
};
```

The sampler runs on the thread that steps the world, once for every collision body in the volume's
bounds. The world stays deterministic as long as the sampler's result depends only on the point and
the simulation state. A body within a sampled volume's horizontal bounds never falls asleep, wet
or dry, since the surface or the current may change while it rests.

### World files, scenes and Engine 2

A world file holds the fluids in its `fluids` element and each body's
coefficients in a `fluid` child (see [World Configuration File](./WorldConfigurationFile.md#fluids));
`World::exportToXml()` writes them. A `.rscene` scene holds them in its
`ambient_fluid` and `fluid_volume` records and the `fluid*` keys of its objects
(see [.rscene File](./RsceneFile.md#fluid-volume-record)), and rayrai draws the
water of its volumes. In RaiSim Engine 2, a FluidVolume node is a pool or
the sea, and the Physics settings hold the ambient fluid; the Fluid Pool
template starts from a pool with a current. Its exported `world.xml` carries
the fluids to raisimGymTorch.

## Buoyancy

The fluid pushes a body up with the weight of the fluid it displaces, and this force acts at the
centroid of the displaced volume:

$$
\boldsymbol{f}_b = -\rho V_s \boldsymbol{g}.
$$

$V_s$ is the volume of the part of the shape below the surface. It is exact for every shape:
- the cap of a sphere in closed form;
- a box or a mesh by cutting its faces at the surface;
- a cylinder or a capsule by integrating the circular segments under the surface along its axis.

Because the force acts at the centroid of the volume below the surface, the restoring moment of a
floating body is that of hydrostatics. A body is stable where its metacenter lies above its center of
mass. The half-dense wooden cube of the [fluid pool example](./examples/server/fluid_pool.md) turns
an edge up: a cube of density between 0.28 and 0.72 times that of water does not float on a face.
The part of a body above the surface displaces the ambient fluid, so a light, large body in air also
feels the air's buoyancy.

A body bobbing at the surface makes waves that carry its energy away, so it settles within a few
periods. The volume's `waveRadiation` scales this damping, and 0 turns it off. At the body's natural
heave frequency $\omega_h$, with waterplane area $A$, the damping is

$$
b = \frac{b_s b_w}{b_s + b_w},\qquad b_s = \frac{\rho\,\omega_h^3 A^2}{2g},\qquad b_w = \frac{\rho g A}{2\omega_h}.
$$

$b_s$ holds for a body small against the waves it makes (Haskind's relation for deep water). $b_w$
caps the damping for a wide body.

A body that falls into the water sets the water under it moving. As it sinks in, its added mass
(see below) grows, and the body hands that new fluid its momentum (von Kármán's water entry): a flat
plate that falls flat into the water slows at once to $m v_0 / (m + m_A)$.

![A plate and a ball falling into a pool every 50 ms: the plate stops on the surface, the ball plunges and bobs](../../rsc/docs/image/rayrai/fluid_entry.png)

## Drag and lift

The surface of each shape is made of small faces. The flow pushes on a face that meets it with a
pressure along its normal $\boldsymbol{n}$, and rubs along the faces it passes:

$$
\boldsymbol{f} = -\tfrac{1}{2}\rho A \left(C_p\, |\boldsymbol{w}|\,\max(0, \boldsymbol{w}\cdot\boldsymbol{n})\,\boldsymbol{n} + C_f\,|\boldsymbol{w}_t|\,\boldsymbol{w}_t\right).
$$

$\boldsymbol{w}$ is the velocity of the face relative to the fluid, $\boldsymbol{w}_t$ its part along
the face, and $A$ the face's area. Integrated over a shape, the pressure gives the textbook drag
$\tfrac{1}{2}\rho C_D A_p |\boldsymbol{w}|\boldsymbol{w}$ on the area $A_p$ the shape shows to the
flow. Each shape takes its drag coefficient $C_D$ from measurements (Hoerner):

| Shape | $C_D$ |
| --- | --- |
| Sphere | Brown and Lawler's correlation in the Reynolds number: 0.41 at $10^4$, 0.47 at $10^5$, Stokes' law below 1 |
| Box face | 1.05 |
| Cylinder across its axis | 1.0 |
| Cylinder along its axis | 0.82 |
| Capsule | 1.0 across its axis, the sphere's on its ends |
| Mesh triangle | 1.0 |

A plate that meets the flow at an angle of attack $\alpha$ feels the pressure on its lower face along
its normal. This gives it lift as well as drag, $C_L = C_D \sin\alpha \cos\alpha$ and
$C_D(\alpha) = C_D \sin^2\alpha$. A fin, a flipper or a leg sweeping through water pushes the body
this way. Because the faces move with the body, a body that turns meets the flow faster on its
outer faces. A long link that swings is braked as strip theory says. A sphere that spins feels only
its skin friction, plus its Magnus lift, $\boldsymbol{f} = C_M \rho V (\boldsymbol{\omega}\times
\boldsymbol{w})$. Cylinders and capsules feel Magnus lift from their spin about their axis.

Every shape also feels the Stokes drag of the viscosity, $3\pi\mu d\,\boldsymbol{w}$, and the Stokes
torque $\pi\mu d^3\boldsymbol{\omega}$ of a sphere of the same volume, with diameter $d$. They
dominate at low Reynolds number, where a grain settles at Stokes' speed. Drag and skin friction are
applied implicitly: a light body in a viscous fluid settles stably even when the drag would stop it
within a fraction of a time step, and a thin rod sliding along its length through water slows down
without turning around.

## Added mass

A body that accelerates in a fluid also accelerates the fluid around it, so it moves as if it were
heavier. The extra inertia $\boldsymbol{M}_A$ depends on the shape and the direction:
- half the displaced mass for a sphere;
- the displaced mass for a long cylinder moving across its axis;
- far more than its own mass for a thin plate moving across its face.

RaiSim adds $\boldsymbol{M}_A$ to the inertia of each body for the whole step. Contacts, joints,
rolling and spinning friction and tendons therefore see it too, and a body much lighter than the
water it displaces stays stable. A ping-pong
ball let go under water starts up at 15.5 m/s², as it does in a tank. Without its added mass it would
start at 112 m/s².

| Shape | Added mass |
| --- | --- |
| Sphere | $\tfrac{1}{2}\rho V$ in every direction |
| Ellipsoid | Lamb's coefficients (matches DNV-RP-C205, Table D-2, to 0.1 %) |
| Box | DNV-RP-C205 plates and prisms: $0.68\rho a^3$ for a cube, $0.579\rho\frac{\pi}{4}a^3$ for a square plate |
| Cylinder | across its axis within 5 % of DNV-RP-C205; along it, its flat ends |
| Capsule, mesh | the ellipsoid that shows the same projected areas, at the shape's volume |

The fluid carried along also turns the body as it moves (Kirchhoff's equations). A long body that
moves at an angle to its axis feels the Munk moment, which turns it toward broadside to its motion. A
body wholly in one still, inviscid fluid, with no other force on it, keeps its kinetic energy with its
added mass exactly. Part of a body above the surface carries the added mass of the fluid there.

![A floating log shoved along at an angle to its axis, yawing and curving as it glides](../../rsc/docs/image/rayrai/fluid_munk.png)

## Fluid coefficients

`FluidCoefficients` sets how fluids act on a collision body:

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | whether fluids act on the shape |
| `buoyancy` | `true` | whether the fluid holds it up |
| `drag` | $-1$ | $C_D$ on the projected area; negative takes the shape's own (table above) |
| `skinFriction` | 0.005 | $C_f$ of the flow along the surface |
| `magnusLift` | 0.5 | $C_M$ of a spinning sphere, cylinder or capsule |
| `addedMass` | 1 | scale of the added mass |

```cpp
raisim::FluidCoefficients hull;
hull.drag = 0.6;
world.setFluidCoefficients(boat, hull);          // every collision body of an object
robot->getCollisionBody("fin/0").setFluidCoefficients(hull);   // one body of an articulated system
```

`Object::getFluidForce(localIdx)` and `Object::getFluidTorque(localIdx)` return what the fluids did
to a body in the last step, about its center of mass in the world frame. This covers buoyancy, drag,
lift and the turn the added mass gives a body that moves, such as the Munk moment, at the velocity
the step started with. The added mass that resists acceleration is not part of it.

## Accuracy

Each value below is checked by RaiSim's tests:

| Case | Expected | RaiSim |
| --- | --- | --- |
| Sphere, box, cylinder, capsule cut by the surface at any tilt | buoyancy of the displaced volume | exact (cylinder and capsule to $10^{-10}$ of their volume) |
| Ping-pong ball let go under water | 15.5 m/s² | 15.5 m/s² |
| Sphere on a spring in water, $T = 2\pi\sqrt{(m + \frac{1}{2}\rho V)/k}$ | 0.706 s | within 0.2 % |
| 10 mm steel ball sinking in water | 1.45 m/s | to $10^{-6}$ of the drag correlation |
| 1 mm steel grain settling in glycerol at a step three times its relaxation time | 2.5 mm/s, without overshoot | within 0.5 % |
| Box half as dense as water | floats at half its height | within 0.01 mm |
| Capsule moving at an angle in still, inviscid water | keeps its kinetic energy | to $10^{-9}$ |
| Plate falling flat into water, $m_A = 11\,m$ | $m v_0 / (m + m_A)$ | within 3.3 % at 1 ms, 0.16 % at 0.25 ms |
| Light float on a 20 m wave of amplitude $a$ | heaves by $2a$ | within 0.5 % |
| 2 m box held at the surface of a 4 m wave of amplitude 0.06 m, at an angle to it | buoyancy and yaw moment of the columns of water over its bottom | force within 0.3 %, moment within 1 % (the tangent plane at its centroid: 3 %, 18 %) |
| Sphere sinking into a pool, turned any way | the same force; no force across its motion | to $10^{-9}$ |
| Sphere crossing the surface | drag without a jump as it sinks under or first touches | to $10^{-6}$ |
| Upright cylinder a quarter of a wave across, held there | buoyancy of the columns of water over its disc | within 0.1 % |
| Box rolled by 0.3 rad, and one lying on its side, held at the surface of a 3 m wave | buoyancy of the water below the surface within it | within 0.15 % |
| Particle as dense as water 2 m under that wave | moves on a circle of radius $a e^{-k z}$ | within 5 % horizontally, 8 % vertically |
| Particle as dense as water 1 m or 3 m under a 10 m wave of amplitude 0.2 m, over 50 periods at a 2 ms step, as a single body or an articulated system integrating with `TRAPEZOID` or `SEMI_IMPLICIT` | keeps its depth; drifts forward at $a^2 \Omega k\, e^{-2 k z}$ | depth within 15 % of the circle's radius, drift within 5 % |

## Limits

- The fluid does not move. Bodies do not stir it, so they leave no wake. A body in another body's
  wake, or between two walls, feels the free stream.
- A body sees the surface as one plane: under waves, the waves averaged over its extent; under a
  sampler, the plane through the sample at its centroid, tilted by the sampler's normal. Waves that
  wet a body along part of its length and leave the rest dry, as over a raft thinner than they are
  high, are seen through that one plane, so its buoyancy is approximate. The velocity and
  acceleration of the waves are taken at the centroid, so a body longer than about a quarter of a
  wave feels their motion approximately.
- A floating body drifts over many waves mostly because of the waves it reflects, which these forces
  leave out, so its drift is not reliable. A body under the surface drifts with the Stokes drift.
- Forces are quasi-steady. The added mass of a body that pierces the surface is that of the part
  below it, which is approximate within a factor of order one at most wave frequencies.
- Deformable and granular objects do not feel these fluids. Cloth has its own `airDrag`.
- An articulated system integrating with `EULER` drifts off the orbits of the waves over many periods;
  `SEMI_IMPLICIT` (the default) and `TRAPEZOID` keep them.
- The TCP viewer draws the water of every volume without a sampler as a still pool or sea, and the
  water of a scene file it receives in that scene's look. The waves of a sampled volume are not
  drawn there; rayrai draws them with `RayraiWindow::setWaterBody()`.

## Examples

- [Fluid pool](./examples/server/fluid_pool.md): bodies of different densities float, tilt and sink.
- [Fluid stream](./examples/server/fluid_stream.md): a stream, whose speed a slider in the viewer
  sets, carries crates, logs and balls off and slides bricks along its bed when strong.
- [Fluid stream with ANYmal](./examples/server/fluid_stream_anymal.md): a stream strong enough tips a
  standing ANYmal over and carries it off.
- [Floating on waves](./examples/rayrai/rayrai_fluid_waves.md): crates, a raft and barrels ride the swell
  rayrai draws, through a sampler.
- [Underwater ROV](./examples/server/underwater_rov.md): a vehicle holds its station against a turning
  current.
