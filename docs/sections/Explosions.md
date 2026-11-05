# Explosions and Projectiles

A `raisim::World` sets off explosions and flies projectiles. An explosion's blast wave spreads from the
charge and pushes the dynamic bodies it reaches. A projectile flies under gravity and the drag of the
world's fluids, hits the first body on its path, hands it its momentum and, if it is explosive,
detonates there. Together they make shells, grenades, bullets, rockets after burnout, demolition
charges and exploding barrels. The [Explosion and Projectile](examples/rayrai/rayrai_explosion_projectile.md)
example shows the least code that uses both; the [Tank Survival](examples/rayrai/rayrai_tank_survival.md) game and the [Tank Battle](examples/rayrai/rayrai_tank_battle.md) example use
them at length, and rayrai draws them ([Explosion effects](rayrai/Explosions.md)).

```cpp
#include "raisim/World.hpp"

raisim::World world;
world.setAmbientFluid(raisim::FluidMedium::air());  // air for projectile drag; its velocity is the wind

// 2 kg of TNT on the ground, digging a 1.2 m crater in the heightmaps under it.
raisim::ExplosionProperties blast;
blast.chargeMass = 2.0;
blast.surfaceBurst = true;
blast.craterRadius = 1.2;
blast.craterDepth = 0.4;
world.addExplosion({10.0, 0.0, 0.2}, blast);

// A 12 kg shell at 450 m/s that detonates where it hits; the tank that fires it is not hit.
raisim::ProjectileProperties shell;
shell.mass = 12.0;
shell.radius = 0.06;
shell.dragCoefficient = 0.25;
shell.explosion.chargeMass = 2.0;
shell.ignoredObject = tank->getIndexInWorld();
world.addProjectile(muzzle, direction * 450.0, shell);
```

## Explosions

`World::addExplosion(position, properties)` sets off a charge at the current world time and returns
its identifier. `ExplosionProperties` holds:

| Field | Meaning |
| --- | --- |
| `chargeMass` | TNT-equivalent mass (kg). Multiply the mass of another explosive by its TNT equivalence (about 1.2 for Composition B, 1.3 for C-4). |
| `surfaceBurst` | The charge lies on the ground: the ground reflects its hemispherical wave, so it loads objects like a free-air charge 1.8 times as heavy (`blast::kSurfaceBurstFactor`). |
| `collisionMask` | Collision groups the blast loads. |
| `lineOfSight` | Bodies hidden from the charge take only the wave that wraps around them (default on). |
| `craterRadius`, `craterDepth` | A bowl-shaped crater dug in each heightmap under the charge (zero digs none). |

`World::getExplosions()` lists the last 256 explosions, and older ones whose blast still spreads,
oldest first, with increasing identifiers, so a renderer or a game can act on those newer than the last
it saw. Each records its position, world
time, properties and the projectile that set it off, if any.

### The blast wave

`raisim::blast::wave(distance, chargeMass)` describes the blast at a distance from a free-air charge
in a sea-level atmosphere. Every quantity follows Hopkinson–Cranz scaling: a charge of mass W at
distance R behaves as a 1 kg charge at the scaled distance Z = R / W^(1/3), with times and impulses
scaled by W^(1/3).

- **Peak overpressure** Ps and **positive-phase impulse** follow the free-air fits of Kinney and
  Graham, *Explosive Shocks in Air* (1985).
- **Arrival time**: the shock front moves at the Rankine–Hugoniot shock speed U = a0 √(1 + 6 Ps / 7 P0),
  integrated outward from the charge.
- **Positive-phase duration**: the Kinney–Graham hypothesis that the overpressure phase ends when a
  boundary moving at the sound speed behind the shock arrives, with one constant calibrated to
  Kingery–Bulmash surface bursts.
- **Pressure history**: the modified Friedlander pulse p(t) = Ps (1 − t/td) e^(−b t/td), its decay b
  chosen so that the pulse carries exactly the fitted impulse.
- **Reflected overpressure** on a surface facing the blast, Pr = 2 Ps (7 P0 + 4 Ps) / (7 P0 + Ps), and
  **dynamic pressure** of the blast wind, q = 5 Ps² / 2 (7 P0 + Ps), from the Rankine–Hugoniot
  relations for air.

Scaled distances under 0.1 m/kg^(1/3), inside the fireball, take the values at 0.1. Blasts load objects
out to 40 m/kg^(1/3) (`blast::reach()`), where the overpressure has fallen to about 2 kPa.

The table compares the model, for surface bursts of the charge, with the Kingery–Bulmash values that
the UN guideline IATG 01.80 (2021) tabulates (model / reference). The model's near-field impulse falls
below the Kingery–Bulmash data, as Kinney and Graham's does.

| Charge, distance | Overpressure (bar) | Impulse (bar ms) | Duration (ms) | Arrival (ms) | Shock speed (m/s) |
| --- | --- | --- | --- | --- | --- |
| 1000 kg at 50 m | 0.425 / 0.43 | 5.71 / 5.9 | 39.9 / 37.9 | 80.2 / 82.4 | 396 / 398 |
| 10000 kg at 50 m | 2.32 / 2.02 | 23.4 / 25.2 | 59.8 / 46.8 | 45.2 / 48.1 | 585 / 559 |
| 100000 kg at 50 m | 13.1 / 11.5 | 67.7 / 106 | 82.8 / 93 | 22.5 / 24.8 | 1182 / 1114 |
| 1000 kg at 222 m | 0.049 / 0.054 | 1.31 / 1.43 | 55.2 / 61.1 | – | – |

### How the blast pushes bodies

Every dynamic body in reach takes the blast on its collision geometry: rigid bodies, the links of
articulated systems and the parts of compounds. Each shape is divided into panels (the same panels the
[fluid forces](Fluids.md) use), and each panel takes the pressure of the wave from the moment the shock
reaches it, so the wave reaches the far side of a body later than the near side:

- every panel takes the incident overpressure;
- a panel facing the charge also takes the excess of the reflected overpressure, weighted by the square
  of the cosine of its incidence and cleared over the time the shock takes to travel three equivalent
  radii of the body, and the dynamic pressure of the blast wind, weighted the same way.

The force on each panel is integrated over the step exactly, so a body gains the same momentum whatever
the time step, even from a pulse shorter than a step; the step applies the resulting force and torque
at the body's center of mass. A body is pushed only while the wave passes it, and a sleeping body
wakes.

With `lineOfSight`, five rays from the charge to the center of a collision body and around it decide
how much of it the charge sees; the reflected pressure and the blast wind scale with that fraction. A
body behind a wall, a hill, another body or another part of its own object (a link of the same robot)
then takes only the incident pressure, whose net push comes from the difference in arrival times across
it. The fraction is found when the wave reaches the body and kept while it passes.

Static and kinematic objects are not pushed. Deformable objects and granular particles are not loaded
yet.

A 1 kg charge pushes a 20 kg, 0.5 m crate at 1 m from its face to about 6.7 m/s, at 2 m to 2.4 m/s and
at 5 m to 0.43 m/s; a 1 t, 1.5 m crate at 1 m to 1 m/s. The blast wave alone moves heavy objects little
beyond a few charge radii: fragments, which are not modelled, do the most damage of real munitions.

### Craters

With `craterRadius` and `craterDepth`, the explosion lowers every heightmap whose surface lies within
the crater radius of the charge by a paraboloid bowl: by the depth at the center and nothing at the
radius, following the terrain's slope. The heightmap's collision surface is rebuilt at once. In
[deformable terrain](DeformableTerrain.md) the crater is permanent sinkage of the soil
(`DeformableTerrain::getSoilState()`).

## Projectiles

`World::addProjectile(position, velocity, properties)` launches a projectile at the current world time
and returns its identifier. `ProjectileProperties` holds:

| Field | Meaning |
| --- | --- |
| `mass`, `radius`, `dragCoefficient` | Mass (kg), radius of its cross section (m) and the drag coefficient of that section. |
| `explosion` | The explosion it sets off; a `chargeMass` of zero makes an inert projectile. |
| `detonateOnImpact` | An explosive projectile detonates where it hits; otherwise it stops there. |
| `fuseTime` | Seconds after launch at which it detonates in flight (zero: no time fuse). |
| `maxLifetime` | Seconds after launch at which it is removed if it hit nothing. |
| `collisionMask` | Collision groups it hits. |
| `ignoredObject` | World index of an object it passes through, such as the vehicle that fired it. `World::removeObject()` keeps it pointing at the same object, or clears it when that object is removed. |

Each step moves a projectile under gravity and the quadratic drag of the fluid around it,
F = −½ ρ Cd π r² |u| u, where u is its velocity relative to the fluid:

- above the liquid, the world's ambient fluid (`World::setAmbientFluid()`): its density sets the drag
  and its velocity is the wind, which carries projectiles downwind;
- below the surface of a volume of liquid (`World::addFluidVolume()`), the liquid and its current, so a
  projectile slows sharply when it enters water. Where the volume has a sampler or waves, the surface
  and the current are those it gives at the projectile (a simulated river, a wave crest); otherwise the
  mean surface and the liquid's velocity.

A world without an ambient fluid is a vacuum.

Each step sweeps the projectile's path with a ray, so a fast projectile hits a thin plate it would
otherwise pass through. At the first body on its path it stops and hands the body the impulse of a
perfectly inelastic impact at the point it hits: the impulse that leaves that point of the body moving
with the projectile. It depends on the projectile's velocity relative to that point, so a shell
overtaking a moving target pushes it less, and on the mass the body presents there, which includes its
rotation and, for a link of an articulated system, the links and joints behind it; an off-center hit
spins the body. Projectiles that hit the same body in one step take turns: each meets it moving as the
earlier ones left it. The time and velocity of a hit follow the projectile's path through the step,
which gravity and drag bend. An explosive projectile then detonates there, just off the surface. A
time fuse detonates it where it has flown when the fuse runs out, part way through a step if need be.

`World::getProjectiles()` lists the projectiles in flight with their positions and velocities and
where, when and how fast they were launched.
`World::getProjectileHits()` lists the last 256 hits, oldest first, with increasing identifiers: the
object and body hit, the point of impact, the surface normal there, the projectile's velocity and mass,
the time, and whether it detonated. `World::getProjectileLaunches()` lists the last 256 projectiles
launched, as they were launched, whatever became of them: a renderer draws a muzzle flash from it also
for a round that came down before it looked. `World::removeProjectile(id)` removes one in flight.

`RayCollisionItem::getNormal()` gives the normal of the surface a ray test hits; projectile hits use
it.

## Checkpoints and performance

`World::captureCheckpoint()` captures the blasts still spreading, the projectiles in flight, the
explosion, hit and launch lists and the heightmaps; `restoreCheckpoint()` restores them, craters included, so replays are
reproducible.

The `explosion` benchmark compares RaiSim with MuJoCo, which has neither explosions nor projectiles and
gets the same face-on blast load through `xfrc_applied` and rounds flown with `mj_ray`:

```bash
./build/benchmark/benchmarks --bench explosion
```

It reports the momentum a blast hands a free crate against the panel integral of the model, the cost of
a step of a yard of 200 crates with a 2 kg charge going off among them every 0.25 s, the cost of a
round in flight, the cost of a volley of 100 rounds at the crates, and the cost of sustained fire: two
rounds a step for a second, swept across the crates, so that dozens fly at once and crates take several
in a step. MuJoCo's blast is the face-on approximation and its rounds hand over the momentum of a crate at
rest, so its timings come from less work than RaiSim's panel loads, line of sight and impacts.

## Python

```python
import numpy as np
import raisimpy as r

world = r.World()
world.setAmbientFluid(r.FluidMedium.air())
shell = r.ProjectileProperties()
shell.explosion.chargeMass = 2.0
world.addProjectile(np.array([0.0, 0.0, 1.0]), np.array([300.0, 0.0, 20.0]), shell)
wave = r.blast.wave(5.0, 1.0)  # overpressure, impulse, arrivalTime, duration, ...
```
