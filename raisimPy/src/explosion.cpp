/**
 * Python wrappers for the explosions and projectiles of raisim (raisim/Explosion.hpp) using nanobind.
 */

#include "nanobind_helpers.hpp"

#include "converter.hpp"
#include "raisim/Explosion.hpp"

namespace py = nanobind;
using namespace raisim;

namespace {

template<class T, class Member>
void vectorProperty(py::class_<T>& type, const char* name, Member member, const char* doc) {
  type.def_prop_rw(name,
      [member](const T& self) { return convert_vec_to_np(self.*member); },
      [member](T& self, const NDArray& array) { self.*member = convert_np_to_vec<3>(array); }, doc,
      py::for_getter(py::rv_policy::move));
}

}  // namespace

void init_explosion(py::module_ &m) {
  auto blast = m.def_submodule("blast", R"(The air blast of a spherical TNT charge in a sea-level atmosphere.

Peak overpressure and positive-phase impulse follow the free-air fits of Kinney and Graham (1985), the
arrival time the Rankine-Hugoniot shock speed, the positive-phase duration the Kinney-Graham sound-speed
hypothesis calibrated to Kingery-Bulmash surface bursts, and the pressure history the modified
Friedlander pulse that carries the fitted impulse.)");
  blast.attr("kAmbientPressure") = blast::kAmbientPressure;
  blast.attr("kAmbientSoundSpeed") = blast::kAmbientSoundSpeed;
  blast.attr("kMinScaledDistance") = blast::kMinScaledDistance;
  blast.attr("kMaxScaledDistance") = blast::kMaxScaledDistance;
  blast.attr("kSurfaceBurstFactor") = blast::kSurfaceBurstFactor;
  py::class_<blast::Wave>(blast, "Wave", "The blast at one distance from a charge.")
      .def(py::init<>())
      .def_rw("overpressure", &blast::Wave::overpressure, "Peak side-on overpressure (Pa).")
      .def_rw("impulse", &blast::Wave::impulse, "Positive-phase side-on impulse per unit area (Pa s).")
      .def_rw("arrivalTime", &blast::Wave::arrivalTime, "Time from the detonation until the shock arrives (s).")
      .def_rw("duration", &blast::Wave::duration, "Duration of the positive phase (s).")
      .def_rw("decay", &blast::Wave::decay, "Decay coefficient b of the Friedlander pulse.")
      .def_rw("shockSpeed", &blast::Wave::shockSpeed, "Speed of the shock front (m/s).")
      .def_rw("reflectedOverpressure", &blast::Wave::reflectedOverpressure,
              "Peak overpressure on a surface facing the blast squarely (Pa).")
      .def_rw("dynamicPressure", &blast::Wave::dynamicPressure, "Peak dynamic pressure of the blast wind (Pa).")
      .def("overpressureAt", &blast::Wave::overpressureAt, py::arg("sinceArrival"),
           "Side-on overpressure (Pa) a time sinceArrival (s) after the shock arrived; zero outside the positive phase.");
  blast.def("wave", &blast::wave, py::arg("distance"), py::arg("chargeMass"),
            "The blast of chargeMass kg of TNT in free air at distance m.");
  blast.def("reach", &blast::reach, py::arg("chargeMass"),
            "Distance (m) at which a charge's blast stops loading objects.");

  py::class_<ExplosionProperties>(m, "ExplosionProperties", "What an explosion is and does.")
      .def(py::init<>())
      .def_rw("chargeMass", &ExplosionProperties::chargeMass, "TNT-equivalent mass of the charge (kg).")
      .def_rw("surfaceBurst", &ExplosionProperties::surfaceBurst,
              "The charge lies on the ground: it loads objects like a free-air charge 1.8 times heavier.")
      .def_rw("collisionMask", &ExplosionProperties::collisionMask, "Collision groups the blast loads.")
      .def_rw("lineOfSight", &ExplosionProperties::lineOfSight,
              "Bodies hidden from the charge take only the wave that wraps around them.")
      .def_rw("craterRadius", &ExplosionProperties::craterRadius,
              "Radius (m) of the crater dug in each heightmap under the charge; zero digs none.")
      .def_rw("craterDepth", &ExplosionProperties::craterDepth, "Depth (m) of the crater at its center.");

  py::class_<Explosion> explosion(m, "Explosion", "An explosion the world has set off.");
  explosion.def(py::init<>())
      .def_rw("id", &Explosion::id, "Increasing identifier, unique within the world.")
      .def_rw("time", &Explosion::time, "World time of the detonation (s).")
      .def_rw("properties", &Explosion::properties, "The properties it was set off with.")
      .def_rw("projectileId", &Explosion::projectileId, "Identifier of the projectile that set it off, or -1 as size_t.");
  vectorProperty(explosion, "position", &Explosion::position, "Position of the charge in the world frame (m).");

  py::class_<ProjectileProperties>(m, "ProjectileProperties", R"(A ballistic projectile: a shell, a bullet, a rocket after burnout.

It flies through the world's fluids: drag acts on its velocity relative to the ambient fluid (whose velocity is the wind)
or, below the mean surface of a volume of liquid, to the liquid. A world without an ambient fluid is a vacuum. An
explosive projectile (explosion.chargeMass > 0) detonates where it hits; an inert one stops there. Either hands a
dynamic body it hits the momentum of an inelastic impact.)")
      .def(py::init<>())
      .def_rw("mass", &ProjectileProperties::mass, "Mass (kg).")
      .def_rw("radius", &ProjectileProperties::radius, "Radius (m) of its cross section, which sets its drag area.")
      .def_rw("dragCoefficient", &ProjectileProperties::dragCoefficient, "Drag coefficient of its cross section.")
      .def_rw("maxLifetime", &ProjectileProperties::maxLifetime, "Seconds after launch at which it is removed.")
      .def_rw("fuseTime", &ProjectileProperties::fuseTime,
              "Seconds after launch at which it detonates in flight; zero or negative: no time fuse.")
      .def_rw("detonateOnImpact", &ProjectileProperties::detonateOnImpact,
              "It detonates where it hits something, or else stops there.")
      .def_rw("explosion", &ProjectileProperties::explosion, "The explosion it sets off; a chargeMass of zero is inert.")
      .def_rw("collisionMask", &ProjectileProperties::collisionMask, "Collision groups the projectile hits.")
      .def_rw("ignoredObject", &ProjectileProperties::ignoredObject, "World index of an object it passes through.");

  py::class_<Projectile> projectile(m, "Projectile", "A projectile in flight.");
  projectile.def(py::init<>())
      .def_rw("id", &Projectile::id, "Increasing identifier, unique within the world.")
      .def_rw("launchTime", &Projectile::launchTime, "World time of the launch (s).")
      .def_rw("properties", &Projectile::properties, "The properties it was launched with.");
  vectorProperty(projectile, "position", &Projectile::position, "Position in the world frame (m).");
  vectorProperty(projectile, "velocity", &Projectile::velocity, "Velocity in the world frame (m/s).");
  vectorProperty(projectile, "launchPosition", &Projectile::launchPosition, "Position at the launch (m).");
  vectorProperty(projectile, "launchVelocity", &Projectile::launchVelocity, "Velocity at the launch (m/s).");

  py::class_<ProjectileHit> hit(m, "ProjectileHit", "A projectile striking an object.");
  hit.def(py::init<>())
      .def_rw("id", &ProjectileHit::id, "Increasing identifier, unique within the world.")
      .def_rw("projectileId", &ProjectileHit::projectileId, "Identifier of the projectile.")
      .def_rw("objectIndex", &ProjectileHit::objectIndex, "World index of the object hit.")
      .def_rw("localIdx", &ProjectileHit::localIdx, "Local body index of the body hit.")
      .def_rw("mass", &ProjectileHit::mass, "Mass of the projectile (kg).")
      .def_rw("time", &ProjectileHit::time, "World time of the impact (s).")
      .def_rw("detonated", &ProjectileHit::detonated, "The projectile detonated there.");
  vectorProperty(hit, "position", &ProjectileHit::position, "Point of impact in the world frame (m).");
  vectorProperty(hit, "normal", &ProjectileHit::normal, "Surface normal at the point of impact.");
  vectorProperty(hit, "velocity", &ProjectileHit::velocity, "Velocity of the projectile at impact (m/s).");
}
