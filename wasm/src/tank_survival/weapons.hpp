// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#pragma once

#include "common.hpp"

namespace tank {

enum class WeaponId { AP, HE, MG, Missile, Artillery };
inline constexpr int kWeaponCount = 5;

// Armor penetration is in millimetres of rolled steel at the muzzle; kinetic rounds lose it with speed.
struct WeaponSpec {
  const char* name;
  const char* shortName;
  const char* role;
  double speed, mass, radius, dragCoefficient;
  double charge;
  double penetration, damage;
  double reload;
  bool kinetic;
  glm::vec4 color;
};

inline const std::array<WeaponSpec, kWeaponCount> kWeapons{{
    {"APFSDS", "AP", "Armor-piercing dart: flat, fast, beats armor head-on", 1500, 4.5, 0.03, 0.15, 0.0, 230, 60, 3.0,
     true, {1.0f, 0.86f, 0.35f, 1.0f}},
    {"HE-FRAG", "HE", "High explosive: blast and craters, wrecks light vehicles", 700, 15.0, 0.06, 0.25, 2.2, 45, 30, 3.0,
     false, {1.0f, 0.45f, 0.2f, 1.0f}},
    {"Coaxial MG", "MG", "7.62 mm machine gun: shreds scouts, bounces off tanks", 850, 0.0095, 0.0039, 0.3, 0.0, 14, 30.0,
     0.085, true, {0.85f, 0.85f, 0.85f, 1.0f}},
    {"ATGM", "Missile", "Wire-guided missile: follows your aim, tandem warhead", 150, 25.0, 0.08, 0.3, 2.5, 520, 95, 5.0,
     false, {0.4f, 0.85f, 1.0f, 1.0f}},
    {"Artillery strike", "Artillery", "Six 155 mm shells on the aimed point after 3 s", 0, 40.0, 0.08, 0.0, 8.0, 80, 60, 45.0,
     false, {0.55f, 1.0f, 0.45f, 1.0f}},
}};

inline const WeaponSpec& weapon(WeaponId id) { return kWeapons[size_t(id)]; }

// The direction to fire a round from `origin` so that it passes through `target` as the world flies it:
// raisim's projectile step (gravity, and the drag of air of `density` moving at `wind`), replayed at the
// game's time step. The straight line is corrected, a few times, by where a round fired along the last
// guess passes the target. A target out of reach keeps the last guess.
inline Vec3 launchDirection(const Vec3& origin, const Vec3& target, const WeaponSpec& round, const Vec3& wind,
                            double density = 1.205) {
  const Vec3 line = target - origin;
  const double range = line.norm();
  if (!(range > 1e-6) || !(round.speed > 0.0)) return line.normalized();
  const Vec3 axis = line / range, gravity(0.0, 0.0, -kGravity);
  const double drag = 0.5 * density * round.dragCoefficient * kPi * round.radius * round.radius / round.mass;
  Vec3 aimAt = target, direction = axis;
  for (int iteration = 0; iteration < 6; ++iteration) {
    direction = (aimAt - origin).normalized();
    Vec3 x = origin, v = direction * round.speed, passed = origin;
    bool reached = false;
    for (int step = 0; step < 20000; ++step) {
      const Vec3 relative = v - wind;
      const Vec3 v1 = wind + (relative + gravity * kTimeStep) / (1.0 + drag * relative.norm() * kTimeStep);
      const Vec3 x1 = x + 0.5 * kTimeStep * (v + v1);
      const double before = (x - origin).dot(axis), after = (x1 - origin).dot(axis);
      if (after >= range) {
        passed = x + (after > before ? (range - before) / (after - before) : 1.0) * (x1 - x);
        reached = true;
        break;
      }
      if (after < before) break;
      x = x1;
      v = v1;
    }
    if (!reached) break;
    const Vec3 miss = target - passed;
    if (miss.norm() < 1e-4) break;
    aimAt += miss - miss.dot(axis) * axis;
  }
  return direction;
}

}  // namespace tank
