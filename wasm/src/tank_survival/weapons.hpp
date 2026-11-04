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
    {"Coaxial MG", "MG", "7.62 mm machine gun: shreds scouts, bounces off tanks", 850, 0.0095, 0.0039, 0.3, 0.0, 14, 2.5,
     0.085, true, {0.85f, 0.85f, 0.85f, 1.0f}},
    {"ATGM", "Missile", "Wire-guided missile: follows your aim, tandem warhead", 150, 25.0, 0.08, 0.3, 2.5, 520, 95, 5.0,
     false, {0.4f, 0.85f, 1.0f, 1.0f}},
    {"Artillery strike", "Artillery", "Six 155 mm shells on the aimed point after 3 s", 0, 40.0, 0.08, 0.0, 8.0, 80, 60, 45.0,
     false, {0.55f, 1.0f, 0.45f, 1.0f}},
}};

inline const WeaponSpec& weapon(WeaponId id) { return kWeapons[size_t(id)]; }

}  // namespace tank
