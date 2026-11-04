// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#pragma once

#include "vehicles.hpp"

namespace tank {

enum class Terrain { Hills, Dunes, Town, Ridges, Fortress };

struct Squad {
  Role role;
  int count;
};

struct LevelSpec {
  const char* name;
  const char* subtitle;
  const char* briefing;
  Terrain terrain;
  Vec3 wind;
  // Weather: the sun's azimuth and elevation (degrees), cloud cover, visibility (m), haze (turbidity),
  // snow cover, rain and lightning (strikes per minute), and the color of the haze.
  float sunAzimuth, sunElevation, clouds, visibility, turbidity, snow, rain, lightning;
  glm::vec3 fogColor;
  double enemyAccuracy, enemyReload;
  Vec3 playerStart;
  Vec3 enemyCenter;
  double enemyRadius;
  std::vector<std::vector<Squad>> waves;
  bool endless;
};

inline const std::vector<LevelSpec>& levels() {
  static const std::vector<LevelSpec> list{
      {"Green Valley", "Training ground, clear noon",
       "Scouts and Raiders probe the valley. Use MG on scouts and AP on tanks.", Terrain::Hills, {3.0, 1.0, 0.0}, 210.f, 52.f, 0.25f, 9000.f, 2.2f, 0.f, 0.f, 0.f, {0.66f, 0.72f, 0.78f}, 1.5, 1.25, {0, 0, 0}, {0, 0, 0}, 0.0,
       {{{Role::Scout, 3}}, {{Role::Medium, 2}, {Role::Scout, 2}}, {{Role::Medium, 3}}}, false},
      {"Dust Storm", "Desert dunes, 12 m/s crosswind",
       "Strong wind drifts slow rounds: lead with HE, or use flat AP. Tank destroyers snipe from range; mortars "
       "mark their target on the ground.",
       Terrain::Dunes, {11.0, 4.0, 0.0}, 120.f, 34.f, 0.05f, 420.f, 6.0f, 0.f, 0.f, 0.f, {0.80f, 0.68f, 0.52f}, 1.25, 1.1, {0, 0, 0},
       {0, 0, 0}, 0.0,
       {{{Role::Scout, 2}, {Role::Medium, 2}},
        {{Role::Destroyer, 2}, {Role::Medium, 1}},
        {{Role::Artillery, 2}, {Role::Medium, 2}},
        {{Role::Destroyer, 2}, {Role::Scout, 3}}},
       false},
      {"Ruined Town", "Street fighting under an overcast sky",
       "Brutes hold the streets. Their front armor stops AP: flank them or hit them with missiles.", Terrain::Town,
       {2.0, -3.0, 0.0}, 250.f, 26.f, 0.85f, 1100.f, 3.0f, 0.f, 0.35f, 0.f, {0.60f, 0.62f, 0.66f}, 1.15, 1.0, {0, 0, 0}, {0, 0, 0}, 0.0,
       {{{Role::Medium, 3}},
        {{Role::Heavy, 1}, {Role::Scout, 3}},
        {{Role::Heavy, 2}, {Role::Medium, 2}},
        {{Role::Destroyer, 2}, {Role::Artillery, 1}, {Role::Medium, 2}}},
       false},
      {"Frozen Ridge", "Snow ridges in the late afternoon",
       "Mixed armor and mortars on high ground. Use the ridges as cover and smoke when you cross open snow.",
       Terrain::Ridges, {-6.0, 5.0, 0.0}, 255.f, 17.f, 0.3f, 2200.f, 3.0f, 0.85f, 0.f, 0.f, {0.66f, 0.64f, 0.76f}, 1.05, 0.95, {0, 0, 0},
       {0, 0, 0}, 0.0,
       {{{Role::Medium, 3}, {Role::Scout, 2}},
        {{Role::Heavy, 2}, {Role::Artillery, 2}},
        {{Role::Destroyer, 3}, {Role::Medium, 2}},
        {{Role::Heavy, 2}, {Role::Medium, 3}, {Role::Artillery, 1}}},
       false},
      {"Iron Fortress", "Assault the Behemoth's stronghold",
       "Break into the fortress and destroy the Behemoth. Its rear armor is its weakness; call artillery on its "
       "escorts.",
       Terrain::Fortress, {4.0, 4.0, 0.0}, 300.f, 18.f, 0.7f, 1300.f, 4.0f, 0.f, 0.f, 1.5f, {0.56f, 0.44f, 0.42f}, 1.0, 0.9, {0, -70, 0},
       {0, 12, 0}, 22.0,
       {{{Role::Medium, 3}, {Role::Destroyer, 1}},
        {{Role::Heavy, 2}, {Role::Artillery, 2}},
        {{Role::Boss, 1}, {Role::Medium, 2}}},
       false},
      {"Survival", "Endless waves of every kind",
       "Hold out as long as you can. Every fifth wave brings a T-14 Armata.", Terrain::Hills, {5.0, 2.5, 0.0}, 160.f, 42.f, 0.4f, 7000.f, 2.5f, 0.f, 0.f, 0.f, {0.68f, 0.72f, 0.78f}, 1.2, 1.1, {0, 0, 0}, {0, 0, 0}, 0.0, {}, true},
  };
  return list;
}

// The enemies of an endless wave.
inline std::vector<Squad> endlessWave(int wave) {
  std::vector<Squad> squads{{Role::Medium, 1 + wave / 2}, {Role::Scout, 1 + wave % 3}};
  if (wave >= 2) squads.push_back({Role::Destroyer, wave / 3});
  if (wave >= 3) squads.push_back({Role::Heavy, (wave - 1) / 3});
  if (wave >= 4) squads.push_back({Role::Artillery, std::min(3, wave / 4)});
  if (wave % 5 == 0) squads.push_back({Role::Boss, 1});
  return squads;
}

inline constexpr double kArena = 86.0;
inline constexpr double kExtent = 220.0;

// Small-scale relief: octaves of sine waves at incommensurate angles and phases, from 24 m swells to
// 1.5 m bumps (sand and town ground are smoother).
inline double roughness(Terrain terrain, double x, double y) {
  struct Octave {
    double cosine, sine, phase;
  };
  static const std::array<Octave, 7> octaves = [] {
    std::array<Octave, 7> table{};
    for (int octave = 0; octave < 7; ++octave) {
      const double angle = 2.399963 * octave + 0.7;
      table[size_t(octave)] = {std::cos(angle), std::sin(angle), 1.7 * octave * octave + 0.3};
    }
    return table;
  }();
  const double amplitude = terrain == Terrain::Dunes ? 0.35 : terrain == Terrain::Town ? 0.12 : 1.0;
  double h = 0.0, a = 0.55, wavelength = 24.0;
  for (const auto& [cosine, sine, phase] : octaves) {
    const double u = x * cosine + y * sine, v = -x * sine + y * cosine;
    h += a * std::sin(2.0 * kPi * u / wavelength + phase) * std::cos(2.0 * kPi * v / (1.37 * wavelength) + 0.5 * phase);
    a *= 0.52;
    wavelength *= 0.58;
  }
  return amplitude * h;
}

// A river crossing the arena along +x (none in the desert); its middle wanders around y = offset.
struct River {
  bool present = false;
  double offset = 0.0, amplitude = 0.0;
};

inline River riverOf(Terrain terrain) {
  switch (terrain) {
    case Terrain::Hills: return {true, 30.0, 5.0};
    case Terrain::Ridges: return {true, -34.0, 5.0};
    case Terrain::Fortress: return {true, -44.0, 4.0};
    case Terrain::Town: return {true, 24.0, 0.0};
    default: return {};
  }
}

// The river's bed: a flat bottom kRiverBottom to each side of the middle, kRiverDepth under the water,
// with banks rising kRiverBank metres per metre (steeper in the town's canal). The water falls
// kRiverSlope metres per metre downstream.
inline constexpr double kRiverDepth = 1.1, kRiverBottom = 2.5, kRiverBank = 0.55, kRiverSlope = 0.001;
inline constexpr double kRiverFlow = 7.0, kRiverSpeed = 1.1;

inline double riverY(const River& river, double x) { return river.offset + river.amplitude * std::sin(0.031 * x + 0.6); }
inline double riverSurface(double x) { return -0.75 - kRiverSlope * x; }
inline double riverBank(Terrain terrain) { return terrain == Terrain::Town ? 1.6 : kRiverBank; }
// Half the width of the water's surface.
inline double riverHalfWidth(Terrain terrain) { return kRiverBottom + kRiverDepth / riverBank(terrain); }

// How far (x, y) is from the middle of the river (across it), or infinity without one.
inline double riverDistance(Terrain terrain, double x, double y) {
  const River river = riverOf(terrain);
  return river.present ? std::abs(y - riverY(river, x)) : std::numeric_limits<double>::infinity();
}

// The ground h with the river's bed cut into it; the ground beside the river is raised above the
// water so that it stays in its bed.
inline double carveRiver(Terrain terrain, double x, double y, double h) {
  const double d = riverDistance(terrain, x, y);
  if (!std::isfinite(d)) return h;
  const double surface = riverSurface(x);
  const double bed = surface - kRiverDepth + std::max(0.0, d - kRiverBottom) * riverBank(terrain);
  const double levee = surface + 0.3 + 0.02 * d - 0.5 * std::max(0.0, d - 18.0);
  return std::min(std::max(h, levee), bed);
}

// The hills around the arena, which nothing reaches: they rise from the flat ground outside the walls
// (zero at the edge of the heightmap) toward the horizon.
inline double outerHeight(Terrain terrain, double x, double y) {
  const double r = std::max(0.0, std::max(std::abs(x), std::abs(y)) - 0.5 * kExtent);
  const double rise = terrain == Terrain::Ridges ? 0.2 : terrain == Terrain::Hills ? 0.1 : terrain == Terrain::Fortress ? 0.08
                    : terrain == Terrain::Dunes ? 0.05 : 0.03;
  const double swell = 0.6 + 0.4 * std::sin(0.013 * x + 0.9) * std::cos(0.011 * y - 0.3) + 0.2 * std::sin(0.031 * x - 0.027 * y);
  double h = rise * r * swell;
  if (terrain == Terrain::Dunes) h += std::min(1.0, r / 40.0) * 4.0 * std::sin(0.03 * x + 0.8 * std::sin(0.012 * y));
  return carveRiver(terrain, x, y, h + std::min(1.0, r / 25.0) * 2.5 * roughness(terrain, x, y));
}

// The ground of a level; the middle (or the player's start) is gentler and the edges fall to the arena
// walls.
inline double groundShape(Terrain terrain, double x, double y) {
  double h = 0.0, flatRadius = 14.0;
  Vec3 flatCenter(0, 0, 0);
  switch (terrain) {
    case Terrain::Hills:
      h = 2.0 * std::sin(0.05 * x) * std::cos(0.043 * y) + 0.8 * std::sin(0.12 * x + 0.7) * std::sin(0.1 * y + 0.3) +
          0.25 * std::sin(0.33 * x) * std::cos(0.29 * y);
      break;
    case Terrain::Dunes:
      h = 3.0 * std::sin(0.075 * x + 0.8 * std::sin(0.031 * y)) * (0.65 + 0.35 * std::cos(0.047 * y)) +
          0.6 * std::sin(0.21 * y + 0.4 * x * 0.05) + 1.5;
      break;
    case Terrain::Town:
      h = 0.25 * std::sin(0.08 * x) * std::cos(0.07 * y);
      flatRadius = 1e9;
      break;
    case Terrain::Ridges:
      h = 5.5 * std::pow(std::abs(std::sin(0.042 * x + 0.35 * std::sin(0.05 * y))), 1.5) +
          1.2 * std::sin(0.13 * y) * std::cos(0.09 * x);
      break;
    case Terrain::Fortress:
      h = 1.6 * std::sin(0.06 * x) * std::cos(0.05 * y) + 0.5 * std::sin(0.17 * x + 0.3);
      flatRadius = 40.0;
      flatCenter = Vec3(0, 12, 0);
      break;
  }
  const double r2 = (x - flatCenter.x()) * (x - flatCenter.x()) + (y - flatCenter.y()) * (y - flatCenter.y());
  const double flat = flatRadius > 1e8 ? 1.0 : std::exp(-r2 / (2.0 * flatRadius * flatRadius));
  const double edge = std::clamp((kArena - 3.0 - std::max(std::abs(x), std::abs(y))) / 12.0, 0.0, 1.0);
  const double fade = edge * edge * (3.0 - 2.0 * edge);
  return carveRiver(terrain, x, y, (h * (1.0 - flat) + roughness(terrain, x, y) * (1.0 - 0.6 * flat)) * fade);
}

}  // namespace tank
