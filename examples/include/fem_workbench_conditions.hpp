// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once
#include <array>
#include <string>
#include <vector>

#include "raisim/fem/LinearElasticity.hpp"
namespace fem_workbench {
inline constexpr double kSmoothSurfaceCosine = .81915204428899178968;
using raisim::Vec;
using raisim::fem::LinearElasticity;
enum class Kind {
  Fixed,
  Displacement,
  Force,
  Traction,
  Pressure,
  Gravity,
  Directional,
  Roller,
  Slider,
  Bearing,
  BoltHole,
  BoltConnector,
  Remote,
  Ball,
  Spring,
  Tie
};
inline const char* kindName(Kind kind) {
  constexpr const char* names[] = {"Fixed support",
                                   "Axis displacement",
                                   "Total force",
                                   "Surface traction",
                                   "Pressure",
                                   "Gravity",
                                   "Directional / symmetry",
                                   "Normal roller",
                                   "Slider / guide",
                                   "Cylindrical bearing",
                                   "Grounded bolt hole",
                                   "Bolt connector",
                                   "Remote displacement",
                                   "Ball / spherical support",
                                   "Elastic foundation",
                                   "Bonded surface tie"};
  return names[static_cast<int>(kind)];
}
struct Condition {
  std::string name;
  Kind kind = Kind::Fixed;
  bool enabled = true;
  std::array<bool, 3> axes{true, true, true};
  Vec<3> value{0., 0., 0.};
  std::vector<unsigned int> nodes;
  std::vector<size_t> faces;
  Vec<3> axis{0., 0., 1.}, center{0., 0., 0.}, rotation{0., 0., 0.};
  double stiffness = 1e6, tolerance = 1e-4;
  LinearElasticity::BoltProperties bolt;
  std::vector<unsigned int> targetNodes;
  std::vector<size_t> targetFaces;
};

}  // namespace fem_workbench
