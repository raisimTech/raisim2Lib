// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once
#include <array>
#include <string>
#include <memory>
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
  Tie,
  Hinge,
  BallJoint,
  SliderJoint,
  RigidLink
};
inline bool isConnection(Kind kind) {
  return kind == Kind::Tie || kind == Kind::BoltConnector ||
         static_cast<int>(kind) >= static_cast<int>(Kind::Hinge);
}
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
                                   "Bonded surface tie",
                                   "Bearing / pin joint",
                                   "Ball joint",
                                   "Slider joint",
                                   "Rigid link"};
  return names[static_cast<int>(kind)];
}
enum class ForceDirection { Vector, FaceNormal, ReferenceNormal };
struct SelectionAnchor;
struct Condition {
  std::string name;
  Kind kind = Kind::Fixed;
  ForceDirection direction = ForceDirection::Vector;
  bool enabled = true;
  std::array<bool, 3> axes{true, true, true};
  Vec<3> value{0., 0., 0.};
  std::vector<unsigned int> nodes;
  std::vector<size_t> faces;
  Vec<3> axis{0., 0., 1.}, center{0., 0., 0.}, rotation{0., 0., 0.};
  double stiffness = 1e6, tolerance = 1e-4;
  /// Glue: lap shear strength of the adhesive; bolt: yield strength of the
  /// bolt. In Pa, for the connection's factor of safety; zero is unknown.
  double strength = 0.;
  LinearElasticity::BoltProperties bolt;
  std::vector<unsigned int> targetNodes;
  std::vector<size_t> targetFaces;
  std::vector<int> cadFaces, targetCadFaces;
  std::shared_ptr<const SelectionAnchor> primaryGeometry, targetGeometry;
};

}  // namespace fem_workbench
