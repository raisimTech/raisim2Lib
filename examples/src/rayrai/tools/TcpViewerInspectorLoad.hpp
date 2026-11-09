// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once
#include <string>
#include <vector>

#include "raisim/object/ArticulatedSystem/JointAndBodies.hpp"

namespace raisim {
class ArticulatedSystem;
class Object;
class World;
} // namespace raisim

namespace raisin::tcp_viewer {
// A robot file the inspector loaded into its world.
struct InspectorModel {
  // The file's root element is <mujoco>; URDF otherwise.
  bool mjcf = false;
  // Null when the file held no articulated system; the load then added nothing.
  raisim::ArticulatedSystem* system = nullptr;
  // Everything else the load added, such as an MJCF file's ground.
  std::vector<raisim::Object*> sideObjects;
};

// Loads a URDF or MJCF file for the inspector, which only poses the model and
// draws collision bodies from their mesh files. URDF collision meshes therefore
// load as convex hulls: building a hull is faster than building the BVH of a
// high-poly triangle mesh. A file RaiSim cannot load, including one that stops
// RaiSim with a fatal error, throws std::exception and leaves the world without
// any object of the load.
InspectorModel loadInspectorModel(raisim::World& world, const std::string& path);

// A movable joint of an inspected model, as its pose slider edits it.
struct InspectorJoint {
  std::string name;
  raisim::Joint::Type type = raisim::Joint::Type::FIXED;
  int gcOffset = 0;       // start index into the ArticulatedSystem GC vector
  int gcDim = 0;          // 1 (revolute/prismatic) / 4 (spherical) / 7 (floating)
  double minLimit = 0.0;  // for revolute/prismatic
  double maxLimit = 0.0;
  bool hasLimits = false;
};

// The movable joints of `system` in joint order, with their generalized-coordinate ranges and
// the position limits of the revolute and prismatic ones (getJointLimits() is indexed by
// generalized velocity).
std::vector<InspectorJoint> inspectorJoints(const raisim::ArticulatedSystem& system);
} // namespace raisin::tcp_viewer
