// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>

#include "rayrai/example_common.hpp"

#ifndef TANK_SURVIVAL_ASSETS
#define TANK_SURVIVAL_ASSETS "/assets"
#endif

namespace tank {

using Viewer = raisin::RayraiWindow;
using Vec3 = Eigen::Vector3d;
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTimeStep = 0.002;
inline constexpr double kGravity = 9.81;
// The collision group of the trees in the arena: the tanks' turrets and guns swing through them.
inline constexpr raisim::CollisionGroup kTreeCollisionGroup = raisim::CollisionGroup(1) << 40;

inline raisim::Vec<3> vec(const Vec3& value) {
  raisim::Vec<3> v;
  v.e() = value;
  return v;
}

inline double wrapAngle(double angle) { return std::remainder(angle, 2.0 * kPi); }

inline ImU32 rgba(const glm::vec4& c, float alpha = 1.0f) {
  return IM_COL32(int(c.r * 255), int(c.g * 255), int(c.b * 255), int(c.a * alpha * 255));
}

inline std::string readText(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Cannot read " + path);
  std::stringstream text;
  text << file.rdbuf();
  return text.str();
}

inline double uniform(std::mt19937& random, double low, double high) {
  return std::uniform_real_distribution<double>(low, high)(random);
}

}  // namespace tank
