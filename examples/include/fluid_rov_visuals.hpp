// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#ifndef RAISIM_EXAMPLES_INCLUDE_FLUID_ROV_VISUALS_HPP_
#define RAISIM_EXAMPLES_INCLUDE_FLUID_ROV_VISUALS_HPP_

#include <array>
#include <cmath>
#include <mutex>
#include <string>

#include "fluid_rov.hpp"
#include "raisim/RaisimServer.hpp"

namespace raisim_examples::fluid_rov {

class VehicleVisuals {
 public:
  VehicleVisuals(Scene& scene, raisim::RaisimServer& server, const std::string& mesh)
      : scene_(scene), shell_(server.addVisualMesh("rov_shell", mesh)),
        current_(server.addVisualPolyLine("current")), tether_(server.addVisualPolyLine("rov_tether")) {
    current_->color = {0.15, 0.8, 0.9, 1.0};
    current_->width = 0.014;
    current_->points.resize(5);
    tether_->color = {0.95, 0.4, 0.06, 1.0};
    tether_->width = 0.009;
    tether_->points.resize(49);
    update();
  }

  // Call before launching the server or inside integrateWorldThreadSafe().
  void update() {
    const Eigen::Vector3d position = scene_.vehicle()->getPosition();
    const Eigen::Matrix3d rotation = scene_.vehicle()->getRotationMatrix();
    const auto q = scene_.vehicle()->getQuaternion();
    {
      std::lock_guard<raisim::Visuals> lock(*shell_);
      shell_->setPosition(position);
      shell_->setOrientation(q[0], q[1], q[2], q[3]);
    }
    {
      std::lock_guard<raisim::PolyLine> lock(*current_);
      const Eigen::Vector3d flow = scene_.current().normalized();
      const Eigen::Vector3d side(-flow[1], flow[0], 0.0);
      const Eigen::Vector3d start(-0.85, -0.4, kSurface - kDepth + 0.25);
      const Eigen::Vector3d tip = start + 0.65 * flow;
      const std::array<Eigen::Vector3d, 5> points{
          start, tip, tip - 0.16 * flow + 0.08 * side, tip, tip - 0.16 * flow - 0.08 * side};
      for (size_t i = 0; i < points.size(); ++i)
        current_->points[i] = {points[i][0], points[i][1], points[i][2]};
    }
    {
      std::lock_guard<raisim::PolyLine> lock(*tether_);
      const Eigen::Vector3d start = position + rotation * Eigen::Vector3d(-0.2, 0.0, 0.275);
      const Eigen::Vector3d end(-2.8, 2.0, kSurface + 0.25);
      for (size_t i = 0; i < tether_->points.size(); ++i) {
        const double t = static_cast<double>(i) / (tether_->points.size() - 1);
        Eigen::Vector3d p = (1.0 - t) * start + t * end;
        p[0] -= 0.6 * std::sin(kPi * t);
        p[2] -= 0.35 * std::sin(kPi * t);
        tether_->points[i] = {p[0], p[1], p[2]};
      }
    }
  }

 private:
  Scene& scene_;
  raisim::Visuals* shell_;
  raisim::PolyLine* current_;
  raisim::PolyLine* tether_;
};

}

#endif
