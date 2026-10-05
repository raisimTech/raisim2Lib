// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rayrai/TcpRemoteSceneState.hpp"

namespace raisin::tcp_viewer
{

/**
 * Recent motor operating points of the actuators of a selected articulated system, one sample per
 * server update, for the operating-region plots of the Actuators tab. Owned by a viewer pane.
 */
class ActuatorTraces {
 public:
  /// Seconds of simulation time kept as the trail of each plot.
  static constexpr double kTrailSeconds = 1.0;
  /// Points outside the region by more than this fraction of the peak torque are flagged.
  static constexpr double kOutsideTolerance = 0.01;

  struct Sample {
    double time = 0.0;
    float speed = 0.0f;
    float torque = 0.0f;
    int32_t saturation = 0;
    bool outside = false;
  };

  struct MotorTrace {
    std::vector<Sample> samples; ///< Oldest first, within kTrailSeconds of the newest.
    size_t sampleCount = 0;      ///< All samples since the selection or the last reset.
    size_t saturatedCount = 0;
    size_t outsideCount = 0;
    double maxOutside = 0.0;     ///< Largest distance outside the region [Nm].
    float maxBusVoltage = 0.0f;  ///< Fixes the speed axis so a voltage drop shrinks the region.
  };

  /// Records the operating points of @p info if @p serverTime is a new server update.
  void update(const SelectedObjectInfo& info, double serverTime);
  /// Trace of the motor of actuator @p actuatorName of the current selection, or nullptr.
  const MotorTrace* find(const std::string& actuatorName) const;
  void reset();

 private:
  uint32_t tag_ = 0;
  double lastTime_ = -1.0;
  std::vector<std::pair<std::string, MotorTrace>> traces_;
};

/**
 * Actuators tab of the selected-object panel, laid out like the rayrai_motor_operating_region
 * example: per actuator the velocity and actuator torque of its joint, and its motor's operating
 * region in the torque-speed plane (green), the Box-MOR of peak torque and no-load speed (dashed)
 * and the operating points of the last second.
 */
void drawObjectActuators(const SelectedObjectInfo& info, ActuatorTraces& traces);

} // namespace raisin::tcp_viewer
