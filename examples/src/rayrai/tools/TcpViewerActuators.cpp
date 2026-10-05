// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerActuators.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <imgui/imgui.h>

#include "raisim/object/ArticulatedSystem/MotorOperatingRegion.hpp"

namespace raisin::tcp_viewer
{
namespace
{

raisim::DcMotorParameters motorOf(const SelectedActuatorInfo& actuator) {
  raisim::DcMotorParameters motor;
  motor.resistance = actuator.resistance;
  motor.torqueConstant = actuator.torqueConstant;
  motor.velocityConstant = actuator.velocityConstant;
  motor.busVoltage = actuator.busVoltage;
  motor.peakTorque = actuator.peakTorque;
  return motor;
}

/// Torque that scales the plot: the peak torque, or the stall torque of a motor without one.
double torqueScale(const raisim::DcMotorParameters& motor, double busVoltage) {
  return std::isfinite(motor.peakTorque) ? motor.peakTorque :
                                           motor.torquePerVolt() * busVoltage;
}

ImU32 sampleColor(const ActuatorTraces::Sample& sample) {
  if (sample.outside) return IM_COL32(255, 70, 70, 255);
  if (sample.saturation == int32_t(raisim::MotorSaturation::PEAK_TORQUE))
    return IM_COL32(255, 175, 60, 230);
  if (sample.saturation == int32_t(raisim::MotorSaturation::VOLTAGE))
    return IM_COL32(205, 125, 255, 230);
  return IM_COL32(110, 175, 255, 200);
}

// Same drawing as rayrai_motor_operating_region: the region in the motor's torque-speed plane, the
// dashed Box-MOR and the trail of operating points. The speed axis is fixed by the highest bus
// voltage seen, so a lower voltage visibly shrinks the region.
void drawMotorPlot(const std::string& name, const raisim::DcMotorParameters& motor,
                   const ActuatorTraces::MotorTrace* trace, ImVec2 size) {
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImVec2 lo(origin.x + 4.f, origin.y + ImGui::GetTextLineHeightWithSpacing());
  const ImVec2 hi(origin.x + size.x - 4.f, origin.y + size.y - 4.f);

  const double perVolt = motor.torquePerVolt(), kv = motor.velocityConstant;
  const double axisVoltage = std::max(motor.busVoltage, trace ? double(trace->maxBusVoltage) : 0.);
  const double peak = torqueScale(motor, axisVoltage);
  const double xRange = 1.12 * kv * (axisVoltage + peak / perVolt);
  const double yRange = 1.4 * peak;
  const auto toScreen = [&](double speed, double torque) {
    const double x = std::clamp(speed / xRange, -1., 1.), y = std::clamp(torque / yRange, -1., 1.);
    return ImVec2(float(0.5 * (lo.x + hi.x) + 0.5 * x * (hi.x - lo.x)),
                  float(0.5 * (lo.y + hi.y) - 0.5 * y * (hi.y - lo.y)));
  };

  const ImU32 axisColor = IM_COL32(120, 125, 135, 120), labelColor = IM_COL32(150, 155, 165, 255);
  draw->AddRectFilled(lo, hi, IM_COL32(22, 25, 31, 235), 3.f);
  draw->AddLine(toScreen(-xRange, 0.), toScreen(xRange, 0.), axisColor);
  draw->AddLine(toScreen(0., -yRange), toScreen(0., yRange), axisColor);

  // EM-MOR: the bus-voltage band cut by the peak torque, a parallelogram. Without a peak torque the
  // band is cut by the plot instead.
  const double volt = motor.busVoltage;
  const double margin = peak / perVolt;
  const ImVec2 region[4] = {
      toScreen(kv * (-volt - margin), peak), toScreen(kv * (volt - margin), peak),
      toScreen(kv * (volt + margin), -peak), toScreen(kv * (-volt + margin), -peak)};
  draw->AddConvexPolyFilled(region, 4, IM_COL32(70, 190, 105, 55));
  draw->AddPolyline(region, 4, IM_COL32(95, 215, 125, 255), ImDrawFlags_Closed, 1.5f);

  // Box-MOR: peak torque and no-load speed, dashed.
  const double noLoad = motor.noLoadSpeed();
  const ImVec2 box[4] = {toScreen(-noLoad, peak), toScreen(noLoad, peak), toScreen(noLoad, -peak),
                         toScreen(-noLoad, -peak)};
  for (int side = 0; side < 4; ++side) {
    const ImVec2 a = box[side], b = box[(side + 1) % 4];
    for (float t = 0.f; t < 1.f; t += 0.04f)
      draw->AddLine(ImVec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t),
                    ImVec2(a.x + (b.x - a.x) * (t + 0.02f), a.y + (b.y - a.y) * (t + 0.02f)),
                    IM_COL32(210, 210, 210, 150));
  }

  if (trace) {
    for (size_t i = 0; i < trace->samples.size(); ++i) {
      const auto& sample = trace->samples[i];
      const ImVec2 p = toScreen(sample.speed, sample.torque);
      // The newest point is drawn larger so the current operating point stands out.
      const bool newest = i + 1 == trace->samples.size();
      const float r = sample.outside || newest ? 2.5f : 1.2f;
      draw->AddRectFilled(ImVec2(p.x - r, p.y - r), ImVec2(p.x + r, p.y + r), sampleColor(sample));
    }
  }

  char text[160];
  const double maxOutside = trace ? trace->maxOutside : 0.;
  const double saturated = trace && trace->sampleCount ?
    double(trace->saturatedCount) / double(trace->sampleCount) : 0.;
  std::snprintf(text, sizeof(text), "%s  max out %.2f%%  sat %.0f%%", name.c_str(),
                100. * maxOutside / peak, 100. * saturated);
  draw->AddText(origin, trace && trace->outsideCount ? IM_COL32(255, 110, 110, 255) :
                                                       IM_COL32(225, 228, 235, 255), text);
  std::snprintf(text, sizeof(text), "%.0f rad/s", xRange);
  draw->AddText(ImVec2(hi.x - ImGui::CalcTextSize(text).x - 2.f,
                       hi.y - ImGui::GetTextLineHeight() - 1.f), labelColor, text);
  std::snprintf(text, sizeof(text), "%.1f Nm", yRange);
  draw->AddText(ImVec2(0.5f * (lo.x + hi.x) + 3.f, lo.y + 1.f), labelColor, text);
  ImGui::Dummy(size);
}

} // namespace

void ActuatorTraces::update(const SelectedObjectInfo& info, double serverTime) {
  if (info.tag != tag_ || serverTime < lastTime_) {
    reset();
    tag_ = info.tag;
  }
  if (serverTime == lastTime_) return;
  lastTime_ = serverTime;
  for (const auto& actuator : info.actuators) {
    auto it = std::find_if(traces_.begin(), traces_.end(),
                           [&](const auto& entry) { return entry.first == actuator.name; });
    if (it == traces_.end()) {
      traces_.emplace_back(actuator.name, MotorTrace());
      it = std::prev(traces_.end());
    }
    MotorTrace& trace = it->second;
    const auto motor = motorOf(actuator);
    if (!motor.isValid()) continue;
    // The torque is constant over a step while the speed changes, so compare it with the region
    // at the step-averaged speed.
    const double violation = raisim::motorOperatingRegionViolation(
        motor, actuator.averageMotorSpeed, actuator.motorTorque);
    Sample sample;
    sample.time = serverTime;
    sample.speed = actuator.averageMotorSpeed;
    sample.torque = actuator.motorTorque;
    sample.saturation = actuator.saturation;
    sample.outside = violation > kOutsideTolerance * torqueScale(motor, actuator.busVoltage);
    trace.samples.push_back(sample);
    const auto stale = std::find_if(trace.samples.begin(), trace.samples.end(),
        [&](const Sample& kept) { return kept.time >= serverTime - kTrailSeconds; });
    trace.samples.erase(trace.samples.begin(), stale);
    ++trace.sampleCount;
    trace.saturatedCount += actuator.saturation != int32_t(raisim::MotorSaturation::NONE);
    trace.outsideCount += sample.outside;
    trace.maxOutside = std::max(trace.maxOutside, violation);
    trace.maxBusVoltage = std::max(trace.maxBusVoltage, actuator.busVoltage);
  }
}

const ActuatorTraces::MotorTrace* ActuatorTraces::find(const std::string& actuatorName) const {
  const auto it = std::find_if(traces_.begin(), traces_.end(),
                               [&](const auto& entry) { return entry.first == actuatorName; });
  return it == traces_.end() ? nullptr : &it->second;
}

void ActuatorTraces::reset() {
  tag_ = 0;
  lastTime_ = -1.0;
  traces_.clear();
}

void drawObjectActuators(const SelectedObjectInfo& info, ActuatorTraces& traces) {
  double worst = 0.;
  size_t outside = 0, samples = 0;
  for (const auto& actuator : info.actuators) {
    const auto motor = motorOf(actuator);
    const auto* trace = traces.find(actuator.name);
    if (!trace || !motor.isValid()) continue;
    worst = std::max(worst, trace->maxOutside / torqueScale(motor, actuator.busVoltage));
    outside += trace->outsideCount;
    samples += trace->sampleCount;
  }
  ImGui::Text("%s: worst point %.2f%% of peak torque outside",
              info.motorRegionEnforced ? "EM-MOR" : "Box-MOR", 100. * worst);
  ImGui::TextColored(outside ? ImVec4(1.f, 0.45f, 0.45f, 1.f) : ImVec4(0.55f, 0.9f, 0.6f, 1.f),
                     "%zu of %zu samples outside by > %.0f%%", outside, samples,
                     100. * ActuatorTraces::kOutsideTolerance);
  if (ImGui::Button("Reset statistics")) traces.reset();
  ImGui::TextDisabled("Green: EM-MOR. Dashed: Box-MOR (peak torque, no-load speed).\n"
                      "Points: blue inside, orange at the peak torque,\n"
                      "purple at the voltage limit, red outside.");

  for (size_t a = 0; a < info.actuators.size(); ++a) {
    const auto& actuator = info.actuators[a];
    ImGui::PushID(static_cast<int>(a));
    std::string header = actuator.name;
    if (!actuator.model.empty()) header += "  (" + actuator.model + ")";
    if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
      const ImVec4 valueColor(0.85f, 0.9f, 1.0f, 1.0f);
      ImGui::TextUnformatted(actuator.joint.c_str());
      ImGui::SameLine();
      ImGui::TextColored(valueColor, "velocity %.3f  torque %.3f  gear %.3g", actuator.jointVelocity,
                         actuator.jointTorque, actuator.gearRatio);
      const float width = ImGui::GetContentRegionAvail().x;
      const ImVec2 cell(width, std::round(width * 0.62f));
      const auto motor = motorOf(actuator);
      if (motor.isValid()) {
        drawMotorPlot("motor", motor, traces.find(actuator.name), cell);
      } else {
        ImGui::TextDisabled("invalid motor parameters");
      }
    }
    ImGui::PopID();
  }
}

} // namespace raisin::tcp_viewer
