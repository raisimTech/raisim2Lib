// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerSignals.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>

#include "raisim/object/ArticulatedSystem/JointAndBodies.hpp"

namespace raisin::tcp_viewer
{

namespace
{

int32_t jointGvDim(int32_t jointType) {
  switch (static_cast<raisim::Joint::Type>(jointType)) {
    case raisim::Joint::Type::FIXED: return 0;
    case raisim::Joint::Type::REVOLUTE:
    case raisim::Joint::Type::PRISMATIC: return 1;
    case raisim::Joint::Type::SPHERICAL: return 3;
    case raisim::Joint::Type::FLOATING: return 6;
  }
  return -1;
}

/** Joint names are user-controlled, so keep them safe for keys and CSV headers. */
std::string sanitizeChannelToken(const std::string& token) {
  std::string sanitized;
  sanitized.reserve(token.size());
  for (const char c : token) {
    const unsigned char raw = static_cast<unsigned char>(c);
    if (std::isalnum(raw) != 0 || c == '_' || c == '-' || c == '.') {
      sanitized.push_back(c);
    } else {
      sanitized.push_back('_');
    }
  }
  if (sanitized.empty()) {
    sanitized = "unnamed";
  }
  return sanitized;
}

float channelValue(const SignalChannelDesc& channel, const SignalSampleInputs& inputs,
                   const std::vector<int32_t>& jointGvOffsets);

} // namespace

std::vector<int32_t> deriveJointGvOffsets(const std::vector<int32_t>& jointTypes, int32_t dof) {
  std::vector<int32_t> offsets;
  offsets.reserve(jointTypes.size());
  int32_t running = 0;
  for (const int32_t jointType : jointTypes) {
    const int32_t dimension = jointGvDim(jointType);
    if (dimension < 0) {
      return {};
    }
    offsets.push_back(running);
    running += dimension;
  }
  if (running != dof) {
    return {};
  }
  return offsets;
}

std::vector<SignalChannelDesc> buildSignalChannels(
    bool isArticulated, bool hasContactTags, const SelectedObjectInfo* selectedInfo) {
  std::vector<SignalChannelDesc> channels;
  channels.push_back({kSignalChannelLinearSpeed, "Linear speed (m/s)",
                      SignalChannelScope::SceneWide, "%.3f m/s"});
  channels.push_back({kSignalChannelAngularSpeed, "Angular speed (rad/s)",
                      SignalChannelScope::SceneWide, "%.3f rad/s"});
  channels.push_back({"pos.x", "Position X (m)", SignalChannelScope::SceneWide, "%.3f m"});
  channels.push_back({"pos.y", "Position Y (m)", SignalChannelScope::SceneWide, "%.3f m"});
  channels.push_back({"pos.z", "Position Z (m)", SignalChannelScope::SceneWide, "%.3f m"});
  channels.push_back({"vel.x", "Linear velocity X (m/s)", SignalChannelScope::SceneWide,
                      "%.3f m/s"});
  channels.push_back({"vel.y", "Linear velocity Y (m/s)", SignalChannelScope::SceneWide,
                      "%.3f m/s"});
  channels.push_back({"vel.z", "Linear velocity Z (m/s)", SignalChannelScope::SceneWide,
                      "%.3f m/s"});
  if (hasContactTags) {
    channels.push_back({kSignalChannelContacts, "Contacts (count)",
                        SignalChannelScope::SceneWide, "%.0f"});
  }
  if (isArticulated) {
    channels.push_back({kSignalChannelGeneralizedSpeed, "Generalized speed (mixed units)",
                        SignalChannelScope::SelectionOnly, "%.3f"});
  }

  if (selectedInfo == nullptr || !selectedInfo->valid || !selectedInfo->isArticulated) {
    return channels;
  }

  const std::vector<int32_t> gvOffsets =
    deriveJointGvOffsets(selectedInfo->jointTypes, selectedInfo->dof);
  for (size_t i = 0; i < selectedInfo->jointNames.size(); ++i) {
    const std::string token = sanitizeChannelToken(selectedInfo->jointNames[i]);
    const int32_t gcDim = i < selectedInfo->jointGcDims.size() ? selectedInfo->jointGcDims[i] : 0;
    if (i < selectedInfo->jointAngles.size() && gcDim == 1) {
      channels.push_back({"joint." + token + ".q", selectedInfo->jointNames[i] + " angle",
                          SignalChannelScope::SelectionOnly, "%.5g"});
    }
    if (i < gvOffsets.size() &&
        jointGvDim(i < selectedInfo->jointTypes.size() ? selectedInfo->jointTypes[i] : 0) == 1) {
      channels.push_back({"joint." + token + ".qd", selectedInfo->jointNames[i] + " velocity",
                          SignalChannelScope::SelectionOnly, "%.5g"});
    }
  }
  return channels;
}

std::vector<std::string> defaultSignalChannelKeys(
    const std::vector<SignalChannelDesc>& channels) {
  static const std::vector<std::string> preferred{
    kSignalChannelLinearSpeed, kSignalChannelAngularSpeed, kSignalChannelGeneralizedSpeed,
    kSignalChannelContacts};
  std::vector<std::string> keys;
  for (const std::string& candidate : preferred) {
    const auto match = std::find_if(channels.begin(), channels.end(),
      [&candidate](const SignalChannelDesc& channel) { return channel.key == candidate; });
    if (match != channels.end()) {
      keys.push_back(candidate);
    }
  }
  return keys;
}

namespace
{

float selectedScalarAt(const SelectedObjectInfo* info, const std::vector<float>& values,
                       int32_t offset) {
  if (info == nullptr || offset < 0 || static_cast<size_t>(offset) >= values.size()) {
    return 0.0f;
  }
  return values[static_cast<size_t>(offset)];
}

float channelValue(const SignalChannelDesc& channel, const SignalSampleInputs& inputs,
                   const std::vector<int32_t>& jointGvOffsets) {
  const SelectedObjectInfo* info = inputs.selectedInfo;
  if (channel.key == kSignalChannelLinearSpeed) {
    return inputs.hasMotionEstimate ? glm::length(inputs.linearVelocity) : 0.0f;
  }
  if (channel.key == kSignalChannelAngularSpeed) {
    return inputs.hasMotionEstimate ? inputs.angularSpeed : 0.0f;
  }
  if (channel.key == "pos.x") return inputs.position.x;
  if (channel.key == "pos.y") return inputs.position.y;
  if (channel.key == "pos.z") return inputs.position.z;
  if (channel.key == "vel.x") return inputs.hasMotionEstimate ? inputs.linearVelocity.x : 0.0f;
  if (channel.key == "vel.y") return inputs.hasMotionEstimate ? inputs.linearVelocity.y : 0.0f;
  if (channel.key == "vel.z") return inputs.hasMotionEstimate ? inputs.linearVelocity.z : 0.0f;
  if (channel.key == kSignalChannelContacts) return inputs.contactCount;
  if (channel.key == kSignalChannelGeneralizedSpeed) {
    if (info == nullptr || !info->valid) return 0.0f;
    double squared = 0.0;
    for (const float velocity : info->generalizedVelocities) {
      squared += static_cast<double>(velocity) * static_cast<double>(velocity);
    }
    return static_cast<float>(std::sqrt(squared));
  }

  // joint.<name>.q / joint.<name>.qd
  if (info == nullptr || !info->valid || channel.key.rfind("joint.", 0) != 0) {
    return 0.0f;
  }
  const size_t suffixStart = channel.key.rfind('.');
  if (suffixStart == std::string::npos) {
    return 0.0f;
  }
  const std::string suffix = channel.key.substr(suffixStart + 1);
  const std::string token = channel.key.substr(6, suffixStart - 6);
  for (size_t i = 0; i < info->jointNames.size(); ++i) {
    if (sanitizeChannelToken(info->jointNames[i]) != token) {
      continue;
    }
    if (suffix == "q") {
      return i < info->jointAngles.size() ? info->jointAngles[i] : 0.0f;
    }
    if (suffix == "qd" && i < jointGvOffsets.size()) {
      return selectedScalarAt(info, info->generalizedVelocities, jointGvOffsets[i]);
    }
    return 0.0f;
  }
  return 0.0f;
}

} // namespace

std::vector<float> sampleSignalChannels(
    const std::vector<SignalChannelDesc>& channels, const SignalSampleInputs& inputs) {
  std::vector<int32_t> jointGvOffsets;
  if (inputs.selectedInfo != nullptr && inputs.selectedInfo->valid) {
    jointGvOffsets =
      deriveJointGvOffsets(inputs.selectedInfo->jointTypes, inputs.selectedInfo->dof);
  }
  std::vector<float> values;
  values.reserve(channels.size());
  for (const SignalChannelDesc& channel : channels) {
    const float value = channelValue(channel, inputs, jointGvOffsets);
    values.push_back(std::isfinite(value) ? value : 0.0f);
  }
  return values;
}

void SignalHistory::setChannels(std::vector<SignalChannelDesc> channels) {
  const bool identical = channels.size() == channels_.size() &&
    std::equal(channels.begin(), channels.end(), channels_.begin(),
      [](const SignalChannelDesc& lhs, const SignalChannelDesc& rhs) {
        return lhs.key == rhs.key;
      });
  if (identical) {
    return;
  }
  channels_ = std::move(channels);
  clear();
}

bool SignalHistory::append(double worldTime, std::vector<float> values, bool selectionAvailable) {
  if (values.size() != channels_.size() || !std::isfinite(worldTime)) {
    return false;
  }
  if (!times_.empty() && !(worldTime > times_.back())) {
    return false;
  }
  times_.push_back(worldTime);
  samples_.push_back(std::move(values));
  selectionAvailable_.push_back(selectionAvailable);
  while (times_.size() > kMaxSamples) {
    times_.pop_front();
    samples_.pop_front();
    selectionAvailable_.pop_front();
  }
  return true;
}

void SignalHistory::clear() {
  times_.clear();
  samples_.clear();
  selectionAvailable_.clear();
}

double SignalHistory::lastTime() const {
  return times_.empty() ? -std::numeric_limits<double>::infinity() : times_.back();
}

int SignalHistory::channelIndex(const std::string& key) const {
  for (size_t i = 0; i < channels_.size(); ++i) {
    if (channels_[i].key == key) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

bool SignalHistory::sampleAvailable(size_t row, size_t column) const {
  return row < samples_.size() && column < channels_.size() &&
    (channels_[column].scope != SignalChannelScope::SelectionOnly || selectionAvailable_[row]);
}

bool SignalHistory::series(const std::string& key, std::vector<float>& out) const {
  const int index = channelIndex(key);
  if (index < 0) {
    return false;
  }
  out.clear();
  out.reserve(samples_.size());
  bool measured = false;
  float last = 0.0f;
  for (size_t row = 0; row < samples_.size(); ++row) {
    if (sampleAvailable(row, static_cast<size_t>(index))) {
      last = samples_[row][static_cast<size_t>(index)];
      measured = true;
    }
    if (measured) out.push_back(last);
  }
  return true;
}

bool writeSignalCsv(
    const std::filesystem::path& path, const SignalHistory& history,
    const std::vector<std::string>& channelKeys, std::string& status) {
  std::vector<int> columns;
  std::vector<std::string> headers;
  for (const std::string& key : channelKeys) {
    const int index = history.channelIndex(key);
    if (index >= 0) {
      columns.push_back(index);
      headers.push_back(key);
    }
  }
  if (columns.empty()) {
    status = "signal export failed: no channels selected";
    return false;
  }
  if (history.empty()) {
    status = "signal export failed: no samples recorded yet";
    return false;
  }

  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      status = "signal export failed: cannot create " + path.parent_path().string();
      return false;
    }
  }
  std::ofstream file(path);
  if (!file) {
    status = "signal export failed: cannot open " + path.string();
    return false;
  }

  file << "time";
  for (const std::string& header : headers) {
    file << ',' << header;
  }
  file << '\n';

  file << std::setprecision(9);
  const auto& times = history.times();
  const auto& samples = history.samples();
  for (size_t row = 0; row < times.size(); ++row) {
    file << times[row];
    for (const int column : columns) {
      file << ',';
      if (history.sampleAvailable(row, static_cast<size_t>(column)))
        file << samples[row][static_cast<size_t>(column)];
    }
    file << '\n';
  }
  if (!file) {
    status = "signal export failed: write error on " + path.string();
    return false;
  }
  status = "exported " + std::to_string(times.size()) + " sample(s) to " + path.string();
  return true;
}

std::filesystem::path timestampedSignalCsvPath(
    const std::filesystem::path& directory, const std::string& objectName, std::time_t now) {
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &now);
#else
  localtime_r(&now, &tm);
#endif
  std::ostringstream name;
  name << "rayrai_signals_" << sanitizeChannelToken(objectName) << "_"
       << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".csv";
  return directory / name.str();
}

} // namespace raisin::tcp_viewer
