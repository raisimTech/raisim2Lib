// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "raisim/sensors/SegmentationSensor.hpp"
#include "raisim/sensors/Sensors.hpp"

namespace raisin::tcp_viewer {

/// @return True for the image sensors the viewer renders: RGB, depth and segmentation cameras.
inline bool isCameraSensor(raisim::Sensor::Type type) {
  return type == raisim::Sensor::Type::RGB || type == raisim::Sensor::Type::DEPTH ||
         type == raisim::Sensor::Type::SEGMENTATION;
}

/// Near plane a camera with this clip range renders with, the rule of rayrai's Camera: a clipNear
/// of 0 (at most 1e-6 m) renders with clipFar * 1e-4.
inline double sensorRenderNearClip(double clipNear, double clipFar) {
  return clipNear > 1.0e-6 ? clipNear : clipFar * 1.0e-4;
}

/// @return True when a camera with this clip range can render: its near plane (see
///   sensorRenderNearClip()) exceeds 1e-6 m and lies before clipFar.
inline bool validSensorClipRange(double clipNear, double clipFar) {
  const double nearClip = sensorRenderNearClip(clipNear, clipFar);
  return nearClip > 1.0e-6 && clipFar > nearClip;
}

inline bool validSensorDimensions(int width, int height) {
  constexpr int kMaxSide = 8192;
  constexpr int64_t kMaxPixels = 8 * 1024 * 1024;
  return width > 0 && height > 0 && width <= kMaxSide &&
         height <= kMaxSide &&
         static_cast<int64_t>(width) * height <= kMaxPixels;
}

// RaiSimServer's 32 MiB receive buffer includes a four-byte frame length.
constexpr size_t kMaxSensorUpdatePayloadBytes = 33554432u - 4u;

/// @return Bytes per pixel of a camera's reply: four for RGB and depth, four per image for segmentation.
inline size_t sensorReplyBytesPerPixel(raisim::Sensor::Type type,
                                       raisim::SegmentationCamera::SegmentationType segmentationType) {
  return type == raisim::Sensor::Type::SEGMENTATION
             ? 4u * size_t(raisim::SegmentationCamera::imageCount(segmentationType)) : 4u;
}

inline std::vector<size_t> selectSensorUpdateIndices(
    const std::vector<size_t>& entryBytes, size_t start,
    size_t headerBytes = 20u) {
  std::vector<size_t> selected;
  if (entryBytes.empty()) return selected;
  size_t used = headerBytes;
  for (size_t offset = 0; offset < entryBytes.size(); ++offset) {
    const size_t index = (start + offset) % entryBytes.size();
    const size_t bytes = entryBytes[index];
    if (bytes <= kMaxSensorUpdatePayloadBytes - used) {
      selected.push_back(index);
      used += bytes;
    }
  }
  return selected;
}

} // namespace raisin::tcp_viewer
