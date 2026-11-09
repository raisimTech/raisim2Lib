// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <cmath>

#include "rayrai/Camera.hpp"

namespace fem_workbench {
inline void updateMeshCameraClipping(raisin::Camera& camera) {
  const float distance =
      glm::distance(camera.position, camera.getTargetPosition());
  camera.nearPlane = camera.zNear = std::max(1e-6f, distance * .001f);
  camera.farPlane = camera.zFar = std::max(1e-4f, distance * 10.f);
}
inline void frameMeshCamera(raisin::Camera& camera, const glm::vec3& low,
                            const glm::vec3& high) {
  camera.releaseTarget();
  const float extent = std::max(glm::length(high - low), 1e-9f);
  camera.setMinimumOrbitDistance(std::max(1e-6f, extent * .01f));
  const glm::vec3 offset = extent * glm::vec3(1.1f, -1.45f, .85f) /
                           std::min(1.f, std::max(camera.aspect, .1f));
  camera.target = low + .5f * (high - low);
  camera.position = camera.target + offset;
  const glm::vec3 direction = glm::normalize(camera.target - camera.position);
  camera.yaw = glm::degrees(std::atan2(direction.y, direction.x));
  camera.pitch = glm::degrees(std::asin(direction.z));
  camera.zoom = 45.f;
  camera.setCameraFixedTarget(true);
  camera.setCameraFixedDistance(true);
  camera.update(false);
  updateMeshCameraClipping(camera);
}
}  // namespace fem_workbench
