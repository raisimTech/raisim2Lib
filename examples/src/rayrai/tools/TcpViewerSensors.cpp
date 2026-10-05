// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerSensors.hpp"
#include "TcpViewerSensorLimits.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

#include <SDL.h>
#include <glbinding/gl/gl.h>
#include <glm/gtc/quaternion.hpp>

#include "rayrai/Camera.hpp"
#include "rayrai/RayraiWindow.hpp"
#include "raisim/sensors/DepthSensor.hpp"
#include "raisim/sensors/RGBSensor.hpp"

namespace raisin::tcp_viewer
{
namespace
{

std::string sensorKey(const SensorInfo& info) {
  return std::to_string(info.parentTag) + ":" + std::to_string(static_cast<int>(info.type)) +
         ":" + info.name;
}

void configureCamera(Camera& camera, const PendingSensorUpdate& update) {
  const auto& info = update.info;
  camera.ensureRenderTargets(info.width, info.height);
  camera.aspect = static_cast<float>(info.width) / static_cast<float>(info.height);
  camera.nearPlane = camera.zNear = static_cast<float>(std::max(1.0e-4, info.clipNear));
  camera.farPlane = camera.zFar = static_cast<float>(std::max(info.clipFar, info.clipNear + 1.0e-3));
  camera.setLensModel(info.lens, info.hFov);
  const double rasterHFov = info.lens.isFisheye() ? std::min(info.hFov, 3.124139361) : info.hFov;
  camera.zoom = glm::degrees(static_cast<float>(
    2.0 * std::atan(std::tan(0.5 * rasterHFov) / std::max(1.0e-6f, camera.aspect))));
  camera.setProjectionMode(Camera::ProjectionMode::PERSPECTIVE);

  glm::quat orientation(update.orientation.w, update.orientation.x,
                        update.orientation.y, update.orientation.z);
  const float norm = glm::length(orientation);
  orientation = std::isfinite(norm) && norm > 1.0e-6f
                  ? orientation / norm
                  : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  const glm::mat3 rotation = glm::mat3_cast(orientation);
  camera.position = update.position;
  camera.front = glm::normalize(rotation[0]);
  camera.up = glm::normalize(rotation[2]);
  camera.right = glm::normalize(glm::cross(camera.front, camera.up));
  camera.worldUp = camera.up;
  camera.target = camera.position + camera.front;
}

/** RaiSim camera properties matching a remote sensor, for reading back its image. */
template <typename Properties>
Properties sensorCameraProperties(const SensorInfo& info) {
  Properties properties;
  properties.name = info.name;
  properties.full_name = info.name;
  properties.width = info.width;
  properties.height = info.height;
  properties.clipNear = info.clipNear;
  properties.clipFar = info.clipFar;
  properties.hFOV = info.hFov;
  properties.lens = info.lens;
  return properties;
}

void uploadDepthPreview(unsigned int& texture, int width, int height,
                        const std::vector<float>& depth, float nearPlane, float farPlane,
                        float& minimumDepth, float& maximumDepth) {
  const auto rgba = detail::makeDepthPreviewRgba(
    depth, nearPlane, farPlane, minimumDepth, maximumDepth);

  if (texture == 0) {
    gl::glGenTextures(1, &texture);
  }
  gl::glBindTexture(gl::GL_TEXTURE_2D, texture);
  gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MIN_FILTER, gl::GL_LINEAR);
  gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MAG_FILTER, gl::GL_LINEAR);
  gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_S, gl::GL_CLAMP_TO_EDGE);
  gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_T, gl::GL_CLAMP_TO_EDGE);
  gl::glPixelStorei(gl::GL_UNPACK_ALIGNMENT, 1);
  gl::glTexImage2D(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA8, width, height, 0,
                   gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
}

} // namespace

namespace detail
{

std::vector<unsigned char> makeDepthPreviewRgba(const std::vector<float>& depth,
  float clipMinimum, float clipMaximum, float& minimumDepth, float& maximumDepth) {
  std::vector<unsigned char> rgba(depth.size() * 4u, 0);
  minimumDepth = std::numeric_limits<float>::infinity();
  maximumDepth = 0.0f;
  for (float value : depth) {
    if (std::isfinite(value) && value > 0.0f) {
      minimumDepth = std::min(minimumDepth, value);
      maximumDepth = std::max(maximumDepth, value);
    }
  }
  if (!std::isfinite(minimumDepth)) {
    minimumDepth = 0.0f;
  }

  const float scaleMinimum = std::isfinite(clipMinimum)
    ? std::max(clipMinimum, 1.0e-6f)
    : 1.0e-6f;
  const float scaleMaximum = std::isfinite(clipMaximum) && clipMaximum > scaleMinimum
    ? clipMaximum
    : scaleMinimum + 1.0e-6f;
  const float inverseLogRange = 1.0f / std::log(scaleMaximum / scaleMinimum);
  for (size_t i = 0; i < depth.size(); ++i) {
    const float value = depth[i];
    unsigned char shade = 0;
    if (std::isfinite(value) && value > 0.0f) {
      const float scaledValue = std::clamp(value, scaleMinimum, scaleMaximum);
      const float normalized = std::log(scaledValue / scaleMinimum) * inverseLogRange;
      shade = static_cast<unsigned char>(std::lround(255.0f * normalized));
    }
    rgba[i * 4u + 0u] = shade;
    rgba[i * 4u + 1u] = shade;
    rgba[i * 4u + 2u] = shade;
    rgba[i * 4u + 3u] = 255;
  }
  return rgba;
}

} // namespace detail

struct SensorRenderer::CachedSensor {
  std::unique_ptr<Camera> camera;
  SensorPreviewInfo preview;
  unsigned int depthPreviewTexture = 0;
};

SensorRenderer::SensorRenderer() = default;

SensorRenderer::~SensorRenderer() {
  clear();
}

bool SensorRenderer::render(RayraiWindow& viewer, std::vector<PendingSensorUpdate>& updates,
                            std::string& status) {
  // Keep the update handshake alive even when a remote sensor advertises an
  // unusable size. The caller sends an empty acknowledgment if all updates
  // are filtered, and the server can continue serving scene frames.
  updates.erase(std::remove_if(updates.begin(), updates.end(),
      [](const PendingSensorUpdate& update) {
        return !validSensorDimensions(update.info.width, update.info.height);
      }), updates.end());
  RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;
  overrides.drawCoordinateFrames = false;
  overrides.drawPointClouds = false;
  overrides.drawVisualizationObjects = true;
  overrides.drawWeatherEffects = false;
  overrides.postProcess = false;
  overrides.allowViewerMsaa = false;
  overrides.allowTemporalAa = false;
  overrides.allowViewerUpscale = false;

  for (auto& update : updates) {
    const auto& info = update.info;
    if ((info.type != raisim::Sensor::Type::RGB &&
         info.type != raisim::Sensor::Type::DEPTH) ||
        !validSensorDimensions(info.width, info.height)) {
      status = "unsupported or invalid sensor request: " + info.name;
      return false;
    }

    const std::string key = sensorKey(info);
    auto& cached = sensors_[key];
    if (!cached) {
      cached = std::make_unique<CachedSensor>();
      cached->camera = std::make_unique<Camera>();
    }
    configureCamera(*cached->camera, update);

    const auto started = std::chrono::steady_clock::now();
    raisim::Vec<3> position;
    position.setZero();
    raisim::Mat<3, 3> rotation;
    rotation.setIdentity();
    if (info.type == raisim::Sensor::Type::RGB) {
      viewer.renderWithExternalCamera(*cached->camera, overrides);
      raisim::RGBCamera sensor(
        sensorCameraProperties<raisim::RGBCamera::RGBCameraProperties>(info), nullptr, position,
        rotation);
      update.colorBgra.resize(static_cast<size_t>(info.width) *
                              static_cast<size_t>(info.height) * 4u);
      cached->camera->getRawImage(sensor, Camera::SensorStorageMode::CUSTOM_BUFFER,
        reinterpret_cast<char*>(update.colorBgra.data()), update.colorBgra.size(), true);
      cached->preview.texture = cached->camera->getFinalTexture();
    } else {
      cached->camera->ensureLinearDepthRenderTarget();
      // TCP physical geometry is marked detectable when RemoteScene reconstructs
      // it. Use the same filter as RGB so viewer-only helpers do not leak into
      // either sensor stream.
      viewer.renderDepthPlaneDistance(*cached->camera, nullptr, true, true);
      raisim::DepthCamera sensor(
        sensorCameraProperties<raisim::DepthCamera::DepthCameraProperties>(info), nullptr,
        position, rotation);
      update.depth.resize(static_cast<size_t>(info.width) * static_cast<size_t>(info.height));
      cached->camera->getRawImage(sensor, Camera::SensorStorageMode::CUSTOM_BUFFER,
        update.depth.data(), update.depth.size(), true);
      uploadDepthPreview(cached->depthPreviewTexture, info.width, info.height, update.depth,
        static_cast<float>(info.clipNear), static_cast<float>(info.clipFar),
        cached->preview.minimumDepth, cached->preview.maximumDepth);
      cached->preview.texture = cached->depthPreviewTexture;
    }
    update.renderMilliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    cached->preview.parentTag = info.parentTag;
    cached->preview.type = info.type;
    cached->preview.name = info.name;
    cached->preview.width = info.width;
    cached->preview.height = info.height;
    cached->preview.renderMilliseconds = update.renderMilliseconds;
  }
  status = "rendered " + std::to_string(updates.size()) + " sensor request(s)";
  return true;
}

std::vector<SensorPreviewInfo> SensorRenderer::previews() const {
  std::vector<SensorPreviewInfo> result;
  result.reserve(sensors_.size());
  for (const auto& item : sensors_) {
    result.push_back(item.second->preview);
  }
  std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.name < rhs.name;
  });
  return result;
}

std::vector<SensorPreviewInfo> SensorRenderer::previewsForTag(uint32_t parentTag) const {
  auto result = previews();
  result.erase(std::remove_if(result.begin(), result.end(), [parentTag](const auto& preview) {
    return preview.parentTag != parentTag;
  }), result.end());
  return result;
}

void SensorRenderer::clear() {
  if (SDL_GL_GetCurrentContext() != nullptr) {
    for (const auto& item : sensors_) {
      if (item.second && item.second->depthPreviewTexture != 0) {
        gl::glDeleteTextures(1, &item.second->depthPreviewTexture);
      }
    }
  }
  sensors_.clear();
}

} // namespace raisin::tcp_viewer
