// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerScreenshot.hpp"

#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

#include "TcpViewerPathText.hpp"

#include <glbinding/gl/gl.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "rayrai/RayraiWindow.hpp"

namespace raisin::tcp_viewer
{

void flipRgbaRows(std::vector<unsigned char>& rgba, int width, int height) {
  const size_t stride = static_cast<size_t>(width) * 4u;
  std::vector<unsigned char> row(stride);
  for (int y = 0; y < height / 2; ++y) {
    auto* top = rgba.data() + static_cast<size_t>(y) * stride;
    auto* bottom = rgba.data() + static_cast<size_t>(height - 1 - y) * stride;
    std::memcpy(row.data(), top, stride);
    std::memcpy(top, bottom, stride);
    std::memcpy(bottom, row.data(), stride);
  }
}

std::filesystem::path timestampedCapturePath(const std::filesystem::path& dir, const char* prefix) {
  const auto now = std::chrono::system_clock::now();
  const std::time_t raw = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &raw);
#else
  localtime_r(&raw, &tm);
#endif
  std::ostringstream name;
  name << (prefix ? prefix : "rayrai") << "_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".png";
  return nextAvailableOutputPath(dir / name.str());
}

std::filesystem::path nextAvailableOutputPath(const std::filesystem::path& requested) {
  std::error_code ec;
  if (!std::filesystem::exists(requested, ec) && !ec) return requested;
  for (unsigned int suffix = 1; suffix < 100000; ++suffix) {
    std::filesystem::path name = requested.stem();
    name += "_" + std::to_string(suffix);
    name += requested.extension();
    const auto candidate = requested.parent_path() / name;
    ec.clear();
    if (!std::filesystem::exists(candidate, ec) && !ec) return candidate;
  }
  return requested;
}

bool captureViewerRgba(RayraiWindow& viewer, std::vector<unsigned char>& rgba, int& width,
                       int& height, std::string& status) {
  auto& camera = viewer.getCamera();
  width = camera.rtWidth();
  height = camera.rtHeight();
  if (width <= 0 || height <= 0) {
    status = "capture failed: invalid render target";
    return false;
  }

  rgba.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
  gl::GLint previousPackAlignment = 4;
  gl::glGetIntegerv(gl::GL_PACK_ALIGNMENT, &previousPackAlignment);
  gl::glBindTexture(gl::GL_TEXTURE_2D, camera.getFinalTexture());
  gl::glPixelStorei(gl::GL_PACK_ALIGNMENT, 1);
  gl::glGetTexImage(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
  gl::glPixelStorei(gl::GL_PACK_ALIGNMENT, previousPackAlignment);
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
  flipRgbaRows(rgba, width, height);
  return true;
}

bool saveRgbaPng(const std::vector<unsigned char>& rgba, int width, int height,
                 const std::filesystem::path& path, std::string& status) {
  if (width <= 0 || height <= 0 ||
      rgba.size() != static_cast<size_t>(width) * static_cast<size_t>(height) * 4u) {
    status = "capture failed: incomplete frame buffer";
    return false;
  }

  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      status = "capture failed: cannot create directory";
      return false;
    }
  }

  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  const auto write = [](void* stream, void* data, int size) {
    static_cast<std::ofstream*>(stream)->write(static_cast<const char*>(data), size);
  };
  if (!file || !stbi_write_png_to_func(write, &file, width, height, 4, rgba.data(), width * 4) ||
      !file.flush()) {
    status = "capture failed: PNG write failed";
    return false;
  }
  status = "saved " + utf8PathText(path);
  return true;
}

bool saveViewerTexturePng(RayraiWindow& viewer, const std::filesystem::path& path, std::string& status) {
  std::vector<unsigned char> rgba;
  int width = 0;
  int height = 0;
  if (!captureViewerRgba(viewer, rgba, width, height, status)) {
    return false;
  }
  return saveRgbaPng(rgba, width, height, path, status);
}

} // namespace raisin::tcp_viewer
