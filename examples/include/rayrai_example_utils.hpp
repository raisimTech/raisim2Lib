// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

// What the rayrai examples share: PNG files of rendered images for their --screenshot options,
// sensor images in panels and conversions from RaiSim to glm types. stb_image_write is compiled
// into each example that includes this header, privately, so examples need no copy of their own.

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <system_error>
#include <vector>

#include <glbinding/gl/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui/imgui.h>

#include "raisim/math.hpp"
#include "rayrai/Camera.hpp"
#include "rayrai/TextureBindingCache.hpp"

// An example uses a few of the private stb functions; the others would warn as unused.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4505)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace raisim_examples {

/// Saves an RGBA8 texture as a PNG file, top row first, creating its folder. Returns false when
/// the file cannot be written.
inline bool saveTexturePng(unsigned int texture, int width, int height, const std::filesystem::path& path) {
  const size_t stride = static_cast<size_t>(width) * 4;
  std::vector<unsigned char> rgba(stride * static_cast<size_t>(height));
  gl::glBindTexture(gl::GL_TEXTURE_2D, texture);
  gl::glGetTexImage(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
  raisin::resetTextureBindingCache();
  // OpenGL stores the bottom row first.
  for (int y = 0; y < height / 2; ++y)
    std::swap_ranges(rgba.begin() + y * stride, rgba.begin() + (y + 1) * stride, rgba.begin() + (height - 1 - y) * stride);
  std::error_code error;
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), error);
  return stbi_write_png(path.string().c_str(), width, height, 4, rgba.data(), static_cast<int>(stride)) != 0;
}

/// Saves a camera's final image, after post-processing, as a PNG file.
inline bool saveCameraPng(raisin::Camera& camera, const std::filesystem::path& path) {
  return saveTexturePng(camera.getFinalTexture(), camera.rtWidth(), camera.rtHeight(), path);
}

/// The largest size with the aspect ratio (width / height) that fits in avail.
inline ImVec2 fitToAspect(ImVec2 avail, float aspect) {
  if (aspect <= 0.0f) return avail;
  ImVec2 size = avail;
  size.y = size.x / aspect;
  if (size.y > avail.y) {
    size.y = avail.y;
    size.x = size.y * aspect;
  }
  return size;
}

/// Draws a texture (bottom row first, as rendered) in the current window, as large as it fits with
/// its aspect ratio while textLines lines of text below it still fit.
inline void drawFittedImage(unsigned int texture, float aspect, int textLines) {
  ImVec2 avail = ImGui::GetContentRegionAvail();
  avail.y = std::max(1.0f, avail.y - float(textLines) * ImGui::GetTextLineHeightWithSpacing());
  ImGui::Image((ImTextureID)(intptr_t)texture, fitToAspect(avail, aspect), ImVec2(0, 1), ImVec2(1, 0));
}

inline glm::vec3 toGlm(const raisim::Vec<3>& v) { return glm::vec3(v[0], v[1], v[2]); }
inline glm::mat3 toGlm(const raisim::Mat<3, 3>& R) { return glm::make_mat3(R.ptr()); }

}  // namespace raisim_examples
