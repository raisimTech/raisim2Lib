// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

// What the rayrai examples share: sensor images in panels and conversions from RaiSim to glm types.

#include <algorithm>
#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui/imgui.h>

#include "raisim/math.hpp"

namespace raisim_examples {

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
