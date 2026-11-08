#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>

#include <imgui/imgui.h>

/// Loading bar for the meshes a rayrai viewer reads asynchronously.
///
/// Call update() with RayraiWindow::pendingAsyncMeshLoadCount() once per frame
/// and draw() after rendering the viewer. The bar shows the finished share of
/// the largest queue seen since the last time it emptied, and hides itself
/// once nothing is pending. It is drawn on ImGui's foreground layer, so it
/// stays above the viewer without taking input or focus.
struct RayraiLoadingProgress {
  explicit RayraiLoadingProgress(std::string label = "Loading assets") : label(std::move(label)) {}

  size_t total = 0, pending = 0;
  std::string label;

  void update(size_t count) {
    if (pending == 0 && count != 0) total = count;  // A new load queue.
    total = std::max(total, count);
    pending = count;
  }

  bool visible() const { return pending != 0; }

  float fraction() const { return total ? 1.f - float(pending) / float(total) : 1.f; }

  /// @return Whether the bar was drawn.
  bool draw() const {
    if (!visible()) return false;
    const auto* viewport = ImGui::GetMainViewport();
    auto* list = ImGui::GetForegroundDrawList();
    const ImVec2 lo(viewport->Pos.x + 8, viewport->Pos.y + 8);
    const ImVec2 hi(viewport->Pos.x + std::max(9.f, viewport->Size.x - 8), lo.y + 24);
    // Colors sampled from rsc/docs/image/logo.png, independent of the ImGui theme.
    constexpr ImU32 ink = IM_COL32(71, 14, 89, 255);
    constexpr std::array<ImU32, 6> gradient = {
        IM_COL32(204, 246, 34, 255), IM_COL32(218, 177, 56, 255),
        IM_COL32(239, 77, 88, 255), IM_COL32(211, 76, 208, 255),
        IM_COL32(165, 94, 215, 255), IM_COL32(136, 102, 214, 255)};
    list->AddRectFilled(viewport->Pos,
                        ImVec2(viewport->Pos.x + viewport->Size.x, hi.y + 8),
                        IM_COL32(255, 255, 255, 245));
    list->AddRectFilled(lo, hi, IM_COL32(235, 226, 240, 255), 3);
    const float segmentWidth = (hi.x - lo.x) / float(gradient.size() - 1);
    const float fillEnd = lo.x + (hi.x - lo.x) * fraction();
    for (size_t i = 0; i + 1 < gradient.size(); ++i) {
      const float start = lo.x + float(i) * segmentWidth;
      const float end = std::min(start + segmentWidth, fillEnd);
      if (end <= start) break;
      const auto left = ImGui::ColorConvertU32ToFloat4(gradient[i]);
      const auto right = ImGui::ColorConvertU32ToFloat4(gradient[i + 1]);
      const float blend = (end - start) / segmentWidth;
      const ImU32 endColor = ImGui::ColorConvertFloat4ToU32(
          ImVec4(left.x + (right.x - left.x) * blend,
                 left.y + (right.y - left.y) * blend,
                 left.z + (right.z - left.z) * blend, 1));
      list->AddRectFilledMultiColor(ImVec2(start, lo.y), ImVec2(end, hi.y),
                                   gradient[i], endColor, endColor, gradient[i]);
    }
    char text[96];
    std::snprintf(text, sizeof(text), "%s: %zu / %zu", label.c_str(), total - pending, total);
    const ImVec2 size = ImGui::CalcTextSize(text);
    const ImVec2 textPosition((lo.x + hi.x - size.x) * .5f, (lo.y + hi.y - size.y) * .5f);
    // Keep the label readable over every color of the logo gradient.
    list->AddRectFilled(ImVec2(textPosition.x - 6, textPosition.y - 2),
                        ImVec2(textPosition.x + size.x + 6, textPosition.y + size.y + 2),
                        IM_COL32(255, 255, 255, 255), 3);
    list->AddText(textPosition, ink, text);
    return true;
  }
};
