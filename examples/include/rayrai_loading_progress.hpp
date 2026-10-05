#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>

#include <imgui/imgui.h>

/// Loading bar for the meshes a rayrai viewer reads asynchronously.
///
/// Call update() with RayraiWindow::pendingAsyncMeshLoadCount() once per frame
/// and draw() after rendering the viewer. The bar shows the finished share of
/// the largest queue seen since the last time it emptied, and hides itself
/// once nothing is pending. It is drawn on ImGui's foreground layer, so it
/// stays above the viewer without taking input or focus.
struct RayraiLoadingProgress {
  explicit RayraiLoadingProgress(const char* label = "Loading assets") : label(label) {}

  size_t total = 0, pending = 0;
  const char* label;

  void update(size_t count) {
    if (pending == 0 && count != 0) total = count;  // A new load queue.
    total = std::max(total, count);
    pending = count;
  }

  bool visible() const { return pending != 0; }

  float fraction() const {
    return total ? 1.f - float(pending) / float(total) : 1.f;
  }

  /// @return Whether the bar was drawn.
  bool draw() const {
    if (!visible()) return false;
    const auto* viewport = ImGui::GetMainViewport();
    auto* list = ImGui::GetForegroundDrawList();
    const ImVec2 lo(viewport->Pos.x + 8, viewport->Pos.y + 8);
    const ImVec2 hi(viewport->Pos.x + std::max(9.f, viewport->Size.x - 8), lo.y + 24);
    list->AddRectFilled(viewport->Pos, ImVec2(viewport->Pos.x + viewport->Size.x, hi.y + 8),
                        IM_COL32(18, 22, 28, 245));
    list->AddRectFilled(lo, hi, ImGui::GetColorU32(ImGuiCol_FrameBg), 3);
    if (fraction() > 0)
      list->AddRectFilled(lo, ImVec2(lo.x + (hi.x - lo.x) * fraction(), hi.y),
                          ImGui::GetColorU32(ImGuiCol_ButtonActive), 3);
    char text[96];
    std::snprintf(text, sizeof(text), "%s: %zu / %zu", label, total - pending, total);
    const ImVec2 size = ImGui::CalcTextSize(text);
    list->AddText(ImVec2((lo.x + hi.x - size.x) * .5f, (lo.y + hi.y - size.y) * .5f),
                  ImGui::GetColorU32(ImGuiCol_Text), text);
    return true;
  }
};
