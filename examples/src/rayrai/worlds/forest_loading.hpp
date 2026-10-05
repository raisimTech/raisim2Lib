#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <imgui/imgui.h>

// The forest queues its assets once before entering the event/render loop.
struct ForestLoadingProgress {
  size_t total = 0, pending = 0;
  const char* label = "Loading forest assets";
  void update(size_t count) {
    if (pending == 0 && count != 0) total = count;
    total = std::max(total, count);
    pending = count;
  }
  bool visible() const { return pending != 0; }
  float fraction() const {
    return total ? 1.f - float(pending) / float(total) : 1.f;
  }
  bool draw() const {
    if (!visible()) return false;
    const auto* viewport = ImGui::GetMainViewport();
    auto* draw = ImGui::GetForegroundDrawList();
    const ImVec2 lo(viewport->Pos.x+8, viewport->Pos.y+8);
    const ImVec2 hi(viewport->Pos.x+std::max(9.f,viewport->Size.x-8),lo.y+24);
    // Foreground drawing keeps the bar above the viewer without taking input/focus.
    draw->AddRectFilled(viewport->Pos,ImVec2(viewport->Pos.x+viewport->Size.x,hi.y+8),
                        IM_COL32(18,22,28,245));
    draw->AddRectFilled(lo,hi,ImGui::GetColorU32(ImGuiCol_FrameBg),3);
    if (fraction()>0)
      draw->AddRectFilled(lo,ImVec2(lo.x+(hi.x-lo.x)*fraction(),hi.y),
                          ImGui::GetColorU32(ImGuiCol_ButtonActive),3);
    char text[96];
    std::snprintf(text,sizeof(text),"%s: %zu / %zu",label,total-pending,total);
    const ImVec2 size = ImGui::CalcTextSize(text);
    draw->AddText(ImVec2((lo.x+hi.x-size.x)*.5f,(lo.y+hi.y-size.y)*.5f),
                  ImGui::GetColorU32(ImGuiCol_Text),text);
    return true;
  }
};
