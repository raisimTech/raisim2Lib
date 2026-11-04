// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#pragma once

#include "game.hpp"

namespace tank {

struct Screen {
  glm::mat4 viewProjection{1.0f}, inverse{1.0f};
  ImVec2 origin, size;

  bool project(const Vec3& p, ImVec2& out) const {
    const glm::vec4 clip = viewProjection * glm::vec4(float(p.x()), float(p.y()), float(p.z()), 1.0f);
    if (clip.w <= 1e-4f) return false;
    out = ImVec2(origin.x + (clip.x / clip.w * 0.5f + 0.5f) * size.x, origin.y + (0.5f - clip.y / clip.w * 0.5f) * size.y);
    return true;
  }

  Vec3 ray(const ImVec2& point, const glm::vec3& eye) const {
    const float x = 2.0f * (point.x - origin.x) / std::max(1.0f, size.x) - 1.0f;
    const float y = 1.0f - 2.0f * (point.y - origin.y) / std::max(1.0f, size.y);
    const glm::vec4 p = inverse * glm::vec4(x, y, 0.5f, 1.0f);
    const glm::vec3 direction = glm::normalize(glm::vec3(p) / p.w - eye);
    return {direction.x, direction.y, direction.z};
  }
};

struct Fonts {
  ImFont* text = nullptr;
  ImFont* title = nullptr;
};

inline void bar(ImDrawList* draw, ImVec2 at, float width, float height, float fraction, ImU32 color) {
  draw->AddRectFilled(at, ImVec2(at.x + width, at.y + height), IM_COL32(0, 0, 0, 150), 3.0f);
  draw->AddRectFilled(at, ImVec2(at.x + width * std::clamp(fraction, 0.0f, 1.0f), at.y + height), color, 3.0f);
  draw->AddRect(at, ImVec2(at.x + width, at.y + height), IM_COL32(255, 255, 255, 70), 3.0f);
}

inline void text(ImDrawList* draw, ImFont* font, float size, ImVec2 at, ImU32 color, const std::string& value,
                 float align = 0.0f) {
  at.x -= align * font->CalcTextSizeA(size, 1e6f, 0.0f, value.c_str()).x;
  draw->AddText(font, size, ImVec2(at.x + 2, at.y + 2), IM_COL32(0, 0, 0, 170), value.c_str());
  draw->AddText(font, size, at, color, value.c_str());
}

inline ImU32 fade(ImU32 color, float alpha) {
  return (color & 0x00FFFFFFu) | (ImU32(std::clamp(alpha, 0.0f, 1.0f) * float(color >> 24)) << 24);
}

inline std::string clock(double seconds) {
  char line[32];
  std::snprintf(line, sizeof(line), "%d:%02d", int(seconds) / 60, int(seconds) % 60);
  return line;
}

inline void drawHud(Game& game, const Screen& screen, const Fonts& fonts, double heading, bool help) {
  ImDrawList* draw = ImGui::GetForegroundDrawList();
  ImFont* font = fonts.text;
  const ImVec2 o = screen.origin, s = screen.size;
  const double t = game.now();
  char line[200];

  for (const auto& enemy : game.enemies()) {
    if (!enemy->system) continue;
    ImVec2 at;
    if (!screen.project(enemy->position() + Vec3(0, 0, 3.2 * enemy->scale()), at)) continue;
    const float width = enemy->spec->role == Role::Boss ? 140.0f : 60.0f;
    bar(draw, ImVec2(at.x - 0.5f * width, at.y), width, 6, float(enemy->health / enemy->spec->health), rgba(enemy->spec->marker));
    if (enemy->spec->role != Role::Medium && enemy->spec->role != Role::Scout)
      text(draw, font, 14.0f, ImVec2(at.x, at.y - 18), rgba(enemy->spec->marker, 0.9f), enemy->spec->name, 0.5f);
    if (t < enemy->tracksBrokenUntil) text(draw, font, 13.0f, ImVec2(at.x, at.y + 8), IM_COL32(255, 200, 120, 220), "IMMOBILIZED", 0.5f);
  }
  for (const auto& pickup : game.pickups()) {
    ImVec2 at;
    if (screen.project(pickup.position + Vec3(0, 0, 2.4), at))
      text(draw, font, 15.0f, at, rgba(kBonusStyles[size_t(pickup.type)].color), kBonusStyles[size_t(pickup.type)].label, 0.5f);
  }
  for (const auto& note : game.floating()) {
    ImVec2 at;
    const double age = t - note.time;
    if (screen.project(note.position + Vec3(0, 0, 2.0 + 2.0 * age), at))
      text(draw, font, 18.0f, at, fade(note.color, float(1.6 - age)), note.text, 0.5f);
  }

  const float x0 = o.x + 18, y0 = o.y + 18;
  draw->AddRectFilled(ImVec2(x0 - 8, y0 - 8), ImVec2(x0 + 300, y0 + 150), IM_COL32(8, 10, 12, 160), 8.0f);
  text(draw, font, 20.0f, ImVec2(x0, y0), IM_COL32(255, 255, 255, 255), "ARMOR");
  const float health = float(game.playerHealth() / vehicle(Role::Player).health);
  bar(draw, ImVec2(x0 + 80, y0 + 3), 200, 18, health,
      health > 0.5f ? IM_COL32(90, 220, 110, 255) : health > 0.25f ? IM_COL32(240, 200, 60, 255) : IM_COL32(240, 70, 50, 255));
  std::snprintf(line, sizeof(line), "%d", int(std::ceil(game.playerHealth())));
  text(draw, font, 15.0f, ImVec2(x0 + 180, y0 + 4), IM_COL32(255, 255, 255, 230), line, 0.5f);
  float y = y0 + 30;
  std::string status;
  if (game.tracksBroken()) status += "TRACKS BROKEN  ";
  if (game.gunDamaged()) status += "GUN DAMAGED  ";
  text(draw, font, 16.0f, ImVec2(x0, y), status.empty() ? IM_COL32(150, 220, 150, 255) : IM_COL32(255, 160, 90, 255),
       status.empty() ? "ALL SYSTEMS GO" : status);
  y += 24;
  std::string extras;
  if (game.reactive() > 0) extras += "REACTIVE x" + std::to_string(game.reactive()) + "  ";
  if (game.overdriveLeft() > 0) extras += "OVERDRIVE " + std::to_string(int(std::ceil(game.overdriveLeft()))) + "  ";
  extras += "SMOKE x" + std::to_string(game.smokeCharges());
  text(draw, font, 16.0f, ImVec2(x0, y), IM_COL32(200, 200, 255, 255), extras);
  y += 26;
  std::snprintf(line, sizeof(line), "SCORE %d", game.score());
  text(draw, font, 26.0f, ImVec2(x0, y), IM_COL32(255, 255, 255, 255), line);
  std::snprintf(line, sizeof(line), "%s   kills %d", clock(game.elapsed()).c_str(), game.stats().kills);
  text(draw, font, 16.0f, ImVec2(x0, y + 34), IM_COL32(210, 210, 210, 255), line);

  const LevelSpec& level = game.levelSpec();
  if (game.waves() > 0)
    std::snprintf(line, sizeof(line), "%s  -  WAVE %d / %d  -  %d hostile", level.name, std::max(1, game.wave()),
                  game.waves(), game.enemiesLeft());
  else
    std::snprintf(line, sizeof(line), "%s  -  WAVE %d  -  %d hostile", level.name, std::max(1, game.wave()), game.enemiesLeft());
  text(draw, font, 18.0f, ImVec2(o.x + 0.5f * s.x, o.y + 14), IM_COL32(255, 240, 200, 235), line, 0.5f);
  const Vec3 wind = level.wind;
  std::snprintf(line, sizeof(line), "wind %.0f m/s", wind.norm());
  text(draw, font, 14.0f, ImVec2(o.x + 0.5f * s.x, o.y + 38), IM_COL32(200, 220, 255, 200), line, 0.5f);

  const float radius = 92.0f;
  const ImVec2 radar(o.x + s.x - radius - 22, o.y + radius + 22);
  draw->AddCircleFilled(radar, radius, IM_COL32(8, 18, 12, 170), 48);
  draw->AddCircle(radar, radius, IM_COL32(120, 255, 150, 120), 48, 2.0f);
  draw->AddCircle(radar, radius * 0.5f, IM_COL32(120, 255, 150, 50), 48);
  const float sweep = float(std::fmod(t * 2.0, 2.0 * kPi));
  draw->AddLine(radar, ImVec2(radar.x + radius * std::cos(sweep), radar.y + radius * std::sin(sweep)), IM_COL32(120, 255, 150, 90), 2.0f);
  const Vec3 center = game.playerTank() && game.playerTank()->system ? game.playerTank()->position() : Vec3::Zero();
  const double c = std::cos(-heading + kPi / 2), sn = std::sin(-heading + kPi / 2);
  const auto toRadar = [&](const Vec3& p) {
    const Vec3 d = p - center;
    const float scale = radius / 90.0f;
    ImVec2 at(radar.x + float(c * d.x() - sn * d.y()) * scale, radar.y - float(sn * d.x() + c * d.y()) * scale);
    const float r = std::hypot(at.x - radar.x, at.y - radar.y);
    if (r > radius - 4) {
      at.x = radar.x + (at.x - radar.x) * (radius - 4) / r;
      at.y = radar.y + (at.y - radar.y) * (radius - 4) / r;
    }
    return at;
  };
  for (const auto& barrel : game.barrels())
    if (barrel.body) draw->AddCircleFilled(toRadar(barrel.body->getPosition()), 1.5f, IM_COL32(220, 90, 40, 160));
  for (const auto& pickup : game.pickups()) {
    const ImVec2 at = toRadar(pickup.position);
    draw->AddRectFilled(ImVec2(at.x - 3, at.y - 3), ImVec2(at.x + 3, at.y + 3), rgba(kBonusStyles[size_t(pickup.type)].color));
  }
  for (const auto& enemy : game.enemies())
    if (enemy->system) draw->AddCircleFilled(toRadar(enemy->position()), enemy->spec->role == Role::Boss ? 7.0f : 4.5f, rgba(enemy->spec->marker));
  draw->AddTriangleFilled(ImVec2(radar.x, radar.y - 7), ImVec2(radar.x - 5, radar.y + 5), ImVec2(radar.x + 5, radar.y + 5), IM_COL32(255, 255, 255, 255));
  const double hurtAge = t - game.hurtTime();
  if (hurtAge < 1.2 && game.hurtFrom().head<2>().norm() > 1e-3) {
    const Vec3 d = game.hurtFrom();
    const float a = float(std::atan2(-(sn * d.x() + c * d.y()), c * d.x() - sn * d.y()));
    draw->PathArcTo(radar, radius + 6.0f, a - 0.35f, a + 0.35f, 16);
    draw->PathStroke(IM_COL32(255, 60, 40, int(230 * (1.2 - hurtAge) / 1.2)), 0, 6.0f);
  }
  float feedY = radar.y + radius + 14;
  for (auto it = game.kills().rbegin(); it != game.kills().rend(); ++it) {
    const double age = t - it->time;
    if (age > 6.0) continue;
    text(draw, font, 15.0f, ImVec2(o.x + s.x - 20, feedY), fade(it->color, float(6.0 - age)), it->text, 1.0f);
    feedY += 20;
  }

  float noteY = o.y + s.y * 0.2f;
  for (const auto& note : game.notices()) {
    const double age = t - note.time;
    if (age > 3.0) continue;
    const bool big = note.text.rfind("WAVE", 0) == 0 || note.text.find(" - ") != std::string::npos;
    text(draw, font, big ? 40.0f : 24.0f, ImVec2(o.x + 0.5f * s.x, noteY), fade(note.color, float(3.0 - age)), note.text, 0.5f);
    noteY += big ? 48.0f : 30.0f;
  }

  const float slot = 108.0f, gap = 8.0f, total = kWeaponCount * slot + (kWeaponCount - 1) * gap;
  const float bx = o.x + 0.5f * (s.x - total), by = o.y + s.y - 84.0f;
  for (int i = 0; i < kWeaponCount; ++i) {
    const WeaponId id = WeaponId(i);
    const WeaponSpec& w = weapon(id);
    const ImVec2 a(bx + i * (slot + gap), by), b(a.x + slot, a.y + 64);
    const bool selected = game.selected() == id;
    draw->AddRectFilled(a, b, selected ? IM_COL32(40, 46, 30, 220) : IM_COL32(8, 10, 12, 170), 6.0f);
    draw->AddRect(a, b, selected ? rgba(w.color) : IM_COL32(255, 255, 255, 50), 6.0f, 0, selected ? 2.5f : 1.0f);
    text(draw, font, 13.0f, ImVec2(a.x + 6, a.y + 4), IM_COL32(200, 200, 200, 200), std::to_string(i + 1));
    text(draw, font, 16.0f, ImVec2(a.x + 0.5f * slot, a.y + 8), rgba(w.color), w.shortName, 0.5f);
    std::snprintf(line, sizeof(line), "%d", game.ammo(id));
    text(draw, font, 20.0f, ImVec2(a.x + 0.5f * slot, a.y + 28), game.ammo(id) > 0 ? IM_COL32(255, 255, 255, 255) : IM_COL32(255, 90, 70, 255), line, 0.5f);
    bar(draw, ImVec2(a.x + 8, a.y + 54), slot - 16, 4, float(game.ready(id)), rgba(w.color, 0.8f));
  }
  text(draw, font, 15.0f, ImVec2(o.x + 0.5f * s.x, by - 22), rgba(weapon(game.selected()).color, 0.9f), weapon(game.selected()).role, 0.5f);

  const double hurt = t - game.hurtTime();
  if (hurt < 0.6) {
    const int alpha = int(140 * (1.0 - hurt / 0.6));
    const ImU32 edge = IM_COL32(255, 30, 20, alpha), clear = IM_COL32(255, 30, 20, 0);
    const float w = 0.18f * s.x, h = 0.18f * s.y;
    draw->AddRectFilledMultiColor(o, ImVec2(o.x + w, o.y + s.y), edge, clear, clear, edge);
    draw->AddRectFilledMultiColor(ImVec2(o.x + s.x - w, o.y), ImVec2(o.x + s.x, o.y + s.y), clear, edge, edge, clear);
    draw->AddRectFilledMultiColor(o, ImVec2(o.x + s.x, o.y + h), edge, edge, clear, clear);
    draw->AddRectFilledMultiColor(ImVec2(o.x, o.y + s.y - h), ImVec2(o.x + s.x, o.y + s.y), clear, clear, edge, edge);
  }

  const ImVec2 mouse = ImGui::GetIO().MousePos;
  if (game.phase() == Phase::Playing && mouse.x > o.x && mouse.y > o.y && mouse.x < o.x + s.x && mouse.y < o.y + s.y) {
    const double readiness = game.ready(game.selected());
    const ImU32 color = readiness >= 1.0 ? IM_COL32(255, 255, 255, 230) : IM_COL32(255, 160, 80, 230);
    draw->AddCircle(mouse, 14.0f, color, 32, 2.0f);
    draw->PathArcTo(mouse, 19.0f, float(-kPi / 2), float(-kPi / 2 + 2 * kPi * readiness), 32);
    draw->PathStroke(color, 0, 3.0f);
    for (int i = 0; i < 4; ++i) {
      const float a = float(i * kPi / 2);
      draw->AddLine(ImVec2(mouse.x + 6 * std::cos(a), mouse.y + 6 * std::sin(a)), ImVec2(mouse.x + 11 * std::cos(a), mouse.y + 11 * std::sin(a)), color, 2.0f);
    }
    const double marker = t - game.hitMarkerTime();
    if (marker < 0.35) {
      const ImU32 hit = fade(game.hitMarkerColor(), float(1.0 - marker / 0.35));
      for (const auto& [dx, dy] : {std::array<float, 2>{1, 1}, std::array<float, 2>{1, -1}, std::array<float, 2>{-1, 1}, std::array<float, 2>{-1, -1}})
        draw->AddLine(ImVec2(mouse.x + 18 * dx, mouse.y + 18 * dy), ImVec2(mouse.x + 28 * dx, mouse.y + 28 * dy), hit, 3.0f);
    }
  }
  if (help) {
    const char* rows[] = {"W/S drive   A/D turn   mouse aims", "Click/Space fire   1-5 or Q/E weapon",
                          "F smoke screen   wheel zoom", "Hit tanks from the side or rear", "Esc/P pause   R restart"};
    float hy = o.y + s.y - 24.0f * 5 - 104.0f;
    for (const char* row : rows) {
      text(draw, font, 16.0f, ImVec2(o.x + 20, hy), IM_COL32(230, 230, 230, 210), row);
      hy += 22.0f;
    }
  }
}

enum class Choice { None, Play, Resume, Retry, Next, Menu };

struct MenuResult {
  Choice choice = Choice::None;
  int level = 0;
};

inline void beginPanel(const char* id, const Screen& screen, ImVec2 size) {
  ImGui::SetNextWindowPos(ImVec2(screen.origin.x + 0.5f * screen.size.x, screen.origin.y + 0.5f * screen.size.y),
                          ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(size);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(10, 12, 14, 215));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(22, 18));
  ImGui::Begin(id, nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
}

inline void endPanel() {
  ImGui::End();
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor();
}

inline void title(const Fonts& fonts, const std::string& value, ImU32 color) {
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const ImVec2 at = ImGui::GetCursorScreenPos();
  const float width = ImGui::GetContentRegionAvail().x;
  text(draw, fonts.title, 44.0f, ImVec2(at.x + 0.5f * width, at.y), color, value, 0.5f);
  ImGui::Dummy(ImVec2(width, 54));
}

inline MenuResult drawMainMenu(const Screen& screen, const Fonts& fonts, const std::vector<int>& best, int& quality) {
  MenuResult result;
  beginPanel("Main menu", screen, ImVec2(620, 640));
  title(fonts, "RAISIM TANK SURVIVAL", IM_COL32(255, 210, 120, 255));
  ImGui::TextWrapped("Command a tank against armored columns. Every shell is a RaiSim projectile; every blast pushes, "
                     "flips and wrecks what it reaches.");
  ImGui::Spacing();
  for (size_t i = 0; i < levels().size(); ++i) {
    const LevelSpec& level = levels()[i];
    ImGui::PushID(int(i));
    char label[160];
    std::snprintf(label, sizeof(label), "%s%s\n%s", level.endless ? "" : (std::to_string(i + 1) + ".  ").c_str(), level.name,
                  level.subtitle);
    if (ImGui::Button(label, ImVec2(410, 58))) result = {Choice::Play, int(i)};
    ImGui::SameLine();
    if (best[i] > 0) ImGui::Text("best\n%d", best[i]);
    ImGui::PopID();
  }
  ImGui::Spacing();
  const char* qualities[] = {"Fast", "Balanced", "High", "Ultra"};
  ImGui::SetNextItemWidth(200);
  ImGui::Combo("Render quality", &quality, qualities, 4);
  ImGui::TextDisabled("W/S A/D drive, mouse aims, click fires, 1-5 weapons, F smoke");
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  ImGui::TextWrapped("Vehicles by MirzaArrafiERV_45, 42manako and KojfDiscord (Sketchfab, CC BY 4.0); props, plants and "
                     "textures from Poly Haven (CC0).");
  ImGui::PopStyleColor();
  endPanel();
  return result;
}

inline MenuResult drawPauseMenu(const Screen& screen, const Fonts& fonts) {
  MenuResult result;
  beginPanel("Paused", screen, ImVec2(360, 270));
  title(fonts, "PAUSED", IM_COL32(255, 255, 255, 255));
  if (ImGui::Button("Resume", ImVec2(-1, 44))) result.choice = Choice::Resume;
  if (ImGui::Button("Restart mission", ImVec2(-1, 44))) result.choice = Choice::Retry;
  if (ImGui::Button("Main menu", ImVec2(-1, 44))) result.choice = Choice::Menu;
  endPanel();
  return result;
}

inline MenuResult drawResult(const Screen& screen, const Fonts& fonts, const Game& game) {
  MenuResult result;
  const bool won = game.phase() == Phase::Victory;
  const Stats& stats = game.stats();
  beginPanel("Result", screen, ImVec2(470, 420));
  title(fonts, won ? "MISSION COMPLETE" : "DESTROYED", won ? IM_COL32(150, 255, 150, 255) : IM_COL32(255, 90, 60, 255));
  const double accuracy = stats.shots ? 100.0 * stats.hits / stats.shots : 0.0;
  if (won) {
    const int stars = 1 + (game.playerHealth() > 80.0) + (accuracy > 35.0);
    std::string row;
    for (int i = 0; i < 3; ++i) row += i < stars ? "* " : "- ";
    ImGui::SetWindowFontScale(1.6f);
    ImGui::Text("   %s", row.c_str());
    ImGui::SetWindowFontScale(1.0f);
  }
  ImGui::Text("%s  -  wave %d", game.levelSpec().name, std::max(1, game.wave()));
  ImGui::Separator();
  ImGui::Text("Score            %d", game.score());
  ImGui::Text("Time             %s", clock(game.elapsed()).c_str());
  ImGui::Text("Enemies destroyed  %d", stats.kills);
  ImGui::Text("Shots %d   hits %d   penetrations %d   accuracy %.0f%%", stats.shots, stats.hits, stats.penetrations, accuracy);
  ImGui::Text("Damage taken     %.0f", stats.damageTaken);
  ImGui::Spacing();
  const bool more = won && size_t(game.level() + 1) < levels().size();
  if (more && ImGui::Button("Next mission", ImVec2(-1, 42))) result = {Choice::Next, game.level() + 1};
  if (ImGui::Button(won ? "Play again" : "Retry", ImVec2(-1, 42))) result = {Choice::Retry, game.level()};
  if (ImGui::Button("Main menu", ImVec2(-1, 42))) result = {Choice::Menu, game.level()};
  endPanel();
  return result;
}

}  // namespace tank
