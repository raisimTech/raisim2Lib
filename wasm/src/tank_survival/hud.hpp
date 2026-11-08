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
  ImFont* heading = nullptr;
  ImFont* small = nullptr;
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

// The loading screen: the game's title over a dark field, what is being prepared and how far it has
// come.
inline void drawLoading(const Fonts& fonts, double fraction, const char* what) {
  ImDrawList* draw = ImGui::GetBackgroundDrawList();
  const ImVec2 size = ImGui::GetIO().DisplaySize;
  draw->AddRectFilled(ImVec2(0, 0), size, IM_COL32(7, 8, 6, 255));
  const ImVec2 center(0.5f * size.x, 0.5f * size.y);
  text(draw, fonts.title, 46.0f, ImVec2(center.x, center.y - 90.0f), IM_COL32(240, 226, 180, 255), "TANK SURVIVAL", 0.5f);
  const float width = std::min(520.0f, 0.7f * size.x);
  bar(draw, ImVec2(center.x - 0.5f * width, center.y), width, 10.0f, float(std::clamp(fraction, 0.0, 1.0)),
      IM_COL32(214, 186, 96, 255));
  text(draw, fonts.text, 19.0f, ImVec2(center.x, center.y + 24.0f), IM_COL32(200, 200, 190, 255), what, 0.5f);
}

inline ImU32 fade(ImU32 color, float alpha) {
  return (color & 0x00FFFFFFu) | (ImU32(std::clamp(alpha, 0.0f, 1.0f) * float(color >> 24)) << 24);
}

inline std::string clock(double seconds) {
  char line[32];
  std::snprintf(line, sizeof(line), "%d:%02d", int(seconds) / 60, int(seconds) % 60);
  return line;
}

// A horizontal circle around center, seen in perspective.
inline void groundRing(ImDrawList* draw, const Screen& screen, const Vec3& center, double radius, ImU32 color, float thickness) {
  std::array<ImVec2, 33> points;
  for (int i = 0; i <= 32; ++i) {
    const double a = 2.0 * kPi * i / 32;
    if (!screen.project(center + radius * Vec3(std::cos(a), std::sin(a), 0.0), points[size_t(i)])) return;
  }
  draw->AddPolyline(points.data(), int(points.size()), color, ImDrawFlags_None, thickness);
}

// The aiming guide. The sight marks the point the cursor rests on in the world, or brackets the enemy it
// locked on; a dotted arc shows the flight of a round fired now, from the muzzle as it points at this
// moment (the turret takes a moment to follow the cursor), to the ring where it would strike: red when it
// would hit a vehicle.
inline void drawAimGuide(ImDrawList* draw, Game& game, const Screen& screen, const Fonts& fonts, const Tank* locked) {
  const Tank* hero = game.player();
  if (!hero) return;
  if (locked && locked->system) {
    const Eigen::Matrix3d R = locked->hullRotation();
    const VehicleModel& m = locked->model;
    ImVec2 low(1e9f, 1e9f), high(-1e9f, -1e9f);
    for (int corner = 0; corner < 8; ++corner) {
      const Vec3 local((corner & 1 ? m.hullMax : m.hullMin).x(), (corner & 2 ? m.hullMax : m.hullMin).y(),
                       (corner & 4 ? m.turretMax : m.hullMin).z());
      ImVec2 at;
      if (!screen.project(locked->position() + R * local, at)) return;
      low = ImVec2(std::min(low.x, at.x), std::min(low.y, at.y));
      high = ImVec2(std::max(high.x, at.x), std::max(high.y, at.y));
    }
    const ImU32 color = IM_COL32(255, 80, 60, 230);
    const float arm = std::max(8.0f, 0.25f * std::min(high.x - low.x, high.y - low.y));
    for (const auto& [cx, cy, sx, sy] : {std::array<float, 4>{low.x, low.y, 1, 1}, std::array<float, 4>{high.x, low.y, -1, 1},
                                         std::array<float, 4>{low.x, high.y, 1, -1}, std::array<float, 4>{high.x, high.y, -1, -1}}) {
      draw->AddLine(ImVec2(cx, cy), ImVec2(cx + sx * arm, cy), color, 2.5f);
      draw->AddLine(ImVec2(cx, cy), ImVec2(cx, cy + sy * arm), color, 2.5f);
    }
    text(draw, fonts.text, 13.0f, ImVec2(0.5f * (low.x + high.x), low.y - 18.0f), color, "LOCKED", 0.5f);
  } else {
    const Vec3 at = game.aim();
    groundRing(draw, screen, Vec3(at.x(), at.y(), game.height(at.x(), at.y()) + 0.08), 0.9, IM_COL32(255, 255, 255, 150), 2.0f);
  }
  const WeaponId id = game.selected();
  if (id == WeaponId::Missile || id == WeaponId::Artillery) {
    const Vec3 at = game.aim();
    groundRing(draw, screen, Vec3(at.x(), at.y(), game.height(at.x(), at.y()) + 0.08), id == WeaponId::Artillery ? 7.0 : 1.6,
               rgba(weapon(id).color, 0.8f), 2.5f);
    return;
  }
  const WeaponSpec& round = weapon(id);
  Vec3 position;
  const Vec3 direction = Game::roundPath(*hero, id, &game.aim(), game.levelSpec().wind, position);
  Vec3 velocity = direction * round.speed;
  const Vec3 wind = game.levelSpec().wind;
  const double drag = 0.5 * 1.2 * round.dragCoefficient * kPi * round.radius * round.radius / round.mass;
  const ImU32 dots = rgba(round.color, 0.9f), outline = IM_COL32(0, 0, 0, 120);
  double travelled = 0.0;
  for (int i = 0; i < 120 && travelled < 320.0; ++i) {
    const double dt = 5.0 / std::max(velocity.norm(), 1.0);
    const Vec3 relative = velocity - wind;
    velocity += dt * (Vec3(0.0, 0.0, -kGravity) - drag * relative.norm() * relative);
    const Vec3 next = position + dt * velocity, step = next - position;
    const double length = step.norm();
    Vec3 struck;
    bool vehicle = false;
    if (game.sweep(position, step / length, length, struck, vehicle)) {
      groundRing(draw, screen, struck + Vec3(0.0, 0.0, 0.08), 1.3, vehicle ? IM_COL32(255, 70, 50, 240) : IM_COL32(255, 190, 90, 230), 3.0f);
      return;
    }
    ImVec2 at;
    if (i > 0 && screen.project(next, at)) {
      draw->AddCircleFilled(at, 3.6f, outline, 10);
      draw->AddCircleFilled(at, 2.6f, dots, 10);
    }
    position = next;
    travelled += length;
  }
}

// A button under the status panel that opens the help.
inline bool drawHelpButton(const Screen& screen) {
  ImGui::SetNextWindowPos(ImVec2(screen.origin.x + 10, screen.origin.y + 176));
  ImGui::SetNextWindowBgAlpha(0.0f);
  ImGui::Begin("Help button", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing);
  const bool pressed = ImGui::Button("?  How to play  (H)", ImVec2(196, 32));
  ImGui::End();
  return pressed;
}

// While a panel (help or pause) is open, the sight, the notices and the hints stay off it.
// The direction on a screen of size pixels (y down) toward a point out of view at clip-space position clip, or
// zero straight ahead or behind. Clip x and y are not divided by w, which would turn points behind the camera
// around.
inline glm::vec2 offscreenDirection(const glm::vec4& clip, const ImVec2& size) {
  const glm::vec2 direction(clip.x * size.x, -clip.y * size.y);
  return glm::length(direction) < 1e-6f ? glm::vec2(0.0f) : glm::normalize(direction);
}

// The direction (y down) on the radar, which turns with the camera's heading, of a horizontal world vector.
inline glm::vec2 radarDirection(double heading, const Vec3& vector) {
  const double c = std::cos(-heading + kPi / 2), sn = std::sin(-heading + kPi / 2);
  return {float(c * vector.x() - sn * vector.y()), float(-(sn * vector.x() + c * vector.y()))};
}

// Where the sight is drawn: the middle of the view when the mouse turns the camera, or the cursor; whether
// it is over an enemy, whether to offer the mouse for aiming, and the sniper sight's magnification (1 when
// it is not in use).
struct Sight {
  ImVec2 at;
  bool centered = false, onEnemy = false, offerMouse = false;
  double magnification = 1.0;
};

// Aiming down the sight, as in a shooter: while the right button is held the view looks from the gun,
// magnified, and the wheel steps the magnification (it starts at the step last used).
struct AimDownSights {
  static constexpr std::array<double, 4> kMagnifications{2.0, 3.0, 5.0, 8.0};
  bool held = false;
  int step = 1;
  double notches = 0.0;

  // Follows the button and the wheel (fractions of a notch from a touchpad add up).
  void update(bool down, double wheel) {
    held = down;
    if (!held) {
      notches = 0.0;
      return;
    }
    if (wheel * notches < 0.0) notches = 0.0;
    notches += wheel;
    while (std::abs(notches) >= 1.0) {
      const bool in = notches > 0.0;
      notches -= in ? 1.0 : -1.0;
      step = std::clamp(step + (in ? 1 : -1), 0, int(kMagnifications.size()) - 1);
    }
  }
  double magnification() const { return held ? kMagnifications[size_t(step)] : 1.0; }
};

// Index of the point nearest `to` within `radius`, or -1.
inline int nearestWithin(const std::vector<ImVec2>& points, ImVec2 to, float radius) {
  int best = -1;
  for (size_t i = 0; i < points.size(); ++i) {
    const float d = std::hypot(points[i].x - to.x, points[i].y - to.y);
    if (d < radius) {
      radius = d;
      best = int(i);
    }
  }
  return best;
}

inline void drawHud(Game& game, const Screen& screen, const Fonts& fonts, double heading, bool help, const Tank* locked,
                    bool panel, const Sight& sight) {
  ImDrawList* draw = ImGui::GetForegroundDrawList();
  ImFont* font = fonts.text;
  const ImVec2 o = screen.origin, s = screen.size;
  const double t = game.now();
  char line[200];

  for (const auto& enemy : game.enemies()) {
    if (panel || !enemy->system) continue;
    ImVec2 at;
    if (!screen.project(enemy->position() + Vec3(0, 0, 3.2 * enemy->scale()), at)) continue;
    const float width = enemy->spec->role == Role::Boss ? 140.0f : 60.0f;
    bar(draw, ImVec2(at.x - 0.5f * width, at.y), width, 6, float(enemy->health / enemy->spec->health), rgba(enemy->spec->marker));
    if (enemy->spec->role != Role::Medium && enemy->spec->role != Role::Scout)
      text(draw, font, 14.0f, ImVec2(at.x, at.y - 18), rgba(enemy->spec->marker, 0.9f), enemy->spec->name, 0.5f);
    if (t < enemy->tracksBrokenUntil) text(draw, font, 13.0f, ImVec2(at.x, at.y + 8), IM_COL32(255, 200, 120, 220), "IMMOBILIZED", 0.5f);
  }
  // Enemies out of view: an arrow at the edge of the screen points at each, with its distance.
  if (!panel && game.phase() == Phase::Playing) {
    const ImVec2 center(o.x + 0.5f * s.x, o.y + 0.5f * s.y);
    const float halfX = 0.5f * s.x - 70.0f, halfY = 0.5f * s.y - 110.0f;
    const Vec3 from = game.playerTank() && game.playerTank()->system ? game.playerTank()->position() : Vec3::Zero();
    for (const auto& enemy : game.enemies()) {
      if (!enemy->alive()) continue;
      const Vec3 p = enemy->position() + Vec3(0, 0, 0.6 * enemy->scale());
      const glm::vec4 clip = screen.viewProjection * glm::vec4(float(p.x()), float(p.y()), float(p.z()), 1.0f);
      if (clip.w > 1e-4f && std::abs(clip.x) < 0.97f * clip.w && std::abs(clip.y) < 0.97f * clip.w) continue;
      const glm::vec2 direction = offscreenDirection(clip, s);
      if (direction == glm::vec2(0.0f)) continue;
      const float reach = std::min(halfX / std::max(std::abs(direction.x), 1e-4f), halfY / std::max(std::abs(direction.y), 1e-4f));
      const ImVec2 at(center.x + direction.x * reach, center.y + direction.y * reach);
      const glm::vec2 side(-direction.y, direction.x);
      const ImU32 color = rgba(enemy->spec->marker, 0.9f);
      draw->AddTriangleFilled(ImVec2(at.x + 16 * direction.x, at.y + 16 * direction.y),
                              ImVec2(at.x - 8 * direction.x + 10 * side.x, at.y - 8 * direction.y + 10 * side.y),
                              ImVec2(at.x - 8 * direction.x - 10 * side.x, at.y - 8 * direction.y - 10 * side.y), color);
      std::snprintf(line, sizeof(line), "%.0f m", (enemy->position() - from).head<2>().norm());
      text(draw, font, 13.0f, ImVec2(at.x - 18 * direction.x, at.y - 18 * direction.y - 7), color, line, 0.5f);
    }
  }
  for (const auto& pickup : game.pickups()) {
    ImVec2 at;
    if (!panel && screen.project(pickup.position + Vec3(0, 0, 2.4), at))
      text(draw, font, 15.0f, at, rgba(kBonusStyles[size_t(pickup.type)].color), kBonusStyles[size_t(pickup.type)].label, 0.5f);
  }
  for (const auto& note : game.floating()) {
    if (panel) break;
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
  // An arrow beside it shows where the wind blows, as the radar shows the arena.
  if (wind.head<2>().norm() > 0.1) {
    const glm::vec2 d = glm::normalize(radarDirection(heading, wind)), side(-d.y, d.x);
    const ImVec2 at(o.x + 0.5f * s.x + 52.0f, o.y + 47.0f);
    const ImU32 color = IM_COL32(200, 220, 255, 200);
    draw->AddLine(ImVec2(at.x - 8 * d.x, at.y - 8 * d.y), ImVec2(at.x + 4 * d.x, at.y + 4 * d.y), color, 2.0f);
    draw->AddTriangleFilled(ImVec2(at.x + 9 * d.x, at.y + 9 * d.y), ImVec2(at.x + 2 * d.x + 4 * side.x, at.y + 2 * d.y + 4 * side.y),
                            ImVec2(at.x + 2 * d.x - 4 * side.x, at.y + 2 * d.y - 4 * side.y), color);
  }

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
    const glm::vec2 d = radarDirection(heading, p - center) * (radius / 90.0f);
    ImVec2 at(radar.x + d.x, radar.y + d.y);
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
    if (panel) break;
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

  const ImVec2 mouse = sight.at;
  // With the mouse captured the sight alone aims, as in a shooter; aiming at the cursor shows where a round
  // would fly.
  if (game.phase() == Phase::Playing && !panel && !sight.centered) drawAimGuide(draw, game, screen, fonts, locked);
  if (game.phase() == Phase::Playing && !panel && sight.offerMouse)
    text(draw, font, 15.0f, ImVec2(o.x + 0.5f * s.x, o.y + 0.5f * s.y + 44.0f), IM_COL32(255, 255, 255, 210),
         "Click to aim with the mouse", 0.5f);
  if (game.phase() == Phase::Playing && !panel && sight.magnification > 1.5) {
    const ImU32 lens = IM_COL32(0, 0, 0, 150);
    const float radius = 0.46f * std::min(s.x, s.y);
    draw->AddCircle(mouse, radius, lens, 96, 6.0f);
    draw->AddLine(ImVec2(mouse.x - radius, mouse.y), ImVec2(mouse.x - 30.0f, mouse.y), lens, 2.0f);
    draw->AddLine(ImVec2(mouse.x + 30.0f, mouse.y), ImVec2(mouse.x + radius, mouse.y), lens, 2.0f);
    draw->AddLine(ImVec2(mouse.x, mouse.y + 30.0f), ImVec2(mouse.x, mouse.y + radius), lens, 2.0f);
    std::snprintf(line, sizeof(line), "x%.0f", sight.magnification);
    text(draw, font, 18.0f, ImVec2(mouse.x + 0.7f * radius, mouse.y + 0.7f * radius), IM_COL32(255, 255, 255, 200), line, 0.5f);
  }
  if (game.phase() == Phase::Playing && !panel && mouse.x > o.x && mouse.y > o.y && mouse.x < o.x + s.x && mouse.y < o.y + s.y) {
    const double readiness = game.ready(game.selected());
    const ImU32 color = sight.onEnemy       ? IM_COL32(255, 80, 60, 240)
                        : readiness >= 1.0 ? IM_COL32(255, 255, 255, 230)
                                           : IM_COL32(255, 160, 80, 230);
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
  if (help && !panel) {
    const char* rows[] = {"W/S drive   A/D turn   Mouse aims   Click/Space fire",
                          "Hold right: aim down the sight   Wheel: zoom",
                          "1-5 or Q/E weapon", "Hit tanks from the side or rear", "H how to play   Esc/P pause"};
    float hy = o.y + s.y - 24.0f * 5 - 104.0f;
    for (const char* row : rows) {
      text(draw, font, 16.0f, ImVec2(o.x + 20, hy), IM_COL32(230, 230, 230, 210), row);
      hy += 22.0f;
    }
  }
}

enum class Choice { None, Play, Resume, Retry, Next, Menu, Help };

struct MenuResult {
  Choice choice = Choice::None;
  int level = 0;
  // The player picked a render quality, maybe the one already set.
  bool qualityPicked = false;
};

inline void beginPanel(const char* id, const Screen& screen, ImVec2 size) {
  ImGui::SetNextWindowPos(ImVec2(screen.origin.x + 0.5f * screen.size.x, screen.origin.y + 0.5f * screen.size.y),
                          ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  // A height of 0 fits the panel to what it holds, at most the height of the view.
  ImGui::SetNextWindowSize(size);
  ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(FLT_MAX, std::max(120.0f, screen.size.y - 20.0f)));
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

// The direction the camera looks at the player's tank from: its heading and how far it looks down
// (radians). The captured mouse turns it, as in World of Tanks; without it a drag turns the heading. It
// stays put while the hull turns under it; asked to, it swings quickly behind the hull. The autopilot's
// camera follows the hull.
struct OrbitHeading;

// The kick of a shot on the view: it turns the view up and settles back by as much as it turned (at the top
// of the view's range, by nothing).
struct Recoil {
  double left = 0.0;
  void kick(OrbitHeading& orbit, double angle);
  void settle(OrbitHeading& orbit, double dt);
};

struct OrbitHeading {
  static constexpr double kPitch = -0.12, kLowest = -1.0, kHighest = 0.35;
  // Radians the view turns per pixel the mouse moves.
  static constexpr double kSensitivity = 0.0022;
  double heading = 0.0, pitch = kPitch;
  bool behindHull = false;

  void drag(double pixels) {
    heading -= 0.005 * pixels;
    behindHull = false;
  }
  // Turns the view by a mouse movement; `scale` slows it down in the sniper sight.
  void look(double dx, double dy, double scale) {
    heading = wrapAngle(heading - kSensitivity * scale * dx);
    pitch = std::clamp(pitch - kSensitivity * scale * dy, kLowest, kHighest);
    if (dx != 0.0) behindHull = false;
  }
  void place(double hull) {
    heading = hull;
    pitch = kPitch;
    behindHull = false;
  }
  void update(double hull, double dt, bool follow) {
    if (follow) {
      heading += wrapAngle(hull - heading) * std::min(1.0, dt * 2.5);
    } else if (behindHull) {
      const double error = wrapAngle(hull - heading);
      heading += error * std::min(1.0, dt * 12.0);
      behindHull = std::abs(error) > 0.002;
    }
  }
};

inline void Recoil::kick(OrbitHeading& orbit, double angle) {
  const double before = orbit.pitch;
  orbit.look(0.0, -angle / OrbitHeading::kSensitivity, 1.0);
  left += orbit.pitch - before;
}

inline void Recoil::settle(OrbitHeading& orbit, double dt) {
  if (left <= 0.0) return;
  const double back = left * std::min(1.0, dt * 12.0);
  orbit.look(0.0, back / OrbitHeading::kSensitivity, 1.0);
  left -= back;
}

// A key that acts once it has been held for a moment, so that a slip onto it does nothing; it acts again
// only after it is let go.
struct HoldToConfirm {
  double seconds = 0.8, held = 0.0;
  bool released = true;

  // Whether the key, down (or not) for another dt seconds, has just been held long enough.
  bool update(bool down, double dt) {
    if (!down) {
      held = 0.0;
      released = true;
      return false;
    }
    if (!released) return false;
    held += dt;
    if (held < seconds) return false;
    held = 0.0;
    released = false;
    return true;
  }
  double fraction() const { return std::min(1.0, held / seconds); }
};

// The progress of a held key over the weapon bar, as it fills.
inline void drawHold(const Screen& screen, const Fonts& fonts, const char* what, double fraction) {
  ImDrawList* draw = ImGui::GetForegroundDrawList();
  const ImVec2 at(screen.origin.x + 0.5f * screen.size.x, screen.origin.y + screen.size.y - 150.0f);
  text(draw, fonts.text, 18.0f, ImVec2(at.x, at.y - 24.0f), IM_COL32(255, 230, 190, 240), what, 0.5f);
  bar(draw, ImVec2(at.x - 90.0f, at.y), 180.0f, 6.0f, float(fraction), IM_COL32(255, 200, 120, 255));
}

// Lowers the render quality while the menu's backdrop draws slowly, until the player picks one: once a load
// has settled, the median of 30 frames longer than 1/30 s steps the quality down, at most to Fast. Below 25
// frames a second the game slows down, as a frame's physics steps stop keeping up with the clock.
struct QualityTuner {
  bool active = true;
  int settle = 60;
  std::vector<double> frames;

  // The quality to draw with after a frame of frameSeconds at quality.
  int update(int quality, double frameSeconds) {
    if (!active || quality <= 0) return quality;
    if (settle > 0) {
      --settle;
      return quality;
    }
    frames.push_back(frameSeconds);
    if (frames.size() < 30) return quality;
    std::nth_element(frames.begin(), frames.begin() + 15, frames.end());
    const bool slow = frames[15] > 1.0 / 30.0;
    frames.clear();
    settle = 60;
    active = slow;
    return slow ? quality - 1 : quality;
  }

  // Starts timing anew once the next frames settle, after frames that are not the menu's.
  void restart() {
    frames.clear();
    settle = 60;
  }
};

// Text wrapped to the panel's width, each line centered.
inline void centeredText(const char* value, ImU32 color) {
  ImFont* font = ImGui::GetFont();
  const float width = ImGui::GetContentRegionAvail().x, size = ImGui::GetFontSize(), left = ImGui::GetCursorPosX();
  const char* end = value + std::strlen(value);
  ImGui::PushStyleColor(ImGuiCol_Text, color);
  while (value < end) {
    const char* cut = font->CalcWordWrapPositionA(size / font->FontSize, value, end, width);
    if (cut == value) cut = value + 1;
    ImGui::SetCursorPosX(left + 0.5f * (width - font->CalcTextSizeA(size, FLT_MAX, 0.0f, value, cut).x));
    ImGui::TextUnformatted(value, cut);
    for (value = cut; value < end && *value == ' ';) ++value;
  }
  ImGui::PopStyleColor();
}

// A mission on the main menu: its number, name and setting on the left, its waves and the best score on
// the right. Returns true when picked.
inline bool missionCard(const Fonts& fonts, int index, const LevelSpec& level, int best) {
  constexpr float kHeight = 66.0f, kBadge = 46.0f, kPad = 10.0f;
  const ImVec2 at = ImGui::GetCursorScreenPos();
  const float width = ImGui::GetContentRegionAvail().x;
  ImGui::PushID(index);
  const bool picked = ImGui::InvisibleButton("mission", ImVec2(width, kHeight));
  ImGui::PopID();
  const bool hovered = ImGui::IsItemHovered(), pressed = ImGui::IsItemActive();
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const ImVec2 end(at.x + width, at.y + kHeight);
  draw->AddRectFilled(at, end, pressed ? IM_COL32(78, 88, 48, 250) : hovered ? IM_COL32(58, 66, 38, 245) : IM_COL32(32, 36, 27, 235),
                      8.0f);
  draw->AddRect(at, end, hovered ? IM_COL32(255, 210, 120, 210) : IM_COL32(255, 255, 255, 22), 8.0f, 0, hovered ? 1.5f : 1.0f);
  const ImVec2 badge(at.x + kPad, at.y + 0.5f * (kHeight - kBadge));
  draw->AddRectFilled(badge, ImVec2(badge.x + kBadge, badge.y + kBadge), IM_COL32(10, 12, 9, 230), 6.0f);
  const std::string mark = level.endless ? "\xE2\x88\x9E" : std::to_string(index + 1);
  const float markSize = level.endless ? 38.0f : 28.0f;
  const ImVec2 markExtent = fonts.title->CalcTextSizeA(markSize, FLT_MAX, 0.0f, mark.c_str());
  draw->AddText(fonts.title, markSize,
                ImVec2(badge.x + 0.5f * (kBadge - markExtent.x), badge.y + 0.5f * (kBadge - markExtent.y)),
                IM_COL32(255, 210, 120, 255), mark.c_str());
  const float left = badge.x + kBadge + 14.0f, right = end.x - 14.0f;
  draw->AddText(fonts.heading, 24.0f, ImVec2(left, at.y + 9.0f), IM_COL32(244, 240, 226, 255), level.name);
  draw->AddText(fonts.text, 19.0f, ImVec2(left, at.y + 37.0f), IM_COL32(186, 192, 170, 255), level.subtitle);
  const std::string waves = level.endless ? "ENDLESS" : std::to_string(level.waves.size()) + " WAVES";
  const float wavesWidth = fonts.small->CalcTextSizeA(15.0f, FLT_MAX, 0.0f, waves.c_str()).x;
  if (best > 0) {
    const std::string score = std::to_string(best);
    const float scoreWidth = fonts.heading->CalcTextSizeA(24.0f, FLT_MAX, 0.0f, score.c_str()).x;
    draw->AddText(fonts.small, 15.0f, ImVec2(right - wavesWidth, at.y + 11.0f), IM_COL32(150, 156, 138, 255), waves.c_str());
    draw->AddText(fonts.small, 15.0f, ImVec2(right - 112.0f, at.y + 39.0f), IM_COL32(150, 156, 138, 255), "BEST");
    draw->AddText(fonts.heading, 24.0f, ImVec2(right - scoreWidth, at.y + 33.0f), IM_COL32(255, 210, 120, 255), score.c_str());
  } else {
    draw->AddText(fonts.small, 15.0f, ImVec2(right - wavesWidth, at.y + 0.5f * (kHeight - 15.0f)), IM_COL32(150, 156, 138, 255),
                  waves.c_str());
  }
  return picked;
}

inline MenuResult drawMainMenu(const Screen& screen, const Fonts& fonts, const std::vector<int>& best, int& quality,
                               const char* note = nullptr) {
  MenuResult result;
  beginPanel("Main menu", screen, ImVec2(std::min(640.0f, screen.size.x - 24.0f), 0.0f));
  title(fonts, "RAISIM TANK SURVIVAL", IM_COL32(255, 210, 120, 255));
  centeredText("Command a tank against armored columns. Every shell is a RaiSim projectile; every blast pushes, flips "
               "and wrecks what it reaches.",
               IM_COL32(214, 216, 204, 255));
  ImGui::Dummy(ImVec2(0.0f, 10.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 8.0f));
  for (size_t i = 0; i < levels().size(); ++i)
    if (missionCard(fonts, int(i), levels()[i], best[i])) result = {Choice::Play, int(i)};
  ImGui::PopStyleVar();
  ImGui::Dummy(ImVec2(0.0f, 4.0f));
  const char* qualities[] = {"Fast", "Balanced", "High", "Ultra"};
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Render quality");
  ImGui::SameLine(0.0f, 12.0f);
  ImGui::SetNextItemWidth(150.0f);
  result.qualityPicked = ImGui::Combo("##quality", &quality, qualities, 4);
  constexpr float kHelpWidth = 150.0f;
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - kHelpWidth);
  if (ImGui::Button("How to play", ImVec2(kHelpWidth, 0.0f))) result.choice = Choice::Help;
  if (note) ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.5f, 1.0f), "%s", note);
  ImGui::Dummy(ImVec2(0.0f, 2.0f));
  ImGui::PushFont(fonts.small);
  ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(140, 144, 132, 255));
  ImGui::TextWrapped("Vehicles by MirzaArrafiERV_45, 42manako and KojfDiscord (Sketchfab, CC BY 4.0); props, plants and "
                     "textures from Poly Haven (CC0).");
  ImGui::PopStyleColor();
  ImGui::PopFont();
  endPanel();
  return result;
}

inline MenuResult drawPauseMenu(const Screen& screen, const Fonts& fonts) {
  MenuResult result;
  beginPanel("Paused", screen, ImVec2(360, 322));
  title(fonts, "PAUSED", IM_COL32(255, 255, 255, 255));
  if (ImGui::Button("Resume", ImVec2(-1, 44))) result.choice = Choice::Resume;
  if (ImGui::Button("How to play", ImVec2(-1, 44))) result.choice = Choice::Help;
  if (ImGui::Button("Restart mission", ImVec2(-1, 44))) result.choice = Choice::Retry;
  if (ImGui::Button("Main menu", ImVec2(-1, 44))) result.choice = Choice::Menu;
  endPanel();
  return result;
}

// How to play: the controls, how aiming works and a few tips, in two columns on a wide screen. Returns true
// when closed.
inline bool drawHelp(const Screen& screen, const Fonts& fonts) {
  const bool wide = screen.size.x >= 1100.0f;
  beginPanel("How to play", screen,
             ImVec2(std::min(wide ? 1040.0f : 760.0f, screen.size.x - 40.0f), std::min(wide ? 600.0f : 700.0f, screen.size.y - 40.0f)));
  title(fonts, "HOW TO PLAY", IM_COL32(255, 210, 120, 255));
  const auto section = [](const char* name) {
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 210, 120, 255));
    ImGui::TextUnformatted(name);
    ImGui::PopStyleColor();
    ImGui::Separator();
  };
  const auto row = [](const char* keys, const char* what) {
    const float x = ImGui::GetCursorPosX();
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(150, 220, 255, 255));
    ImGui::TextUnformatted(keys);
    ImGui::PopStyleColor();
    ImGui::SameLine(x + 150.0f);
    ImGui::TextWrapped("%s", what);
  };
  const auto controls = [&] {
    ImGui::TextWrapped("Destroy every enemy vehicle, wave after wave, and keep your tank alive.");
    section("DRIVE");
    row("W / S, arrows", "Drive forward and back.");
    row("A / D", "Turn the hull. The turret keeps aiming where you look.");
    row("", "Your tank flattens small trees: drive through the woods.");
    section("AIM AND FIRE");
    row("Mouse", "Click the view once, then move the mouse to look and aim, as in a shooter: every round lands "
                 "where the crosshair in the middle of the view is, near or far. It turns red over an enemy.");
    row("Click / Space", "Fire. The ring around the crosshair fills while the gun reloads.");
    row("Right (hold)", "Aim down the sight: the view from the gun, magnified three times; the wheel zooms from "
                        "two to eight times.");
    row("Esc", "Free the mouse and pause. On a touch screen, tap where to aim: a dotted arc shows where a round "
               "would fly.");
  };
  const auto more = [&] {
    section("WEAPONS");
    row("1 - 5,  Q / E", "Armor-piercing (AP), high explosive (HE), machine gun (MG), guided missile, artillery "
                         "strike (lands under the sight after a delay).");
    section("CAMERA AND MORE");
    row("Mouse wheel", "Move the camera closer or farther.");
    row("C", "Camera behind the tank. The view stays where you leave it while the tank turns.");
    row("F", "Smoke screen: enemies cannot see through it.");
    row("Esc / P", "Pause.");
    row("R (hold)", "Restart the mission.");
    row("H", "This help.");
    section("TIPS");
    for (const char* tip : {"Armor is thickest in front: hit enemies in the side or rear.",
                            "HE wrecks light vehicles, trees and crates; AP pierces heavy armor.",
                            "Drive over supply drops for ammunition, repairs and power-ups.",
                            "Keep moving: artillery shells land where you were."}) {
      ImGui::Bullet();
      ImGui::TextWrapped("%s", tip);
    }
  };
  if (wide) {
    const float half = 0.5f * (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x);
    ImGui::BeginChild("help controls", ImVec2(half, -54));
    controls();
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("help more", ImVec2(half, -54));
    more();
    ImGui::EndChild();
  } else {
    ImGui::BeginChild("help text", ImVec2(0, -54));
    controls();
    more();
    ImGui::EndChild();
  }
  const bool close = ImGui::Button("Got it", ImVec2(-1, 44)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false) ||
                     ImGui::IsKeyPressed(ImGuiKey_Enter, false);
  endPanel();
  return close;
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
