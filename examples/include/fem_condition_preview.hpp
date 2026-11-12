// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <imgui/imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

#include "fem_workbench_conditions.hpp"

namespace fem_workbench {
inline const char* conditionDescription(Kind kind) {
  switch (kind) {
    case Kind::Fixed:
      return "Holds the selected faces in place: they cannot move or rotate, "
             "and the rest of the part deforms under the loads.";
    case Kind::Displacement:
      return "Moves the selected faces by a set distance along the chosen "
             "axes; the other axes stay free.";
    case Kind::Force:
      return "Applies a total force spread over the selected faces by area, "
             "normal to them, along X/Y/Z or normal to a reference face.";
    case Kind::Traction:
      return "Applies a force per area (Pa) in one fixed direction at every "
             "point of the selected faces.";
    case Kind::Pressure:
      return "Pushes normal to the selected faces with a force per area (Pa); "
             "positive pressure presses into the part.";
    case Kind::Gravity:
      return "Adds the part's weight from its density and the acceleration. "
             "It acts on the whole part and needs no selection.";
    case Kind::Directional:
      return "Blocks motion along one direction only. On a symmetry plane it "
             "lets you model half of a symmetric part.";
    case Kind::Roller:
      return "Stops motion normal to the selected faces; they can still slide "
             "along their surface.";
    case Kind::Slider:
      return "Lets the selected faces move only along the chosen direction, "
             "like a carriage on a rail.";
    case Kind::Bearing:
      return "Holds a cylindrical face on its axis while it can still turn "
             "about that axis, like a shaft in a bearing.";
    case Kind::BoltHole:
      return "Bolts a hole to the ground: the shaft and the seat around the "
             "hole are held, and the rest of the part is free.";
    case Kind::BoltConnector:
      return "Joins two hole patches with an elastic bolt that carries "
             "tension, shear and bending, with optional preload.";
    case Kind::Remote:
      return "Moves the selected faces rigidly with a remote point by the "
             "given translation and rotation.";
    case Kind::Ball:
      return "Holds the center of the selected faces at a point; the part can "
             "rotate freely about it.";
    case Kind::Spring:
      return "Supports the selected faces with distributed springs of the "
             "given stiffness instead of holding them rigidly.";
    case Kind::Tie:
      return "Glues two touching surfaces so that they move together, even "
             "where their meshes do not match.";
    case Kind::Hinge:
      return "Joins a pin or shaft to its hole: the parts turn relative to "
             "each other about the axis and share all other motion.";
    case Kind::BallJoint:
      return "Joins a ball to its socket: the faces stay together but the "
             "parts turn freely relative to each other.";
    case Kind::SliderJoint:
      return "Lets one part slide along another in one direction; rotation "
             "and sideways motion are shared.";
    case Kind::RigidLink:
      return "Connects two face sets rigidly, like a weld between the parts, "
             "without making the parts themselves rigid.";
  }
  return "";
}

namespace preview {
constexpr ImU32 kBackground = IM_COL32(222, 223, 226, 255);
constexpr ImU32 kSolid = IM_COL32(96, 148, 200, 255);
constexpr ImU32 kEdge = IM_COL32(28, 44, 62, 255);
constexpr ImU32 kRest = IM_COL32(110, 118, 130, 150);
constexpr ImU32 kSupport = IM_COL32(40, 170, 90, 255);
constexpr ImU32 kLoad = IM_COL32(205, 70, 205, 255);
constexpr ImU32 kGround = IM_COL32(95, 102, 112, 255);
constexpr float kLength = 1.f, kHeight = .25f;

struct Vec2 {
  float x, y;
};
inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(Vec2 a, float s) { return {a.x * s, a.y * s}; }
inline Vec2 rotate(Vec2 p, float angle) {
  const float c = std::cos(angle), s = std::sin(angle);
  return {c * p.x - s * p.y, s * p.x + c * p.y};
}

struct Piece {
  float x0, x1, y0, y1;
  std::function<Vec2(Vec2)> displacement;
};

class Canvas {
 public:
  Canvas(ImDrawList* draw, ImVec2 origin, float scale)
      : draw_(draw), origin_(origin), scale_(scale) {}
  ImVec2 operator()(Vec2 p) const {
    return {origin_.x + p.x * scale_, origin_.y - p.y * scale_};
  }
  float pixels(float length) const { return length * scale_; }
  void piece(const Piece& piece) const {
    const int nx = std::max(2, int(std::lround((piece.x1 - piece.x0) * 16.f)));
    constexpr int ny = 3;
    const auto at = [&](int i, int j) {
      const Vec2 p{piece.x0 + (piece.x1 - piece.x0) * float(i) / float(nx),
                   piece.y0 + (piece.y1 - piece.y0) * float(j) / float(ny)};
      return p + piece.displacement(p);
    };
    draw_->AddRect((*this)(Vec2{piece.x0, piece.y1}),
                   (*this)(Vec2{piece.x1, piece.y0}), kRest, 0.f, 0, 1.f);
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        draw_->AddQuadFilled((*this)(at(i, j)), (*this)(at(i + 1, j)),
                             (*this)(at(i + 1, j + 1)), (*this)(at(i, j + 1)),
                             kSolid);
    std::vector<ImVec2> outline;
    for (int i = 0; i <= nx; ++i) outline.push_back((*this)(at(i, 0)));
    for (int j = 1; j <= ny; ++j) outline.push_back((*this)(at(nx, j)));
    for (int i = nx - 1; i >= 0; --i) outline.push_back((*this)(at(i, ny)));
    for (int j = ny - 1; j > 0; --j) outline.push_back((*this)(at(0, j)));
    draw_->AddPolyline(outline.data(), int(outline.size()), kEdge,
                       ImDrawFlags_Closed, 1.5f);
  }
  void arrow(Vec2 from, Vec2 to, ImU32 color, float width = 2.f) const {
    const ImVec2 a = (*this)(from), b = (*this)(to);
    const float dx = b.x - a.x, dy = b.y - a.y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length < 2.f) return;
    const float ux = dx / length, uy = dy / length, head = std::min(8.f, length);
    const ImVec2 base(b.x - ux * head, b.y - uy * head);
    draw_->AddLine(a, base, color, width);
    draw_->AddTriangleFilled(
        b, ImVec2(base.x - uy * head * .5f, base.y + ux * head * .5f),
        ImVec2(base.x + uy * head * .5f, base.y - ux * head * .5f), color);
  }
  void wall(float x, float y0, float y1, bool left, ImU32 color) const {
    draw_->AddLine((*this)(Vec2{x, y0}), (*this)(Vec2{x, y1}), color, 3.f);
    const float step = .04f, depth = left ? -.05f : .05f;
    for (float y = y0; y < y1 - 1e-4f; y += step)
      draw_->AddLine((*this)(Vec2{x, y + step}),
                     (*this)(Vec2{x + depth, y}), color, 1.f);
  }
  void floor(float x0, float x1, float y, ImU32 color) const {
    draw_->AddLine((*this)(Vec2{x0, y}), (*this)(Vec2{x1, y}), color, 3.f);
    for (float x = x0; x < x1 - 1e-4f; x += .05f)
      draw_->AddLine((*this)(Vec2{x, y}), (*this)(Vec2{x + .04f, y - .05f}),
                     color, 1.f);
  }
  void circle(Vec2 center, float radius, ImU32 color, bool filled,
              float width = 2.f) const {
    if (filled)
      draw_->AddCircleFilled((*this)(center), pixels(radius), color);
    else
      draw_->AddCircle((*this)(center), pixels(radius), color, 0, width);
  }
  void spring(Vec2 bottom, Vec2 top, ImU32 color) const {
    constexpr int turns = 6;
    std::array<ImVec2, 2 * turns + 2> points;
    points[0] = (*this)(bottom);
    for (int i = 1; i <= 2 * turns; ++i) {
      const float s = float(i) / float(2 * turns + 1);
      const Vec2 p = bottom + (top - bottom) * s;
      points[size_t(i)] = (*this)(Vec2{p.x + (i % 2 ? .02f : -.02f), p.y});
    }
    points.back() = (*this)(top);
    draw_->AddPolyline(points.data(), int(points.size()), color, 0, 1.5f);
  }
  void dashed(Vec2 from, Vec2 to, ImU32 color) const {
    constexpr int dashes = 8;
    for (int i = 0; i < dashes; ++i) {
      const Vec2 a = from + (to - from) * (float(2 * i) / float(2 * dashes)),
                 b = from + (to - from) * (float(2 * i + 1) / float(2 * dashes));
      draw_->AddLine((*this)(a), (*this)(b), color, 1.f);
    }
  }
  void label(Vec2 at, ImU32 color, const char* text) const {
    draw_->AddText((*this)(at), color, text);
  }

 private:
  ImDrawList* draw_;
  ImVec2 origin_;
  float scale_;
};

inline Vec2 cantilever(Vec2 p, float tip) {
  const float x = p.x / kLength;
  const float v = tip * x * x * (3.f - x) * .5f,
              slope = tip * 3.f * x * (2.f - x) * .5f / kLength;
  return {-(p.y - kHeight * .5f) * slope, v};
}
inline Vec2 uniformLoad(Vec2 p, float tip) {
  const float x = p.x / kLength;
  const float v = tip * x * x * (6.f - 4.f * x + x * x) / 3.f,
              slope = tip * (12.f * x - 12.f * x * x + 4.f * x * x * x) / 3.f /
                      kLength;
  return {-(p.y - kHeight * .5f) * slope, v};
}
inline Vec2 stretch(Vec2 p, float amount) {
  return {amount * p.x / kLength,
          -.3f * amount * (p.y - kHeight * .5f) / kLength};
}
inline Vec2 spin(Vec2 p, Vec2 center, float angle) {
  return rotate(p - center, angle) - (p - center);
}
}  // namespace preview

/// Draws a looping sketch of what a support or load does on a bar, followed by
/// a short description. Call inside a tooltip or any ImGui window.
inline void drawConditionPreview(Kind kind, double time) {
  using namespace preview;
  const ImVec2 size(250.f, 128.f), start = ImGui::GetCursorScreenPos();
  ImGui::Dummy(size);
  auto* draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(start, ImVec2(start.x + size.x, start.y + size.y),
                      kBackground, 4.f);
  draw->PushClipRect(start, ImVec2(start.x + size.x, start.y + size.y), true);
  const Canvas c(draw, ImVec2(start.x + 38.f, start.y + 84.f), 150.f);
  const float t = .5f - .5f * std::cos(float(time) * 2.4f);
  const auto at = [](const Piece& piece, Vec2 p) {
    return p + piece.displacement(p);
  };
  std::vector<Piece> pieces;
  const auto bar = [&](std::function<Vec2(Vec2)> u) {
    pieces.push_back({0.f, kLength, 0.f, kHeight, std::move(u)});
    return pieces.back();
  };
  switch (kind) {
    case Kind::Fixed: {
      const auto p = bar([&](Vec2 q) { return cantilever(q, -.16f * t); });
      c.piece(p);
      c.wall(0.f, -.05f, kHeight + .05f, true, kSupport);
      const Vec2 tip = at(p, {kLength, kHeight * .5f});
      c.arrow(tip + Vec2{0.f, .2f}, tip, kRest);
      break;
    }
    case Kind::Displacement: {
      const auto p = bar([&](Vec2 q) { return stretch(q, .14f * t); });
      c.piece(p);
      c.wall(0.f, -.05f, kHeight + .05f, true, kGround);
      const Vec2 end = at(p, {kLength, kHeight * .5f});
      c.arrow({kLength, kHeight * .5f}, end + Vec2{.05f, 0.f}, kSupport, 2.5f);
      c.label({kLength - .42f, kHeight + .14f}, kSupport, "set distance");
      break;
    }
    case Kind::Force:
    case Kind::Traction: {
      const auto p = bar([&](Vec2 q) { return cantilever(q, -.16f * t); });
      c.piece(p);
      c.wall(0.f, -.05f, kHeight + .05f, true, kGround);
      if (kind == Kind::Force) {
        const Vec2 tip = at(p, {kLength, kHeight * .5f});
        c.arrow(tip + Vec2{0.f, .08f + .2f * t}, tip, kLoad, 3.f);
      } else {
        for (float y = .03f; y < kHeight; y += .06f) {
          const Vec2 point = at(p, {kLength, y});
          c.arrow(point + Vec2{.02f, .1f}, point + Vec2{.02f, 0.f}, kLoad,
                  1.5f);
        }
      }
      break;
    }
    case Kind::Pressure:
    case Kind::Gravity: {
      const auto p = bar([&](Vec2 q) { return uniformLoad(q, -.12f * t); });
      c.piece(p);
      c.wall(0.f, -.05f, kHeight + .05f, true, kGround);
      if (kind == Kind::Pressure) {
        for (float x = .08f; x < kLength; x += .12f) {
          const Vec2 point = at(p, {x, kHeight});
          c.arrow(point + Vec2{0.f, .12f}, point, kLoad, 1.5f);
        }
      } else {
        for (float x = .15f; x < kLength; x += .2f) {
          const Vec2 point = at(p, {x, kHeight * .75f});
          c.arrow(point, point - Vec2{0.f, .1f}, kLoad, 1.5f);
        }
        c.label({kLength + .03f, kHeight + .12f}, kLoad, "g");
        c.arrow({kLength + .07f, kHeight + .05f},
                {kLength + .07f, kHeight - .12f}, kLoad, 2.f);
      }
      break;
    }
    case Kind::Directional: {
      const auto p = bar([&](Vec2 q) { return stretch(q, .14f * t); });
      c.piece(p);
      c.wall(-.06f, -.06f, kHeight + .06f, true, kSupport);
      for (float y = .04f; y < kHeight; y += .085f)
        c.circle(at(p, {0.f, y}) - Vec2{.03f, 0.f}, .025f, kSupport, false);
      const Vec2 end = at(p, {kLength, kHeight * .5f});
      c.arrow(end, end + Vec2{.12f, 0.f}, kLoad, 3.f);
      break;
    }
    case Kind::Roller: {
      const float slide = .16f * t;
      const auto p = bar([&](Vec2) { return Vec2{slide, 0.f}; });
      c.piece(p);
      c.floor(-.1f, kLength + .3f, -.06f, kGround);
      for (float x = .1f; x < kLength; x += .2f)
        c.circle(at(p, {x, 0.f}) - Vec2{0.f, .03f}, .028f, kSupport, false);
      const Vec2 end = at(p, {0.f, kHeight * .5f});
      c.arrow(end - Vec2{.16f, 0.f}, end, kLoad, 3.f);
      break;
    }
    case Kind::Slider: {
      const float slide = .12f * t;
      const auto p = bar([&](Vec2 q) {
        const auto bend = cantilever(q, -.09f * t);
        return Vec2{slide + bend.x, bend.y};
      });
      c.piece(p);
      for (const float y : {-.03f, kHeight + .03f})
        draw->AddLine(c(Vec2{-.15f, y}), c(Vec2{.35f, y}), kSupport, 3.f);
      const Vec2 tip = at(p, {kLength, kHeight * .5f});
      c.arrow(tip + Vec2{-.1f, .14f}, tip, kLoad, 3.f);
      c.label({-.15f, kHeight + .2f}, kSupport, "rail");
      break;
    }
    case Kind::Bearing:
    case Kind::Ball: {
      const Vec2 pivot{.15f, kHeight * .5f};
      const float angle = -.35f * t;
      const auto p = bar([&](Vec2 q) { return spin(q, pivot, angle); });
      c.piece(p);
      if (kind == Kind::Bearing) {
        c.circle(pivot, .07f, kBackground, true);
        c.circle(pivot, .055f, kSupport, true);
        c.circle(pivot, .07f, kEdge, false, 1.5f);
        const float swing = angle * 3.f;
        draw->PathArcTo(c(pivot), c.pixels(.11f), -1.2f - swing, -.2f - swing);
        draw->PathStroke(kSupport, 0, 1.5f);
      } else {
        c.circle(pivot, .055f, kSupport, true);
        c.circle(pivot - Vec2{-.015f, -.015f}, .015f,
                 IM_COL32(255, 255, 255, 180), true);
      }
      const Vec2 tip = at(p, {kLength, kHeight * .5f});
      c.arrow(tip + Vec2{0.f, .16f}, tip, kLoad, 3.f);
      break;
    }
    case Kind::BoltHole: {
      const Vec2 hole{.15f, kHeight * .5f};
      const auto p = bar([&](Vec2 q) {
        if (q.x <= .3f) return Vec2{0.f, 0.f};
        Vec2 shifted{q.x - .3f, q.y};
        return cantilever(Vec2{shifted.x / .7f, q.y}, -.12f * t) * 1.f;
      });
      c.piece(p);
      c.floor(-.1f, .45f, -.07f, kGround);
      c.circle(hole, .06f, kBackground, true);
      c.circle(hole, .045f, kSupport, true);
      draw->AddRectFilled(c(Vec2{hole.x - .07f, kHeight + .05f}),
                          c(Vec2{hole.x + .07f, kHeight}), kSupport);
      draw->AddLine(c(Vec2{hole.x, kHeight + .05f}), c(Vec2{hole.x, -.07f}),
                    kSupport, 2.f);
      const Vec2 tip = at(p, {kLength, kHeight * .5f});
      c.arrow(tip + Vec2{0.f, .16f}, tip, kLoad, 3.f);
      break;
    }
    case Kind::BoltConnector: {
      const float gap = .1f, pull = .12f * t;
      pieces.push_back({0.f, .45f, 0.f, kHeight,
                        [](Vec2) { return Vec2{0.f, 0.f}; }});
      pieces.push_back({.45f + gap, kLength, 0.f, kHeight,
                        [pull](Vec2) { return Vec2{pull, 0.f}; }});
      for (const auto& piece : pieces) c.piece(piece);
      c.wall(0.f, -.05f, kHeight + .05f, true, kGround);
      const Vec2 a{.35f, kHeight * .5f}, b = Vec2{.65f + pull, kHeight * .5f};
      draw->AddLine(c(a), c(b), kSupport, 4.f);
      for (const Vec2 head : {a, b})
        draw->AddRectFilled(c(head + Vec2{-.02f, .06f}),
                            c(head + Vec2{.02f, -.06f}), kSupport);
      const Vec2 end{kLength + pull, kHeight * .5f};
      c.arrow(end, end + Vec2{.12f, 0.f}, kLoad, 3.f);
      break;
    }
    case Kind::Remote: {
      const float d = -.12f * t, theta = -.45f * t;
      const auto p = bar([&](Vec2 q) {
        const float x = q.x / kLength;
        const float v = d * (3.f * x * x - 2.f * x * x * x) +
                        theta * (x * x * x - x * x);
        const float slope = (d * (6.f * x - 6.f * x * x) +
                             theta * (3.f * x * x - 2.f * x)) /
                            kLength;
        return Vec2{-(q.y - kHeight * .5f) * slope, v};
      });
      c.piece(p);
      c.wall(0.f, -.05f, kHeight + .05f, true, kGround);
      const Vec2 remote = Vec2{kLength + .18f, kHeight * .5f + d} +
                          rotate(Vec2{0.f, 0.f}, theta);
      for (const float y : {0.f, kHeight})
        c.dashed(at(p, {kLength, y}), remote, kSupport);
      c.circle(remote, .025f, kSupport, true);
      draw->PathArcTo(c(remote), c.pixels(.07f), -.6f, .6f - theta * 2.f);
      draw->PathStroke(kSupport, 0, 1.5f);
      break;
    }
    case Kind::Spring: {
      const auto p = bar([&](Vec2 q) {
        const auto bend = uniformLoad(Vec2{q.x, q.y}, 0.f);
        return Vec2{bend.x, -.07f * t};
      });
      c.piece(p);
      c.floor(-.05f, kLength + .05f, -.2f, kGround);
      for (float x = .06f; x < kLength; x += .11f)
        c.spring({x, -.2f}, at(p, {x, 0.f}), kSupport);
      for (float x = .1f; x < kLength; x += .2f) {
        const Vec2 point = at(p, {x, kHeight});
        c.arrow(point + Vec2{0.f, .1f}, point, kLoad, 1.5f);
      }
      break;
    }
    case Kind::Hinge:
    case Kind::BallJoint:
    case Kind::RigidLink: {
      const float joint = .5f;
      const Vec2 pin{joint, kHeight * .5f};
      const bool rigid = kind == Kind::RigidLink;
      const float angle = rigid ? 0.f : -.4f * t;
      pieces.push_back({0.f, joint, 0.f, kHeight, [&](Vec2 q) {
                          return rigid ? cantilever(q, -.12f * t)
                                       : Vec2{0.f, 0.f};
                        }});
      pieces.push_back({joint, kLength, 0.f, kHeight, [&, angle](Vec2 q) {
                          return rigid ? cantilever(q, -.12f * t)
                                       : spin(q, pin, angle);
                        }});
      for (const auto& piece : pieces) c.piece(piece);
      c.wall(0.f, -.05f, kHeight + .05f, true, kGround);
      if (kind == Kind::Hinge) {
        c.circle(pin, .045f, kSupport, true);
        c.circle(pin, .045f, kEdge, false, 1.5f);
      } else if (kind == Kind::BallJoint) {
        c.circle(pin, .05f, kSupport, true);
        c.circle(pin + Vec2{-.015f, .015f}, .014f,
                 IM_COL32(255, 255, 255, 180), true);
      } else {
        const Vec2 low = at(pieces[0], {joint, -.02f}),
                   high = at(pieces[0], {joint, kHeight + .02f});
        draw->AddLine(c(low), c(high), kSupport, 5.f);
      }
      const Vec2 tip = at(pieces[1], {kLength, kHeight * .5f});
      c.arrow(tip + Vec2{0.f, .16f}, tip, kLoad, 3.f);
      break;
    }
    case Kind::SliderJoint: {
      const float slide = .16f * t;
      pieces.push_back({0.f, kLength, .08f, .17f,
                        [](Vec2) { return Vec2{0.f, 0.f}; }});
      pieces.push_back({.15f, .45f, .17f, .3f,
                        [slide](Vec2) { return Vec2{slide, 0.f}; }});
      pieces.push_back({.15f, .45f, -.05f, .08f,
                        [slide](Vec2) { return Vec2{slide, 0.f}; }});
      for (const auto& piece : pieces) c.piece(piece);
      c.wall(0.f, .03f, .22f, true, kGround);
      for (const float y : {.08f, .17f})
        draw->AddLine(c(Vec2{.15f + slide, y}), c(Vec2{.45f + slide, y}),
                      kSupport, 3.f);
      const Vec2 end{.45f + slide, .3f};
      c.arrow(end, end + Vec2{.12f, 0.f}, kLoad, 3.f);
      break;
    }
    case Kind::Tie: {
      const auto field = [&](Vec2 q) { return stretch(q, .13f * t); };
      pieces.push_back({0.f, .5f, 0.f, kHeight, field});
      pieces.push_back({.5f, kLength, 0.f, kHeight, field});
      for (const auto& piece : pieces) c.piece(piece);
      c.wall(0.f, -.05f, kHeight + .05f, true, kGround);
      draw->AddLine(c(at(pieces[0], {.5f, 0.f})),
                    c(at(pieces[0], {.5f, kHeight})), kSupport, 3.5f);
      const Vec2 end = at(pieces[1], {kLength, kHeight * .5f});
      c.arrow(end, end + Vec2{.12f, 0.f}, kLoad, 3.f);
      c.label({.4f, kHeight + .14f}, kSupport, "bonded");
      break;
    }
  }
  draw->PopClipRect();
  ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + size.x);
  ImGui::TextUnformatted(conditionDescription(kind));
  ImGui::PopTextWrapPos();
}

/// A tooltip with the name, animated sketch and description of a condition.
inline void conditionTooltip(Kind kind) {
  ImGui::BeginTooltip();
  ImGui::TextUnformatted(kindName(kind));
  drawConditionPreview(kind, ImGui::GetTime());
  ImGui::EndTooltip();
}
}  // namespace fem_workbench
