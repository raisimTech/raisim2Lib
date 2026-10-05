#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace forest {
// Placement broad phase only. The final test keeps the original hypot and
// strict radius comparison, so it cannot change placement or RNG consumption.
class TreeProximityIndex {
 public:
  TreeProximityIndex() { heads_.fill(end); }
  void add(double x, double y) {
    auto& head = x >= -40 && x < 40 && y >= -40 && y < 40
      ? heads_[coordinate(y) * cells + coordinate(x)] : outside_;
    points_.push_back({x, y, head});
    head = points_.size() - 1;
  }
  bool anyWithin(double x, double y, double radius) const {
    if (radius <= 0) return false;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(radius)) {
      return std::any_of(points_.begin(), points_.end(), [&](const Point& p) {
        return std::hypot(x - p.x, y - p.y) < radius;
      });
    }
    // Expand by a cell on each side so rounding at a bin/radius boundary can
    // only add candidates. Trees outside the template are checked separately.
    const int firstX = std::max(0, coordinate(x - radius) - 1);
    const int lastX = std::min(cells - 1, coordinate(x + radius) + 1);
    const int firstY = std::max(0, coordinate(y - radius) - 1);
    const int lastY = std::min(cells - 1, coordinate(y + radius) + 1);
    for (int iy = firstY; iy <= lastY; ++iy)
      for (int ix = firstX; ix <= lastX; ++ix)
        if (anyWithin(heads_[iy * cells + ix], x, y, radius)) return true;
    return anyWithin(outside_, x, y, radius);
  }
 private:
  static constexpr int cells = 40;
  static constexpr size_t end = std::numeric_limits<size_t>::max();
  struct Point { double x, y; size_t next; };
  static int coordinate(double value) {
    if (value <= -40) return 0;
    if (value >= 40) return cells - 1;
    return std::min(cells - 1, int(std::floor((value + 40) / 2)));
  }
  bool anyWithin(size_t head, double x, double y, double radius) const {
    for (size_t i = head; i != end; i = points_[i].next) {
      const auto& p = points_[i];
      if (std::hypot(x - p.x, y - p.y) < radius) return true;
    }
    return false;
  }
  std::array<size_t, cells * cells> heads_;
  size_t outside_ = end;
  std::vector<Point> points_;
};
} // namespace forest
