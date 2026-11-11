// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fem_workbench_fixtures.hpp"
#include "raisim/fem/LinearElasticity.hpp"

namespace fem_workbench {
/// Rigid placement x -> rotation * x + translation.
struct Pose {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Vector3d operator()(const Eigen::Vector3d& x) const {
    return rotation * x + translation;
  }
  Pose operator*(const Pose& other) const {
    return {rotation * other.rotation,
            rotation * other.translation + translation};
  }
  Pose inverse() const {
    return {rotation.transpose(), -(rotation.transpose() * translation)};
  }
  bool operator==(const Pose& other) const {
    return rotation == other.rotation && translation == other.translation;
  }
  bool isIdentity() const { return *this == Pose{}; }
  bool valid() const {
    return rotation.allFinite() && translation.allFinite() &&
           (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                   .cwiseAbs()
                   .maxCoeff() < 1e-9 &&
           rotation.determinant() > 0.;
  }
  /// Rotation by @p rotation about @p pivot.
  static Pose about(const Eigen::Matrix3d& rotation,
                    const Eigen::Vector3d& pivot) {
    return {rotation, pivot - rotation * pivot};
  }
};
inline FaceShape transformed(FaceShape shape, const Pose& pose) {
  shape.point = pose(shape.point);
  shape.direction = pose.rotation * shape.direction;
  return shape;
}
/// Moves the vertices of each listed region by its pose.
inline void transformRegions(raisim::fem::TetMesh& mesh,
                             const std::vector<std::pair<int, Pose>>& poses) {
  if (poses.empty()) return;
  std::vector<unsigned char> moved(mesh.vertices.size(), 0);
  for (size_t t = 0; t < mesh.tetrahedra.size(); ++t) {
    const int region = mesh.regions.empty() ? 0 : mesh.regions[t];
    for (const auto& [target, pose] : poses) {
      if (target != region) continue;
      for (const auto v : mesh.tetrahedra[t])
        if (!moved[v]) {
          mesh.vertices[v].e() = pose(mesh.vertices[v].e());
          moved[v] = 1;
        }
    }
  }
}
/// Assembly mates that place one part against another, after the standard and
/// advanced mates of SolidWorks.
enum class MateType {
  Coincident,
  Parallel,
  Perpendicular,
  Tangent,
  Concentric,
  Distance,
  Angle,
  Lock,
  ProfileCenter,
  Width
};
inline constexpr MateType kMateTypes[] = {
    MateType::Coincident, MateType::Parallel,   MateType::Perpendicular,
    MateType::Tangent,    MateType::Concentric, MateType::Distance,
    MateType::Angle,      MateType::Lock,       MateType::ProfileCenter,
    MateType::Width};
inline const char* mateTypeName(MateType type) {
  constexpr const char* names[] = {
      "Coincident", "Parallel", "Perpendicular", "Tangent",        "Concentric",
      "Distance",   "Angle",    "Lock",          "Profile center", "Width"};
  return names[static_cast<int>(type)];
}
inline const char* mateTypeHelp(MateType type) {
  constexpr const char* help[] = {
      "The two planes lie on each other",
      "Planes, axes, or a plane and an axis become parallel",
      "Planes, axes, or a plane and an axis become perpendicular",
      "A plane, cylinder or sphere touches a cylinder or sphere",
      "Two cylinders share an axis, or two spheres a center",
      "Planes, axes or centers keep a distance",
      "Two planes or axes keep an angle",
      "The part keeps its place on the other part",
      "The planes lie on each other with their centers on each other",
      "A tab between two planes is centered in a slot between two planes"};
  return help[static_cast<int>(type)];
}
/// Settings of a position mate.
struct MateOptions {
  /// The normals or axes point the same way. Unset chooses facing planes for
  /// Coincident, Profile center and planar Distance, and the nearer way
  /// otherwise.
  std::optional<bool> aligned;
  /// Distance: metres; Angle: radians; Profile center: offset in metres. Unset
  /// keeps the current distance or angle.
  std::optional<double> value;
  /// Distance, Profile center and Tangent: the moving face goes to the other
  /// side.
  bool flip = false;
};
/// A mate between face sets of two parts. The faces are stored in the CAD
/// frame of their part, so the mate holds when either part moves. Width uses
/// two planes on each side; Lock stores three points of the moving part and
/// where they sit on the reference part.
struct PositionMate {
  MateType type = MateType::Coincident;
  std::string name;
  int moving = 0, reference = 0;
  std::vector<FaceShape> movingFaces, referenceFaces;
  MateOptions options;
  /// Largest distance at which the mate holds; angles hold within 2e-3 rad.
  double tolerance = 1e-6;
};
/// The faces of a mate placed in the world.
struct PlacementTarget {
  const PositionMate* mate = nullptr;
  std::vector<FaceShape> moving, reference;
};

namespace placement {
using Row = Eigen::Matrix<double, 1, 6>;
using Block = Eigen::Matrix<double, 3, 6>;
/// A scalar or vector with its derivative with respect to a rotation (scaled
/// by the model length) and a translation of the moving part.
struct Scalar {
  double value = 0.;
  Row gradient = Row::Zero();
};
struct Vector {
  Eigen::Vector3d value = Eigen::Vector3d::Zero();
  Block jacobian = Block::Zero();
};
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0., -v.z(), v.y(), v.z(), 0., -v.x(), -v.y(), v.x(), 0.;
  return m;
}
inline Vector operator-(const Vector& a, const Vector& b) {
  return {a.value - b.value, a.jacobian - b.jacobian};
}
inline Vector operator+(const Vector& a, const Vector& b) {
  return {a.value + b.value, a.jacobian + b.jacobian};
}
inline Vector operator*(double s, const Vector& a) {
  return {s * a.value, s * a.jacobian};
}
inline Scalar dot(const Vector& a, const Vector& b) {
  return {a.value.dot(b.value),
          a.value.transpose() * b.jacobian + b.value.transpose() * a.jacobian};
}
/// Length of @p a; at zero length the derivative is taken along @p fallback,
/// so a distance can grow from coincident points.
inline Scalar norm(const Vector& a, const Eigen::Vector3d& fallback) {
  const double length = a.value.norm();
  if (!(length > 0.)) return {0., fallback.transpose() * a.jacobian};
  return {length, a.value.transpose() * a.jacobian / length};
}
/// The part of @p a perpendicular to the unit vector @p u.
inline Vector across(const Vector& a, const Vector& u) {
  const Scalar along = dot(u, a);
  return {a.value - along.value * u.value,
          a.jacobian - u.value * along.gradient - along.value * u.jacobian};
}
/// Points and directions of the moving part after it moves by @p motion, with
/// their derivatives; fixed ones have none.
class Builder {
 public:
  Builder(const Pose& motion, const Eigen::Vector3d& pivot, double length)
      : motion_(motion), pivot_(pivot), length_(length) {}
  Vector point(const Eigen::Vector3d& x, bool moving) const {
    Vector v;
    v.value = moving ? motion_(x) : x;
    if (moving) {
      v.jacobian.leftCols<3>() = -skew(v.value - pivot_) / length_;
      v.jacobian.rightCols<3>() = Eigen::Matrix3d::Identity();
    }
    return v;
  }
  Vector direction(const Eigen::Vector3d& d, bool moving) const {
    Vector v;
    v.value = moving ? Eigen::Vector3d(motion_.rotation * d) : d;
    if (moving) v.jacobian.leftCols<3>() = -skew(v.value) / length_;
    return v;
  }
  double length() const { return length_; }

 private:
  Pose motion_;
  Eigen::Vector3d pivot_;
  double length_;
};
/// Residual rows, in lengths, the angles they set and the correspondences
/// that start the solve.
struct Rows {
  enum class Kind { Distance, Direction, Cosine };
  struct Entry {
    Scalar row;
    Kind kind;
  };
  std::vector<Entry> entries;
  /// Angle between two directions and the angle it should have.
  std::vector<std::pair<double, double>> angles;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> directions, points;
  void add(const Scalar& row, bool angular) {
    entries.push_back({row, angular ? Kind::Direction : Kind::Distance});
  }
};
using Shape = FaceShape::Type;
inline bool isAxis(const FaceShape& s) { return s.type == Shape::Cylinder; }
inline bool isPlane(const FaceShape& s) { return s.type == Shape::Plane; }
inline bool isSphere(const FaceShape& s) { return s.type == Shape::Sphere; }
/// Emits the residuals of @p target after its part moves by the builder's
/// motion. Throws std::invalid_argument when the faces do not fit the mate.
inline void mateRows(const PlacementTarget& target, const Builder& b,
                     Rows& rows) {
  const auto& mate = *target.mate;
  const auto& o = mate.options;
  const double length = b.length();
  const double sign = o.aligned.value_or(false) ? 1. : -1.,
               side = o.flip ? -1. : 1., value = o.value.value_or(0.);
  const auto unfit = []() {
    throw std::invalid_argument("These faces do not fit this mate");
  };
  if (target.moving.empty() || target.reference.empty()) unfit();
  const FaceShape &m = target.moving[0], &r = target.reference[0];
  const auto align = [&](const FaceShape& moving, const FaceShape& fixed,
                         double s) {
    const Vector a = b.direction(moving.direction, true);
    for (int k = 0; k < 3; ++k)
      rows.add({(a.value[k] - s * fixed.direction[k]) * length,
                a.jacobian.row(k) * length},
               true);
    rows.directions.emplace_back(moving.direction, s * fixed.direction);
  };
  const auto angle = [&](const FaceShape& x, bool xMoving, const FaceShape& y,
                         double cosine) {
    const Scalar d = dot(b.direction(x.direction, xMoving),
                         b.direction(y.direction, !xMoving));
    rows.entries.push_back({{(d.value - cosine) * length, d.gradient * length},
                            Rows::Kind::Cosine});
    rows.angles.emplace_back(std::acos(std::clamp(d.value, -1., 1.)),
                             std::acos(std::clamp(cosine, -1., 1.)));
  };
  const auto planeOffset = [&](const FaceShape& plane, bool planeMoving,
                               const Eigen::Vector3d& q, double offset) {
    Scalar d = dot(b.direction(plane.direction, planeMoving),
                   b.point(q, !planeMoving) - b.point(plane.point, planeMoving));
    d.value -= offset;
    rows.add(d, false);
  };
  const auto axisDistance = [&](const FaceShape& axis, bool axisMoving,
                                const Eigen::Vector3d& q, double distance) {
    const Vector u = b.direction(axis.direction, axisMoving);
    const Vector off =
        across(b.point(q, !axisMoving) - b.point(axis.point, axisMoving), u);
    if (distance == 0.) {
      for (int k = 0; k < 3; ++k)
        rows.add({off.value[k], off.jacobian.row(k)}, false);
      return;
    }
    Scalar d = norm(off, u.value.unitOrthogonal());
    d.value -= distance;
    rows.add(d, false);
  };
  const auto pointDistance = [&](const Eigen::Vector3d& moving,
                                 const Eigen::Vector3d& fixed,
                                 double distance) {
    const Vector off = b.point(moving, true) - b.point(fixed, false);
    if (distance == 0.) {
      for (int k = 0; k < 3; ++k)
        rows.add({off.value[k], off.jacobian.row(k)}, false);
      rows.points.emplace_back(moving, fixed);
      return;
    }
    Scalar d = norm(off, Eigen::Vector3d::UnitX());
    d.value -= distance;
    rows.add(d, false);
  };
  const bool pp = isPlane(m) && isPlane(r), cc = isAxis(m) && isAxis(r),
             ss = isSphere(m) && isSphere(r);
  const bool pc = isPlane(m) && isAxis(r), cp = isAxis(m) && isPlane(r);
  const bool ps = isPlane(m) && isSphere(r), sp = isSphere(m) && isPlane(r);
  const bool cs = isAxis(m) && isSphere(r), sc = isSphere(m) && isAxis(r);
  const auto planeAndAxis = [&](auto&& each) {
    if (pc) each(m, true, r);
    if (cp) each(r, false, m);
  };
  const auto planeAndSphere = [&](auto&& each) {
    if (ps) each(m, true, r);
    if (sp) each(r, false, m);
  };
  const auto axisAndSphere = [&](auto&& each) {
    if (cs) each(m, true, r);
    if (sc) each(r, false, m);
  };
  const double outer = m.radius + r.radius,
               inner = std::abs(m.radius - r.radius);
  switch (mate.type) {
    case MateType::Coincident:
      if (!pp) unfit();
      align(m, r, sign);
      planeOffset(r, false, m.point, 0.);
      return;
    case MateType::ProfileCenter:
      if (!pp) unfit();
      align(m, r, sign);
      planeOffset(r, false, m.point, side * value);
      axisDistance(r, false, m.point, 0.);
      return;
    case MateType::Parallel:
      if (pp || cc) {
        align(m, r, sign);
      } else if (pc || cp) {
        planeAndAxis([&](const FaceShape& plane, bool planeMoving,
                         const FaceShape& axis) {
          angle(plane, planeMoving, axis, 0.);
        });
      } else {
        unfit();
      }
      return;
    case MateType::Perpendicular:
      if (pp || cc) {
        angle(m, true, r, 0.);
      } else if (pc || cp) {
        align(m, r, sign);
      } else {
        unfit();
      }
      return;
    case MateType::Angle: {
      if (!pp && !cc) unfit();
      const double cosine = std::cos(value);
      if (std::abs(cosine) > 1. - 1e-12)
        align(m, r, cosine > 0. ? 1. : -1.);
      else
        angle(m, true, r, cosine);
      return;
    }
    case MateType::Distance:
      if (pp) {
        align(m, r, sign);
        planeOffset(r, false, m.point, side * value);
      } else if (cc) {
        align(m, r, sign);
        axisDistance(r, false, m.point, value);
      } else if (ss) {
        pointDistance(m.point, r.point, value);
      } else if (pc || cp) {
        planeAndAxis([&](const FaceShape& plane, bool planeMoving,
                         const FaceShape& axis) {
          angle(plane, planeMoving, axis, 0.);
          planeOffset(plane, planeMoving, axis.point, side * value);
        });
      } else if (ps || sp) {
        planeAndSphere([&](const FaceShape& plane, bool planeMoving,
                           const FaceShape& sphere) {
          planeOffset(plane, planeMoving, sphere.point, side * value);
        });
      } else if (cs || sc) {
        axisAndSphere([&](const FaceShape& axis, bool axisMoving,
                          const FaceShape& sphere) {
          axisDistance(axis, axisMoving, sphere.point, value);
        });
      } else {
        unfit();
      }
      return;
    case MateType::Tangent:
      if (pc || cp) {
        planeAndAxis([&](const FaceShape& plane, bool planeMoving,
                         const FaceShape& axis) {
          angle(plane, planeMoving, axis, 0.);
          planeOffset(plane, planeMoving, axis.point, side * axis.radius);
        });
      } else if (ps || sp) {
        planeAndSphere([&](const FaceShape& plane, bool planeMoving,
                           const FaceShape& sphere) {
          planeOffset(plane, planeMoving, sphere.point, side * sphere.radius);
        });
      } else if (cc) {
        align(m, r, sign);
        axisDistance(r, false, m.point, o.flip ? inner : outer);
      } else if (ss) {
        pointDistance(m.point, r.point, o.flip ? inner : outer);
      } else if (cs || sc) {
        axisAndSphere([&](const FaceShape& axis, bool axisMoving,
                          const FaceShape& sphere) {
          axisDistance(axis, axisMoving, sphere.point, o.flip ? inner : outer);
        });
      } else {
        unfit();
      }
      return;
    case MateType::Concentric:
      if (cc) {
        align(m, r, sign);
        axisDistance(r, false, m.point, 0.);
      } else if (ss) {
        pointDistance(m.point, r.point, 0.);
      } else if (cs || sc) {
        axisAndSphere([&](const FaceShape& axis, bool axisMoving,
                          const FaceShape& sphere) {
          axisDistance(axis, axisMoving, sphere.point, 0.);
        });
      } else {
        unfit();
      }
      return;
    case MateType::Lock:
      if (target.moving.size() != 3 || target.reference.size() != 3) unfit();
      for (size_t i = 0; i < 3; ++i)
        pointDistance(target.moving[i].point, target.reference[i].point, 0.);
      return;
    case MateType::Width: {
      if (target.moving.size() != 2 || target.reference.size() != 2) unfit();
      const double parallel = std::cos(3. * std::acos(-1.) / 180.);
      for (const auto* faces : {&target.moving, &target.reference})
        if (!isPlane((*faces)[0]) || !isPlane((*faces)[1]) ||
            !(std::abs((*faces)[0].direction.dot((*faces)[1].direction)) >
              parallel))
          unfit();
      align(m, r, sign);
      const Vector tab = .5 * (b.point(target.moving[0].point, true) +
                               b.point(target.moving[1].point, true));
      const Eigen::Vector3d slot =
          .5 * (target.reference[0].point + target.reference[1].point);
      rows.add(dot(b.direction(r.direction, false),
                   tab - b.point(slot, false)),
               false);
      return;
    }
  }
  unfit();
}
/// Pseudo-inverse solution of the symmetric system @p a x = @p b, the
/// smallest x in the null space of @p a.
template <int N>
Eigen::Matrix<double, N, 1> smallestSolution(
    const Eigen::Matrix<double, N, N>& a, const Eigen::Matrix<double, N, 1>& b) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigen(a);
  Eigen::Matrix<double, N, 1> x = Eigen::Matrix<double, N, 1>::Zero();
  const double largest = eigen.eigenvalues().cwiseAbs().maxCoeff();
  if (eigen.info() != Eigen::Success || !(largest > 0.)) return x;
  for (int i = 0; i < N; ++i) {
    const double value = eigen.eigenvalues()[i];
    if (value > 1e-12 * largest)
      x += eigen.eigenvectors().col(i) *
           (eigen.eigenvectors().col(i).dot(b) / value);
  }
  return x;
}
inline Rows evaluate(const std::vector<PlacementTarget>& targets,
                     const Pose& motion, const Eigen::Vector3d& pivot,
                     double length) {
  const Builder builder(motion, pivot, length);
  Rows rows;
  for (const auto& target : targets) mateRows(target, builder, rows);
  return rows;
}
inline double cost(const Rows& rows) {
  double sum = 0.;
  for (const auto& e : rows.entries) sum += e.row.value * e.row.value;
  return sum;
}
}  // namespace placement

/// Throws std::invalid_argument when the faces do not fit the mate.
inline void checkMateFaces(const PositionMate& mate) {
  const PlacementTarget target{&mate, mate.movingFaces, mate.referenceFaces};
  placement::Rows rows;
  placement::mateRows(target, placement::Builder({}, Eigen::Vector3d::Zero(), 1.),
                      rows);
}
/// The current distance or angle between the faces of a Distance or Angle
/// mate given in the world: a signed distance along the plane normal when a
/// plane is involved, otherwise a distance or an angle in [0, pi].
inline double measureMate(MateType type, const std::vector<FaceShape>& moving,
                          const std::vector<FaceShape>& reference) {
  using namespace placement;
  const FaceShape &m = moving.at(0), &r = reference.at(0);
  if (type == MateType::Angle)
    return std::acos(std::clamp(m.direction.dot(r.direction), -1., 1.));
  if (isPlane(r)) return r.direction.dot(m.point - r.point);
  if (isPlane(m)) return m.direction.dot(r.point - m.point);
  if (isAxis(r)) {
    const Eigen::Vector3d d = m.point - r.point;
    return (d - r.direction * r.direction.dot(d)).norm();
  }
  if (isAxis(m)) {
    const Eigen::Vector3d d = r.point - m.point;
    return (d - m.direction * m.direction.dot(d)).norm();
  }
  return (m.point - r.point).norm();
}
/// Number of independent motions of a part that its mates leave free, from
/// 0 (fully positioned) to 6.
inline int freeMotions(const std::vector<PlacementTarget>& targets,
                       const Eigen::Vector3d& pivot, double length) {
  if (targets.empty()) return 6;
  const auto rows = placement::evaluate(targets, {}, pivot, length);
  Eigen::Matrix<double, 6, 6> normal = Eigen::Matrix<double, 6, 6>::Zero();
  for (const auto& e : rows.entries)
    normal += e.row.gradient.transpose() * e.row.gradient;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigen(normal);
  const double largest = eigen.eigenvalues().cwiseAbs().maxCoeff();
  int rank = 0;
  for (int i = 0; i < 6; ++i)
    if (largest > 0. && eigen.eigenvalues()[i] > 1e-9 * largest) ++rank;
  return 6 - rank;
}
/// The smallest motion of the moving part after which every target holds,
/// turning about @p pivot; @p length is the size of the model. The identity
/// when the targets already hold exactly. Throws std::runtime_error when the
/// targets conflict and std::invalid_argument when faces do not fit a mate.
inline Pose solvePlacement(const std::vector<PlacementTarget>& targets,
                           const Eigen::Vector3d& pivot, double length) {
  using namespace placement;
  if (targets.empty()) return {};
  if (!(length > 0.) || !std::isfinite(length) || !pivot.allFinite())
    throw std::invalid_argument("Invalid placement scale");
  const auto total = [&](const Pose& p) {
    return cost(evaluate(targets, p, pivot, length));
  };
  const auto start = evaluate(targets, {}, pivot, length);
  const double converged = 1e-24 * length * length;
  if (cost(start) <= converged) return {};
  const auto turn = [&](const Pose& p, const Eigen::Matrix3d& rotation) {
    return Pose{rotation * p.rotation,
                rotation * (p.translation - pivot) + pivot};
  };
  const auto shift = [&](const Pose& p) {
    const auto rows = evaluate(targets, p, pivot, length);
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    for (const auto& e : rows.entries) {
      const Eigen::Vector3d j = e.row.gradient.rightCols<3>().transpose();
      normal += j * j.transpose();
      gradient += j * e.row.value;
    }
    Pose shifted = p;
    shifted.translation -= smallestSolution<3>(normal, gradient);
    return total(shifted) < cost(rows) ? shifted : p;
  };
  const auto refine = [&](Pose p) {
    for (int iteration = 0; iteration < 100; ++iteration) {
      const auto rows = evaluate(targets, p, pivot, length);
      Eigen::Matrix<double, 6, 6> normal = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> gradient =
          Eigen::Matrix<double, 6, 1>::Zero();
      for (const auto& e : rows.entries) {
        normal += e.row.gradient.transpose() * e.row.gradient;
        gradient += e.row.gradient.transpose() * e.row.value;
      }
      const double current = cost(rows);
      if (current <= converged) break;
      const Eigen::Matrix<double, 6, 1> x =
          -smallestSolution<6>(normal, gradient);
      bool accepted = false;
      double scale = 1.;
      for (int k = 0; k < 40 && !accepted; ++k, scale *= .5) {
        const Eigen::Vector3d spin = scale * x.head<3>() / length;
        const double angle = spin.norm();
        Pose trial = p;
        if (angle > 0.)
          trial =
              turn(p, Eigen::AngleAxisd(angle, spin / angle).toRotationMatrix());
        trial.translation += scale * x.tail<3>();
        if (total(trial) < current) {
          p = trial;
          accepted = true;
        }
      }
      if (!accepted) break;
    }
    p.rotation = Eigen::Quaterniond(p.rotation).normalized().toRotationMatrix();
    return p;
  };
  const auto holds = [&](const Pose& p) {
    const Builder builder(p, pivot, length);
    for (const auto& target : targets) {
      Rows rows;
      mateRows(target, builder, rows);
      for (const auto& e : rows.entries) {
        const double r = std::abs(e.row.value);
        if ((e.kind == Rows::Kind::Distance && !(r <= target.mate->tolerance)) ||
            (e.kind == Rows::Kind::Direction && !(r / length <= 2e-3)))
          return false;
      }
      for (const auto& [actual, wanted] : rows.angles)
        if (!(std::abs(actual - wanted) <= 2e-3)) return false;
    }
    return true;
  };
  const auto size = [&](const Pose& p) {
    return (p(pivot) - pivot).norm() +
           length * Eigen::AngleAxisd(p.rotation).angle();
  };
  Eigen::Matrix3d spread = Eigen::Matrix3d::Zero();
  for (const auto& [moving, fixed] : start.directions)
    spread += fixed * moving.transpose();
  if (start.points.size() >= 2) {
    Eigen::Vector3d meanMoving = Eigen::Vector3d::Zero(),
                    meanFixed = Eigen::Vector3d::Zero();
    for (const auto& [moving, fixed] : start.points) {
      meanMoving += moving / double(start.points.size());
      meanFixed += fixed / double(start.points.size());
    }
    for (const auto& [moving, fixed] : start.points)
      spread += (fixed - meanFixed) * (moving - meanMoving).transpose() /
                (length * length);
  }
  std::vector<Eigen::Matrix3d> starts;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      spread, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto& singular = svd.singularValues();
  if (singular[0] > 0.) {
    if (singular[1] > 1e-2 * singular[0]) {
      Eigen::Vector3d diagonal(1., 1., 1.);
      if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.)
        diagonal[2] = -1.;
      starts.push_back(svd.matrixU() * diagonal.asDiagonal() *
                       svd.matrixV().transpose());
    } else {
      starts.push_back(Eigen::Quaterniond::FromTwoVectors(
                           Eigen::Vector3d(svd.matrixV().col(0)),
                           Eigen::Vector3d(svd.matrixU().col(0)))
                           .toRotationMatrix());
    }
  }
  starts.push_back(Eigen::Matrix3d::Identity());
  const Eigen::Matrix3d first = starts.front();
  for (int axis = 0; axis < 3; ++axis)
    starts.push_back(
        Eigen::AngleAxisd(std::acos(-1.), Eigen::Vector3d::Unit(axis))
            .toRotationMatrix() *
        first);
  std::optional<Pose> best;
  for (const auto& rotation : starts) {
    const Pose candidate = refine(shift(turn({}, rotation)));
    if (holds(candidate) && (!best || size(candidate) < size(*best)))
      best = candidate;
  }
  if (!best)
    throw std::runtime_error("conflicts with the other mates of the part");
  return *best;
}
}  // namespace fem_workbench
