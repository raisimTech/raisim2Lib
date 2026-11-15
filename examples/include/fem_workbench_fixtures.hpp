// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "fem_surface_projection.hpp"
#include "fem_workbench_conditions.hpp"

namespace fem_workbench {
class FixtureCache {
 public:
  std::vector<size_t> boltConditions;
  size_t tieProjectionBuilds = 0;
  void clear() {
    rows_.clear();
    bolts_.clear();
    springs_.clear();
    boltConditions.clear();
    ties_.clear();
    tieProjectionBuilds = 0;
  }
  void configure(LinearElasticity& solver,
                 const std::vector<Condition>& conditions) {
    try {
      configureImpl(solver, conditions);
    } catch (...) {
      clear();
      solver.clearLinearConstraints();
      solver.clearBolts();
      solver.clearElasticSupports();
      throw;
    }
  }

 private:
  void configureImpl(LinearElasticity& solver,
                     const std::vector<Condition>& conditions) {
    ties_.resize(conditions.size());
    std::vector<Terms> rows;
    std::vector<double> values;
    std::vector<Bolt> bolts;
    std::vector<Spring> springs;
    boltConditions.clear();
    const auto add = [&](Terms terms, double value = 0.) {
      if (terms.empty() || !std::isfinite(value))
        throw std::runtime_error("Invalid fixture equation");
      rows.push_back(std::move(terms));
      values.push_back(value);
    };
    const auto direction = [&](unsigned int node, const Vec<3>& d,
                               double value = 0.) {
      const auto n = unit(d);
      Terms terms;
      for (unsigned int a = 0; a < 3; ++a)
        if (n[a] != 0.) terms.push_back({node, a, n[a]});
      add(std::move(terms), value);
    };
    const auto average = [&](const LinearElasticity::CouplingPatch& p,
                             const Vec<3>& d, double value = 0.) {
      const auto n = unit(d);
      Terms terms;
      for (size_t i = 0; i < p.nodes.size(); ++i)
        for (unsigned int a = 0; a < 3; ++a)
          if (n[a] != 0.) terms.push_back({p.nodes[i], a, p.weights[i] * n[a]});
      add(std::move(terms), value);
    };
    for (size_t index = 0; index < conditions.size(); ++index) {
      const auto& c = conditions[index];
      if (!c.enabled ||
          static_cast<int>(c.kind) <= static_cast<int>(Kind::Gravity))
        continue;
      if (c.nodes.empty())
        throw std::runtime_error("Fixture has no selected nodes");
      const auto n = unit(c.axis);
      if (c.kind == Kind::Directional)
        for (const auto node : c.nodes) direction(node, n, c.value[0]);
      if (c.kind == Kind::Slider) {
        Vec<3> first, second;
        first.e() = n.e().unitOrthogonal();
        second = n.cross(first);
        for (const auto node : c.nodes) {
          direction(node, first, c.value[0]);
          direction(node, second, c.value[1]);
        }
      }
      if (c.kind == Kind::Roller) {
        std::vector<std::vector<Vec<3>>> normals(solver.mesh().vertices.size());
        for (const auto face : c.faces) {
          const auto& ids = solver.boundaryFaces().at(face).vertices;
          const auto& x = solver.mesh().vertices;
          const Vec<3> normal =
              (x[ids[1]] - x[ids[0]]).cross(x[ids[2]] - x[ids[0]]);
          const auto axis = unit(normal);
          for (const auto node : ids) {
            auto& groups = normals[node];
            bool found = false;
            for (auto& group : groups)
              if (unit(group).e().dot(axis.e()) >= kSmoothSurfaceCosine) {
                group += normal;
                found = true;
                break;
              }
            if (!found) groups.push_back(normal);
          }
        }
        for (const auto node : c.nodes) {
          if (normals.at(node).empty())
            throw std::runtime_error(
                "Normal roller needs a selected boundary surface at every "
                "node");
          for (const auto& normal : normals[node])
            direction(node, normal, c.value[0]);
        }
      }
      if (c.kind == Kind::Bearing || c.kind == Kind::BoltHole) {
        for (const auto node : c.nodes) {
          const Vec<3> r = solver.mesh().vertices.at(node) - c.center;
          const Vec<3> radial = r - n * r.e().dot(n.e());
          if (c.axes[0]) direction(node, radial, c.value[0]);
          if (c.kind == Kind::BoltHole && c.axes[2])
            direction(node, n.cross(unit(radial)), c.value[2]);
        }
        if (c.axes[1]) {
          if (c.kind == Kind::Bearing)
            average(patch(solver, c.nodes, c.faces), n, c.value[1]);
          else {
            if (c.targetNodes.empty())
              throw std::runtime_error(
                  "Bolt-hole fixture needs a captured seat region");
            for (const auto node : c.targetNodes)
              direction(node, n, c.value[1]);
          }
        }
        if (c.kind == Kind::Bearing && c.axes[2])
          add(rotation(solver, patch(solver, c.nodes, c.faces), n), c.value[2]);
      }
      if (c.kind == Kind::Remote)
        for (const auto node : c.nodes) {
          const Vec<3> u =
              c.value +
              c.rotation.cross(solver.mesh().vertices.at(node) - c.center);
          for (unsigned int a = 0; a < 3; ++a)
            if (c.axes[a]) add({{node, a, 1.}}, u[a]);
        }
      if (c.kind == Kind::Ball) {
        const auto p = patch(solver, c.nodes, c.faces);
        for (int a = 0; a < 3; ++a) {
          Vec<3> d{0., 0., 0.};
          d[a] = 1.;
          average(p, d, c.value[a]);
        }
      }
      if (c.kind == Kind::Spring) {
        const auto p = patch(solver, c.nodes, c.faces);
        for (size_t i = 0; i < p.nodes.size(); ++i)
          springs.push_back(
              {p.nodes[i], n, c.stiffness * p.weights[i], c.value[0]});
      }
      if (c.kind == Kind::BoltConnector) {
        if (c.targetNodes.empty())
          throw std::runtime_error(
              "Bolt connector needs a captured second head/nut region");
        bolts.push_back({patch(solver, c.nodes, c.faces),
                         patch(solver, c.targetNodes, c.targetFaces), c.bolt});
        boltConditions.push_back(index);
      }
      if (static_cast<int>(c.kind) >= static_cast<int>(Kind::Hinge)) {
        if (c.targetNodes.empty())
          throw std::runtime_error(std::string(kindName(c.kind)) +
                                   " needs faces on the second part");
        const auto a = patch(solver, c.nodes, c.faces),
                   b = patch(solver, c.targetNodes, c.targetFaces);
        Vec<3> first, second;
        first.e() = n.e().unitOrthogonal();
        second = n.cross(first);
        const auto relative = [&](Terms terms, const Terms& other) {
          for (auto term : other) {
            term.weight = -term.weight;
            terms.push_back(term);
          }
          add(std::move(terms));
        };
        const auto along = [](const LinearElasticity::CouplingPatch& p,
                               const Vec<3>& d) {
          Terms terms;
          for (size_t i = 0; i < p.nodes.size(); ++i)
            for (unsigned int k = 0; k < 3; ++k)
              if (d[k] != 0.)
                terms.push_back({p.nodes[i], k, p.weights[i] * d[k]});
          return terms;
        };
        const auto pointMotion = [&](const LinearElasticity::CouplingPatch& p,
                                     const Vec<3>& d) {
          auto terms = along(p, d);
          Eigen::Vector3d center = Eigen::Vector3d::Zero();
          for (size_t i = 0; i < p.nodes.size(); ++i)
            center += p.weights[i] * solver.mesh().vertices[p.nodes[i]].e();
          const Eigen::Vector3d offset = c.center.e() - center,
                                arm = offset.cross(d.e());
          if (arm.norm() > 1e-12 * offset.norm()) {
            const auto spin = rotation(solver, p, {arm[0], arm[1], arm[2]});
            terms.insert(terms.end(), spin.begin(), spin.end());
          }
          return terms;
        };
        const auto translation = [&](const Vec<3>& d) {
          relative(pointMotion(a, d), pointMotion(b, d));
        };
        const auto spin = [&](const Vec<3>& d) {
          relative(rotation(solver, a, d), rotation(solver, b, d));
        };
        const bool hinge = c.kind == Kind::Hinge,
                   slider = c.kind == Kind::SliderJoint;
        translation(first);
        translation(second);
        if (!slider && (!hinge || c.axes[2])) translation(n);
        if (hinge || slider || c.kind == Kind::RigidLink) {
          spin(first);
          spin(second);
        }
        if (slider || c.kind == Kind::RigidLink) spin(n);
      }
      if (c.kind == Kind::Tie) {
        if (c.targetFaces.empty() || !std::isfinite(c.tolerance) ||
            c.tolerance <= 0.)
          throw std::runtime_error(
              "Bonded tie needs a master surface and positive projection "
              "tolerance");
        auto& cached = ties_[index];
        if (!cached.valid || cached.nodes != c.nodes ||
            cached.faces != c.targetFaces || cached.tolerance != c.tolerance) {
          SurfaceProjection surface(solver, c.targetFaces);
          std::vector<SurfaceProjection::Result> points;
          points.reserve(c.nodes.size());
          for (const auto node : c.nodes)
            points.push_back(surface.project(node));
          cached = {c.nodes, c.targetFaces, c.tolerance, std::move(points),
                    true};
          ++tieProjectionBuilds;
        }
        for (size_t i = 0; i < c.nodes.size(); ++i) {
          const auto node = c.nodes[i];
          const auto& projection = cached.points[i];
          if (projection.distance > c.tolerance * c.tolerance)
            throw std::runtime_error(
                "A tie slave node is outside the master-surface projection "
                "tolerance");
          bool same = false;
          for (int j = 0; j < 3; ++j)
            same |= projection.face[j] == node &&
                    std::abs(projection.weights[j] - 1.) < 1e-12;
          if (same) continue;
          for (unsigned int a = 0; a < 3; ++a) {
            Terms terms{{node, a, 1.}};
            for (int j = 0; j < 3; ++j)
              if (projection.weights[j] != 0.)
                terms.push_back(
                    {projection.face[j], a, -projection.weights[j]});
            add(std::move(terms));
          }
        }
      }
    }
    if (!sameRows(rows, rows_)) {
      solver.clearLinearConstraints();
      for (size_t r = 0; r < rows.size(); ++r)
        solver.addLinearConstraint(rows[r], values[r]);
      rows_ = rows;
    } else
      for (size_t r = 0; r < rows.size(); ++r)
        solver.setLinearConstraintValue(r, values[r]);
    bool sameBolts = bolts.size() == bolts_.size();
    if (sameBolts)
      for (size_t i = 0; i < bolts.size(); ++i)
        sameBolts &= samePatch(bolts[i].head, bolts_[i].head) &&
                     samePatch(bolts[i].nut, bolts_[i].nut);
    if (!sameBolts) {
      solver.clearBolts();
      for (const auto& b : bolts) solver.addBolt(b.head, b.nut, b.properties);
    } else
      for (size_t i = 0; i < bolts.size(); ++i)
        solver.setBoltProperties(i, bolts[i].properties);
    bolts_ = bolts;
    bool sameSprings = springs.size() == springs_.size();
    if (sameSprings)
      for (size_t i = 0; i < springs.size(); ++i) {
        const auto& a = springs[i];
        const auto& b = springs_[i];
        sameSprings &= a.node == b.node &&
                       (a.direction - b.direction).squaredNorm() == 0. &&
                       a.stiffness == b.stiffness;
      }
    if (!sameSprings) {
      solver.clearElasticSupports();
      for (const auto& s : springs)
        solver.addElasticSupport(s.node, s.direction, s.stiffness, s.rest);
    } else
      for (size_t i = 0; i < springs.size(); ++i)
        solver.setElasticSupportRestDisplacement(i, springs[i].rest);
    springs_ = springs;
  }

 private:
  using Terms = std::vector<LinearElasticity::ConstraintTerm>;
  struct Bolt {
    LinearElasticity::CouplingPatch head, nut;
    LinearElasticity::BoltProperties properties;
  };
  struct Spring {
    unsigned int node;
    Vec<3> direction;
    double stiffness, rest;
  };
  struct TieCache {
    std::vector<unsigned int> nodes;
    std::vector<size_t> faces;
    double tolerance = 0.;
    std::vector<SurfaceProjection::Result> points;
    bool valid = false;
  };
  std::vector<TieCache> ties_;
  std::vector<Terms> rows_;
  std::vector<Bolt> bolts_;
  std::vector<Spring> springs_;

  static Vec<3> unit(const Vec<3>& direction) {
    const double norm = direction.e().norm();
    if (!(norm > 0.) || !std::isfinite(norm))
      throw std::runtime_error("Fixture direction must be finite and nonzero");
    return direction / norm;
  }
  static bool samePatch(const LinearElasticity::CouplingPatch& a,
                        const LinearElasticity::CouplingPatch& b) {
    return a.nodes == b.nodes && a.weights == b.weights;
  }
  static bool sameRows(const std::vector<Terms>& a,
                       const std::vector<Terms>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i].size() != b[i].size()) return false;
      for (size_t j = 0; j < a[i].size(); ++j)
        if (a[i][j].node != b[i][j].node || a[i][j].axis != b[i][j].axis ||
            a[i][j].weight != b[i][j].weight)
          return false;
    }
    return true;
  }
  static LinearElasticity::CouplingPatch patch(
      const LinearElasticity& solver, const std::vector<unsigned int>& nodes,
      const std::vector<size_t>& faces) {
    LinearElasticity::CouplingPatch result;
    if (faces.empty()) {
      result.nodes = nodes;
      result.weights.assign(nodes.size(),
                            1. / std::max<size_t>(1, nodes.size()));
      return result;
    }
    std::vector<std::pair<unsigned int, double>> weights;
    for (const auto face : faces) {
      const auto& ids = solver.boundaryFaces().at(face).vertices;
      const auto& x = solver.mesh().vertices;
      const Vec<3> normal =
          (x[ids[1]] - x[ids[0]]).cross(x[ids[2]] - x[ids[0]]);
      for (const auto v : ids) weights.emplace_back(v, normal.e().norm() / 6.);
    }
    std::sort(weights.begin(), weights.end());
    double total = 0.;
    for (const auto& entry : weights) {
      if (!result.nodes.empty() && result.nodes.back() == entry.first)
        result.weights.back() += entry.second;
      else {
        result.nodes.push_back(entry.first);
        result.weights.push_back(entry.second);
      }
      total += entry.second;
    }
    if (!(total > 0.) || !std::isfinite(total))
      throw std::runtime_error("Fixture patch must have positive area");
    for (auto& weight : result.weights) weight /= total;
    return result;
  }
  static Terms rotation(const LinearElasticity& solver,
                        const LinearElasticity::CouplingPatch& p,
                        const Vec<3>& axis) {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < p.nodes.size(); ++i)
      center += p.weights[i] * solver.mesh().vertices[p.nodes[i]].e();
    Eigen::Matrix3d moment = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < p.nodes.size(); ++i) {
      const Eigen::Vector3d r = solver.mesh().vertices[p.nodes[i]].e() - center;
      moment.noalias() +=
          p.weights[i] *
          (r.squaredNorm() * Eigen::Matrix3d::Identity() - r * r.transpose());
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(moment);
    if (eigen.info() != Eigen::Success ||
        eigen.eigenvalues().minCoeff() <=
            1e-12 * eigen.eigenvalues().maxCoeff())
      throw std::runtime_error(
          "A bearing or joint region must not lie on one line");
    const Eigen::Vector3d weighted = moment.inverse() * axis.e();
    Terms terms;
    for (size_t i = 0; i < p.nodes.size(); ++i) {
      const Eigen::Vector3d r = solver.mesh().vertices[p.nodes[i]].e() - center;
      const Eigen::Vector3d coefficients = p.weights[i] * weighted.cross(r);
      for (unsigned int a = 0; a < 3; ++a)
        if (coefficients[a] != 0.)
          terms.push_back({p.nodes[i], a, coefficients[a]});
    }
    return terms;
  }
};
/// Axis shared by the normals of cylindrical or extruded faces: the direction
/// most nearly perpendicular to all of them. Returns false when the faces do
/// not define one direction (a plane, a sphere).
inline bool fitFaceAxis(
    const raisim::fem::TetMesh& mesh,
    const std::vector<LinearElasticity::BoundaryFace>& boundary,
    const std::vector<size_t>& faces, Vec<3>& axis) {
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto f : faces) {
    const auto& ids = boundary.at(f).vertices;
    const auto& x = mesh.vertices;
    const Eigen::Vector3d normal =
        (x[ids[1]].e() - x[ids[0]].e()).cross(x[ids[2]].e() - x[ids[0]].e());
    const double area = normal.norm();
    if (area > 0.) covariance.noalias() += normal * normal.transpose() / area;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance);
  if (eigen.info() != Eigen::Success) return false;
  const auto& values = eigen.eigenvalues();
  if (!(values[2] > 0.) || !(values[0] < .05 * values[1])) return false;
  const Eigen::Vector3d d = eigen.eigenvectors().col(0);
  axis = {d[0], d[1], d[2]};
  return true;
}
/// Point through which the normals of the faces pass in the least-squares
/// sense: the center of a sphere, or a point on the axis of a cylinder at the
/// mean axial position. Returns false for faces with one normal direction.
inline bool fitJointPoint(
    const raisim::fem::TetMesh& mesh,
    const std::vector<LinearElasticity::BoundaryFace>& boundary,
    const std::vector<size_t>& faces, Vec<3>& point) {
  Eigen::Matrix3d lhs = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
  for (const auto f : faces) {
    const auto& ids = boundary.at(f).vertices;
    const auto& x = mesh.vertices;
    const Eigen::Vector3d cross =
        (x[ids[1]].e() - x[ids[0]].e()).cross(x[ids[2]].e() - x[ids[0]].e());
    const double area = cross.norm();
    if (!(area > 0.)) continue;
    const Eigen::Vector3d n = cross / area,
                          centroid = (x[ids[0]].e() + x[ids[1]].e() +
                                      x[ids[2]].e()) /
                                     3.;
    const Eigen::Matrix3d projector =
        area * (Eigen::Matrix3d::Identity() - n * n.transpose());
    lhs += projector;
    rhs += projector * centroid;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(lhs);
  if (eigen.info() != Eigen::Success ||
      !(eigen.eigenvalues()[0] > 1e-3 * eigen.eigenvalues()[2]))
    return false;
  const Eigen::Vector3d c = lhs.ldlt().solve(rhs);
  if (!c.allFinite()) return false;
  point = {c[0], c[1], c[2]};
  return true;
}
/// Shape of a face set: planar (within @p flatness of a plane), cylindrical or
/// spherical when every face normal agrees with that shape.
struct FaceShape {
  enum class Type { Other, Plane, Cylinder, Sphere };
  Type type = Type::Other;
  /// Plane: centroid. Cylinder: a point on the axis. Sphere: the center.
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  /// Plane: mean normal. Cylinder: axis.
  Eigen::Vector3d direction = Eigen::Vector3d::Zero();
  double radius = 0., size = 0.;
};
inline FaceShape describeFaces(
    const raisim::fem::TetMesh& mesh,
    const std::vector<LinearElasticity::BoundaryFace>& boundary,
    const std::vector<size_t>& faces, double flatness) {
  struct Patch {
    Eigen::Vector3d center, normal;
    double area;
  };
  std::vector<Patch> patches;
  FaceShape shape;
  double area = 0.;
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  Eigen::Vector3d normalSum = Eigen::Vector3d::Zero(),
                  centroid = Eigen::Vector3d::Zero(),
                  low = Eigen::Vector3d::Constant(
                      std::numeric_limits<double>::infinity()),
                  high = -low;
  for (const auto f : faces) {
    const auto& ids = boundary.at(f).vertices;
    const Eigen::Vector3d a = mesh.vertices[ids[0]].e(),
                          b = mesh.vertices[ids[1]].e(),
                          c = mesh.vertices[ids[2]].e();
    const Eigen::Vector3d cross = (b - a).cross(c - a);
    const double twice = cross.norm();
    if (!(twice > 0.)) continue;
    const Patch patch{(a + b + c) / 3., cross / twice, .5 * twice};
    patches.push_back(patch);
    area += patch.area;
    covariance.noalias() += patch.area * patch.normal * patch.normal.transpose();
    normalSum += patch.area * patch.normal;
    centroid += patch.area * patch.center;
    low = low.cwiseMin(a).cwiseMin(b).cwiseMin(c);
    high = high.cwiseMax(a).cwiseMax(b).cwiseMax(c);
  }
  if (!(area > 0.)) return shape;
  centroid /= area;
  shape.size = (high - low).norm();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance / area);
  if (eigen.info() != Eigen::Success) return shape;
  const auto& values = eigen.eigenvalues();
  if (values[0] + values[1] < .02 && normalSum.norm() > .95 * area) {
    const Eigen::Vector3d normal = normalSum.normalized();
    double deviation = 0.;
    for (const auto f : faces)
      for (const auto v : boundary.at(f).vertices)
        deviation = std::max(
            deviation, std::abs((mesh.vertices[v].e() - centroid).dot(normal)));
    if (deviation <= flatness) {
      shape.type = FaceShape::Type::Plane;
      shape.point = centroid;
      shape.direction = normal;
      return shape;
    }
  }
  Vec<3> fitted;
  if (!fitJointPoint(mesh, boundary, faces, fitted)) return shape;
  const Eigen::Vector3d center = fitted.e();
  const bool cylinder = values[0] < .02;
  const Eigen::Vector3d axis = eigen.eigenvectors().col(0);
  const auto radial = [&](const Eigen::Vector3d& x) {
    const Eigen::Vector3d d = x - center;
    return cylinder ? Eigen::Vector3d(d - axis * axis.dot(d)) : d;
  };
  double radius = 0., aligned = 0.;
  for (const auto& patch : patches) {
    const Eigen::Vector3d r = radial(patch.center);
    const double length = r.norm();
    radius += patch.area * length;
    if (length > 0.) aligned += patch.area * std::abs(patch.normal.dot(r)) / length;
  }
  radius /= area;
  double spread = 0.;
  for (const auto& patch : patches)
    spread += patch.area * std::abs(radial(patch.center).norm() - radius);
  if (!(radius > 0.) || aligned < .98 * area || spread > .05 * radius * area)
    return shape;
  shape.type =
      cylinder ? FaceShape::Type::Cylinder : FaceShape::Type::Sphere;
  shape.point = center;
  shape.direction = cylinder ? axis : Eigen::Vector3d::Zero();
  shape.radius = radius;
  return shape;
}
/// Connections that fit two face sets on different parts, best first, judged
/// from their shapes and placement like the quick mates of a CAD assembly.
/// A tie (glue) is offered when every node of one set lies on the other set;
/// @p tieReversed means the second set must be the tie's first faces.
struct MateSuggestion {
  std::string relation;
  std::vector<Kind> kinds;
  bool tieReversed = false;
  FaceShape first, second;
};
inline MateSuggestion suggestMates(
    const raisim::fem::TetMesh& mesh,
    const std::vector<LinearElasticity::BoundaryFace>& boundary,
    const std::vector<size_t>& facesA, const std::vector<size_t>& facesB,
    double tolerance) {
  MateSuggestion result;
  if (facesA.empty() || facesB.empty()) return result;
  const auto a = describeFaces(mesh, boundary, facesA, tolerance);
  const auto b = describeFaces(mesh, boundary, facesB, tolerance);
  result.first = a;
  result.second = b;
  const auto facesWithinTolerance = [&](const std::vector<size_t>& from,
                                        const std::vector<size_t>& to, bool all) {
    std::vector<size_t> targets;
    for (const auto f : to) {
      const auto& ids = boundary.at(f).vertices;
      const auto& x = mesh.vertices;
      if ((x[ids[1]].e() - x[ids[0]].e())
              .cross(x[ids[2]].e() - x[ids[0]].e())
              .squaredNorm() > 0.)
        targets.push_back(f);
    }
    const SurfaceProjection projection(mesh, boundary, targets);
    for (const auto f : from)
      for (const auto v : boundary.at(f).vertices)
        if ((projection.project(v).distance <= tolerance * tolerance) != all)
          return !all;
    return all;
  };
  const bool touching = facesWithinTolerance(facesA, facesB, false) ||
                        facesWithinTolerance(facesB, facesA, false);
  const bool bondForward = touching && facesWithinTolerance(facesA, facesB, true);
  result.tieReversed =
      touching && !bondForward && facesWithinTolerance(facesB, facesA, true);
  const bool bondable = bondForward || result.tieReversed;
  const double parallel = std::cos(3. * std::acos(-1.) / 180.);
  const auto suggest = [&](const char* relation,
                           std::initializer_list<Kind> kinds) {
    result.relation = relation;
    for (const auto kind : kinds)
      if (kind != Kind::Tie || bondable) result.kinds.push_back(kind);
    return result;
  };
  using Type = FaceShape::Type;
  if (a.type == Type::Cylinder && b.type == Type::Cylinder &&
      std::abs(a.direction.dot(b.direction)) > parallel) {
    const Eigen::Vector3d d = b.point - a.point;
    if ((d - a.direction * a.direction.dot(d)).norm() <=
        std::max(tolerance, .05 * std::max(a.radius, b.radius)))
      return suggest("Concentric cylinders",
                     {Kind::Hinge, Kind::SliderJoint, Kind::Tie,
                      Kind::RigidLink});
  }
  if (a.type == Type::Sphere && b.type == Type::Sphere &&
      (b.point - a.point).norm() <=
          std::max(tolerance, .05 * std::max(a.radius, b.radius)))
    return suggest("Concentric spheres",
                   {Kind::BallJoint, Kind::Tie, Kind::RigidLink});
  if (a.type == Type::Plane && b.type == Type::Plane &&
      a.direction.dot(b.direction) < -parallel) {
    const Eigen::Vector3d d = b.point - a.point;
    const double gap = d.dot(a.direction);
    if (std::abs(gap) <= tolerance && bondable)
      return suggest("Coincident planes", {Kind::Tie, Kind::RigidLink});
    if (gap < -tolerance &&
        (d - a.direction * gap).norm() <= .25 * std::max(a.size, b.size))
      return suggest("Opposite seats",
                     {Kind::BoltConnector, Kind::RigidLink});
    return suggest("Parallel planes", {Kind::RigidLink});
  }
  return suggest(touching ? "Touching faces" : "Separate faces",
                 {Kind::Tie, Kind::RigidLink});
}
}  // namespace fem_workbench

