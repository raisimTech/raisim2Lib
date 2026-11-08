// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
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
      throw std::runtime_error("Bearing region must be non-collinear");
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
}  // namespace fem_workbench
