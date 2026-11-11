// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "fem_workbench_conditions.hpp"

namespace fem_workbench {
enum class SurfaceSelection { Triangle, Planar, Smooth };
class SurfaceSelector {
 public:
  explicit SurfaceSelector(const LinearElasticity& solver)
      : SurfaceSelector(solver.mesh(), solver.boundaryFaces()) {}
  SurfaceSelector(const raisim::fem::TetMesh& mesh,
                  const std::vector<LinearElasticity::BoundaryFace>& faces)
      : mesh_(mesh), faces_(faces) {}
  std::vector<size_t> collect(size_t seed, SurfaceSelection mode) {
    const auto& faces = faces_;
    (void)faces.at(seed);
    if (mode == SurfaceSelection::Triangle) return {seed};
    if (neighbors_.empty()) build();
    const auto& x = mesh_.vertices;
    const Eigen::Vector3d origin = x[faces[seed].vertices[0]].e();
    std::vector<unsigned char> visited(faces.size(), 0);
    std::vector<size_t> pending{seed}, result;
    visited[seed] = 1;
    while (!pending.empty()) {
      const size_t current = pending.back();
      pending.pop_back();
      result.push_back(current);
      for (const auto neighbor : neighbors_[current]) {
        if (visited[neighbor]) continue;
        bool matches =
            normals_[current].dot(normals_[neighbor]) >= kSmoothSurfaceCosine;
        if (mode == SurfaceSelection::Planar) {
          matches = normals_[seed].dot(normals_[neighbor]) >= 1. - 1e-6;
          for (const auto node : faces[neighbor].vertices)
            matches &= std::abs(normals_[seed].dot(x[node].e() - origin)) <=
                       1e-5 * extent_;
        }
        if (matches) {
          visited[neighbor] = 1;
          pending.push_back(neighbor);
        }
      }
    }
    return result;
  }
  std::vector<int> partition(const std::vector<int>& parts) {
    if (parts.size() != faces_.size())
      throw std::invalid_argument("Surface parts do not match its faces");
    if (faces_.empty()) return {};
    if (neighbors_.empty()) build();
    std::vector<int> labels(faces_.size(), 0);
    std::vector<size_t> pending;
    int count = 0;
    for (size_t seed = 0; seed < faces_.size(); ++seed) {
      if (labels[seed]) continue;
      labels[seed] = ++count;
      pending.push_back(seed);
      while (!pending.empty()) {
        const size_t current = pending.back();
        pending.pop_back();
        for (const auto neighbor : neighbors_[current])
          if (!labels[neighbor] && parts[neighbor] == parts[current] &&
              normals_[current].dot(normals_[neighbor]) >=
                  kSmoothSurfaceCosine) {
            labels[neighbor] = count;
            pending.push_back(neighbor);
          }
      }
    }
    return labels;
  }

 private:
  const raisim::fem::TetMesh& mesh_;
  const std::vector<LinearElasticity::BoundaryFace>& faces_;
  std::vector<std::vector<size_t>> neighbors_;
  std::vector<Eigen::Vector3d> normals_;
  double extent_ = 0.;
  void build() {
    struct Edge {
      unsigned int a, b;
      size_t face;
    };
    std::vector<Edge> edges;
    const auto& faces = faces_;
    const auto& x = mesh_.vertices;
    neighbors_.resize(faces.size());
    normals_.resize(faces.size());
    edges.reserve(3 * faces.size());
    Eigen::Vector3d low = x[0].e(), high = low;
    for (const auto& point : x) {
      low = low.cwiseMin(point.e());
      high = high.cwiseMax(point.e());
    }
    extent_ = (high - low).stableNorm();
    for (size_t f = 0; f < faces.size(); ++f) {
      const auto& ids = faces[f].vertices;
      normals_[f] = (x[ids[1]].e() - x[ids[0]].e())
                        .cross(x[ids[2]].e() - x[ids[0]].e())
                        .normalized();
      for (int i = 0; i < 3; ++i)
        edges.push_back({std::min(ids[i], ids[(i + 1) % 3]),
                         std::max(ids[i], ids[(i + 1) % 3]), f});
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
      return a.a < b.a ||
             (a.a == b.a && (a.b < b.b || (a.b == b.b && a.face < b.face)));
    });
    for (size_t i = 0; i < edges.size();) {
      size_t end = i + 1;
      while (end < edges.size() && edges[end].a == edges[i].a &&
             edges[end].b == edges[i].b)
        ++end;
      if (end == i + 2) {
        neighbors_[edges[i].face].push_back(edges[i + 1].face);
        neighbors_[edges[i + 1].face].push_back(edges[i].face);
      }
      i = end;
    }
  }
};
}  // namespace fem_workbench
