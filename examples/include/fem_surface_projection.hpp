// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "fem_workbench_conditions.hpp"

namespace fem_workbench {
class SurfaceProjection {
 public:
  struct Result {
    std::array<unsigned int, 3> face{};
    Eigen::Vector3d weights = Eigen::Vector3d::Zero();
    double distance = std::numeric_limits<double>::infinity();
    size_t order = std::numeric_limits<size_t>::max();
  };
  SurfaceProjection(const LinearElasticity& solver,
                    const std::vector<size_t>& faces)
      : solver_(solver), faces_(faces) {
    order_.resize(faces.size());
    std::iota(order_.begin(), order_.end(), 0u);
    if (!order_.empty()) build(0, order_.size());
  }
  Result project(unsigned int node) const {
    Result result;
    if (!tree_.empty()) query(0, solver_.mesh().vertices.at(node).e(), result);
    return result;
  }

 private:
  struct Node {
    Eigen::Vector3d low, high;
    size_t begin, end;
    int left = -1, right = -1;
  };
  const LinearElasticity& solver_;
  const std::vector<size_t>& faces_;
  std::vector<size_t> order_;
  std::vector<Node> tree_;
  Eigen::Vector3d center(size_t i) const {
    const auto& ids = solver_.boundaryFaces().at(faces_[i]).vertices;
    const auto& x = solver_.mesh().vertices;
    return x[ids[0]].e() +
           ((x[ids[1]].e() - x[ids[0]].e()) + (x[ids[2]].e() - x[ids[0]].e())) /
               3.;
  }
  int build(size_t begin, size_t end) {
    Node node;
    node.begin = begin;
    node.end = end;
    node.low.setConstant(std::numeric_limits<double>::infinity());
    node.high = -node.low;
    for (size_t i = begin; i < end; ++i)
      for (const auto v :
           solver_.boundaryFaces().at(faces_[order_[i]]).vertices) {
        node.low = node.low.cwiseMin(solver_.mesh().vertices[v].e());
        node.high = node.high.cwiseMax(solver_.mesh().vertices[v].e());
      }
    const int index = static_cast<int>(tree_.size());
    tree_.push_back(node);
    if (end - begin <= 8) return index;
    Eigen::Index axis;
    (node.high - node.low).maxCoeff(&axis);
    const size_t middle = begin + (end - begin) / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + middle,
                     order_.begin() + end, [&](size_t a, size_t b) {
                       const double x = center(a)[axis], y = center(b)[axis];
                       return x < y || (x == y && a < b);
                     });
    const int left = build(begin, middle), right = build(middle, end);
    tree_[index].left = left;
    tree_[index].right = right;
    return index;
  }
  double distance(int index, const Eigen::Vector3d& p) const {
    const auto& node = tree_[index];
    return (node.low - p).cwiseMax(p - node.high).cwiseMax(0.).squaredNorm();
  }
  void triangle(size_t index, const Eigen::Vector3d& point,
                Result& best) const {
    const auto& ids = solver_.boundaryFaces().at(faces_[index]).vertices;
    const auto& x = solver_.mesh().vertices;
    const Eigen::Vector3d a = x[ids[0]].e(), e0 = x[ids[1]].e() - a,
                          e1 = x[ids[2]].e() - a;
    const Eigen::Vector3d normal = e0.cross(e1), offset = point - a;
    const double denominator = normal.squaredNorm();
    if (!(denominator > 0.) || !std::isfinite(denominator))
      throw std::runtime_error(
          "Tie surface is degenerate or outside floating-point range");
    const double u = offset.dot(e1.cross(normal)) / denominator,
                 v = offset.dot(normal.cross(e0)) / denominator;
    const auto candidate = [&](const Eigen::Vector3d& weights) {
      const double distance =
          (weights[1] * e0 + weights[2] * e1 - offset).squaredNorm();
      if (distance < best.distance ||
          (distance == best.distance && index < best.order))
        best = {ids, weights, distance, index};
    };
    if (u >= 0. && v >= 0. && u + v <= 1.) candidate({1. - u - v, u, v});
    const Eigen::Vector3d vertices[3] = {Eigen::Vector3d::Zero(), e0, e1};
    for (int edge = 0; edge < 3; ++edge) {
      const Eigen::Vector3d direction =
          vertices[(edge + 1) % 3] - vertices[edge];
      const double t = std::clamp(
          (offset - vertices[edge]).dot(direction) / direction.squaredNorm(),
          0., 1.);
      Eigen::Vector3d weights = Eigen::Vector3d::Zero();
      weights[edge] = 1. - t;
      weights[(edge + 1) % 3] = t;
      candidate(weights);
    }
  }
  void query(int index, const Eigen::Vector3d& point, Result& best) const {
    if (distance(index, point) > best.distance) return;
    const auto& node = tree_[index];
    if (node.left < 0) {
      for (size_t i = node.begin; i < node.end; ++i)
        triangle(order_[i], point, best);
      return;
    }
    const bool leftFirst =
        distance(node.left, point) <= distance(node.right, point);
    query(leftFirst ? node.left : node.right, point, best);
    query(leftFirst ? node.right : node.left, point, best);
  }
};
}  // namespace fem_workbench
