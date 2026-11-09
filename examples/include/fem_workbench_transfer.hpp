// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <memory>
#include <numeric>

#include "fem_surface_projection.hpp"
#include "fem_surface_selection.hpp"
#include "fem_workbench_geometry.hpp"

namespace fem_workbench {
struct RegionGeometry {
  struct Point {
    Vec<3> position;
    unsigned int original;
    int part;
  };
  struct Group {
    int part;
    std::vector<size_t> faces;
    std::unique_ptr<SurfaceProjection> tree;
  };
  SurfaceGeometry surface;
  std::vector<Point> points;
  std::vector<Group> groups;
  bool partialPlane = false;
  double deviation = 0.;
  std::shared_ptr<const RegionGeometry> reference;
  std::vector<size_t> referenceFaces;
};
struct SelectionAnchor {
  std::shared_ptr<const RegionGeometry> geometry;
  std::vector<unsigned int> nodes;
  std::vector<size_t> faces;
};
inline void indexRegion(const std::shared_ptr<RegionGeometry>& data) {
  const auto& surface = data->surface;
  std::vector<int> tags = surface.parts;
  std::sort(tags.begin(), tags.end());
  tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
  for (const auto tag : tags) {
    RegionGeometry::Group group;
    group.part = tag;
    for (size_t f = 0; f < surface.faces.size(); ++f)
      if (surface.parts[f] == tag) group.faces.push_back(f);
    data->groups.push_back(std::move(group));
  }
  for (auto& group : data->groups)
    group.tree = std::make_unique<SurfaceProjection>(
        surface.mesh, surface.faces, group.faces);
}
inline std::shared_ptr<const SelectionAnchor> captureRegion(
    const raisim::fem::TetMesh& mesh,
    const std::vector<LinearElasticity::BoundaryFace>& boundary,
    const std::vector<int>& parts, const std::vector<unsigned int>& nodes,
    const std::vector<size_t>& faces,
    std::shared_ptr<const SelectionAnchor> previous,
    std::shared_ptr<const RegionGeometry> reference) {
  if (previous && previous->nodes == nodes && previous->faces == faces)
    return previous;
  auto data = std::make_shared<RegionGeometry>();
  data->reference = std::move(reference);
  data->deviation = data->reference ? data->reference->deviation : 0.;
  data->referenceFaces = faces;
  std::sort(data->referenceFaces.begin(), data->referenceFaces.end());
  auto& surface = data->surface;
  std::vector<unsigned int> used;
  for (const auto f : faces)
    for (const auto v : boundary.at(f).vertices) used.push_back(v);
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  for (const auto v : used)
    surface.mesh.vertices.push_back(mesh.vertices.at(v));
  for (const auto f : faces) {
    auto triangle = boundary.at(f);
    for (auto& v : triangle.vertices)
      v = unsigned(std::lower_bound(used.begin(), used.end(), v) -
                   used.begin());
    triangle.element = 0;
    surface.faces.push_back(triangle);
    surface.parts.push_back(parts.at(f));
  }
  std::vector<int> owners(mesh.vertices.size(), 0);
  for (size_t f = 0; f < boundary.size(); ++f)
    for (const auto v : boundary[f].vertices) owners[v] = parts[f];
  for (const auto v : nodes)
    if (!std::binary_search(used.begin(), used.end(), v))
      data->points.push_back({mesh.vertices.at(v), v, owners.at(v)});
  if (!faces.empty()) {
    const auto normal = [&](size_t f) {
      const auto& v = boundary[f].vertices;
      return (mesh.vertices[v[1]].e() - mesh.vertices[v[0]].e())
          .cross(mesh.vertices[v[2]].e() - mesh.vertices[v[0]].e())
          .normalized()
          .eval();
    };
    bool planar = true;
    for (const auto f : faces)
      planar &= normal(faces[0]).dot(normal(f)) > 1. - 1e-6;
    if (planar) {
      SurfaceSelector selector(mesh, boundary);
      auto selected = faces;
      std::sort(selected.begin(), selected.end());
      std::vector<unsigned char> visited(boundary.size(), 0);
      for (const auto seed : faces)
        if (!visited[seed])
          for (const auto f :
               selector.collect(seed, SurfaceSelection::Planar)) {
            visited[f] = 1;
            if (!std::binary_search(selected.begin(), selected.end(), f))
              data->partialPlane = true;
          }
    }
  }
  indexRegion(data);
  return std::make_shared<SelectionAnchor>(SelectionAnchor{data, nodes, faces});
}
inline double clippedTriangleArea(
    const std::array<Eigen::Vector2d, 3>& clip,
    const std::array<Eigen::Vector2d, 3>& triangle) {
  const auto cross = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a[0] * b[1] - a[1] * b[0];
  };
  const double orientation =
      cross(clip[1] - clip[0], clip[2] - clip[0]) >= 0. ? 1. : -1.;
  std::vector<Eigen::Vector2d> polygon(triangle.begin(), triangle.end());
  for (int edge = 0; edge < 3 && !polygon.empty(); ++edge) {
    std::vector<Eigen::Vector2d> next;
    const auto distance = [&](const Eigen::Vector2d& p) {
      return orientation *
             cross(clip[(edge + 1) % 3] - clip[edge], p - clip[edge]);
    };
    for (size_t i = 0; i < polygon.size(); ++i) {
      const auto a = polygon[i], b = polygon[(i + 1) % polygon.size()];
      const double da = distance(a), db = distance(b);
      if (da >= 0.) next.push_back(a);
      if ((da < 0.) != (db < 0.))
        next.push_back(a + (b - a) * (da / (da - db)));
    }
    polygon = std::move(next);
  }
  double area = 0.;
  for (size_t i = 0; i < polygon.size(); ++i)
    area += cross(polygon[i], polygon[(i + 1) % polygon.size()]);
  return std::abs(area) * .5;
}
inline std::shared_ptr<const SelectionAnchor> mapRegion(
    const std::shared_ptr<const SelectionAnchor>& anchor,
    const raisim::fem::TetMesh& mesh,
    const std::vector<LinearElasticity::BoundaryFace>& boundary,
    const std::vector<int>& parts, double tolerance,
    double destinationDeviation, std::vector<unsigned int>& nodes,
    std::vector<size_t>& faces) {
  if (!anchor) return {};
  const auto& data = *anchor->geometry;
  tolerance = std::max(tolerance, 4. * (data.deviation + destinationDeviation));
  std::vector<unsigned char> selected(mesh.vertices.size(), 0);
  faces.clear();
  nodes.clear();
  const double limit = tolerance * tolerance;
  for (size_t f = 0; f < boundary.size(); ++f) {
    const auto group =
        std::lower_bound(data.groups.begin(), data.groups.end(), parts[f],
                         [](const auto& a, int b) { return a.part < b; });
    if (group == data.groups.end() || group->part != parts[f]) continue;
    const auto& ids = boundary[f].vertices;
    const Eigen::Vector3d p[3]{mesh.vertices[ids[0]].e(),
                               mesh.vertices[ids[1]].e(),
                               mesh.vertices[ids[2]].e()};
    const Eigen::Vector3d normal =
        (p[1] - p[0]).cross(p[2] - p[0]).normalized();
    const RegionGeometry::Group* canonical = nullptr;
    if (data.reference) {
      const auto found = std::lower_bound(
          data.reference->groups.begin(), data.reference->groups.end(),
          parts[f], [](const auto& a, int b) { return a.part < b; });
      if (found != data.reference->groups.end() && found->part == parts[f])
        canonical = &*found;
    }
    const auto inside = [&](const Eigen::Vector3d& q) {
      const auto projection = group->tree->project(q, &normal);
      if (!(projection.distance <= limit)) return false;
      if (canonical) {
        const double nearest = canonical->tree->project(q, &normal).distance;
        const double roundoff =
            std::max(1e-24 * (p[1] - p[0]).squaredNorm(),
                     1e-10 * std::max(nearest, projection.distance));
        if (projection.distance > nearest + roundoff) return false;
      }
      const auto& v = projection.face;
      const auto& x = data.surface.mesh.vertices;
      const Eigen::Vector3d n = (x[v[1]].e() - x[v[0]].e())
                                    .cross(x[v[2]].e() - x[v[0]].e())
                                    .normalized();
      return normal.dot(n) > .5;
    };
    bool covered = true;
    for (int a = 0; a < 3; ++a) {
      const bool onRegion = inside(p[a]);
      if (onRegion) selected[ids[a]] = 1;
      covered &= onRegion && inside((p[a] + p[(a + 1) % 3]) * .5);
    }
    covered &= inside((p[0] + p[1] + p[2]) / 3.);
    bool samePlane = false;
    if (data.partialPlane) {
      const auto& v = data.surface.faces[group->faces[0]].vertices;
      const auto& x = data.surface.mesh.vertices;
      const Eigen::Vector3d n = (x[v[1]].e() - x[v[0]].e())
                                    .cross(x[v[2]].e() - x[v[0]].e())
                                    .normalized();
      samePlane = normal.dot(n) > 1. - 1e-5;
      for (const auto& q : p)
        samePlane &= std::abs(n.dot(q - x[v[0]].e())) <= tolerance;
    }
    if (samePlane) {
      Eigen::Index drop;
      normal.cwiseAbs().maxCoeff(&drop);
      const auto project = [&](const Eigen::Vector3d& point) {
        const Eigen::Vector3d q = point - p[0];
        return Eigen::Vector2d(q[(drop + 1) % 3], q[(drop + 2) % 3]);
      };
      const std::array<Eigen::Vector2d, 3> triangle{
          project(p[0]), project(p[1]), project(p[2])};
      const double area = clippedTriangleArea(triangle, triangle);
      const Eigen::Vector3d low =
          (p[0].cwiseMin(p[1]).cwiseMin(p[2]).array() - tolerance).matrix();
      const Eigen::Vector3d high =
          (p[0].cwiseMax(p[1]).cwiseMax(p[2]).array() + tolerance).matrix();
      double overlap = 0.;
      for (const auto source :
           group->tree->overlapping(low.matrix(), high.matrix())) {
        const auto& v = data.surface.faces[source].vertices;
        overlap += clippedTriangleArea(
            triangle, {project(data.surface.mesh.vertices[v[0]].e()),
                       project(data.surface.mesh.vertices[v[1]].e()),
                       project(data.surface.mesh.vertices[v[2]].e())});
      }
      if (overlap > area * 1e-6 && overlap < area * (1. - 1e-6))
        throw std::runtime_error(
            "The mesh does not resolve a fixture boundary; select a complete "
            "surface, split the CAD face, or use a conforming volume mesh");
      covered &= overlap >= area * (1. - 1e-6);
    }
    if (covered && canonical) {
      Eigen::Index drop;
      normal.cwiseAbs().maxCoeff(&drop);
      const auto project = [&](const Eigen::Vector3d& point) {
        const Eigen::Vector3d q = point - p[0];
        return Eigen::Vector2d(q[(drop + 1) % 3], q[(drop + 2) % 3]);
      };
      const std::array<Eigen::Vector2d, 3> triangle{
          project(p[0]), project(p[1]), project(p[2])};
      const double area = clippedTriangleArea(triangle, triangle);
      const Eigen::Vector3d low =
          (p[0].cwiseMin(p[1]).cwiseMin(p[2]).array() - tolerance).matrix();
      const Eigen::Vector3d high =
          (p[0].cwiseMax(p[1]).cwiseMax(p[2]).array() + tolerance).matrix();
      for (const auto source : canonical->tree->overlapping(low, high)) {
        if (std::binary_search(data.referenceFaces.begin(),
                               data.referenceFaces.end(), source))
          continue;
        const auto& vertices = data.reference->surface.mesh.vertices;
        const auto& v = data.reference->surface.faces[source].vertices;
        const Eigen::Vector3d q[3]{vertices[v[0]].e(), vertices[v[1]].e(),
                                   vertices[v[2]].e()};
        const Eigen::Vector3d n = (q[1] - q[0]).cross(q[2] - q[0]).normalized();
        if (normal.dot(n) <= .5) continue;
        bool nearby = true;
        for (const auto& point : q)
          nearby &= std::abs(normal.dot(point - p[0])) <= tolerance;
        if (!nearby) continue;
        if (clippedTriangleArea(triangle, {project(q[0]), project(q[1]),
                                           project(q[2])}) > area * 1e-8)
          throw std::runtime_error(
              "The mesh crosses an unselected fixture region; use a conforming "
              "mesh or split the CAD face");
      }
    }
    if (covered) faces.push_back(f);
  }
  for (const auto& group : data.groups) {
    std::vector<size_t> mapped;
    for (const auto face : faces)
      if (parts[face] == group.part) mapped.push_back(face);
    SurfaceProjection target(mesh, boundary, mapped);
    for (const auto face : group.faces) {
      const auto& ids = data.surface.faces[face].vertices;
      const auto& vertices = data.surface.mesh.vertices;
      const Eigen::Vector3d p[3]{vertices[ids[0]].e(), vertices[ids[1]].e(),
                                 vertices[ids[2]].e()};
      const Eigen::Vector3d normal =
          (p[1] - p[0]).cross(p[2] - p[0]).normalized();
      const auto covered = [&](const Eigen::Vector3d& point) {
        if (!(target.project(point, &normal).distance <= limit))
          throw std::runtime_error(
              "The mesh does not preserve the complete fixture region; select "
              "a complete surface or use a conforming mesh");
      };
      for (int a = 0; a < 3; ++a) {
        covered(p[a]);
        covered((p[a] + p[(a + 1) % 3]) * .5);
      }
      covered((p[0] + p[1] + p[2]) / 3.);
    }
  }
  std::vector<std::pair<int, unsigned int>> membership;
  if (!data.points.empty()) {
    for (size_t f = 0; f < boundary.size(); ++f)
      for (const auto v : boundary[f].vertices)
        membership.emplace_back(parts[f], v);
    std::sort(membership.begin(), membership.end());
    membership.erase(std::unique(membership.begin(), membership.end()),
                     membership.end());
  }
  std::array<std::vector<unsigned int>, 3> coordinates;
  bool indexed = false;
  for (const auto& point : data.points) {
    double distance = std::numeric_limits<double>::infinity();
    unsigned int nearest = 0;
    const auto compatible = [&](unsigned int v) {
      if (!point.part) return true;
      return std::binary_search(membership.begin(), membership.end(),
                                std::make_pair(point.part, v));
    };
    if (point.original < mesh.vertices.size() &&
        (mesh.vertices[point.original].e() - point.position.e())
                .squaredNorm() <= limit &&
        compatible(point.original)) {
      nearest = point.original;
      distance =
          (mesh.vertices[nearest].e() - point.position.e()).squaredNorm();
    } else {
      const std::vector<unsigned int>* candidates = nullptr;
      size_t begin = 0, end = mesh.vertices.size();
      if (data.points.size() >= 16) {
        if (!indexed) {
          for (size_t a = 0; a < 3; ++a) {
            auto& index = coordinates[a];
            index.resize(mesh.vertices.size());
            std::iota(index.begin(), index.end(), 0u);
            std::sort(index.begin(), index.end(), [&](auto x, auto y) {
              return mesh.vertices[x][a] < mesh.vertices[y][a] ||
                     (mesh.vertices[x][a] == mesh.vertices[y][a] && x < y);
            });
          }
          indexed = true;
        }
        for (size_t a = 0; a < 3; ++a) {
          const auto& index = coordinates[a];
          const auto low = std::lower_bound(
              index.begin(), index.end(), point.position[a] - tolerance,
              [&](auto v, double x) { return mesh.vertices[v][a] < x; });
          const auto high = std::upper_bound(
              index.begin(), index.end(), point.position[a] + tolerance,
              [&](double x, auto v) { return x < mesh.vertices[v][a]; });
          if (!candidates || size_t(high - low) < end - begin) {
            candidates = &index;
            begin = size_t(low - index.begin());
            end = size_t(high - index.begin());
          }
        }
      }
      for (size_t i = begin; i < end; ++i) {
        const auto v = candidates ? (*candidates)[i] : unsigned(i);
        const double d =
            (mesh.vertices[v].e() - point.position.e()).squaredNorm();
        if (d < distance && d <= limit && compatible(v)) {
          nearest = v;
          distance = d;
        }
      }
    }
    if (!(distance <= limit))
      throw std::runtime_error(
          "A point fixture could not be mapped; select a surface or use a "
          "finer mesh");
    selected[nearest] = 1;
  }
  for (unsigned int v = 0; v < selected.size(); ++v)
    if (selected[v]) nodes.push_back(v);
  if ((!anchor->nodes.empty() && nodes.empty()) ||
      (!anchor->faces.empty() && faces.empty()))
    throw std::runtime_error(
        "A selected fixture region could not be resolved; reduce the element "
        "size");
  return std::make_shared<SelectionAnchor>(
      SelectionAnchor{anchor->geometry, nodes, faces});
}
inline void writeAnchor(
    std::ostream& out, const std::shared_ptr<const SelectionAnchor>& anchor,
    const std::vector<std::shared_ptr<const RegionGeometry>>& references) {
  out << bool(anchor) << '\n';
  if (!anchor) return;
  const auto& data = *anchor->geometry;
  const auto found =
      std::find(references.begin(), references.end(), data.reference);
  out << (found == references.end() ? -1
                                    : std::distance(references.begin(), found))
      << ' ' << data.referenceFaces.size() << ' ';
  for (auto face : data.referenceFaces) out << face << ' ';
  out << data.deviation << ' ' << data.partialPlane << ' '
      << std::quoted(data.surface.encode()) << ' ' << data.points.size()
      << '\n';
  for (const auto& p : data.points)
    out << p.position[0] << ' ' << p.position[1] << ' ' << p.position[2] << ' '
        << p.original << ' ' << p.part << '\n';
}
inline std::shared_ptr<const SelectionAnchor> readAnchor(
    std::istream& in, const std::vector<unsigned int>& nodes,
    const std::vector<size_t>& faces, size_t budget,
    const std::vector<std::shared_ptr<const RegionGeometry>>& references) {
  bool present = false;
  in >> present;
  if (!in) throw std::runtime_error("Truncated fixture geometry");
  if (!present) return {};
  auto data = std::make_shared<RegionGeometry>();
  long long reference = -1;
  size_t selected = 0;
  in >> reference >> selected;
  if (!in || reference < -1 ||
      (reference >= 0 && size_t(reference) >= references.size()) ||
      selected > budget)
    throw std::runtime_error("Invalid fixture reference geometry");
  if (reference >= 0) data->reference = references[size_t(reference)];
  data->referenceFaces.resize(selected);
  for (auto& face : data->referenceFaces) {
    in >> face;
    if (!in || !data->reference ||
        face >= data->reference->surface.faces.size())
      throw std::runtime_error("Invalid reference face");
  }
  std::sort(data->referenceFaces.begin(), data->referenceFaces.end());
  std::string bytes;
  size_t count = 0;
  in >> data->deviation >> data->partialPlane >> std::quoted(bytes) >> count;
  if (!in || !std::isfinite(data->deviation) || data->deviation < 0. ||
      count > budget || count > 10000000)
    throw std::runtime_error("Invalid fixture geometry count");
  data->surface = SurfaceGeometry::decode(bytes, 1., false);
  data->points.resize(count);
  for (auto& p : data->points) {
    in >> p.position[0] >> p.position[1] >> p.position[2] >> p.original >>
        p.part;
    if (!in || !p.position.e().allFinite())
      throw std::runtime_error("Invalid fixture point");
  }
  indexRegion(data);
  return std::make_shared<SelectionAnchor>(SelectionAnchor{data, nodes, faces});
}
}
