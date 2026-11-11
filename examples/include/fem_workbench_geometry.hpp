// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <utility>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "raisim/PathUtils.hpp"
#include "raisim/fem/LinearElasticity.hpp"

namespace fem_workbench {
struct SurfaceGeometry {
  raisim::fem::TetMesh mesh;
  std::vector<raisim::fem::LinearElasticity::BoundaryFace> faces;
  std::vector<int> parts;
  std::vector<int> brepFaces;

  void normalize() {
    if (mesh.vertices.empty() || faces.empty() || parts.size() != faces.size())
      throw std::runtime_error("Surface contains no complete triangles");
    if (!brepFaces.empty() && brepFaces.size() != faces.size())
      throw std::runtime_error(
          "Surface B-rep faces do not match its triangles");
    auto low = mesh.vertices[0].e().eval(), high = low;
    for (const auto& v : mesh.vertices) {
      if (!v.e().allFinite())
        throw std::runtime_error("Surface coordinates must be finite");
      low = low.cwiseMin(v.e());
      high = high.cwiseMax(v.e());
    }
    const double extent = (high - low).stableNorm();
    if (!(extent > 0.) || !std::isfinite(extent))
      throw std::runtime_error("Invalid surface extent");
    struct Corner {
      int part;
      std::array<long long, 3> key;
      unsigned int node;
      size_t corner;
    };
    std::vector<Corner> corners;
    corners.reserve(3 * faces.size());
    for (size_t f = 0; f < faces.size(); ++f)
      for (int a = 0; a < 3; ++a) {
        const auto node = faces[f].vertices[a];
        if (node >= mesh.vertices.size())
          throw std::runtime_error("Invalid surface triangle index");
        Corner c{parts[f], {}, node, 3 * f + size_t(a)};
        for (int b = 0; b < 3; ++b)
          c.key[b] = std::llround((mesh.vertices[node][b] - low[b]) /
                                  (extent * 1e-10));
        corners.push_back(c);
      }
    std::sort(corners.begin(), corners.end(),
              [](const Corner& a, const Corner& b) {
                return std::tie(a.part, a.key, a.node) <
                       std::tie(b.part, b.key, b.node);
              });
    std::vector<raisim::Vec<3>> vertices;
    unsigned int node = 0;
    for (size_t i = 0; i < corners.size(); ++i) {
      if (!i || corners[i].part != corners[i - 1].part ||
          corners[i].key != corners[i - 1].key) {
        node = static_cast<unsigned int>(vertices.size());
        vertices.push_back(mesh.vertices[corners[i].node]);
      }
      faces[corners[i].corner / 3].vertices[corners[i].corner % 3] = node;
    }
    mesh.vertices = std::move(vertices);
    for (auto& face : faces) {
      const auto& v = face.vertices;
      const Eigen::Vector3d cross =
          (mesh.vertices[v[1]].e() - mesh.vertices[v[0]].e())
              .cross(mesh.vertices[v[2]].e() - mesh.vertices[v[0]].e());
      if (!(cross.stableNorm() > 0.))
        throw std::runtime_error("Surface contains a collapsed triangle");
      face.element = 0;
    }
    std::vector<int> tags = parts;
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    struct Edge {
      int part;
      unsigned int a, b;
      int direction;
    };
    std::vector<Edge> edges;
    std::vector<bool> closed(tags.size(), true);
    std::vector<long double> volumes(tags.size(), 0.);
    std::vector<Eigen::Vector3d> origins(tags.size());
    std::vector<bool> initialized(tags.size(), false);
    for (size_t f = 0; f < faces.size(); ++f) {
      const size_t group = size_t(
          std::lower_bound(tags.begin(), tags.end(), parts[f]) - tags.begin());
      const auto& ids = faces[f].vertices;
      if (!initialized[group]) {
        origins[group] = mesh.vertices[ids[0]].e();
        initialized[group] = true;
      }
      const Eigen::Vector3d a = mesh.vertices[ids[0]].e() - origins[group],
                            b = mesh.vertices[ids[1]].e() - origins[group],
                            c = mesh.vertices[ids[2]].e() - origins[group];
      volumes[group] += static_cast<long double>(a.dot(b.cross(c)));
      for (size_t e = 0; e < 3; ++e) {
        const auto u = ids[e], v = ids[(e + 1) % 3];
        edges.push_back(
            {parts[f], std::min(u, v), std::max(u, v), u < v ? 1 : -1});
      }
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
      return std::tie(a.part, a.a, a.b) < std::tie(b.part, b.a, b.b);
    });
    for (size_t i = 0; i < edges.size();) {
      size_t end = i + 1;
      int direction = edges[i].direction;
      while (end < edges.size() && edges[end].part == edges[i].part &&
             edges[end].a == edges[i].a && edges[end].b == edges[i].b) {
        direction += edges[end].direction;
        ++end;
      }
      if (end - i != 2 || direction != 0)
        closed[size_t(
            std::lower_bound(tags.begin(), tags.end(), edges[i].part) -
            tags.begin())] = false;
      i = end;
    }
    for (size_t f = 0; f < faces.size(); ++f) {
      const auto group = size_t(
          std::lower_bound(tags.begin(), tags.end(), parts[f]) - tags.begin());
      if (closed[group] && volumes[group] < 0.)
        std::swap(faces[f].vertices[1], faces[f].vertices[2]);
    }
  }
  std::string encode() const {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    const bool brep = !brepFaces.empty();
    out << "RAISIM_FEM_SURFACE " << (brep ? 2 : 1) << '\n'
        << mesh.vertices.size() << ' ' << faces.size() << '\n';
    for (const auto& p : mesh.vertices)
      out << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
    for (size_t f = 0; f < faces.size(); ++f) {
      out << faces[f].vertices[0] << ' ' << faces[f].vertices[1] << ' '
          << faces[f].vertices[2] << ' ' << parts[f];
      if (brep) out << ' ' << brepFaces[f];
      out << '\n';
    }
    return out.str();
  }
  static SurfaceGeometry decode(const std::string& bytes, double scale = 1.,
                                bool normalize = true) {
    if (!(scale > 0.) || !std::isfinite(scale))
      throw std::runtime_error("Invalid surface scale");
    std::istringstream in(bytes);
    in.imbue(std::locale::classic());
    std::string magic;
    int version;
    size_t points = 0, triangles = 0;
    in >> magic >> version >> points >> triangles;
    if (!in || magic != "RAISIM_FEM_SURFACE" ||
        (version != 1 && version != 2) ||
        points > 10000000 || triangles > 10000000 || points > bytes.size() ||
        triangles > bytes.size())
      throw std::runtime_error("Invalid surface preview header");
    SurfaceGeometry result;
    result.mesh.vertices.resize(points);
    result.faces.resize(triangles);
    result.parts.resize(triangles);
    if (version == 2) result.brepFaces.resize(triangles);
    for (auto& p : result.mesh.vertices) {
      in >> p[0] >> p[1] >> p[2];
      p *= scale;
    }
    for (size_t f = 0; f < triangles; ++f) {
      in >> result.faces[f].vertices[0] >> result.faces[f].vertices[1] >>
          result.faces[f].vertices[2] >> result.parts[f];
      if (version == 2) {
        in >> result.brepFaces[f];
        if (in && result.brepFaces[f] <= 0)
          throw std::runtime_error("Invalid surface B-rep face");
      }
    }
    if (!in) throw std::runtime_error("Truncated surface preview");
    if (normalize)
      result.normalize();
    else if (!result.faces.empty()) {
      auto check = result;
      check.normalize();
    }
    return result;
  }
};
inline std::string encodeVolume(const raisim::fem::TetMesh& mesh) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "RAISIM_FEM_VOLUME 1\n"
      << mesh.vertices.size() << ' ' << mesh.tetrahedra.size() << '\n';
  for (const auto& p : mesh.vertices)
    out << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
  for (size_t t = 0; t < mesh.tetrahedra.size(); ++t)
    out << mesh.tetrahedra[t][0] << ' ' << mesh.tetrahedra[t][1] << ' '
        << mesh.tetrahedra[t][2] << ' ' << mesh.tetrahedra[t][3] << ' '
        << (mesh.regions.empty() ? 0 : mesh.regions[t]) << '\n';
  return out.str();
}
/// Appends the volume mesh of another CAD file. Its regions, or region 1 when
/// it has none, are numbered after @p regionOffset.
inline void appendVolume(raisim::fem::TetMesh& into, raisim::fem::TetMesh part,
                         int regionOffset) {
  if (into.regions.size() != into.tetrahedra.size())
    into.regions.assign(into.tetrahedra.size(), 1);
  if (!part.regions.empty() && part.regions.size() != part.tetrahedra.size())
    throw std::runtime_error("Volume regions do not match its tetrahedra");
  const auto offset = static_cast<unsigned int>(into.vertices.size());
  into.vertices.insert(into.vertices.end(), part.vertices.begin(),
                       part.vertices.end());
  for (size_t t = 0; t < part.tetrahedra.size(); ++t) {
    auto cell = part.tetrahedra[t];
    for (auto& v : cell) {
      if (v >= part.vertices.size())
        throw std::runtime_error("Invalid volume vertex index");
      v += offset;
    }
    into.tetrahedra.push_back(cell);
    into.regions.push_back((part.regions.empty() ? 1 : part.regions[t]) +
                           regionOffset);
  }
}
/// A CAD file added to an assembly after the first; its solids become regions
/// numbered after @p regionOffset.
struct CadFile {
  std::string name, extension, bytes;
  double scale = 1.;
  int regionOffset = 0;
};
/// Removes the tetrahedra of @p excluded regions and the vertices only they use.
inline void removeRegions(raisim::fem::TetMesh& mesh,
                          const std::vector<int>& excluded) {
  if (excluded.empty() || mesh.regions.size() != mesh.tetrahedra.size())
    return;
  std::vector<unsigned int> index(mesh.vertices.size(), ~0u);
  raisim::fem::TetMesh kept;
  for (size_t t = 0; t < mesh.tetrahedra.size(); ++t) {
    if (std::find(excluded.begin(), excluded.end(), mesh.regions[t]) !=
        excluded.end())
      continue;
    auto cell = mesh.tetrahedra[t];
    for (auto& v : cell) {
      if (index[v] == ~0u) {
        index[v] = static_cast<unsigned int>(kept.vertices.size());
        kept.vertices.push_back(mesh.vertices[v]);
      }
      v = index[v];
    }
    kept.tetrahedra.push_back(cell);
    kept.regions.push_back(mesh.regions[t]);
  }
  mesh = std::move(kept);
}
struct GeometrySource {
  std::string extension, bytes;
  /// Regions of the first file are numbered after this offset.
  int regionOffset = 0;
  std::vector<CadFile> assembly;
  /// Removed parts of files that still have other parts.
  std::vector<int> excludedRegions;
  double scale = 1., maximumEdge = .05, surfaceTolerance = 0., meshedEdge = 0.,
         meshedTolerance = 0., previewTolerance = 0., referenceExtent = 0.;
  std::shared_ptr<const raisim::fem::TetMesh> volume;
  void recordPreviewAccuracy() {
    previewTolerance =
        surfaceTolerance > 0. ? surfaceTolerance : referenceExtent * .001;
  }
};
}
