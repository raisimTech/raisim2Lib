// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fem_surface_selection.hpp"
#include "fem_workbench_fixtures.hpp"
#include "fem_workbench_transfer.hpp"
#include "raisim/PathUtils.hpp"
#include "raisim/fem/LinearElasticity.hpp"

namespace fem_workbench {
using raisim::Vec;
using raisim::fem::LinearElasticity;
using raisim::fem::TetMesh;

class Model {
 public:
  std::unique_ptr<LinearElasticity> solver;
  LinearElasticity::Material material{70e9, .3, 2700.};
  std::vector<Condition> conditions;
  std::vector<unsigned char> selected;
  std::string name = "Cantilever demonstration";
  bool dirty = true, hasSolution = false;
  std::vector<size_t> boltConditions;
  GeometrySource source;
  const TetMesh& mesh() const {
    if (solver) return solver->mesh();
    if (surface_) return surface_->mesh;
    throw std::runtime_error("Import a solid first");
  }
  const std::vector<LinearElasticity::BoundaryFace>& boundaryFaces() const {
    if (solver) return solver->boundaryFaces();
    if (surface_) return surface_->faces;
    throw std::runtime_error("Import a solid first");
  }
  bool hasGeometry() const { return bool(solver) || bool(surface_); }
  const std::vector<int>& faceParts() const {
    return solver ? boundaryParts_ : surface_->parts;
  }
  void setSurface(SurfaceGeometry surface, std::string title) {
    if (surface.mesh.vertices.empty() || surface.faces.empty() ||
        surface.parts.size() != surface.faces.size())
      throw std::runtime_error("Surface contains no valid geometry");
    auto next = std::make_shared<SurfaceGeometry>(std::move(surface));
    auto selector = std::make_unique<SurfaceSelector>(next->mesh, next->faces);
    referenceGeometry_.reset();
    solver.reset();
    surface_ = std::move(next);
    selector_ = std::move(selector);
    source = {};
    name = std::move(title);
    conditions.clear();
    boltConditions.clear();
    selected.assign(mesh().vertices.size(), 0);
    selectedSurfaces_.assign(boundaryFaces().size(), 0);
    previousMask_.assign(3 * selected.size(), 0);
    surfaceSelection_ = false;
    fixtureCache_.clear();
    dirty = true;
    hasSolution = false;
  }
  void setMeshed(TetMesh volume, double tolerance,
                 double destinationDeviation = 0.) {
    if (!(tolerance > 0.) || !std::isfinite(tolerance) ||
        destinationDeviation < 0. || !std::isfinite(destinationDeviation))
      throw std::runtime_error("Invalid fixture mapping tolerance");
    Model next;
    next.material = material;
    next.setMesh(std::move(volume), name);
    next.source = source;
    next.source.meshedTolerance = destinationDeviation;
    next.conditions = conditions;
    for (auto& c : next.conditions) {
      c.primaryGeometry = captureSelection(c.nodes, c.faces, c.primaryGeometry);
      c.targetGeometry =
          captureSelection(c.targetNodes, c.targetFaces, c.targetGeometry);
      c.primaryGeometry = mapRegion(
          c.primaryGeometry, next.mesh(), next.boundaryFaces(),
          next.faceParts(), tolerance, destinationDeviation, c.nodes, c.faces);
      c.targetGeometry = mapRegion(
          c.targetGeometry, next.mesh(), next.boundaryFaces(), next.faceParts(),
          tolerance, destinationDeviation, c.targetNodes, c.targetFaces);
    }
    *this = std::move(next);
  }

  void setMesh(TetMesh volume, std::string title) {
    auto next = std::make_unique<LinearElasticity>(std::move(volume));
    auto selector = std::make_unique<SurfaceSelector>(*next);
    solver = std::move(next);
    surface_.reset();
    referenceGeometry_.reset();
    source = {};
    boundaryParts_.clear();
    for (const auto& f : solver->boundaryFaces())
      boundaryParts_.push_back(solver->mesh().regions.empty()
                                   ? 0
                                   : solver->mesh().regions[f.element]);
    selector_ = std::move(selector);
    selectedSurfaces_.assign(boundaryFaces().size(), 0);
    surfaceSelection_ = false;
    name = std::move(title);
    conditions.clear();
    selected.assign(mesh().vertices.size(), 0);
    previousMask_.assign(3 * selected.size(), 0);
    fixtureCache_.clear();
    boltConditions.clear();
    dirty = true;
    hasSolution = false;
  }
  void demo() {
    setMesh(TetMesh::box({1., .16, .24}, {32, 6, 8}), "Aluminium cantilever");
    selectSlab(0, 0., 1e-10, false);
    add(Kind::Fixed);
    clearSelection();
    selectSlab(0, 1., 1e-10, false);
    add(Kind::Force);
    conditions.back().value = {0., 0., -5000.};
    clearSelection();
    solve();
  }
  void changed() {
    dirty = true;
    hasSolution = false;
  }
  void clearSelection() {
    std::fill(selected.begin(), selected.end(), 0);
    std::fill(selectedSurfaces_.begin(), selectedSurfaces_.end(), 0);
    surfaceSelection_ = false;
  }
  void restoreSelection(const std::vector<unsigned int>& nodes,
                        const std::vector<size_t>& faces) {
    clearSelection();
    surfaceSelection_ = true;
    for (const auto node : nodes) selected.at(node) = 1;
    for (const auto face : faces) selectedSurfaces_.at(face) = 1;
  }
  void useNodeSelection() { surfaceSelection_ = false; }
  size_t tieProjectionBuilds() const {
    return fixtureCache_.tieProjectionBuilds;
  }
  void selectSurface(size_t seed, SurfaceSelection mode, bool append) {
    const auto faces = selector_->collect(seed, mode);
    if (!append)
      clearSelection();
    else if (!surfaceSelection_)
      for (const auto face : selectedFaces()) selectedSurfaces_[face] = 1;
    surfaceSelection_ = true;
    for (const auto face : faces) {
      selectedSurfaces_[face] = 1;
      for (const auto node : boundaryFaces()[face].vertices) selected[node] = 1;
    }
  }
  void selectSlab(unsigned int axis, double coordinate, double tolerance,
                  bool append) {
    if (axis > 2 || !std::isfinite(coordinate) || !std::isfinite(tolerance) ||
        tolerance < 0.)
      throw std::invalid_argument("Invalid selection plane");
    if (!append) clearSelection();
    useNodeSelection();
    for (size_t i = 0; i < selected.size(); ++i)
      if (std::abs(mesh().vertices[i][axis] - coordinate) <= tolerance)
        selected[i] = 1;
  }
  std::vector<size_t> selectedFaces() const {
    std::vector<size_t> faces;
    if (surfaceSelection_) {
      for (size_t f = 0; f < selectedSurfaces_.size(); ++f)
        if (selectedSurfaces_[f]) faces.push_back(f);
      return faces;
    }
    for (size_t f = 0; f < boundaryFaces().size(); ++f) {
      const auto& ids = boundaryFaces()[f].vertices;
      if (selected[ids[0]] && selected[ids[1]] && selected[ids[2]])
        faces.push_back(f);
    }
    return faces;
  }
  void selectCylinder(const Vec<3>& center, const Vec<3>& direction,
                      double radius, double tolerance, double axialMinimum,
                      double axialMaximum, bool append) {
    const double length = direction.e().norm();
    if (!center.e().allFinite() || !std::isfinite(length) || length <= 0. ||
        !std::isfinite(radius) || radius <= 0. || !std::isfinite(tolerance) ||
        tolerance < 0. || !std::isfinite(axialMinimum) ||
        !std::isfinite(axialMaximum) || axialMinimum > axialMaximum)
      throw std::invalid_argument("Invalid cylindrical selection");
    if (!append) clearSelection();
    useNodeSelection();
    const Vec<3> n = direction / length;
    std::vector<unsigned char> surface(selected.size(), 0);
    for (const auto& face : boundaryFaces())
      for (const auto node : face.vertices) surface[node] = 1;
    for (size_t i = 0; i < selected.size(); ++i)
      if (surface[i]) {
        const Vec<3> r = mesh().vertices[i] - center;
        const double axial = r.e().dot(n.e());
        const Vec<3> radial = r - n * axial;
        if (axial >= axialMinimum - tolerance &&
            axial <= axialMaximum + tolerance &&
            std::abs(radial.e().norm() - radius) <= tolerance)
          selected[i] = 1;
      }
  }
  Condition& add(Kind kind) {
    Condition c;
    c.kind = kind;
    c.name = std::string(kindName(kind)) + " " +
             std::to_string(conditions.size() + 1);
    for (unsigned int i = 0; i < selected.size(); ++i)
      if (selected[i]) c.nodes.push_back(i);
    c.faces = selectedFaces();
    if (kind != Kind::Gravity && c.nodes.empty())
      throw std::runtime_error("Select a surface or nodes first");
    if ((kind == Kind::Traction || kind == Kind::Pressure) && c.faces.empty())
      throw std::runtime_error("This load needs selected boundary faces");
    if (kind == Kind::Gravity) {
      c.value = {0., 0., -9.81};
      c.nodes.clear();
      c.faces.clear();
    }
    if (kind == Kind::Force) c.value = {0., 0., -1000.};
    if (kind == Kind::Pressure) c.value[0] = 1e6;
    if (kind == Kind::Bearing) c.axes = {true, true, false};
    if (kind == Kind::BoltHole) c.axes = {true, true, true};
    if (!c.nodes.empty()) {
      for (const auto node : c.nodes) c.center += mesh().vertices[node];
      c.center *= 1. / c.nodes.size();
    }
    conditions.push_back(std::move(c));
    changed();
    return conditions.back();
  }
  void solve() {
    if (!solver) throw std::runtime_error("Import a solid first");
    dirty = false;
    hasSolution = false;
    solver->setMaterial(material);
    std::vector<unsigned char> mask(previousMask_.size(), 0);
    std::vector<double> values(mask.size(), 0.);
    for (const auto& c : conditions)
      if (c.enabled &&
          (c.kind == Kind::Fixed || c.kind == Kind::Displacement)) {
        for (const auto node : c.nodes)
          for (unsigned int a = 0; a < 3; ++a)
            if (c.kind == Kind::Fixed || c.axes[a]) {
              const size_t dof = 3 * static_cast<size_t>(node) + a;
              const double value = c.kind == Kind::Fixed ? 0. : c.value[a];
              if (node >= selected.size() || !std::isfinite(value))
                throw std::runtime_error(
                    "Invalid support node or prescribed value");
              if (mask[dof] && std::abs(values[dof] - value) > 1e-14)
                throw std::runtime_error(
                    "Overlapping supports prescribe different displacements");
              mask[dof] = 1;
              values[dof] = value;
            }
      }
    fixtureCache_.configure(*solver, conditions);
    boltConditions = fixtureCache_.boltConditions;
    for (size_t i = 0; i < mask.size(); ++i) {
      if (mask[i])
        solver->prescribe(static_cast<unsigned int>(i / 3),
                          static_cast<unsigned int>(i % 3), values[i]);
      else if (previousMask_[i])
        solver->release(static_cast<unsigned int>(i / 3),
                        static_cast<unsigned int>(i % 3));
    }
    previousMask_ = mask;
    solver->clearLoads();
    Vec<3> gravity{0., 0., 0.};
    for (const auto& c : conditions)
      if (c.enabled) {
        if (!c.value.e().allFinite())
          throw std::runtime_error("Load and support values must be finite");
        if (c.kind == Kind::Gravity) gravity += c.value;
        if (c.kind == Kind::Pressure)
          for (const auto face : c.faces)
            solver->addSurfacePressure(face, c.value[0]);
        if (c.kind == Kind::Traction)
          for (const auto face : c.faces)
            solver->addSurfaceTraction(face, c.value);
        if (c.kind == Kind::Force) {
          std::vector<double> weights(selected.size(), 0.);
          double total = 0.;
          for (const auto f : c.faces) {
            const auto& ids = boundaryFaces().at(f).vertices;
            const auto& x = mesh().vertices;
            const Vec<3> cross =
                (x[ids[1]] - x[ids[0]]).cross(x[ids[2]] - x[ids[0]]);
            const double areaThird = cross.e().norm() / 6.;
            for (const auto v : ids) weights[v] += areaThird;
            total += 3. * areaThird;
          }
          if (total == 0.) {
            if (c.nodes.empty())
              throw std::runtime_error("Force has no selected nodes");
            for (const auto v : c.nodes) {
              if (v >= weights.size())
                throw std::runtime_error("Invalid force node");
              weights[v] = 1.;
            }
            total = static_cast<double>(c.nodes.size());
          }
          for (unsigned int v = 0; v < weights.size(); ++v)
            if (weights[v] != 0.)
              solver->addNodalForce(v, c.value * (weights[v] / total));
        }
      }
    solver->setGravity(gravity);
    solver->solve();
    hasSolution = true;
    dirty = false;
  }
  void save(const std::string& path) const {
    if (!hasGeometry()) throw std::runtime_error("No model to save");
    std::ofstream out(raisim::pathFromUtf8(path));
    if (!out) throw std::runtime_error("Cannot write project: " + path);
    out.imbue(std::locale::classic());
    out.precision(17);
    out << "RAISIM_FEM_PROJECT 4\n"
        << std::quoted(name) << '\n'
        << material.youngsModulus << ' ' << material.poissonRatio << ' '
        << material.density << '\n';
    const auto& geometry = mesh();
    out << geometry.vertices.size() << ' ' << geometry.tetrahedra.size()
        << '\n';
    for (const auto& x : geometry.vertices)
      out << x[0] << ' ' << x[1] << ' ' << x[2] << '\n';
    for (size_t i = 0; i < geometry.tetrahedra.size(); ++i) {
      for (const auto v : geometry.tetrahedra[i]) out << v << ' ';
      out << (geometry.regions.empty() ? 0 : geometry.regions[i]) << '\n';
    }
    out << boundaryFaces().size() << '\n';
    for (size_t f = 0; f < boundaryFaces().size(); ++f)
      out << boundaryFaces()[f].vertices[0] << ' '
          << boundaryFaces()[f].vertices[1] << ' '
          << boundaryFaces()[f].vertices[2] << ' ' << faceParts()[f] << '\n';
    out << std::quoted(source.extension) << ' ' << source.scale << ' '
        << source.maximumEdge << ' ' << source.surfaceTolerance << ' '
        << source.meshedEdge << ' ' << source.meshedTolerance << ' '
        << source.previewTolerance << ' ' << source.referenceExtent << ' '
        << source.externalGmsh << ' ' << source.meshedExternalGmsh << ' '
        << std::quoted(source.bytes) << '\n';
    out << bool(source.volume) << '\n';
    if (source.volume) {
      const auto& v = *source.volume;
      out << v.vertices.size() << ' ' << v.tetrahedra.size() << '\n';
      for (const auto& x : v.vertices)
        out << x[0] << ' ' << x[1] << ' ' << x[2] << '\n';
      for (size_t t = 0; t < v.tetrahedra.size(); ++t)
        out << v.tetrahedra[t][0] << ' ' << v.tetrahedra[t][1] << ' '
            << v.tetrahedra[t][2] << ' ' << v.tetrahedra[t][3] << ' '
            << (v.regions.empty() ? 0 : v.regions[t]) << '\n';
    }
    auto prepared = conditions;
    std::vector<std::shared_ptr<const RegionGeometry>> references;
    for (auto& c : prepared) {
      c.primaryGeometry = captureSelection(c.nodes, c.faces, c.primaryGeometry);
      c.targetGeometry =
          captureSelection(c.targetNodes, c.targetFaces, c.targetGeometry);
      for (const auto& anchor : {c.primaryGeometry, c.targetGeometry})
        if (anchor && anchor->geometry->reference &&
            std::find(references.begin(), references.end(),
                      anchor->geometry->reference) == references.end())
          references.push_back(anchor->geometry->reference);
    }
    out << references.size() << '\n';
    for (const auto& reference : references)
      out << reference->deviation << ' '
          << std::quoted(reference->surface.encode()) << '\n';
    out << prepared.size() << '\n';
    for (const auto& c : prepared) {
      out << std::quoted(c.name) << ' ' << static_cast<int>(c.kind) << ' '
          << c.enabled;
      for (int a = 0; a < 3; ++a) out << ' ' << c.axes[a] << ' ' << c.value[a];
      out << ' ' << c.nodes.size() << ' ' << c.faces.size() << '\n';
      for (const auto v : c.nodes) out << v << ' ';
      out << '\n';
      for (const auto f : c.faces)
        for (const auto v : boundaryFaces().at(f).vertices) out << v << ' ';
      out << '\n';
      for (int a = 0; a < 3; ++a)
        out << c.axis[a] << ' ' << c.center[a] << ' ' << c.rotation[a] << ' ';
      out << c.stiffness << ' ' << c.tolerance << ' ' << c.bolt.diameter << ' '
          << c.bolt.youngsModulus << ' ' << c.bolt.poissonRatio << ' '
          << c.bolt.gripLength << ' ' << c.bolt.preload << ' '
          << c.targetNodes.size() << ' ' << c.targetFaces.size() << '\n';
      for (const auto v : c.targetNodes) out << v << ' ';
      out << '\n';
      for (const auto f : c.targetFaces)
        for (const auto v : boundaryFaces().at(f).vertices) out << v << ' ';
      out << '\n';
      writeAnchor(out, c.primaryGeometry, references);
      writeAnchor(out, c.targetGeometry, references);
    }
    out.close();
    if (!out) throw std::runtime_error("Failed writing FEM project");
  }
  void loadProject(const std::string& path, bool allowExternalMesher = true) {
    std::ifstream in(raisim::pathFromUtf8(path));
    in.imbue(std::locale::classic());
    std::string magic, title;
    int version = 0;
    in >> magic >> version >> std::quoted(title);
    if (magic != "RAISIM_FEM_PROJECT" || (version < 1 || version > 4))
      throw std::runtime_error("Unsupported FEM project");
    Model next;
    in >> next.material.youngsModulus >> next.material.poissonRatio >>
        next.material.density;
    if (!(next.material.youngsModulus > 0.) ||
        !std::isfinite(next.material.youngsModulus) ||
        !(next.material.poissonRatio > -1. &&
          next.material.poissonRatio < .5) ||
        !(next.material.density >= 0.) || !std::isfinite(next.material.density))
      throw std::runtime_error("Invalid project material");
    const auto budget = std::filesystem::file_size(raisim::pathFromUtf8(path));
    const auto readCount = [&]() {
      long long n = -1;
      in >> n;
      if (!in || n < 0 || n > 10000000 ||
          static_cast<unsigned long long>(n) > budget)
        throw std::runtime_error("Invalid FEM project count");
      return static_cast<size_t>(n);
    };
    TetMesh mesh;
    const auto nodes = readCount(), tets = readCount();
    mesh.vertices.resize(nodes);
    mesh.tetrahedra.resize(tets);
    mesh.regions.resize(tets);
    for (auto& x : mesh.vertices) in >> x[0] >> x[1] >> x[2];
    for (size_t i = 0; i < tets; ++i) {
      for (auto& v : mesh.tetrahedra[i]) in >> v;
      in >> mesh.regions[i];
    }
    if (!in) throw std::runtime_error("Truncated FEM project geometry");
    if (version == 4) {
      SurfaceGeometry surface;
      surface.mesh.vertices = mesh.vertices;
      const auto count = readCount();
      surface.faces.resize(count);
      surface.parts.resize(count);
      for (size_t f = 0; f < count; ++f) {
        auto& face = surface.faces[f];
        in >> face.vertices[0] >> face.vertices[1] >> face.vertices[2] >>
            surface.parts[f];
        for (auto v : face.vertices)
          if (v >= nodes) throw std::runtime_error("Invalid project surface");
      }
      if (mesh.tetrahedra.empty()) {
        if (!count) throw std::runtime_error("Project contains no geometry");
        next.setSurface(std::move(surface), title);
      } else
        next.setMesh(std::move(mesh), title);
      in >> std::quoted(next.source.extension) >> next.source.scale >>
          next.source.maximumEdge >> next.source.surfaceTolerance >>
          next.source.meshedEdge >> next.source.meshedTolerance >>
          next.source.previewTolerance >> next.source.referenceExtent >>
          next.source.externalGmsh >> next.source.meshedExternalGmsh >>
          std::quoted(next.source.bytes);
      if (!in || !(next.source.scale > 0.) ||
          !std::isfinite(next.source.scale) ||
          !(next.source.maximumEdge > 0.) ||
          !std::isfinite(next.source.maximumEdge) ||
          next.source.surfaceTolerance < 0. ||
          !std::isfinite(next.source.surfaceTolerance) ||
          next.source.meshedEdge < 0. ||
          !std::isfinite(next.source.meshedEdge) ||
          next.source.meshedTolerance < 0. ||
          !std::isfinite(next.source.meshedTolerance) ||
          next.source.previewTolerance < 0. ||
          !std::isfinite(next.source.previewTolerance) ||
          next.source.referenceExtent < 0. ||
          !std::isfinite(next.source.referenceExtent))
        throw std::runtime_error("Invalid project meshing settings");
      const std::vector<std::string> formats{"",      ".rfs",  ".step", ".stp",
                                             ".brep", ".iges", ".igs"};
      if (std::find(formats.begin(), formats.end(), next.source.extension) ==
              formats.end() ||
          next.source.bytes.size() > 128u * 1024u * 1024u)
        throw std::runtime_error("Invalid embedded project source");
      bool volumeSource = false;
      in >> volumeSource;
      if (volumeSource) {
        auto v = std::make_shared<TetMesh>();
        const auto nv = readCount(), nt = readCount();
        v->vertices.resize(nv);
        v->tetrahedra.resize(nt);
        v->regions.resize(nt);
        for (auto& x : v->vertices) in >> x[0] >> x[1] >> x[2];
        for (size_t t = 0; t < nt; ++t) {
          for (auto& node : v->tetrahedra[t]) in >> node;
          in >> v->regions[t];
        }
        if (!in) throw std::runtime_error("Truncated source volume");
        LinearElasticity validate(std::move(*v));
        next.source.volume = std::make_shared<TetMesh>(validate.mesh());
      }
    } else
      next.setMesh(std::move(mesh), title);
    std::vector<std::pair<std::array<unsigned int, 3>, size_t>> faceKeys;
    for (size_t f = 0; f < next.boundaryFaces().size(); ++f) {
      auto vertices = next.boundaryFaces()[f].vertices;
      std::sort(vertices.begin(), vertices.end());
      faceKeys.emplace_back(vertices, f);
    }
    std::sort(faceKeys.begin(), faceKeys.end());
    const auto readFace = [&]() {
      size_t face = 0;
      if (version < 3) {
        in >> face;
        if (!in || face >= next.boundaryFaces().size())
          throw std::runtime_error("Invalid FEM condition face");
      } else {
        std::array<unsigned int, 3> vertices{};
        for (auto& v : vertices) in >> v;
        std::sort(vertices.begin(), vertices.end());
        const auto found =
            std::lower_bound(faceKeys.begin(), faceKeys.end(),
                             std::make_pair(vertices, size_t(0)));
        if (!in || found == faceKeys.end() || found->first != vertices)
          throw std::runtime_error(
              "FEM project references a missing boundary triangle");
        face = found->second;
      }
      return face;
    };
    std::vector<std::shared_ptr<const RegionGeometry>> references;
    if (version >= 4) {
      const auto count = readCount();
      for (size_t i = 0; i < count; ++i) {
        std::string bytes;
        auto geometry = std::make_shared<RegionGeometry>();
        in >> geometry->deviation >> std::quoted(bytes);
        if (!in || geometry->deviation < 0. ||
            !std::isfinite(geometry->deviation))
          throw std::runtime_error("Invalid reference deviation");
        geometry->surface = SurfaceGeometry::decode(bytes, 1., false);
        indexRegion(geometry);
        references.push_back(geometry);
      }
    }
    const auto count = readCount();
    next.conditions.resize(count);
    for (auto& c : next.conditions) {
      int kind = -1;
      in >> std::quoted(c.name) >> kind >> c.enabled;
      if (kind < 0 ||
          kind > static_cast<int>(version == 1 ? Kind::Gravity : Kind::Tie))
        throw std::runtime_error("Invalid FEM condition kind");
      c.kind = static_cast<Kind>(kind);
      for (int a = 0; a < 3; ++a) in >> c.axes[a] >> c.value[a];
      const auto nodeCount = readCount(), faceCount = readCount();
      c.nodes.resize(nodeCount);
      c.faces.resize(faceCount);
      for (auto& v : c.nodes) {
        in >> v;
        if (v >= nodes) throw std::runtime_error("Invalid FEM condition node");
      }
      for (auto& f : c.faces) f = readFace();
      if (!in || !c.value.e().allFinite())
        throw std::runtime_error("Truncated or nonfinite FEM condition");
      if (version >= 2) {
        for (int a = 0; a < 3; ++a)
          in >> c.axis[a] >> c.center[a] >> c.rotation[a];
        in >> c.stiffness >> c.tolerance >> c.bolt.diameter >>
            c.bolt.youngsModulus >> c.bolt.poissonRatio >> c.bolt.gripLength >>
            c.bolt.preload;
        const auto targets = readCount(), faces = readCount();
        c.targetNodes.resize(targets);
        c.targetFaces.resize(faces);
        for (auto& node : c.targetNodes) {
          in >> node;
          if (node >= nodes)
            throw std::runtime_error("Invalid fixture target node");
        }
        for (auto& face : c.targetFaces) face = readFace();
        if (!in || !c.axis.e().allFinite() || !c.center.e().allFinite() ||
            !c.rotation.e().allFinite())
          throw std::runtime_error("Truncated or invalid FEM fixture");
      }
      if (version >= 4) {
        c.primaryGeometry =
            readAnchor(in, c.nodes, c.faces, size_t(budget), references);
        c.targetGeometry = readAnchor(in, c.targetNodes, c.targetFaces,
                                      size_t(budget), references);
      }
    }
    if (next.solver)
      next.solver->setMaterial(next.material);
    else {
      SurfaceGeometry validate = *next.surface_;
      validate.normalize();
    }
    if (version < 4) {
      double squared = 0.;
      auto low = next.mesh().vertices[0].e().eval(), high = low;
      for (const auto& p : next.mesh().vertices) {
        low = low.cwiseMin(p.e());
        high = high.cwiseMax(p.e());
      }
      for (const auto& cell : next.mesh().tetrahedra)
        for (size_t a = 0; a < 4; ++a)
          for (size_t b = 0; b < a; ++b)
            squared = std::max(squared, (next.mesh().vertices[cell[a]].e() -
                                         next.mesh().vertices[cell[b]].e())
                                            .squaredNorm());
      next.source.maximumEdge = next.source.meshedEdge = std::sqrt(squared);
      next.source.referenceExtent = (high - low).norm();
      next.source.meshedTolerance = 0.;
      next.source.volume = std::make_shared<TetMesh>(next.mesh());
    }
    if (!allowExternalMesher) next.source.externalGmsh = false;
    *this = std::move(next);
  }

 private:
  std::shared_ptr<const SelectionAnchor> captureSelection(
      const std::vector<unsigned int>& nodes, const std::vector<size_t>& faces,
      const std::shared_ptr<const SelectionAnchor>& previous) const {
    if (previous && previous->nodes == nodes && previous->faces == faces)
      return previous;
    if (nodes.empty() && faces.empty()) return {};
    if (!referenceGeometry_) {
      auto reference = std::make_shared<RegionGeometry>();
      reference->deviation = source.volume ? 0.
                             : solver      ? source.meshedTolerance
                                           : source.previewTolerance;
      reference->surface.mesh.vertices = mesh().vertices;
      reference->surface.faces = boundaryFaces();
      reference->surface.parts = faceParts();
      indexRegion(reference);
      referenceGeometry_ = std::move(reference);
    }
    return captureRegion(mesh(), boundaryFaces(), faceParts(), nodes, faces,
                         previous, referenceGeometry_);
  }
  mutable std::shared_ptr<const RegionGeometry> referenceGeometry_;
  std::shared_ptr<const SurfaceGeometry> surface_;
  std::vector<int> boundaryParts_;
  FixtureCache fixtureCache_;
  std::unique_ptr<SurfaceSelector> selector_;
  std::vector<unsigned char> selectedSurfaces_;
  bool surfaceSelection_ = false;
  std::vector<unsigned char> previousMask_;
};
}  // namespace fem_workbench
