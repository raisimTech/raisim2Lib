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

  void setMesh(TetMesh mesh, std::string title) {
    auto next = std::make_unique<LinearElasticity>(std::move(mesh));
    auto selector = std::make_unique<SurfaceSelector>(*next);
    solver = std::move(next);
    selector_ = std::move(selector);
    selectedSurfaces_.assign(solver->boundaryFaces().size(), 0);
    surfaceSelection_ = false;
    name = std::move(title);
    conditions.clear();
    selected.assign(solver->mesh().vertices.size(), 0);
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
      for (const auto node : solver->boundaryFaces()[face].vertices)
        selected[node] = 1;
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
      if (std::abs(solver->mesh().vertices[i][axis] - coordinate) <= tolerance)
        selected[i] = 1;
  }
  std::vector<size_t> selectedFaces() const {
    std::vector<size_t> faces;
    if (surfaceSelection_) {
      for (size_t f = 0; f < selectedSurfaces_.size(); ++f)
        if (selectedSurfaces_[f]) faces.push_back(f);
      return faces;
    }
    for (size_t f = 0; f < solver->boundaryFaces().size(); ++f) {
      const auto& ids = solver->boundaryFaces()[f].vertices;
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
    for (const auto& face : solver->boundaryFaces())
      for (const auto node : face.vertices) surface[node] = 1;
    for (size_t i = 0; i < selected.size(); ++i)
      if (surface[i]) {
        const Vec<3> r = solver->mesh().vertices[i] - center;
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
      for (const auto node : c.nodes) c.center += solver->mesh().vertices[node];
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
            const auto& ids = solver->boundaryFaces().at(f).vertices;
            const auto& x = solver->mesh().vertices;
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
    if (!solver) throw std::runtime_error("No model to save");
    std::ofstream out(raisim::pathFromUtf8(path));
    if (!out) throw std::runtime_error("Cannot write project: " + path);
    out.imbue(std::locale::classic());
    out.precision(17);
    out << "RAISIM_FEM_PROJECT 3\n"
        << std::quoted(name) << '\n'
        << material.youngsModulus << ' ' << material.poissonRatio << ' '
        << material.density << '\n';
    const auto& mesh = solver->mesh();
    out << mesh.vertices.size() << ' ' << mesh.tetrahedra.size() << '\n';
    for (const auto& x : mesh.vertices)
      out << x[0] << ' ' << x[1] << ' ' << x[2] << '\n';
    for (size_t i = 0; i < mesh.tetrahedra.size(); ++i) {
      for (const auto v : mesh.tetrahedra[i]) out << v << ' ';
      out << (mesh.regions.empty() ? 0 : mesh.regions[i]) << '\n';
    }
    out << conditions.size() << '\n';
    for (const auto& c : conditions) {
      out << std::quoted(c.name) << ' ' << static_cast<int>(c.kind) << ' '
          << c.enabled;
      for (int a = 0; a < 3; ++a) out << ' ' << c.axes[a] << ' ' << c.value[a];
      out << ' ' << c.nodes.size() << ' ' << c.faces.size() << '\n';
      for (const auto v : c.nodes) out << v << ' ';
      out << '\n';
      for (const auto f : c.faces)
        for (const auto v : solver->boundaryFaces().at(f).vertices)
          out << v << ' ';
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
        for (const auto v : solver->boundaryFaces().at(f).vertices)
          out << v << ' ';
      out << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("Failed writing FEM project");
  }
  void loadProject(const std::string& path) {
    std::ifstream in(raisim::pathFromUtf8(path));
    in.imbue(std::locale::classic());
    std::string magic, title;
    int version = 0;
    in >> magic >> version >> std::quoted(title);
    if (magic != "RAISIM_FEM_PROJECT" || (version < 1 || version > 3))
      throw std::runtime_error("Unsupported FEM project");
    Model next;
    in >> next.material.youngsModulus >> next.material.poissonRatio >>
        next.material.density;
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
    next.setMesh(std::move(mesh), title);
    std::vector<std::pair<std::array<unsigned int, 3>, size_t>> faceKeys;
    for (size_t f = 0; f < next.solver->boundaryFaces().size(); ++f) {
      auto vertices = next.solver->boundaryFaces()[f].vertices;
      std::sort(vertices.begin(), vertices.end());
      faceKeys.emplace_back(vertices, f);
    }
    std::sort(faceKeys.begin(), faceKeys.end());
    const auto readFace = [&]() {
      size_t face = 0;
      if (version < 3) {
        in >> face;
        if (!in || face >= next.solver->boundaryFaces().size())
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
    }
    next.solver->setMaterial(next.material);
    *this = std::move(next);
  }

 private:
  FixtureCache fixtureCache_;
  std::unique_ptr<SurfaceSelector> selector_;
  std::vector<unsigned char> selectedSurfaces_;
  bool surfaceSelection_ = false;
  std::vector<unsigned char> previousMask_;
};
}  // namespace fem_workbench
