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
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "fem_surface_selection.hpp"
#include "fem_workbench_fixtures.hpp"
#include "fem_workbench_positioning.hpp"
#include "fem_workbench_transfer.hpp"
#include "raisim/PathUtils.hpp"
#include "raisim/fem/ErrorEstimation.hpp"
#include "raisim/fem/LinearElasticity.hpp"
#include "raisim/fem/MeshResolution.hpp"

namespace fem_workbench {
using raisim::Vec;
using raisim::fem::LinearElasticity;
using raisim::fem::TetMesh;

/// Volume-weighted average of element values at the mesh nodes. NaN values
/// (unknown) are skipped; nodes touched only by infinite values stay infinite
/// and nodes with no known value are NaN.
inline std::vector<double> nodalAverage(const TetMesh& mesh,
                                        const std::vector<double>& values) {
  const auto& x = mesh.vertices;
  std::vector<double> sum(x.size(), 0.), weight(x.size(), 0.);
  std::vector<char> infinite(x.size(), 0);
  for (size_t t = 0; t < mesh.tetrahedra.size(); ++t) {
    const auto& c = mesh.tetrahedra[t];
    if (values[t] == std::numeric_limits<double>::infinity()) {
      for (const auto v : c) infinite[v] = 1;
      continue;
    }
    if (!std::isfinite(values[t])) continue;
    const double volume =
        std::abs((x[c[1]].e() - x[c[0]].e())
                     .dot((x[c[2]].e() - x[c[0]].e())
                              .cross(x[c[3]].e() - x[c[0]].e())));
    for (const auto v : c) {
      sum[v] += volume * values[t];
      weight[v] += volume;
    }
  }
  for (size_t v = 0; v < sum.size(); ++v)
    sum[v] = weight[v] > 0. ? sum[v] / weight[v]
             : infinite[v]  ? std::numeric_limits<double>::infinity()
                            : std::numeric_limits<double>::quiet_NaN();
  return sum;
}

/// Shortens text with "..." until measure(text) fits the width, removing
/// whole UTF-8 code points.
template <class Measure>
std::string fitText(const std::string& text, float width, Measure measure) {
  if (measure(text) <= width) return text;
  std::string fitted = text;
  while (!fitted.empty() && measure(fitted + "...") > width) {
    while (!fitted.empty() &&
           (static_cast<unsigned char>(fitted.back()) & 0xC0) == 0x80)
      fitted.pop_back();
    if (!fitted.empty()) fitted.pop_back();
  }
  return fitted + "...";
}

class Model {
 public:
  using Progress = std::function<void(LinearElasticity::SolveStage, double)>;
  struct Adaptivity {
    double relativeError = .1;
    int maximumSteps = 3;
    double minimumEdge = 0.;
    size_t maximumCells = 400000;
  };
  struct AdaptiveResult {
    int refinements = 0;
    double estimatedError = 0.;
    bool converged = false;
  };
  /// A solid of the model: one mesh region. Without its own material it uses
  /// the model material.
  struct Part {
    int region = 0;
    std::string name;
    std::optional<LinearElasticity::Material> material;
    /// Yield strength (Pa) of the part's own material; zero is unknown.
    double yieldStrength = 0.;
    /// Placement of the part's geometry relative to its CAD file: the rotation
    /// about the CAD origin, then the offset.
    Vec<3> offset{0., 0., 0.};
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    /// A fixed part is never moved: mates move the other part and it cannot
    /// be dragged.
    bool fixed = false;
  };
  std::unique_ptr<LinearElasticity> solver;
  LinearElasticity::Material material{70e9, .3, 2700.};
  /// Yield strength (Pa) of the model material; zero is unknown.
  double yieldStrength = 276e6;
  /// One entry per region, sorted by region.
  std::vector<Part> parts;
  std::vector<Condition> conditions;
  /// Assembly mates that place parts against each other; see addMate().
  std::vector<PositionMate> mates;
  std::vector<unsigned char> selected;
  std::string name = "Cantilever demonstration";
  bool dirty = true, hasSolution = false;
  /// Relative energy-norm error estimated after the last solve, or -1.
  double estimatedError = -1.;
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
        surface.parts.size() != surface.faces.size() ||
        (!surface.brepFaces.empty() &&
         surface.brepFaces.size() != surface.faces.size()))
      throw std::runtime_error("Surface contains no valid geometry");
    auto next = std::make_shared<SurfaceGeometry>(std::move(surface));
    auto selector = std::make_unique<SurfaceSelector>(next->mesh, next->faces);
    auto surfaces = next->brepFaces.empty() ? selector->partition(next->parts)
                                            : next->brepFaces;
    referenceGeometry_.reset();
    solver.reset();
    brepReference_ = next->brepFaces.empty() ? nullptr : next;
    surface_ = std::move(next);
    selector_ = std::move(selector);
    faceSurfaces_ = std::move(surfaces);
    source = {};
    name = std::move(title);
    conditions.clear();
    boltConditions.clear();
    selected.assign(mesh().vertices.size(), 0);
    selectedSurfaces_.assign(boundaryFaces().size(), 0);
    previousMask_.assign(3 * selected.size(), 0);
    surfaceSelection_ = false;
    fixtureCache_.clear();
    refinementLevel_ = 0;
    estimatedError = -1.;
    dirty = true;
    hasSolution = false;
    mates.clear();
    syncParts({});
  }
  void setMeshed(TetMesh volume, double tolerance,
                 double destinationDeviation = 0.) {
    if (!(tolerance > 0.) || !std::isfinite(tolerance) ||
        destinationDeviation < 0. || !std::isfinite(destinationDeviation))
      throw std::runtime_error("Invalid fixture mapping tolerance");
    Model next;
    next.material = material;
    next.yieldStrength = yieldStrength;
    next.setMesh(std::move(volume), name);
    if (brepReference_) next.transferCadFaces(brepReference_);
    next.source = source;
    next.source.meshedTolerance = destinationDeviation;
    transferConditions(next, tolerance, destinationDeviation);
    next.mates = mates;
    next.syncParts(parts);
    *this = std::move(next);
  }
  /// Appends the CAD surface of another file as new parts, renumbering its
  /// regions and CAD faces after the existing ones. The model returns to the
  /// unmeshed surface; supports and loads stay on their faces. Returns the
  /// region offset of the new parts.
  int addSurfacePart(SurfaceGeometry addition, const std::string& partName,
                     double tolerance) {
    if (!hasGeometry()) throw std::runtime_error("Open a CAD solid first");
    if (addition.mesh.vertices.empty() || addition.faces.empty() ||
        addition.parts.size() != addition.faces.size() ||
        (!addition.brepFaces.empty() &&
         addition.brepFaces.size() != addition.faces.size()))
      throw std::runtime_error("Added surface contains no valid geometry");
    if (!(tolerance > 0.) || !std::isfinite(tolerance))
      throw std::runtime_error("Invalid fixture mapping tolerance");
    SurfaceGeometry merged;
    if (brepReference_)
      merged = *brepReference_;
    else if (surface_)
      merged = *surface_;
    else {
      merged.mesh.vertices = mesh().vertices;
      merged.faces = boundaryFaces();
      merged.parts = faceParts();
    }
    int regionOffset =
        *std::max_element(merged.parts.begin(), merged.parts.end());
    for (const auto region : source.excludedRegions)
      regionOffset = std::max(regionOffset, region);
    const int faceOffset =
        merged.brepFaces.empty()
            ? 0
            : *std::max_element(merged.brepFaces.begin(),
                                merged.brepFaces.end());
    const auto vertexOffset =
        static_cast<unsigned int>(merged.mesh.vertices.size());
    const bool cad = !merged.brepFaces.empty() && !addition.brepFaces.empty();
    if (!cad) merged.brepFaces.clear();
    for (const auto& v : addition.mesh.vertices)
      merged.mesh.vertices.push_back(v);
    for (size_t f = 0; f < addition.faces.size(); ++f) {
      auto face = addition.faces[f];
      for (auto& v : face.vertices) v += vertexOffset;
      merged.faces.push_back(face);
      merged.parts.push_back(addition.parts[f] + regionOffset);
      if (cad) merged.brepFaces.push_back(addition.brepFaces[f] + faceOffset);
    }
    Model next;
    next.material = material;
    next.yieldStrength = yieldStrength;
    next.setSurface(std::move(merged), name);
    next.source = source;
    transferConditions(next, tolerance, source.previewTolerance);
    next.mates = mates;
    next.syncParts(parts);
    std::vector<int> added;
    for (const auto& part : next.parts)
      if (part.region > regionOffset) added.push_back(part.region);
    for (auto& part : next.parts)
      if (part.region > regionOffset)
        part.name = added.size() == 1
                        ? partName
                        : partName + " / solid " +
                              std::to_string(part.region - regionOffset);
    *this = std::move(next);
    return regionOffset;
  }
  Part* findPart(int region) {
    for (auto& part : parts)
      if (part.region == region) return &part;
    return nullptr;
  }
  const Part* findPart(int region) const {
    return const_cast<Model*>(this)->findPart(region);
  }
  const LinearElasticity::Material& materialOf(const Part& part) const {
    return part.material ? *part.material : material;
  }
  double yieldOf(const Part& part) const {
    return part.material ? part.yieldStrength : yieldStrength;
  }
  /// Yield strength of each tetrahedron's material, zero where unknown.
  std::vector<double> elementYield() const {
    const auto& regions = mesh().regions;
    std::vector<double> result(mesh().tetrahedra.size(), yieldStrength);
    for (size_t t = 0; t < result.size(); ++t)
      if (const auto* part = findPart(regions.empty() ? 0 : regions[t]))
        result[t] = yieldOf(*part);
    return result;
  }
  /// The model material when no part has its own, otherwise one per
  /// tetrahedron.
  std::vector<LinearElasticity::Material> elementMaterials() const {
    if (std::none_of(parts.begin(), parts.end(),
                     [](const Part& p) { return p.material.has_value(); }))
      return {material};
    const auto& regions = mesh().regions;
    std::vector<LinearElasticity::Material> result(mesh().tetrahedra.size(),
                                                   material);
    for (size_t t = 0; t < result.size(); ++t)
      if (const auto* part = findPart(regions.empty() ? 0 : regions[t]))
        result[t] = materialOf(*part);
    return result;
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
    faceSurfaces_ = selector_->partition(boundaryParts_);
    brepReference_.reset();
    selectedSurfaces_.assign(boundaryFaces().size(), 0);
    surfaceSelection_ = false;
    name = std::move(title);
    conditions.clear();
    selected.assign(mesh().vertices.size(), 0);
    previousMask_.assign(3 * selected.size(), 0);
    fixtureCache_.clear();
    boltConditions.clear();
    refinementLevel_ = 0;
    estimatedError = -1.;
    dirty = true;
    hasSolution = false;
    mates.clear();
    syncParts({});
  }
  void demo() {
    setMesh(TetMesh::box({1., .16, .24}, {32, 6, 8}), "Aluminium cantilever");
    parts.front().name = "Cantilever";
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
  const std::vector<int>& faceSurfaces() const { return faceSurfaces_; }
  std::vector<int> wholeCadFaces(const std::vector<unsigned int>& nodes,
                                 const std::vector<size_t>& faces) const {
    if (!hasCadFaces() || faces.empty()) return {};
    std::vector<int> surfaces;
    for (const auto f : faces) surfaces.push_back(faceSurfaces_.at(f));
    std::sort(surfaces.begin(), surfaces.end());
    surfaces.erase(std::unique(surfaces.begin(), surfaces.end()),
                   surfaces.end());
    std::vector<size_t> expectedFaces;
    std::vector<unsigned int> expectedNodes;
    for (size_t f = 0; f < faceSurfaces_.size(); ++f)
      if (std::binary_search(surfaces.begin(), surfaces.end(),
                             faceSurfaces_[f])) {
        expectedFaces.push_back(f);
        for (const auto v : boundaryFaces()[f].vertices)
          expectedNodes.push_back(v);
      }
    std::sort(expectedNodes.begin(), expectedNodes.end());
    expectedNodes.erase(
        std::unique(expectedNodes.begin(), expectedNodes.end()),
        expectedNodes.end());
    auto sortedFaces = faces;
    auto sortedNodes = nodes;
    std::sort(sortedFaces.begin(), sortedFaces.end());
    std::sort(sortedNodes.begin(), sortedNodes.end());
    if (sortedFaces != expectedFaces || sortedNodes != expectedNodes) return {};
    return surfaces;
  }
  bool hasCadFaces() const { return bool(brepReference_); }
  std::vector<size_t> surfaceFaces(size_t seed) const {
    const int surface = faceSurfaces_.at(seed), part = faceParts().at(seed);
    std::vector<size_t> faces;
    for (size_t f = 0; f < faceSurfaces_.size(); ++f)
      if (faceSurfaces_[f] == surface && faceParts()[f] == part)
        faces.push_back(f);
    return faces;
  }
  bool faceSelected(size_t face) const {
    if (surfaceSelection_) return selectedSurfaces_.at(face) != 0;
    const auto& ids = boundaryFaces().at(face).vertices;
    return selected[ids[0]] && selected[ids[1]] && selected[ids[2]];
  }
  void selectSurface(size_t seed, SurfaceSelection mode, bool append) {
    selectFaces(selector_->collect(seed, mode), append);
  }
  void selectFaces(const std::vector<size_t>& faces, bool append) {
    for (const auto face : faces) (void)boundaryFaces().at(face);
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
  void deselectFaces(const std::vector<size_t>& faces) {
    for (const auto face : faces) (void)boundaryFaces().at(face);
    if (!surfaceSelection_)
      for (const auto face : selectedFaces()) selectedSurfaces_[face] = 1;
    surfaceSelection_ = true;
    for (const auto face : faces) selectedSurfaces_[face] = 0;
    std::vector<unsigned char> kept(selected.size(), 0);
    for (size_t f = 0; f < selectedSurfaces_.size(); ++f)
      if (selectedSurfaces_[f])
        for (const auto node : boundaryFaces()[f].vertices) kept[node] = 1;
    for (const auto face : faces)
      for (const auto node : boundaryFaces()[face].vertices)
        if (!kept[node]) selected[node] = 0;
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
    c.cadFaces = wholeCadFaces(c.nodes, c.faces);
    if (kind != Kind::Gravity && c.nodes.empty())
      throw std::runtime_error("Select a surface or nodes first");
    if ((kind == Kind::Traction || kind == Kind::Pressure) && c.faces.empty())
      throw std::runtime_error("This load needs selected boundary faces");
    if (kind == Kind::Gravity) {
      c.value = {0., 0., -9.81};
      c.nodes.clear();
      c.faces.clear();
      c.cadFaces.clear();
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
  /// Joins two face sets, normally on different parts, with a glue, bolt or
  /// joint. Joint points and bearing and slider axes are fitted to the faces
  /// when they share one; connectionAxisFitted() tells whether the axis was.
  Condition& addConnection(Kind kind, std::vector<size_t> firstFaces,
                           std::vector<size_t> secondFaces,
                           double tieTolerance) {
    if (!isConnection(kind))
      throw std::invalid_argument("Not a connection kind");
    if (firstFaces.empty() || secondFaces.empty())
      throw std::runtime_error("Pick faces on both parts");
    const auto nodesOf = [&](std::vector<size_t>& faces) {
      std::sort(faces.begin(), faces.end());
      faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
      std::vector<unsigned int> nodes;
      for (const auto f : faces)
        for (const auto v : boundaryFaces().at(f).vertices) nodes.push_back(v);
      std::sort(nodes.begin(), nodes.end());
      nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
      return nodes;
    };
    auto nodesA = nodesOf(firstFaces), nodesB = nodesOf(secondFaces);
    Condition c;
    c.kind = kind;
    c.name = std::string(kindName(kind)) + " " +
             std::to_string(conditions.size() + 1);
    c.cadFaces = wholeCadFaces(nodesA, firstFaces);
    c.targetCadFaces = wholeCadFaces(nodesB, secondFaces);
    Vec<3> centerA{0., 0., 0.}, centerB{0., 0., 0.};
    for (const auto node : nodesA) centerA += mesh().vertices[node];
    for (const auto node : nodesB) centerB += mesh().vertices[node];
    c.center = (centerA * (1. / double(nodesA.size())) +
                centerB * (1. / double(nodesB.size()))) *
               .5;
    auto both = firstFaces;
    both.insert(both.end(), secondFaces.begin(), secondFaces.end());
    if (kind == Kind::Hinge || kind == Kind::BallJoint)
      fitJointPoint(mesh(), boundaryFaces(), both, c.center);
    if (kind == Kind::Hinge || kind == Kind::SliderJoint)
      fitFaceAxis(mesh(), boundaryFaces(), both, c.axis);
    c.nodes = std::move(nodesA);
    c.faces = std::move(firstFaces);
    c.targetNodes = std::move(nodesB);
    c.targetFaces = std::move(secondFaces);
    if (kind == Kind::Tie) c.tolerance = tieTolerance;
    conditions.push_back(std::move(c));
    changed();
    return conditions.back();
  }
  /// What a connection carries after a solve. Glue and joints transmit the
  /// reactions of their constraints at the nodes of their first faces; a bolt
  /// reports its shank. Normal stress is positive when it pulls the faces
  /// apart (peel).
  ///
  /// The nominal stresses spread the transmitted force and moment over the
  /// faces as the elastic method for bonded and welded joints does: a traction
  /// F / A + b x r, with b from the moment and the area inertia of the faces.
  /// They capture peel from bending and shear from torsion and do not depend
  /// on the mesh, whereas the node stresses at the edges of a bonded joint grow
  /// as the mesh is refined. Glue compares the equivalent shear
  /// sqrt(shear^2 + normal^2 / 3) with its lap shear strength; a bolt compares
  /// the von Mises stress at its shank surface (axial and bending stress, with
  /// transverse and torsional shear) with its yield strength.
  struct ConnectionLoad {
    /// Force on the first part and moment about @p center.
    Eigen::Vector3d force = Eigen::Vector3d::Zero(),
                    moment = Eigen::Vector3d::Zero(),
                    center = Eigen::Vector3d::Zero();
    /// Glue and joints: area of the first faces. Bolt: shank cross-section.
    double area = 0.;
    /// Force per area along and across the mean normal (bolt: axis).
    double averageNormal = 0., averageShear = 0.;
    /// Largest nominal stresses (bolt: at the shank surface).
    double peel = 0., compression = 0., shear = 0., equivalent = 0.;
    /// Node where the nominal equivalent stress peaks (bolt: the head node
    /// nearest the shank).
    unsigned int peakNode = 0;
    /// Largest stresses at single nodes, from the constraint reactions.
    double nodePeel = 0., nodeCompression = 0., nodeShear = 0.,
           nodeEquivalent = 0.;
    /// Some nodes also carry a support or another connection, so their
    /// reactions mix both.
    bool sharedNodes = false;
    /// Strength divided by the nominal equivalent stress; infinite when either
    /// is unknown or zero.
    double safety = std::numeric_limits<double>::infinity();
  };
  ConnectionLoad connectionLoad(size_t index) const {
    if (!hasSolution || !solver)
      throw std::runtime_error("Solve the model first");
    const auto& c = conditions.at(index);
    if (!isConnection(c.kind))
      throw std::invalid_argument("Not a connection");
    ConnectionLoad load;
    const auto& x = mesh().vertices;
    if (c.kind == Kind::BoltConnector) {
      const auto found =
          std::find(boltConditions.begin(), boltConditions.end(), index);
      if (found == boltConditions.end())
        throw std::runtime_error("This bolt was not part of the last solve");
      const auto& r =
          solver->boltResults().at(size_t(found - boltConditions.begin()));
      const Eigen::Vector3d axis = r.axis.e().normalized();
      const double d = c.bolt.diameter, pi = std::acos(-1.);
      load.force = r.force.e();
      load.moment = r.headMoment.e();
      load.area = .25 * pi * d * d;
      const double polar = pi * d * d * d;
      const double axial = axis.dot(load.force) / load.area,
                   torsion = 16. *
                             std::max(std::abs(axis.dot(r.headMoment.e())),
                                      std::abs(axis.dot(r.nutMoment.e()))) /
                             polar,
                   transverse =
                       (load.force - axis * axis.dot(load.force)).norm() /
                       load.area,
                   shear = transverse + torsion;
      const auto bending = [&](const Eigen::Vector3d& m) {
        return 32. * (m - axis * axis.dot(m)).norm() / polar;
      };
      const double bend =
          std::max(bending(r.headMoment.e()), bending(r.nutMoment.e()));
      load.averageNormal = axial;
      load.averageShear = transverse;
      load.shear = load.nodeShear = shear;
      load.peel = load.nodePeel = std::max(0., axial) + bend;
      load.compression = load.nodeCompression = std::min(0., axial) - bend;
      load.equivalent = load.nodeEquivalent =
          std::sqrt(std::pow(std::abs(axial) + bend, 2) + 3. * shear * shear);
      Eigen::Vector3d center = Eigen::Vector3d::Zero();
      for (const auto v : c.nodes) center += x[v].e();
      for (const auto v : c.targetNodes) center += x[v].e();
      load.center = center / double(std::max<size_t>(
                                 1, c.nodes.size() + c.targetNodes.size()));
      double nearest = std::numeric_limits<double>::infinity();
      for (const auto v : c.nodes)
        if ((x[v].e() - load.center).norm() < nearest) {
          nearest = (x[v].e() - load.center).norm();
          load.peakNode = v;
        }
      if (c.strength > 0. && load.equivalent > 0.)
        load.safety = c.strength / load.equivalent;
      return load;
    }
    const auto& reactions = solver->reactions();
    std::vector<double> nodeArea(x.size(), 0.);
    std::vector<Eigen::Vector3d> nodeNormal(x.size(), Eigen::Vector3d::Zero());
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    for (const auto f : c.faces) {
      const Eigen::Vector3d n = outwardNormal(f);
      const double area = .5 * n.norm();
      normal += .5 * n;
      load.area += area;
      for (const auto v : boundaryFaces().at(f).vertices) {
        nodeArea[v] += area / 3.;
        nodeNormal[v] += .5 * n;
        load.center += area / 3. * x[v].e();
      }
    }
    if (!(load.area > 0.))
      throw std::runtime_error("This connection has no faces with area");
    load.center /= load.area;
    std::vector<unsigned char> elsewhere(x.size(), 0);
    for (size_t i = 0; i < conditions.size(); ++i) {
      const auto& other = conditions[i];
      if (i == index || !other.enabled || other.kind == Kind::Force ||
          other.kind == Kind::Traction || other.kind == Kind::Pressure ||
          other.kind == Kind::Gravity)
        continue;
      for (const auto* nodes : {&other.nodes, &other.targetNodes})
        for (const auto v : *nodes) elsewhere.at(v) = 1;
    }
    normal.normalize();
    const bool glue = c.kind == Kind::Tie;
    const auto equivalentOf = [&](double sigma, double tau) {
      return glue ? std::sqrt(tau * tau + sigma * sigma / 3.)
                  : std::sqrt(sigma * sigma + 3. * tau * tau);
    };
    Eigen::Matrix3d second = Eigen::Matrix3d::Zero();
    for (const auto f : c.faces) {
      Eigen::Matrix3d corners = Eigen::Matrix3d::Zero();
      Eigen::Vector3d sum = Eigen::Vector3d::Zero();
      for (const auto v : boundaryFaces()[f].vertices) {
        const Eigen::Vector3d arm = x[v].e() - load.center;
        corners += arm * arm.transpose();
        sum += arm;
      }
      second += .5 * outwardNormal(f).norm() / 12. *
                (corners + sum * sum.transpose());
    }
    const Eigen::Matrix3d inertia =
        second.trace() * Eigen::Matrix3d::Identity() - second;
    for (const auto v : c.nodes) {
      if (!(nodeArea.at(v) > 0.)) continue;
      const Eigen::Vector3d r = reactions.at(v).e(),
                            arm = x[v].e() - load.center;
      load.force += r;
      load.moment += arm.cross(r);
      load.sharedNodes = load.sharedNodes || elsewhere[v];
      const Eigen::Vector3d n = nodeNormal[v].normalized();
      const double sigma = r.dot(n) / nodeArea[v],
                   tau = (r - n * r.dot(n)).norm() / nodeArea[v];
      load.nodePeel = std::max(load.nodePeel, sigma);
      load.nodeCompression = std::min(load.nodeCompression, sigma);
      load.nodeShear = std::max(load.nodeShear, tau);
      load.nodeEquivalent = std::max(load.nodeEquivalent, equivalentOf(sigma, tau));
    }
    load.averageNormal = load.force.dot(normal) / load.area;
    load.averageShear =
        (load.force - normal * load.force.dot(normal)).norm() / load.area;
    const Eigen::Vector3d spin =
        placement::smallestSolution<3>(inertia, load.moment);
    double best = -1.;
    for (const auto v : c.nodes) {
      if (!(nodeArea.at(v) > 0.)) continue;
      const Eigen::Vector3d t =
          load.force / load.area + spin.cross(x[v].e() - load.center);
      const Eigen::Vector3d n = nodeNormal[v].normalized();
      const double sigma = t.dot(n), tau = (t - n * sigma).norm(),
                   equivalent = equivalentOf(sigma, tau);
      load.peel = std::max(load.peel, sigma);
      load.compression = std::min(load.compression, sigma);
      load.shear = std::max(load.shear, tau);
      if (equivalent > best) {
        best = equivalent;
        load.equivalent = equivalent;
        load.peakNode = v;
      }
    }
    if (glue && c.strength > 0. && load.equivalent > 0.)
      load.safety = c.strength / load.equivalent;
    return load;
  }
  /// Whether the faces of a bearing or slider share one axis.
  bool connectionAxisFitted(const Condition& c) const {
    auto both = c.faces;
    both.insert(both.end(), c.targetFaces.begin(), c.targetFaces.end());
    Vec<3> axis{0., 0., 1.};
    return fitFaceAxis(mesh(), boundaryFaces(), both, axis);
  }
  /// Where parts touch, like the contact sets SolidWorks finds: each entry
  /// pairs whole CAD faces of one part whose nodes all lie within
  /// @p tolerance of another part with the faces of that part they lie on.
  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>
  touchingFaces(double tolerance) const {
    if (!(tolerance > 0.) || !std::isfinite(tolerance))
      throw std::runtime_error("Invalid contact tolerance");
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> result;
    const auto facesOf = [&](int region) {
      std::vector<size_t> faces;
      const auto& x = mesh().vertices;
      for (size_t f = 0; f < boundaryFaces().size(); ++f) {
        const auto& v = boundaryFaces()[f].vertices;
        if (faceParts()[f] == region &&
            (x[v[1]].e() - x[v[0]].e())
                    .cross(x[v[2]].e() - x[v[0]].e())
                    .squaredNorm() > 0.)
          faces.push_back(f);
      }
      return faces;
    };
    for (size_t i = 0; i < parts.size(); ++i)
      for (size_t j = i + 1; j < parts.size(); ++j) {
        std::vector<int> claimed;
        for (const auto& [from, to] :
             {std::pair{parts[i].region, parts[j].region},
              std::pair{parts[j].region, parts[i].region}}) {
          const auto targets = facesOf(to);
          if (targets.empty()) continue;
          const SurfaceProjection projection(mesh(), boundaryFaces(), targets);
          std::vector<size_t> first;
          std::vector<int> hit;
          for (const auto& group : cadFaceGroups(facesOf(from))) {
            if (std::find(claimed.begin(), claimed.end(),
                          faceSurfaces_[group.front()]) != claimed.end())
              continue;
            std::vector<int> touched;
            bool on = true;
            for (const auto f : group) {
              for (const auto v : boundaryFaces()[f].vertices)
                on = on && projection.project(v).distance <=
                               tolerance * tolerance;
              if (!on) break;
              Eigen::Vector3d center = Eigen::Vector3d::Zero();
              for (const auto v : boundaryFaces()[f].vertices)
                center += mesh().vertices[v].e() / 3.;
              touched.push_back(
                  faceSurfaces_[targets[projection.project(center).order]]);
            }
            if (!on) continue;
            first.insert(first.end(), group.begin(), group.end());
            hit.insert(hit.end(), touched.begin(), touched.end());
          }
          if (first.empty()) continue;
          std::sort(hit.begin(), hit.end());
          hit.erase(std::unique(hit.begin(), hit.end()), hit.end());
          std::vector<size_t> second;
          for (const auto f : targets)
            if (std::binary_search(hit.begin(), hit.end(), faceSurfaces_[f]))
              second.push_back(f);
          claimed.insert(claimed.end(), hit.begin(), hit.end());
          result.emplace_back(std::move(first), std::move(second));
        }
      }
    return result;
  }
  /// Glues every pair of touching faces that no glue joins yet and returns
  /// how many glues it added.
  size_t bondTouchingFaces(double tolerance) {
    size_t added = 0;
    for (auto& [first, second] : touchingFaces(tolerance)) {
      const bool tied = std::any_of(
          conditions.begin(), conditions.end(), [&](const Condition& c) {
            return c.kind == Kind::Tie &&
                   ((c.faces == first && c.targetFaces == second) ||
                    (c.faces == second && c.targetFaces == first));
          });
      if (tied) continue;
      addConnection(Kind::Tie, first, second, tolerance);
      ++added;
    }
    return added;
  }
  /// Removes a part with the supports, loads and connections that touch it and
  /// returns how many conditions went with it. A meshed model keeps the mesh of
  /// the other parts. A CAD file without parts left is dropped.
  size_t removePart(int region, double tolerance) {
    if (!findPart(region)) throw std::runtime_error("Unknown part");
    if (parts.size() < 2)
      throw std::runtime_error("Use Clear to remove the last part");
    if (!(tolerance > 0.) || !std::isfinite(tolerance))
      throw std::runtime_error("Invalid fixture mapping tolerance");
    std::vector<unsigned char> inside(mesh().vertices.size(), 0);
    for (size_t f = 0; f < boundaryFaces().size(); ++f)
      if (faceParts()[f] == region)
        for (const auto v : boundaryFaces()[f].vertices) inside[v] = 1;
    const auto touches = [&](const std::vector<unsigned int>& nodes,
                             const std::vector<size_t>& faces) {
      return std::any_of(nodes.begin(), nodes.end(),
                         [&](unsigned int v) { return inside.at(v) != 0; }) ||
             std::any_of(faces.begin(), faces.end(), [&](size_t f) {
               return faceParts().at(f) == region;
             });
    };
    std::vector<Condition> kept;
    for (const auto& c : conditions)
      if (!touches(c.nodes, c.faces) && !touches(c.targetNodes, c.targetFaces))
        kept.push_back(c);
    const size_t removed = conditions.size() - kept.size();
    SurfaceGeometry reference;
    if (brepReference_)
      reference = *brepReference_;
    else if (surface_)
      reference = *surface_;
    else {
      reference.mesh.vertices = mesh().vertices;
      reference.faces = boundaryFaces();
      reference.parts = faceParts();
    }
    SurfaceGeometry filtered;
    std::vector<unsigned int> index(reference.mesh.vertices.size(), ~0u);
    for (size_t f = 0; f < reference.faces.size(); ++f) {
      if (reference.parts[f] == region) continue;
      auto face = reference.faces[f];
      for (auto& v : face.vertices) {
        if (index[v] == ~0u) {
          index[v] = static_cast<unsigned int>(filtered.mesh.vertices.size());
          filtered.mesh.vertices.push_back(reference.mesh.vertices[v]);
        }
        v = index[v];
      }
      filtered.faces.push_back(face);
      filtered.parts.push_back(reference.parts[f]);
      if (!reference.brepFaces.empty())
        filtered.brepFaces.push_back(reference.brepFaces[f]);
    }
    Model next;
    next.material = material;
    next.yieldStrength = yieldStrength;
    if (solver) {
      TetMesh volume = mesh();
      removeRegions(volume, {region});
      next.setMesh(std::move(volume), name);
      if (brepReference_)
        next.transferCadFaces(
            std::make_shared<const SurfaceGeometry>(std::move(filtered)));
    } else {
      next.setSurface(std::move(filtered), name);
    }
    next.source = source;
    if (source.volume) {
      auto volume = *source.volume;
      removeRegions(volume, {region});
      next.source.volume = std::make_shared<TetMesh>(std::move(volume));
    }
    transferConditions(next, tolerance, source.meshedTolerance, kept);
    next.refinementLevel_ = solver ? refinementLevel_ : 0;
    next.mates = mates;
    next.syncParts(parts);
    next.dropRegion(region);
    *this = std::move(next);
    return removed;
  }
  static Pose placement(const Part& part) {
    return {part.rotation, part.offset.e()};
  }
  /// Moves a part rigidly by @p delta, given in the world frame, with the
  /// parts positioned on it. Mates then pull the moved parts back onto their
  /// references, so a part keeps only the motion its mates leave free. The
  /// mesh keeps its topology: supports, loads, connections and the selection
  /// keep their faces, and a meshed part needs no remeshing. Remeshing from the
  /// CAD files applies the placements again.
  void movePart(int region, const Pose& delta) {
    applyMotions(previewMove(region, delta));
  }
  void translatePart(int region, const Vec<3>& delta) {
    if (!delta.e().allFinite())
      throw std::runtime_error("Part translation must be finite");
    movePart(region, Pose{Eigen::Matrix3d::Identity(), delta.e()});
  }
  /// Motion of every part, in the order of parts, that movePart() would
  /// apply.
  std::vector<Pose> previewMove(int region, const Pose& delta) const {
    const auto* part = findPart(region);
    if (!part) throw std::runtime_error("Unknown part");
    if (!delta.valid())
      throw std::runtime_error("Part motion must be a finite rigid motion");
    if (part->fixed && !delta.isIdentity())
      throw std::runtime_error(part->name +
                               " is fixed; float it in Parts to move it");
    return relax(region, delta, mates);
  }
  /// A fixed part stays where it is: mates move the other part and the part
  /// cannot be dragged.
  void setFixed(int region, bool fixed) {
    auto* part = findPart(region);
    if (!part) throw std::runtime_error("Unknown part");
    if (fixed && std::any_of(mates.begin(), mates.end(),
                             [&](const PositionMate& m) {
                               return m.moving == region;
                             }))
      throw std::runtime_error(
          part->name +
          " is placed by its mates; remove them before fixing it");
    part->fixed = fixed;
  }
  /// Motions a part can still make, from 0 (fixed or fully positioned) to 6.
  int freeMotions(int region) const {
    const auto* part = findPart(region);
    if (!part) throw std::runtime_error("Unknown part");
    if (part->fixed) return 0;
    std::vector<Pose> motions(parts.size());
    return freeMotionsOf(region, mates, motions);
  }
  /// Center of the bounding box of a part.
  Eigen::Vector3d partCenter(int region) const {
    Eigen::Vector3d low = Eigen::Vector3d::Constant(
                        std::numeric_limits<double>::infinity()),
                    high = -low;
    for (size_t f = 0; f < boundaryFaces().size(); ++f)
      if (faceParts()[f] == region)
        for (const auto v : boundaryFaces()[f].vertices) {
          low = low.cwiseMin(mesh().vertices[v].e());
          high = high.cwiseMax(mesh().vertices[v].e());
        }
    if (!(low.x() <= high.x())) throw std::runtime_error("Unknown part");
    return .5 * (low + high);
  }
  /// Shape of a face set with plane normals pointing out of the solid.
  FaceShape faceShape(const std::vector<size_t>& faces,
                      double flatness) const {
    auto shape = describeFaces(mesh(), boundaryFaces(), faces, flatness);
    if (shape.type == FaceShape::Type::Plane) {
      Eigen::Vector3d outward = Eigen::Vector3d::Zero();
      for (const auto f : faces) outward += outwardNormal(f);
      if (outward.dot(shape.direction) < 0.) shape.direction = -shape.direction;
    }
    return shape;
  }
  /// A mate to add: the faces of the part that moves, the faces it mates to,
  /// and how.
  struct MateRequest {
    MateType type = MateType::Coincident;
    std::vector<size_t> moving, reference;
    MateOptions options;
    /// Largest distance at which the mate holds, and the flatness of planes.
    double tolerance = 1e-6;
  };
  /// A mate before it is added: the mate with its settings resolved, the
  /// motion of every part (in the order of parts) and the motions the moving
  /// part keeps.
  struct MatePreview {
    PositionMate mate;
    std::vector<Pose> motions;
    int freeMotions = 6;
  };
  /// The mates that fit two face sets on different parts, most likely first,
  /// like the mates SolidWorks offers for a selection.
  std::vector<MateType> mateTypes(const std::vector<size_t>& a,
                                  const std::vector<size_t>& b,
                                  double tolerance) const {
    if (a.empty() || b.empty()) return {};
    const int first = faceParts().at(a.front()),
              second = faceParts().at(b.front());
    for (const auto* faces : {&a, &b})
      for (const auto f : *faces)
        if (faceParts().at(f) != faceParts().at(faces->front())) return {};
    if (first == second) return {};
    using Shape = FaceShape::Type;
    const auto sa = faceShape(a, tolerance), sb = faceShape(b, tolerance);
    const auto pair = [&](Shape x, Shape y) {
      return (sa.type == x && sb.type == y) || (sa.type == y && sb.type == x);
    };
    std::vector<MateType> order;
    using T = MateType;
    if (pair(Shape::Plane, Shape::Plane))
      order = {T::Coincident, T::ProfileCenter, T::Distance, T::Parallel,
               T::Perpendicular, T::Angle};
    else if (pair(Shape::Cylinder, Shape::Cylinder))
      order = {T::Concentric, T::Tangent, T::Parallel, T::Perpendicular,
               T::Distance, T::Angle};
    else if (pair(Shape::Sphere, Shape::Sphere))
      order = {T::Concentric, T::Tangent, T::Distance};
    else if (pair(Shape::Plane, Shape::Cylinder))
      order = {T::Tangent, T::Parallel, T::Perpendicular, T::Distance};
    else if (pair(Shape::Plane, Shape::Sphere))
      order = {T::Tangent, T::Distance};
    else if (pair(Shape::Cylinder, Shape::Sphere))
      order = {T::Concentric, T::Tangent, T::Distance};
    const auto ga = cadFaceGroups(a), gb = cadFaceGroups(b);
    if (ga.size() == 2 && gb.size() == 2) {
      PositionMate width;
      width.type = T::Width;
      for (const auto& group : ga)
        width.movingFaces.push_back(faceShape(group, tolerance));
      for (const auto& group : gb)
        width.referenceFaces.push_back(faceShape(group, tolerance));
      try {
        checkMateFaces(width);
        order.insert(order.empty() ? order.end() : order.begin() + 1,
                     T::Width);
      } catch (const std::invalid_argument&) {
      }
    }
    order.push_back(T::Lock);
    return order;
  }
  /// What addMate() would do, without changing the model. When the part of
  /// the moving faces is fixed, or is already positioned on the other part,
  /// the other part moves instead. Throws when the faces do not fit the mate
  /// or the mate conflicts with the others.
  MatePreview previewMate(const MateRequest& request) const {
    if (!(request.tolerance > 0.) || !std::isfinite(request.tolerance))
      throw std::runtime_error("Invalid mate tolerance");
    const auto partOf = [&](const std::vector<size_t>& faces) {
      if (faces.empty()) throw std::runtime_error("Pick faces on both parts");
      const int region = faceParts().at(faces.front());
      for (const auto f : faces)
        if (faceParts().at(f) != region)
          throw std::runtime_error("Pick the faces of one part on each side");
      return region;
    };
    int moving = partOf(request.moving), reference = partOf(request.reference);
    if (moving == reference)
      throw std::runtime_error("Pick faces on two different parts");
    const auto positioned = [&](int region, int on) {
      const auto found = dependents(on, mates);
      return std::find(found.begin(), found.end(), region) != found.end();
    };
    auto movingFaces = request.moving, referenceFaces = request.reference;
    if (findPart(moving)->fixed || positioned(reference, moving)) {
      if (findPart(reference)->fixed)
        throw std::runtime_error(
            "Both parts are fixed; float one of them in Parts");
      if (positioned(moving, reference))
        throw std::runtime_error("These parts are already positioned on "
                                 "each other through other mates");
      std::swap(moving, reference);
      std::swap(movingFaces, referenceFaces);
    }
    const Pose movingPose = placement(*findPart(moving)),
               referencePose = placement(*findPart(reference));
    const double length = modelLength();
    std::vector<FaceShape> a, b;
    if (request.type == MateType::Lock) {
      const Eigen::Vector3d c = partCenter(moving);
      for (const Eigen::Vector3d& q :
           {c, Eigen::Vector3d(c + length * Eigen::Vector3d::UnitX()),
            Eigen::Vector3d(c + length * Eigen::Vector3d::UnitY())}) {
        FaceShape point;
        point.point = q;
        a.push_back(point);
        b.push_back(point);
      }
    } else if (request.type == MateType::Width) {
      for (const auto& group : cadFaceGroups(movingFaces))
        a.push_back(faceShape(group, request.tolerance));
      for (const auto& group : cadFaceGroups(referenceFaces))
        b.push_back(faceShape(group, request.tolerance));
    } else {
      a.push_back(faceShape(movingFaces, request.tolerance));
      b.push_back(faceShape(referenceFaces, request.tolerance));
    }
    PositionMate mate;
    mate.type = request.type;
    mate.moving = moving;
    mate.reference = reference;
    mate.options = request.options;
    mate.tolerance = request.tolerance;
    auto& o = mate.options;
    const bool facing = mate.type == MateType::Coincident ||
                        mate.type == MateType::ProfileCenter ||
                        (mate.type == MateType::Distance &&
                         a[0].type == FaceShape::Type::Plane &&
                         b[0].type == FaceShape::Type::Plane);
    if (!o.aligned)
      o.aligned = !facing && a[0].direction.dot(b[0].direction) >= 0.;
    if (!o.value && (mate.type == MateType::Distance ||
                     mate.type == MateType::Angle)) {
      double measured = measureMate(mate.type, a, b);
      if (mate.type == MateType::Distance) {
        o.flip = measured < 0.;
        measured = std::abs(measured);
      }
      o.value = measured;
    }
    if (!o.value) o.value = 0.;
    if (!std::isfinite(*o.value) || *o.value < 0. ||
        (mate.type == MateType::Angle && *o.value > std::acos(-1.)))
      throw std::runtime_error(
          "Mate distances must be finite and positive, angles between 0 and "
          "180 degrees");
    {
      PositionMate world = mate;
      world.movingFaces = a;
      world.referenceFaces = b;
      try {
        checkMateFaces(world);
      } catch (const std::invalid_argument&) {
        throw std::runtime_error(std::string(mateTypeName(mate.type)) +
                                 " does not fit these faces");
      }
    }
    for (const auto& shape : a)
      mate.movingFaces.push_back(transformed(shape, movingPose.inverse()));
    for (const auto& shape : b)
      mate.referenceFaces.push_back(
          transformed(shape, referencePose.inverse()));
    for (int n = 1;; ++n) {
      mate.name = std::string(mateTypeName(mate.type)) + std::to_string(n);
      if (std::none_of(mates.begin(), mates.end(), [&](const PositionMate& m) {
            return m.name == mate.name;
          }))
        break;
    }
    auto next = mates;
    next.push_back(mate);
    MatePreview preview;
    preview.mate = mate;
    preview.motions = relax(moving, Pose{}, next);
    preview.freeMotions = freeMotionsOf(moving, next, preview.motions);
    return preview;
  }
  /// Adds a mate and moves its part, with the parts positioned on it, by the
  /// smallest motion that makes the faces meet. Throws, leaving the model
  /// unchanged, when the faces do not fit or the mate conflicts with others.
  const PositionMate& addMate(const MateRequest& request) {
    const auto preview = previewMate(request);
    applyMotions(preview.motions);
    mates.push_back(preview.mate);
    return mates.back();
  }
  /// What editMate() would do, without changing the model.
  MatePreview previewMateEdit(size_t index, const MateOptions& options) const {
    if (index >= mates.size()) throw std::runtime_error("Unknown mate");
    auto next = mates;
    auto& mate = next[index];
    mate.options.aligned = options.aligned.value_or(*mate.options.aligned);
    mate.options.flip = options.flip;
    if (options.value) mate.options.value = options.value;
    const double value = *mate.options.value;
    if (!std::isfinite(value) || value < 0. ||
        (mate.type == MateType::Angle && value > std::acos(-1.)))
      throw std::runtime_error(
          "Mate distances must be finite and positive, angles between 0 and "
          "180 degrees");
    MatePreview preview;
    preview.mate = mate;
    preview.motions = relax(mate.moving, Pose{}, next);
    preview.freeMotions = freeMotionsOf(mate.moving, next, preview.motions);
    return preview;
  }
  /// Changes the alignment, side or value of a mate and moves its part to
  /// match.
  void editMate(size_t index, const MateOptions& options) {
    const auto preview = previewMateEdit(index, options);
    applyMotions(preview.motions);
    mates[index] = preview.mate;
  }
  /// The boundary faces where each side of a mate lies now: the moving part's
  /// faces first, then the reference part's. Mates keep shapes rather than
  /// faces so that they survive remeshing; the faces are found again on the
  /// placed shapes within the extent of the stored patch, grouped by surface
  /// (CAD face or smooth patch), keeping the surfaces whose faces best
  /// reproduce the size and center of the stored shape. A whole-surface
  /// selection is found exactly. Part of a plane comes back as the faces of
  /// that surface near it, part of a cylinder as its whole circumference within
  /// the patch's bounding-box diagonal of the stored axis point along the axis,
  /// and part of a sphere as the whole sphere, because curved shapes keep only
  /// their axis or center, not where the patch was.
  std::array<std::vector<size_t>, 2> mateFaces(size_t index) const {
    const auto& mate = mates.at(index);
    std::array<std::vector<size_t>, 2> result;
    const auto& faces = boundaryFaces();
    const auto& x = mesh().vertices;
    const auto collect = [&](int region, const std::vector<FaceShape>& shapes,
                             std::vector<size_t>& out) {
      const auto* part = findPart(region);
      if (!part) return;
      const Pose pose = placement(*part);
      for (const auto& local : shapes) {
        const FaceShape shape = transformed(local, pose);
        const double tolerance = mate.tolerance + 1e-6 * shape.size;
        std::vector<size_t> candidates;
        for (size_t f = 0; f < faces.size(); ++f) {
          if (faceParts()[f] != region) continue;
          const auto& ids = faces[f].vertices;
          const Eigen::Vector3d a = x[ids[0]].e(), b = x[ids[1]].e(),
                                c = x[ids[2]].e();
          const Eigen::Vector3d cross = (b - a).cross(c - a);
          if (!(cross.norm() > 0.)) continue;
          const Eigen::Vector3d normal = cross.normalized();
          const Eigen::Vector3d offset = (a + b + c) / 3. - shape.point;
          bool on = false;
          switch (shape.type) {
            case FaceShape::Type::Plane:
              on = offset.norm() <= shape.size + tolerance &&
                   normal.dot(shape.direction) > .95 &&
                   std::abs(offset.dot(shape.direction)) <= tolerance;
              break;
            case FaceShape::Type::Cylinder: {
              const double along = shape.direction.dot(offset);
              const Eigen::Vector3d radial = offset - shape.direction * along;
              on = std::abs(along) <= shape.size + tolerance &&
                   std::abs(normal.dot(shape.direction)) < .3 &&
                   std::abs(radial.norm() - shape.radius) <=
                       .08 * shape.radius + tolerance;
              break;
            }
            case FaceShape::Type::Sphere:
              on = std::abs(offset.norm() - shape.radius) <=
                       .08 * shape.radius + tolerance &&
                   std::abs(normal.dot(offset)) > .7 * offset.norm();
              break;
            default:
              break;
          }
          if (on) candidates.push_back(f);
        }
        struct Piece {
          int surface = 0;
          std::vector<size_t> faces;
          double area = 0.;
          Eigen::Vector3d moment = Eigen::Vector3d::Zero(),
                          low = Eigen::Vector3d::Constant(
                              std::numeric_limits<double>::infinity()),
                          high = -low;
        };
        std::vector<Piece> pieces;
        for (const auto f : candidates) {
          auto piece = std::find_if(pieces.begin(), pieces.end(),
                                    [&](const Piece& p) {
                                      return p.surface == faceSurfaces_[f];
                                    });
          if (piece == pieces.end()) {
            pieces.emplace_back();
            pieces.back().surface = faceSurfaces_[f];
            piece = pieces.end() - 1;
          }
          const auto& ids = faces[f].vertices;
          const Eigen::Vector3d a = x[ids[0]].e(), b = x[ids[1]].e(),
                                c = x[ids[2]].e();
          const double area = .5 * (b - a).cross(c - a).norm();
          piece->faces.push_back(f);
          piece->area += area;
          piece->moment += area * (a + b + c) / 3.;
          piece->low = piece->low.cwiseMin(a).cwiseMin(b).cwiseMin(c);
          piece->high = piece->high.cwiseMax(a).cwiseMax(b).cwiseMax(c);
        }
        const double scale = std::max(shape.size, 1e-12);
        const auto error = [&](const std::vector<char>& chosen) {
          Piece sum;
          for (size_t k = 0; k < pieces.size(); ++k) {
            if (!chosen[k]) continue;
            sum.area += pieces[k].area;
            sum.moment += pieces[k].moment;
            sum.low = sum.low.cwiseMin(pieces[k].low);
            sum.high = sum.high.cwiseMax(pieces[k].high);
          }
          if (!(sum.area > 0.)) return std::numeric_limits<double>::infinity();
          double e = std::abs((sum.high - sum.low).norm() - shape.size) / scale;
          if (shape.type == FaceShape::Type::Plane)
            e += (sum.moment / sum.area - shape.point).norm() / scale;
          return e;
        };
        std::vector<char> best(pieces.size(), 0);
        double bestError = std::numeric_limits<double>::infinity();
        if (pieces.size() <= 12) {
          std::vector<char> chosen(pieces.size());
          for (size_t mask = 1; mask < (size_t(1) << pieces.size()); ++mask) {
            for (size_t k = 0; k < pieces.size(); ++k)
              chosen[k] = char(mask >> k & 1);
            const double e = error(chosen);
            if (e < bestError) {
              bestError = e;
              best = chosen;
            }
          }
        } else {
          for (bool improved = true; improved;) {
            improved = false;
            auto step = best;
            for (size_t k = 0; k < pieces.size(); ++k) {
              if (best[k]) continue;
              auto trial = best;
              trial[k] = 1;
              const double e = error(trial);
              if (e < bestError) {
                bestError = e;
                step = std::move(trial);
                improved = true;
              }
            }
            best = std::move(step);
          }
        }
        for (size_t k = 0; k < pieces.size(); ++k)
          if (best[k])
            out.insert(out.end(), pieces[k].faces.begin(), pieces[k].faces.end());
      }
      std::sort(out.begin(), out.end());
      out.erase(std::unique(out.begin(), out.end()), out.end());
    };
    collect(mate.moving, mate.movingFaces, result[0]);
    collect(mate.reference, mate.referenceFaces, result[1]);
    return result;
  }
  /// Removes a mate; the parts stay where they are.
  void removeMate(size_t index) {
    if (index >= mates.size()) throw std::runtime_error("Unknown mate");
    mates.erase(mates.begin() + std::ptrdiff_t(index));
  }
  /// Moves every part by its world motion, in the order of parts, without
  /// solving mates, for example to undo the motions of a MatePreview.
  void moveParts(const std::vector<Pose>& motions) {
    if (motions.size() != parts.size() ||
        std::any_of(motions.begin(), motions.end(),
                    [](const Pose& m) { return !m.valid(); }))
      throw std::runtime_error("One rigid motion per part is required");
    applyMotions(motions);
  }
  Eigen::Vector3d outwardNormal(size_t face) const {
    const auto& f = boundaryFaces().at(face);
    const auto& x = mesh().vertices;
    Eigen::Vector3d normal =
        (x[f.vertices[1]].e() - x[f.vertices[0]].e())
            .cross(x[f.vertices[2]].e() - x[f.vertices[0]].e());
    if (solver)
      for (const auto v : mesh().tetrahedra.at(f.element))
        if (v != f.vertices[0] && v != f.vertices[1] && v != f.vertices[2]) {
          if (normal.dot(x[v].e() - x[f.vertices[0]].e()) > 0.)
            normal = -normal;
          break;
        }
    return normal;
  }
  Vec<3> forceVector(const Condition& c) const {
    if (c.direction == ForceDirection::Vector) return c.value;
    if (c.direction != ForceDirection::ReferenceNormal)
      throw std::logic_error("A face-normal force has no single direction");
    if (c.targetFaces.empty())
      throw std::runtime_error(
          "Pick the reference face that sets the force direction");
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    double area = 0.;
    for (const auto f : c.targetFaces) {
      const auto normal = outwardNormal(f);
      sum += normal;
      area += normal.norm();
    }
    if (!(sum.norm() > 1e-6 * area))
      throw std::runtime_error(
          "The force reference face has no single normal direction; pick a "
          "flat face");
    const Eigen::Vector3d force = -c.value[0] * sum.normalized();
    return {force[0], force[1], force[2]};
  }
  void solve(const std::function<void(LinearElasticity::SolveStage, double)>&
                 progress = {}) {
    if (!solver) throw std::runtime_error("Import a solid first");
    dirty = false;
    hasSolution = false;
    estimatedError = -1.;
    const auto materials = elementMaterials();
    if (materials.size() == 1)
      solver->setMaterial(materials[0]);
    else
      solver->setElementMaterials(materials);
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
        if (c.kind == Kind::Force &&
            c.direction == ForceDirection::FaceNormal) {
          double area = 0.;
          for (const auto f : c.faces) area += .5 * outwardNormal(f).norm();
          if (!(area > 0.))
            throw std::runtime_error("A normal force needs selected faces");
          for (const auto f : c.faces)
            solver->addSurfacePressure(f, c.value[0] / area);
        } else if (c.kind == Kind::Force) {
          const Vec<3> force = forceVector(c);
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
              solver->addNodalForce(v, force * (weights[v] / total));
        }
      }
    solver->setGravity(gravity);
    struct ProgressScope {
      LinearElasticity& solver;
      ~ProgressScope() { solver.setProgressCallback({}); }
    } scope{*solver};
    solver->setProgressCallback(progress);
    solver->solve();
    hasSolution = true;
    dirty = false;
  }
  /// Solves, then refines where the recovered-stress error estimate is largest
  /// and solves again until the estimate meets the target, the mesh has been
  /// refined maximumSteps times since it was generated, or it cannot be refined
  /// within the limits. Supports and loads follow the refined faces exactly.
  AdaptiveResult solveAdaptive(
      const Adaptivity& settings,
      const std::function<void(int step, LinearElasticity::SolveStage, double)>&
          progress = {}) {
    if (!(settings.relativeError > 0.) || !(settings.relativeError < 1.) ||
        settings.maximumSteps < 0 || !(settings.minimumEdge >= 0.) ||
        !std::isfinite(settings.minimumEdge))
      throw std::invalid_argument("Invalid adaptive refinement settings");
    const auto step = [&](int index) -> Progress {
      if (!progress) return {};
      return [&progress, index](LinearElasticity::SolveStage stage,
                                double fraction) {
        progress(index, stage, fraction);
      };
    };
    AdaptiveResult result;
    solve(step(0));
    for (;;) {
      const auto estimate = raisim::fem::estimateRecoveryError(
          mesh(), solver->elementResults(), elementMaterials());
      estimatedError = result.estimatedError = estimate.relative();
      if (result.estimatedError <= settings.relativeError) {
        result.converged = true;
        break;
      }
      if (refinementLevel_ >= settings.maximumSteps ||
          !refine(estimate, settings))
        break;
      ++result.refinements;
      solve(step(result.refinements));
    }
    return result;
  }
  /// Refinement steps applied since the mesh was generated or loaded.
  int refinementLevel() const { return refinementLevel_; }
  bool refine(const raisim::fem::RecoveryErrorEstimate& estimate,
              const Adaptivity& settings) {
    const size_t cells = mesh().tetrahedra.size();
    if (cells >= settings.maximumCells) return false;
    size_t budget = std::min(3 * cells, settings.maximumCells);
    while (budget > cells) {
      const auto targets = raisim::fem::refinementTargets(
          mesh(), estimate, settings.minimumEdge, .5, budget);
      std::vector<std::array<unsigned int, 2>> parents;
      try {
        auto refined = raisim::fem::refineElements(
            mesh(), targets, settings.maximumCells, &parents);
        if (parents.empty()) return false;
        applyRefinement(std::move(refined), parents);
        return true;
      } catch (const std::runtime_error&) {
        budget = cells + (budget - cells) / 2;
      }
    }
    return false;
  }
  void save(const std::string& path) const {
    if (!hasGeometry()) throw std::runtime_error("No model to save");
    std::ofstream out(raisim::pathFromUtf8(path));
    if (!out) throw std::runtime_error("Cannot write project: " + path);
    out.imbue(std::locale::classic());
    out.precision(17);
    out << "RAISIM_FEM_PROJECT 10\n"
        << std::quoted(name) << '\n'
        << material.youngsModulus << ' ' << material.poissonRatio << ' '
        << material.density << ' ' << yieldStrength << '\n';
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
        << source.previewTolerance << ' ' << source.referenceExtent << " 0 0 "
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
    out << bool(brepReference_) << '\n';
    if (brepReference_) {
      out << std::quoted(brepReference_->encode()) << '\n';
      for (const auto surface : faceSurfaces_) out << surface << ' ';
      out << '\n';
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
      out << static_cast<int>(c.direction);
      for (const auto* ids : {&c.cadFaces, &c.targetCadFaces}) {
        out << ' ' << ids->size();
        for (const auto id : *ids) out << ' ' << id;
      }
      out << ' ' << c.strength << '\n';
    }
    out << parts.size() << '\n';
    for (const auto& part : parts) {
      out << part.region << ' ' << std::quoted(part.name) << ' '
          << part.material.has_value();
      if (part.material)
        out << ' ' << part.material->youngsModulus << ' '
            << part.material->poissonRatio << ' ' << part.material->density
            << ' ' << part.yieldStrength;
      out << ' ' << part.offset[0] << ' ' << part.offset[1] << ' '
          << part.offset[2];
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out << ' ' << part.rotation(r, c);
      out << ' ' << part.fixed << '\n';
    }
    out << source.assembly.size() << '\n';
    for (const auto& file : source.assembly)
      out << std::quoted(file.name) << ' ' << std::quoted(file.extension) << ' '
          << file.scale << ' ' << file.regionOffset << ' '
          << std::quoted(file.bytes) << '\n';
    out << source.regionOffset << ' ' << source.excludedRegions.size();
    for (const auto region : source.excludedRegions) out << ' ' << region;
    out << '\n';
    out << mates.size() << '\n';
    const auto writeShape = [&](const FaceShape& shape) {
      out << ' ' << static_cast<int>(shape.type);
      for (int a = 0; a < 3; ++a) out << ' ' << shape.point[a];
      for (int a = 0; a < 3; ++a) out << ' ' << shape.direction[a];
      out << ' ' << shape.radius << ' ' << shape.size;
    };
    for (const auto& mate : mates) {
      out << static_cast<int>(mate.type) << ' ' << std::quoted(mate.name)
          << ' ' << mate.moving << ' ' << mate.reference << ' '
          << mate.options.aligned.value_or(false) << ' '
          << mate.options.value.value_or(0.) << ' ' << mate.options.flip
          << ' ' << mate.tolerance << ' ' << mate.movingFaces.size() << ' '
          << mate.referenceFaces.size();
      for (const auto& face : mate.movingFaces) writeShape(face);
      for (const auto& face : mate.referenceFaces) writeShape(face);
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
    if (magic != "RAISIM_FEM_PROJECT" || (version < 1 || version > 10))
      throw std::runtime_error("Unsupported FEM project");
    Model next;
    in >> next.material.youngsModulus >> next.material.poissonRatio >>
        next.material.density;
    if (version >= 9)
      in >> next.yieldStrength;
    else
      next.yieldStrength = 0.;
    if (!(next.yieldStrength >= 0.) || !std::isfinite(next.yieldStrength))
      throw std::runtime_error("Invalid project yield strength");
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
    if (version >= 4) {
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
      bool retiredMesherFlag = false;
      in >> std::quoted(next.source.extension) >> next.source.scale >>
          next.source.maximumEdge >> next.source.surfaceTolerance >>
          next.source.meshedEdge >> next.source.meshedTolerance >>
          next.source.previewTolerance >> next.source.referenceExtent >>
          retiredMesherFlag >> retiredMesherFlag >>
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
      bool cadFaces = false;
      if (version >= 5) in >> cadFaces;
      if (cadFaces) {
        std::string bytes;
        in >> std::quoted(bytes);
        if (!in) throw std::runtime_error("Truncated project CAD faces");
        auto reference =
            std::make_shared<SurfaceGeometry>(SurfaceGeometry::decode(bytes));
        auto known = reference->brepFaces;
        std::sort(known.begin(), known.end());
        known.erase(std::unique(known.begin(), known.end()), known.end());
        std::vector<int> surfaces(next.boundaryFaces().size());
        for (auto& surface : surfaces) in >> surface;
        if (!in || known.empty() ||
            std::any_of(surfaces.begin(), surfaces.end(), [&](int surface) {
              return !std::binary_search(known.begin(), known.end(), surface);
            }))
          throw std::runtime_error("Invalid project CAD faces");
        next.brepReference_ = std::move(reference);
        next.faceSurfaces_ = std::move(surfaces);
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
          kind > static_cast<int>(version == 1   ? Kind::Gravity
                                  : version < 6 ? Kind::Tie
                                                : Kind::RigidLink))
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
      if (version >= 5) {
        int direction = -1;
        in >> direction;
        if (!in || direction < 0 ||
            direction > static_cast<int>(ForceDirection::ReferenceNormal) ||
            (direction != 0 && c.kind != Kind::Force))
          throw std::runtime_error("Invalid FEM force direction");
        c.direction = static_cast<ForceDirection>(direction);
        for (auto* ids : {&c.cadFaces, &c.targetCadFaces}) {
          ids->resize(readCount());
          for (auto& id : *ids) in >> id;
          if (!in || !std::is_sorted(ids->begin(), ids->end()) ||
              std::adjacent_find(ids->begin(), ids->end()) != ids->end() ||
              (!ids->empty() && !next.hasCadFaces()) ||
              std::any_of(ids->begin(), ids->end(), [&](int id) {
                return std::find(next.faceSurfaces_.begin(),
                                 next.faceSurfaces_.end(),
                                 id) == next.faceSurfaces_.end();
              }))
            throw std::runtime_error("Invalid FEM condition CAD faces");
        }
      }
      if (version >= 10) {
        in >> c.strength;
        if (!in || !(c.strength >= 0.) || !std::isfinite(c.strength))
          throw std::runtime_error("Invalid FEM connection strength");
      }
    }
    if (version >= 6) {
      const auto count = readCount();
      for (size_t i = 0; i < count; ++i) {
        Part part;
        bool custom = false;
        in >> part.region >> std::quoted(part.name) >> custom;
        if (custom) {
          LinearElasticity::Material m;
          in >> m.youngsModulus >> m.poissonRatio >> m.density;
          if (version >= 9) in >> part.yieldStrength;
          if (!(part.yieldStrength >= 0.) ||
              !std::isfinite(part.yieldStrength))
            throw std::runtime_error("Invalid FEM project yield strength");
          if (!(m.youngsModulus > 0.) || !std::isfinite(m.youngsModulus) ||
              !(m.poissonRatio > -1. && m.poissonRatio < .5) ||
              !(m.density >= 0.) || !std::isfinite(m.density))
            throw std::runtime_error("Invalid FEM project part material");
          part.material = m;
        }
        if (version >= 8)
          in >> part.offset[0] >> part.offset[1] >> part.offset[2];
        if (version >= 10) {
          for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) in >> part.rotation(r, c);
          in >> part.fixed;
        }
        auto* existing = next.findPart(part.region);
        if (!in || !existing || !placement(part).valid())
          throw std::runtime_error("Invalid FEM project part");
        *existing = part;
      }
      next.source.assembly.resize(readCount());
      for (auto& file : next.source.assembly) {
        in >> std::quoted(file.name) >> std::quoted(file.extension) >>
            file.scale >> file.regionOffset >> std::quoted(file.bytes);
        const std::vector<std::string> cad{".step", ".stp", ".brep", ".iges",
                                           ".igs"};
        if (!in || std::find(cad.begin(), cad.end(), file.extension) ==
                       cad.end() ||
            !(file.scale > 0.) || !std::isfinite(file.scale) ||
            file.regionOffset < 0 || file.bytes.empty() ||
            file.bytes.size() > 128u * 1024u * 1024u ||
            std::find(cad.begin(), cad.end(), next.source.extension) ==
                cad.end())
          throw std::runtime_error("Invalid FEM project assembly file");
      }
      if (version >= 7) {
        in >> next.source.regionOffset;
        next.source.excludedRegions.resize(readCount());
        for (auto& region : next.source.excludedRegions) in >> region;
        if (!in || next.source.regionOffset < 0)
          throw std::runtime_error("Invalid FEM project part offsets");
      }
    }
    if (version >= 10) {
      next.mates.resize(readCount());
      const auto readShape = [&](FaceShape& shape) {
        int type = -1;
        in >> type;
        for (int a = 0; a < 3; ++a) in >> shape.point[a];
        for (int a = 0; a < 3; ++a) in >> shape.direction[a];
        in >> shape.radius >> shape.size;
        const bool directed = type == 1 || type == 2;
        if (!in || type < 0 || type > 3 || !shape.point.allFinite() ||
            !shape.direction.allFinite() ||
            (directed && std::abs(shape.direction.norm() - 1.) > 1e-9) ||
            !(shape.radius >= 0.) || !std::isfinite(shape.radius) ||
            !std::isfinite(shape.size))
          throw std::runtime_error("Invalid FEM project mate");
        shape.type = static_cast<FaceShape::Type>(type);
      };
      for (auto& mate : next.mates) {
        int type = -1;
        bool aligned = false;
        double value = 0.;
        in >> type >> std::quoted(mate.name) >> mate.moving >>
            mate.reference >> aligned >> value >> mate.options.flip >>
            mate.tolerance;
        const auto movingCount = readCount(), referenceCount = readCount();
        if (!in || type < 0 ||
            type > static_cast<int>(MateType::Width) || movingCount > 3 ||
            referenceCount > 3)
          throw std::runtime_error("Invalid FEM project mate");
        mate.type = static_cast<MateType>(type);
        mate.options.aligned = aligned;
        mate.options.value = value;
        mate.movingFaces.resize(movingCount);
        mate.referenceFaces.resize(referenceCount);
        for (auto& face : mate.movingFaces) readShape(face);
        for (auto& face : mate.referenceFaces) readShape(face);
        const auto* moving = next.findPart(mate.moving);
        if (!in || mate.moving == mate.reference || !moving ||
            moving->fixed || !next.findPart(mate.reference) ||
            !std::isfinite(value) || value < 0. ||
            !(mate.tolerance > 0.) || !std::isfinite(mate.tolerance))
          throw std::runtime_error("Invalid FEM project mate");
        try {
          checkMateFaces(mate);
        } catch (const std::invalid_argument&) {
          throw std::runtime_error("Invalid FEM project mate");
        }
      }
      for (const auto& mate : next.mates) {
        const auto found = dependents(mate.moving, next.mates);
        if (std::find(found.begin(), found.end(), mate.reference) !=
            found.end())
          throw std::runtime_error("FEM project mates form a loop");
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
    *this = std::move(next);
  }

 private:
  /// @p region followed by every part positioned on it, directly or through
  /// other parts.
  static std::vector<int> dependents(int region,
                                     const std::vector<PositionMate>& list) {
    std::vector<int> found{region};
    for (size_t i = 0; i < found.size(); ++i)
      for (const auto& mate : list)
        if (mate.reference == found[i] &&
            std::find(found.begin(), found.end(), mate.moving) == found.end())
          found.push_back(mate.moving);
    return found;
  }
  double modelLength() const {
    Eigen::Vector3d low = mesh().vertices.front().e(), high = low;
    for (const auto& x : mesh().vertices) {
      low = low.cwiseMin(x.e());
      high = high.cwiseMax(x.e());
    }
    return std::max((high - low).norm(), 1e-9);
  }
  /// The faces split by the CAD face (or smooth patch) they belong to.
  std::vector<std::vector<size_t>> cadFaceGroups(
      const std::vector<size_t>& faces) const {
    std::vector<int> ids;
    std::vector<std::vector<size_t>> groups;
    for (const auto f : faces) {
      const int id = faceSurfaces_.at(f);
      const auto found = std::find(ids.begin(), ids.end(), id);
      if (found == ids.end()) {
        ids.push_back(id);
        groups.push_back({f});
      } else {
        groups[size_t(found - ids.begin())].push_back(f);
      }
    }
    return groups;
  }
  /// The mates of @p region with their faces in the world after the parts
  /// move by @p motions.
  std::vector<PlacementTarget> targetsOf(
      int region, const std::vector<PositionMate>& list,
      const std::vector<Pose>& motions) const {
    const auto pose = [&](int r) {
      const auto* part = findPart(r);
      if (!part)
        throw std::runtime_error("A mate refers to a removed part");
      return motions[size_t(part - parts.data())] * placement(*part);
    };
    std::vector<PlacementTarget> targets;
    for (const auto& mate : list) {
      if (mate.moving != region) continue;
      PlacementTarget target;
      target.mate = &mate;
      for (const auto& face : mate.movingFaces)
        target.moving.push_back(transformed(face, pose(region)));
      for (const auto& face : mate.referenceFaces)
        target.reference.push_back(transformed(face, pose(mate.reference)));
      targets.push_back(std::move(target));
    }
    return targets;
  }
  static Eigen::Vector3d pivotOf(const std::vector<PlacementTarget>& targets) {
    Eigen::Vector3d pivot = Eigen::Vector3d::Zero();
    size_t count = 0;
    for (const auto& target : targets)
      for (const auto& face : target.moving) {
        pivot += face.point;
        ++count;
      }
    return count ? Eigen::Vector3d(pivot / double(count)) : pivot;
  }
  int freeMotionsOf(int region, const std::vector<PositionMate>& list,
                    const std::vector<Pose>& motions) const {
    const auto targets = targetsOf(region, list, motions);
    return fem_workbench::freeMotions(targets, pivotOf(targets),
                                      modelLength());
  }
  /// World motion of every part, in the order of parts, after @p region moves
  /// by @p delta and the mates of it and its dependents are solved.
  std::vector<Pose> relax(int region, const Pose& delta,
                          const std::vector<PositionMate>& list) const {
    std::vector<Pose> motions(parts.size());
    const auto at = [&](int r) -> Pose& {
      const auto* part = findPart(r);
      if (!part)
        throw std::runtime_error("A mate refers to a removed part");
      return motions[size_t(part - parts.data())];
    };
    const auto carry = [&](int r, const Pose& change, bool self) {
      for (const auto other : dependents(r, list))
        if ((self || other != r) && !findPart(other)->fixed)
          at(other) = change * at(other);
    };
    if (!delta.isIdentity()) carry(region, delta, true);
    auto remaining = dependents(region, list);
    const double length = modelLength();
    while (!remaining.empty()) {
      const auto ready =
          std::find_if(remaining.begin(), remaining.end(), [&](int r) {
            return std::none_of(
                list.begin(), list.end(), [&](const PositionMate& mate) {
                  return mate.moving == r &&
                         std::find(remaining.begin(), remaining.end(),
                                   mate.reference) != remaining.end();
                });
          });
      if (ready == remaining.end())
        throw std::runtime_error("The mates form a loop");
      const int moving = *ready;
      remaining.erase(ready);
      const auto targets = targetsOf(moving, list, motions);
      if (targets.empty()) continue;
      const auto* part = findPart(moving);
      Pose change;
      try {
        change = solvePlacement(targets, pivotOf(targets), length);
      } catch (const std::runtime_error&) {
        throw std::runtime_error("The mates of " + part->name +
                                 " conflict; flip, change or remove one");
      } catch (const std::invalid_argument&) {
        throw std::runtime_error("A mate of " + part->name +
                                 " no longer fits its faces");
      }
      if (change.isIdentity()) continue;
      if (part->fixed)
        throw std::runtime_error(part->name +
                                 " is fixed and cannot follow its mates");
      carry(moving, change, true);
    }
    return motions;
  }
  /// Moves every part by its world motion, in the order of parts.
  void applyMotions(const std::vector<Pose>& motions) {
    const auto motion = [&](int region) -> const Pose* {
      const auto* part = findPart(region);
      if (!part) return nullptr;
      const auto& m = motions.at(size_t(part - parts.data()));
      return m.isIdentity() ? nullptr : &m;
    };
    if (std::all_of(motions.begin(), motions.end(),
                    [](const Pose& m) { return m.isIdentity(); }))
      return;
    if (source.extension == ".rfs")
      throw std::runtime_error(
          "Parts of this project cannot be moved; open its CAD files again");
    const auto shift = [&](const std::vector<int>& owner, auto& vertices) {
      for (size_t v = 0; v < vertices.size(); ++v)
        if (owner[v] != std::numeric_limits<int>::min())
          if (const auto* m = motion(owner[v]))
            vertices[v].e() = (*m)(vertices[v].e());
    };
    const auto owners = [&](size_t count, auto&& each) {
      std::vector<int> owner(count, std::numeric_limits<int>::min());
      each([&](unsigned int v, int r) {
        if (owner[v] != std::numeric_limits<int>::min() && owner[v] != r &&
            (motion(owner[v]) || motion(r)))
          throw std::runtime_error(
              "This part shares nodes with another part and cannot be moved "
              "on its own");
        owner[v] = r;
      });
      return owner;
    };
    const auto ofVolume = [&](const TetMesh& volume) {
      return owners(volume.vertices.size(), [&](auto&& visit) {
        for (size_t t = 0; t < volume.tetrahedra.size(); ++t)
          for (const auto v : volume.tetrahedra[t])
            visit(v, volume.regions.empty() ? 0 : volume.regions[t]);
      });
    };
    const auto ofSurface = [&](const SurfaceGeometry& surface) {
      return owners(surface.mesh.vertices.size(), [&](auto&& visit) {
        for (size_t f = 0; f < surface.faces.size(); ++f)
          for (const auto v : surface.faces[f].vertices)
            visit(v, surface.parts[f]);
      });
    };
    const bool sharedReference = brepReference_ && brepReference_ == surface_;
    std::shared_ptr<const SurfaceGeometry> reference;
    if (brepReference_ && !sharedReference) {
      auto moved = *brepReference_;
      shift(ofSurface(moved), moved.mesh.vertices);
      reference = std::make_shared<const SurfaceGeometry>(std::move(moved));
    }
    std::shared_ptr<TetMesh> volume;
    if (source.volume) {
      volume = std::make_shared<TetMesh>(*source.volume);
      shift(ofVolume(*volume), volume->vertices);
    }
    std::vector<int> owner;
    if (solver) {
      TetMesh moved = mesh();
      owner = ofVolume(moved);
      shift(owner, moved.vertices);
      solver = std::make_unique<LinearElasticity>(std::move(moved));
      selector_ = std::make_unique<SurfaceSelector>(*solver);
      previousMask_.assign(3 * mesh().vertices.size(), 0);
      fixtureCache_.clear();
    } else {
      auto moved = *surface_;
      owner = ofSurface(moved);
      shift(owner, moved.mesh.vertices);
      surface_ = std::make_shared<const SurfaceGeometry>(std::move(moved));
      selector_ = std::make_unique<SurfaceSelector>(surface_->mesh,
                                                    surface_->faces);
      if (sharedReference) brepReference_ = surface_;
    }
    if (reference) brepReference_ = std::move(reference);
    if (volume) source.volume = std::move(volume);
    referenceGeometry_.reset();
    for (auto& c : conditions) {
      c.primaryGeometry.reset();
      c.targetGeometry.reset();
      const Pose* common = nullptr;
      bool rigid = !c.nodes.empty();
      for (const auto* nodes : {&c.nodes, &c.targetNodes})
        for (const auto v : *nodes) {
          const auto* m = owner.at(v) == std::numeric_limits<int>::min()
                              ? nullptr
                              : motion(owner[v]);
          if (!m || (common && !(*m == *common))) rigid = false;
          common = m;
        }
      if (rigid && common) {
        c.center.e() = (*common)(c.center.e());
        c.axis.e() = common->rotation * c.axis.e();
      }
    }
    for (size_t i = 0; i < parts.size(); ++i) {
      const auto& m = motions[i];
      if (m.isIdentity()) continue;
      auto& part = parts[i];
      part.offset.e() = m.rotation * part.offset.e() + m.translation;
      if (m.rotation != Eigen::Matrix3d::Identity())
        part.rotation = Eigen::Quaterniond(m.rotation * part.rotation)
                            .normalized()
                            .toRotationMatrix();
    }
    changed();
  }
  void dropRegion(int region) {
    auto& files = source.assembly;
    const auto owner = [&](int r) {
      int file = -1, offset = source.regionOffset;
      for (size_t i = 0; i < files.size(); ++i)
        if (files[i].regionOffset < r && files[i].regionOffset >= offset) {
          file = int(i);
          offset = files[i].regionOffset;
        }
      return file;
    };
    const int file = owner(region);
    const bool remaining = std::any_of(parts.begin(), parts.end(), [&](const Part& p) {
      return owner(p.region) == file;
    });
    auto& excluded = source.excludedRegions;
    if (remaining) {
      excluded.push_back(region);
      std::sort(excluded.begin(), excluded.end());
      return;
    }
    if (file < 0 && source.bytes.empty()) return;
    std::vector<int> stale;
    for (const auto r : excluded)
      if (owner(r) == file) stale.push_back(r);
    if (file >= 0) {
      files.erase(files.begin() + file);
    } else if (!files.empty()) {
      source.extension = files.front().extension;
      source.bytes = std::move(files.front().bytes);
      source.scale = files.front().scale;
      source.regionOffset = files.front().regionOffset;
      files.erase(files.begin());
    }
    excluded.erase(std::remove_if(excluded.begin(), excluded.end(),
                                  [&](int r) {
                                    return std::find(stale.begin(), stale.end(),
                                                     r) != stale.end();
                                  }),
                   excluded.end());
  }
  void syncParts(const std::vector<Part>& previous) {
    std::vector<int> regions =
        !solver                     ? faceParts()
        : mesh().regions.empty() ? std::vector<int>{0}
                                    : mesh().regions;
    std::sort(regions.begin(), regions.end());
    regions.erase(std::unique(regions.begin(), regions.end()), regions.end());
    parts.clear();
    for (const auto region : regions) {
      const auto found =
          std::find_if(previous.begin(), previous.end(),
                       [&](const Part& p) { return p.region == region; });
      if (found != previous.end()) {
        parts.push_back(*found);
      } else {
        Part part;
        part.region = region;
        part.name = "Part " + std::to_string(region);
        parts.push_back(std::move(part));
      }
    }
    mates.erase(
        std::remove_if(mates.begin(), mates.end(),
                       [&](const PositionMate& mate) {
                         return !findPart(mate.moving) ||
                                !findPart(mate.reference);
                       }),
        mates.end());
  }
  void transferConditions(Model& next, double tolerance,
                          double destinationDeviation) const {
    transferConditions(next, tolerance, destinationDeviation, conditions);
  }
  void transferConditions(Model& next, double tolerance,
                          double destinationDeviation,
                          const std::vector<Condition>& list) const {
    next.conditions = list;
    const auto transfer = [&](const Condition& c, const std::vector<int>& cad,
                              std::shared_ptr<const SelectionAnchor>& anchor,
                              std::vector<unsigned int>& nodes,
                              std::vector<size_t>& faces) {
      if (!cad.empty() && next.hasCadFaces()) {
        next.cadRegion(c, cad, nodes, faces);
        anchor.reset();
        return;
      }
      anchor = captureSelection(nodes, faces, anchor);
      anchor = mapRegion(anchor, next.mesh(), next.boundaryFaces(),
                         next.faceParts(), tolerance, destinationDeviation,
                         nodes, faces);
    };
    for (auto& c : next.conditions) {
      transfer(c, c.cadFaces, c.primaryGeometry, c.nodes, c.faces);
      transfer(c, c.targetCadFaces, c.targetGeometry, c.targetNodes,
               c.targetFaces);
    }
  }
  void applyRefinement(TetMesh volume,
                       const std::vector<std::array<unsigned int, 2>>& parents) {
    constexpr unsigned int kNone = ~0u;
    const size_t original = mesh().vertices.size();
    if (volume.vertices.size() != original + parents.size())
      throw std::logic_error("Refinement parents do not match the new vertices");
    std::vector<std::array<unsigned int, 4>> carrier(volume.vertices.size());
    for (size_t v = 0; v < original; ++v)
      carrier[v] = {static_cast<unsigned int>(v), kNone, kNone, kNone};
    for (size_t i = 0; i < parents.size(); ++i) {
      auto& c = carrier[original + i];
      c.fill(kNone);
      size_t count = 0;
      for (const auto parent : parents[i])
        for (const auto v : carrier.at(parent)) {
          if (v == kNone || std::find(c.begin(), c.begin() + count, v) !=
                                c.begin() + count)
            continue;
          if (count == 4)
            throw std::logic_error("A refined vertex left its tetrahedron");
          c[count++] = v;
        }
      std::sort(c.begin(), c.begin() + count);
    }
    std::vector<std::pair<std::array<unsigned int, 3>, size_t>> keys;
    keys.reserve(boundaryFaces().size());
    for (size_t f = 0; f < boundaryFaces().size(); ++f) {
      auto key = boundaryFaces()[f].vertices;
      std::sort(key.begin(), key.end());
      keys.emplace_back(key, f);
    }
    std::sort(keys.begin(), keys.end());
    Model next;
    next.material = material;
    next.yieldStrength = yieldStrength;
    next.setMesh(std::move(volume), name);
    const auto& faces = next.boundaryFaces();
    std::vector<size_t> parentFace(faces.size());
    std::vector<std::vector<size_t>> children(boundaryFaces().size());
    for (size_t f = 0; f < faces.size(); ++f) {
      std::array<unsigned int, 4> corners;
      corners.fill(kNone);
      size_t count = 0;
      for (const auto v : faces[f].vertices)
        for (const auto corner : carrier[v]) {
          if (corner == kNone || std::find(corners.begin(),
                                           corners.begin() + count, corner) !=
                                     corners.begin() + count)
            continue;
          if (count == 4) break;
          corners[count++] = corner;
        }
      std::sort(corners.begin(), corners.begin() + count);
      const std::array<unsigned int, 3> key{corners[0], corners[1],
                                            corners[2]};
      const auto found = std::lower_bound(
          keys.begin(), keys.end(), std::make_pair(key, size_t(0)));
      if (count != 3 || found == keys.end() || found->first != key)
        throw std::logic_error("A refined boundary face has no parent face");
      parentFace[f] = found->second;
      children[found->second].push_back(f);
    }
    for (size_t f = 0; f < faces.size(); ++f)
      next.faceSurfaces_[f] = faceSurfaces_[parentFace[f]];
    next.brepReference_ = brepReference_;
    next.source = source;
    next.conditions = conditions;
    const auto map = [&](std::vector<unsigned int>& nodes,
                         std::vector<size_t>& selection) {
      if (selection.empty()) return;
      std::vector<size_t> mapped;
      for (const auto f : selection)
        for (const auto child : children.at(f)) mapped.push_back(child);
      std::sort(mapped.begin(), mapped.end());
      for (const auto f : mapped)
        for (const auto v : faces[f].vertices) nodes.push_back(v);
      std::sort(nodes.begin(), nodes.end());
      nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
      selection = std::move(mapped);
    };
    for (auto& c : next.conditions) {
      map(c.nodes, c.faces);
      map(c.targetNodes, c.targetFaces);
    }
    next.refinementLevel_ = refinementLevel_ + 1;
    next.mates = mates;
    next.syncParts(parts);
    next.surfaceSelection_ = surfaceSelection_;
    for (size_t f = 0; f < faces.size(); ++f)
      next.selectedSurfaces_[f] = selectedSurfaces_[parentFace[f]];
    for (size_t v = 0; v < next.selected.size(); ++v)
      next.selected[v] =
          v < original ? selected[v]
          : surfaceSelection_
              ? 0
              : std::all_of(carrier[v].begin(), carrier[v].end(),
                            [&](unsigned int corner) {
                              return corner == kNone || selected[corner];
                            });
    if (surfaceSelection_)
      for (size_t f = 0; f < faces.size(); ++f)
        if (next.selectedSurfaces_[f])
          for (const auto v : faces[f].vertices) next.selected[v] = 1;
    *this = std::move(next);
  }
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
  void cadRegion(const Condition& c, const std::vector<int>& surfaces,
                 std::vector<unsigned int>& nodes,
                 std::vector<size_t>& faces) const {
    faces.clear();
    nodes.clear();
    std::vector<unsigned char> found(surfaces.size(), 0);
    for (size_t f = 0; f < faceSurfaces_.size(); ++f) {
      const auto at =
          std::lower_bound(surfaces.begin(), surfaces.end(), faceSurfaces_[f]);
      if (at == surfaces.end() || *at != faceSurfaces_[f]) continue;
      found[size_t(at - surfaces.begin())] = 1;
      faces.push_back(f);
      for (const auto v : boundaryFaces()[f].vertices) nodes.push_back(v);
    }
    for (size_t i = 0; i < surfaces.size(); ++i)
      if (!found[i])
        throw std::runtime_error(
            "CAD face " + std::to_string(surfaces[i]) + " of '" + c.name +
            "' is smaller than the mesh; reduce the maximum element edge");
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  }
  void transferCadFaces(std::shared_ptr<const SurfaceGeometry> reference) {
    std::vector<int> tags = reference->parts;
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    std::vector<std::vector<size_t>> groups(tags.size() + 1);
    for (size_t f = 0; f < reference->faces.size(); ++f) {
      groups[size_t(std::lower_bound(tags.begin(), tags.end(),
                                     reference->parts[f]) -
                    tags.begin())]
          .push_back(f);
      groups.back().push_back(f);
    }
    std::vector<SurfaceProjection> trees;
    trees.reserve(groups.size());
    for (const auto& group : groups)
      trees.emplace_back(reference->mesh, reference->faces, group);
    const auto& x = mesh().vertices;
    std::vector<int> surfaces(boundaryFaces().size());
    for (size_t f = 0; f < surfaces.size(); ++f) {
      const auto& ids = boundaryFaces()[f].vertices;
      const Eigen::Vector3d a = x[ids[0]].e(), b = x[ids[1]].e(),
                            c = x[ids[2]].e();
      const Eigen::Vector3d center = (a + b + c) / 3.,
                            normal = (b - a).cross(c - a).normalized();
      const auto tag =
          std::lower_bound(tags.begin(), tags.end(), faceParts()[f]);
      size_t group = groups.size() - 1;
      if (tag != tags.end() && *tag == faceParts()[f])
        group = size_t(tag - tags.begin());
      auto nearest = trees[group].project(center, &normal);
      if (!std::isfinite(nearest.distance))
        nearest = trees[group].project(center);
      if (!std::isfinite(nearest.distance))
        throw std::runtime_error("Cannot transfer CAD faces to the mesh");
      surfaces[f] = reference->brepFaces[groups[group][nearest.order]];
    }
    brepReference_ = std::move(reference);
    faceSurfaces_ = std::move(surfaces);
  }
  mutable std::shared_ptr<const RegionGeometry> referenceGeometry_;
  std::shared_ptr<const SurfaceGeometry> surface_;
  std::shared_ptr<const SurfaceGeometry> brepReference_;
  std::vector<int> faceSurfaces_;
  std::vector<int> boundaryParts_;
  FixtureCache fixtureCache_;
  std::unique_ptr<SurfaceSelector> selector_;
  std::vector<unsigned char> selectedSurfaces_;
  bool surfaceSelection_ = false;
  int refinementLevel_ = 0;
  std::vector<unsigned char> previousMask_;
};
}  // namespace fem_workbench
