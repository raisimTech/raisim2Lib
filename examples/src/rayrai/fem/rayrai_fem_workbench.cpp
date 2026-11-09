// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#if !defined(__EMSCRIPTEN__)
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "example_resources.hpp"
#include "fem_workbench_import.hpp"
#include "fem_workbench_model.hpp"
#include "fem_workbench_view.hpp"
#include "raisim/World.hpp"
#include "raisim/fem/MeshResolution.hpp"
#include "rayrai/AssimpMeshLoader.hpp"
#include "rayrai/example_common.hpp"

namespace {
using fem_workbench::Condition;
using fem_workbench::Kind;
using fem_workbench::Model;
using raisim::Vec;
using raisim::fem::TetMesh;

glm::vec3 vector(const Vec<3>& v) {
  return {float(v[0]), float(v[1]), float(v[2])};
}
glm::vec4 heat(double value) {
  if (!std::isfinite(value)) value = 0.;
  const float t = float(std::clamp(value, 0., 1.));
  const glm::vec3 colors[] = {{.08f, .18f, .55f},
                              {.05f, .65f, .86f},
                              {.26f, .83f, .57f},
                              {.98f, .83f, .22f},
                              {.92f, .18f, .10f}};
  const int i = std::min(3, int(t * 4));
  return glm::vec4(glm::mix(colors[i], colors[i + 1], t * 4 - i), 1.f);
}
std::string extension(const std::string& path) {
  auto e = raisim::pathFromUtf8(path).extension().string();
  std::transform(e.begin(), e.end(), e.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  return e;
}

struct Workbench {
  Model model;
  std::shared_ptr<raisin::RayraiWindow> viewer;
  std::shared_ptr<raisin::Visuals> solid;
  std::vector<Vertex> vertices;
  std::unique_ptr<fem_workbench::MeshJob> meshJob;
  fem_workbench::GeometrySource pendingSource;
  bool previewJob = false, pendingGmsh = false;
  double pendingEdge = 0., pendingTolerance = 0., surfaceTolerance = 0.,
         actualMaximumEdge = 0.;
  std::string importName,
      message = "Select a surface, add supports and loads, then Solve.";
  std::array<char, 2048> file{}, project{}, exportPath{}, gmsh{},
      nativeMesher{};
  double importScale = 1., elementSize = .05, pendingScale = 1.,
         deformation = 100., colorMin = 0., colorMax = 1.;
  double boundsMin[3]{}, boundsMax[3]{};
  double plane = 0., tolerance = .0001;
  int planeAxis = 0, selectionMode = 0, field = 0, edited = -1;
  Vec<3> cylinderCenter{0., 0., 0.}, cylinderAxis{0., 0., 1.};
  double cylinderRadius = .05, cylinderMinimum = -1., cylinderMaximum = 1.;
  bool autoSolve = false, autoRange = true, edges = false, undeformed = false,
       visualDirty = true, append = false, externalGmsh = false;
  ImVec2 boxStart{};
  bool selectingBox = false;
  glm::mat4 screenMatrix{1.f};
  float sidebarWidth = 380.f;

  Workbench() {
    const auto executable = fem_workbench::defaultGmshExecutable();
    std::snprintf(gmsh.data(), gmsh.size(), "%s", executable.c_str());
    std::snprintf(project.data(), project.size(), "%s", "analysis.rfem");
    std::snprintf(exportPath.data(), exportPath.size(), "%s", "analysis.vtk");
  }
  template <class F>
  void attempt(F&& operation) {
    try {
      operation();
    } catch (const std::exception& e) {
      message = e.what();
    }
  }
  void fit() {
    const auto& x = model.mesh().vertices;
    for (int a = 0; a < 3; ++a) {
      boundsMin[a] = boundsMax[a] = x[0][a];
      for (const auto& p : x) {
        boundsMin[a] = std::min(boundsMin[a], p[a]);
        boundsMax[a] = std::max(boundsMax[a], p[a]);
      }
    }
    const glm::vec3 low{float(boundsMin[0]), float(boundsMin[1]),
                        float(boundsMin[2])};
    const glm::vec3 high{float(boundsMax[0]), float(boundsMax[1]),
                         float(boundsMax[2])};
    auto& cam = viewer->getCamera();
    const float extent = std::max(glm::length(high - low), .001f);
    fem_workbench::frameMeshCamera(cam, low, high);
    plane = boundsMin[planeAxis];
    tolerance = extent * .001;
    cylinderCenter = {(boundsMin[0] + boundsMax[0]) * .5,
                      (boundsMin[1] + boundsMax[1]) * .5,
                      (boundsMin[2] + boundsMax[2]) * .5};
    cylinderRadius =
        std::min(boundsMax[0] - boundsMin[0], boundsMax[1] - boundsMin[1]) *
        .25;
    cylinderMinimum = -extent;
    cylinderMaximum = extent;
  }
  void replaceVisual(bool frame = true) {
    if (solid) {
      viewer->removeVisualObject("fem_solid");
      solid.reset();
    }
    vertices.clear();
    actualMaximumEdge = 0.;
    if (model.solver) {
      double squared = 0.;
      for (const auto& cell : model.mesh().tetrahedra)
        for (size_t a = 0; a < 4; ++a)
          for (size_t b = 0; b < a; ++b)
            squared = std::max(squared, (model.mesh().vertices[cell[a]].e() -
                                         model.mesh().vertices[cell[b]].e())
                                            .squaredNorm());
      actualMaximumEdge = std::sqrt(squared);
    }
    visualDirty = true;
    if (frame) {
      fit();
      if (!(model.source.referenceExtent > 0.))
        model.source.referenceExtent =
            std::hypot(boundsMax[0] - boundsMin[0], boundsMax[1] - boundsMin[1],
                       boundsMax[2] - boundsMin[2]);
    }
  }
  double effectiveTolerance() const {
    if (model.source.volume) return 0.;
    if (surfaceTolerance > 0.) return surfaceTolerance;
    return std::min(elementSize * .01, model.source.referenceExtent * .001);
  }
  bool needsMesh() const {
    return !model.solver || model.source.meshedEdge != elementSize ||
           model.source.meshedTolerance != effectiveTolerance() ||
           model.source.meshedExternalGmsh !=
               (externalGmsh && !model.source.volume);
  }
  void syncSettings() {
    model.source.maximumEdge = elementSize;
    model.source.surfaceTolerance = surfaceTolerance;
    model.source.externalGmsh = externalGmsh;
  }
  void finishSolve() {
    model.solve();
    visualDirty = true;
    const auto& s = model.solver->statistics();
    char text[256];
    std::snprintf(
        text, sizeof(text), "Solved: %.2f ms%s | residual %.2e",
        s.assemblyMs + s.factorizationMs + s.solveMs,
        s.reusedFactorization
            ? (s.iterative ? " (cached preconditioner)" : " (cached factor)")
            : "",
        s.relativeResidual);
    message = text;
  }
  void solve() {
    if (meshJob)
      throw std::runtime_error(previewJob
                                   ? "Wait for the geometry to load"
                                   : "Volume meshing is already running");
    if (!model.hasGeometry()) throw std::runtime_error("Open a solid first");
    if (!(elementSize > 0.) || !std::isfinite(elementSize) ||
        surfaceTolerance < 0. || !std::isfinite(surfaceTolerance))
      throw std::runtime_error(
          "Element size must be positive; surface tolerance must be finite and "
          "nonnegative");
    syncSettings();
    if (!needsMesh()) {
      finishSolve();
      return;
    }
    auto job = std::make_unique<fem_workbench::MeshJob>();
    pendingSource = model.source;
    if (pendingSource.bytes.empty() && !pendingSource.volume) {
      if (model.solver)
        pendingSource.volume = std::make_shared<TetMesh>(model.mesh());
      else {
        fem_workbench::SurfaceGeometry surface;
        surface.mesh.vertices = model.mesh().vertices;
        surface.faces = model.boundaryFaces();
        surface.parts = model.faceParts();
        pendingSource.bytes = surface.encode();
        pendingSource.extension = ".rfs";
        pendingSource.scale = 1.;
      }
      model.source = pendingSource;
    }
    std::string ext = pendingSource.volume ? ".rvm" : pendingSource.extension;
    auto bytes = pendingSource.volume
                     ? fem_workbench::encodeVolume(*pendingSource.volume)
                     : pendingSource.bytes;
    if (externalGmsh && ext == ".rfs") {
      const auto surface = fem_workbench::SurfaceGeometry::decode(bytes);
      if (*std::max_element(surface.parts.begin(), surface.parts.end()) > 1)
        throw std::runtime_error(
            "Use the built-in mesher for separate surface parts, or import a "
            "STEP assembly for Gmsh");
      std::ostringstream stl;
      stl.imbue(std::locale::classic());
      stl << std::setprecision(17) << "solid source\n";
      for (const auto& face : surface.faces) {
        stl << "facet normal 0 0 0\nouter loop\n";
        for (const auto v : face.vertices) {
          const auto& p = surface.mesh.vertices[v];
          stl << "vertex " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
        }
        stl << "endloop\nendfacet\n";
      }
      stl << "endsolid source\n";
      bytes = stl.str();
      ext = ".stl";
    }
    std::ofstream output(job->directory / ("source" + ext), std::ios::binary);
    output.write(bytes.data(), std::streamsize(bytes.size()));
    output.close();
    if (!output) throw std::runtime_error("Cannot prepare source geometry");
    pendingEdge = elementSize;
    pendingTolerance = effectiveTolerance();
    const bool gmshJob = externalGmsh && !pendingSource.volume;
    pendingGmsh = gmshJob;
    job->start(gmshJob ? gmsh.data() : nativeMesher.data(), ext,
               pendingEdge / pendingSource.scale,
               gmshJob ? fem_workbench::MesherBackend::Gmsh
                       : fem_workbench::MesherBackend::Native,
               pendingTolerance / pendingSource.scale);
    previewJob = false;
    meshJob = std::move(job);
    model.changed();
    visualDirty = true;
    message =
        "Meshing at the current resolution; supports and loads will be "
        "retained...";
  }
  void demo() {
    meshJob.reset();
    model.demo();
    elementSize = .06;
    surfaceTolerance = 0.;
    edited = 1;
    replaceVisual();
    model.source.volume = std::make_shared<TetMesh>(model.mesh());
    syncSettings();
    model.source.meshedEdge = elementSize;
    model.source.meshedTolerance = effectiveTolerance();
    message = "Demo solved. Change the tip force to reuse the factorization.";
  }
  void import(const std::string& path) {
    if (!std::isfinite(importScale) || importScale <= 0.)
      throw std::runtime_error(
          "Import scale must be positive (metres per file unit)");
    const auto ext = extension(path);
    const auto title = raisim::pathUtf8(raisim::pathFromUtf8(path).filename());
    if (ext == ".rfem") {
      Model next;
#if defined(__EMSCRIPTEN__)
      next.loadProject(path, false);
#else
      next.loadProject(path);
#endif
      meshJob.reset();
      model = std::move(next);
      elementSize = model.source.maximumEdge;
      surfaceTolerance = model.source.surfaceTolerance;
      externalGmsh = model.source.externalGmsh;
      edited = -1;
      replaceVisual();
      if (model.solver && !model.source.volume && model.source.bytes.empty())
        model.source.volume = std::make_shared<TetMesh>(model.mesh());
      message = "Project loaded. Solve uses its current meshing settings.";
      return;
    }
    if (ext == ".msh" || ext == ".mesh" || ext == ".vtk") {
      auto mesh = TetMesh::load(path, importScale);
      meshJob.reset();
      model.setMesh(std::move(mesh), title);
      model.source.volume = std::make_shared<TetMesh>(model.mesh());
      syncSettings();
      edited = -1;
      replaceVisual();
      message =
          "Volume displayed. Solve refines to the selected maximum edge; "
          "existing regions and interfaces are retained.";
      return;
    }
    const bool cad = ext == ".step" || ext == ".stp" || ext == ".brep" ||
                     ext == ".iges" || ext == ".igs";
    if (!cad && ext != ".stl" && ext != ".obj" && ext != ".ply" &&
        ext != ".glb" && ext != ".gltf" && ext != ".dae")
      throw std::runtime_error(
          "Supported: .msh .mesh .vtk .rfem; .stl .obj .ply .glb .gltf .dae; "
          "CAD .step .stp .brep .iges .igs");
    if (cad) {
      const auto size = std::filesystem::file_size(raisim::pathFromUtf8(path));
      if (size > 128u * 1024u * 1024u)
        throw std::runtime_error("CAD file exceeds 128 MB");
      std::ifstream input(raisim::pathFromUtf8(path), std::ios::binary);
      fem_workbench::GeometrySource source;
      source.extension = ext;
      source.scale = ext == ".brep" ? importScale : 1.;
      source.maximumEdge = elementSize;
      source.surfaceTolerance = surfaceTolerance;
      source.bytes.assign(std::istreambuf_iterator<char>(input), {});
      if (source.bytes.size() != size)
        throw std::runtime_error("Cannot read CAD file");
      auto job = std::make_unique<fem_workbench::MeshJob>();
      std::ofstream out(job->directory / ("source" + ext), std::ios::binary);
      out.write(source.bytes.data(), std::streamsize(source.bytes.size()));
      out.close();
      if (!out) throw std::runtime_error("Cannot prepare CAD preview");
      meshJob.reset();
      job->start(nativeMesher.data(), ext, 1.,
                 fem_workbench::MesherBackend::Native,
                 source.surfaceTolerance / source.scale, true);
      pendingSource = std::move(source);
      previewJob = true;
      meshJob = std::move(job);
      importName = title;
      message = "Reading CAD surfaces for display...";
      return;
    }
    const auto meshes = raisin::assimp::loadModel(path);
    if (!meshes || meshes->empty())
      throw std::runtime_error("Surface import produced no triangles");
    fem_workbench::SurfaceGeometry surface;
    for (const auto& mesh : *meshes) {
      const auto offset = unsigned(surface.mesh.vertices.size());
      for (const auto& v : mesh->vertices)
        surface.mesh.vertices.push_back({double(v.Position.x) * importScale,
                                         double(v.Position.y) * importScale,
                                         double(v.Position.z) * importScale});
      for (size_t f = 0; f + 2 < mesh->indices.size(); f += 3) {
        surface.faces.push_back(
            {{offset + mesh->indices[f], offset + mesh->indices[f + 1],
              offset + mesh->indices[f + 2]},
             0});
        surface.parts.push_back(1);
      }
    }
    surface.normalize();
    const auto bytes = surface.encode();
    meshJob.reset();
    model.setSurface(std::move(surface), title);
    model.source.extension = ".rfs";
    model.source.bytes = bytes;
    model.source.scale = 1.;
    syncSettings();
    edited = -1;
    replaceVisual();
    message =
        "Surface displayed. Define supports and loads, set resolution, then "
        "Solve.";
  }
  void pollMesh() {
    if (!meshJob || !meshJob->poll()) return;
    auto job = std::move(meshJob);
    if (previewJob) {
      auto surface = job->surfaceResult(pendingSource.scale);
      model.setSurface(std::move(surface), importName);
      model.source = std::move(pendingSource);
      edited = -1;
      replaceVisual();
      model.source.recordPreviewAccuracy();
      syncSettings();
      message =
          "CAD displayed. Define supports and loads, set resolution, then "
          "Solve.";
      return;
    }
    if (elementSize != pendingEdge ||
        effectiveTolerance() != pendingTolerance ||
        pendingGmsh != (externalGmsh && !pendingSource.volume)) {
      message =
          "Meshing settings changed. Press Solve to generate the new "
          "resolution.";
      return;
    }
    auto volume = job->result(pendingSource.scale);
    if (pendingGmsh)
      volume = raisim::fem::refineToMaximumEdge(std::move(volume), pendingEdge);
    const double diagonal =
        std::hypot(boundsMax[0] - boundsMin[0], boundsMax[1] - boundsMin[1],
                   boundsMax[2] - boundsMin[2]);
    const double mappingTolerance =
        std::max({diagonal * 1e-8, pendingTolerance * 4.,
                  model.source.meshedTolerance * 4.,
                  model.source.previewTolerance * 2.});
    model.setMeshed(std::move(volume), mappingTolerance, pendingTolerance);
    model.source.meshedEdge = pendingEdge;
    model.source.meshedTolerance = pendingTolerance;
    model.source.meshedExternalGmsh = pendingGmsh;
    replaceVisual(false);
    finishSolve();
  }
  void pollBrowserErrors() {
#if defined(__EMSCRIPTEN__)
    if (EM_ASM_INT({ return !!globalThis['raisimFemSolveRequest']; })) {
      const bool configureOnly = EM_ASM_INT(
          { return !!globalThis['raisimFemSolveRequest']['configureOnly']; });
      double settings[2]{elementSize, surfaceTolerance};
      EM_ASM(
          {
            const request = globalThis['raisimFemSolveRequest'];
            if (typeof request['maximumEdge'] != 'undefined')
              HEAPF64[$0 / 8] = Number(request['maximumEdge']);
            if (typeof request['surfaceTolerance'] != 'undefined')
              HEAPF64[$0 / 8 + 1] = Number(request['surfaceTolerance']);
            delete globalThis['raisimFemSolveRequest'];
          },
          settings);
      if (!(settings[0] > 0.) || !std::isfinite(settings[0]) ||
          settings[1] < 0. || !std::isfinite(settings[1])) {
        message =
            "Maximum edge must be positive and surface tolerance must be "
            "nonnegative and finite";
        return;
      }
      elementSize = settings[0];
      surfaceTolerance = settings[1];
      if (configureOnly) {
        syncSettings();
        model.changed();
        visualDirty = true;
      } else
        attempt([&] { solve(); });
    }
    const int size = EM_ASM_INT({
      const error = globalThis['raisimFemError'];
      return error ? new TextEncoder().encode(error).length : 0;
    });
    if (size > 0 && size < 16384) {
      std::string error(static_cast<size_t>(size), '\0');
      EM_ASM(
          {
            HEAPU8.set(new TextEncoder().encode(globalThis['raisimFemError']),
                       $0);
            delete globalThis['raisimFemError'];
          },
          error.data());
      message = std::move(error);
    }
    EM_ASM(
        {
          if (globalThis['raisimFemState']) {
            const state = globalThis['raisimFemState'];
            state['message'] = UTF8ToString($0);
            state['meshing'] = !!$1;
            state['preview'] = !!$2;
            state['requestedEdge'] = $3;
            state['meshedEdge'] = $4;
            state['surfaceTolerance'] = $5;
            state['actualMaximumEdge'] = $6;
          }
        },
        message.c_str(), int(bool(meshJob)), int(!model.solver), elementSize,
        model.source.meshedEdge, effectiveTolerance(), actualMaximumEdge);
#endif
  }
  double scalar(size_t element) const {
    if (!model.hasSolution) return 0.;
    const auto& r = model.solver->elementResults()[element];
    if (field == 0) return r.vonMises;
    if (field == 1) {
      double sum = 0.;
      for (const auto v : model.mesh().tetrahedra[element])
        sum += .25 * model.solver->displacements()[v].e().norm();
      return sum;
    }
    if (field == 2) return r.principalStress[2];
    if (field == 3) return r.principalStress[0];
    if (field < 10) return r.stress[field - 4];
    return r.strain[field - 10];
  }
  glm::vec3 position(unsigned int node, bool deformed = true) const {
    auto p = vector(model.mesh().vertices[node]);
    if (deformed && model.hasSolution)
      p += float(deformation) * vector(model.solver->displacements()[node]);
    return p;
  }
  void updateVisual() {
    if (!visualDirty || !model.hasGeometry()) return;
    if (!std::isfinite(deformation) ||
        !autoRange && (!std::isfinite(colorMin) || !std::isfinite(colorMax) ||
                       colorMax <= colorMin))
      throw std::runtime_error(
          "Display scale and color limits must be finite; maximum must exceed "
          "minimum");
    const auto& faces = model.boundaryFaces();
    if (autoRange) {
      colorMin = std::numeric_limits<double>::infinity();
      colorMax = -colorMin;
      for (const auto& f : faces) {
        const double s = scalar(f.element);
        colorMin = std::min(colorMin, s);
        colorMax = std::max(colorMax, s);
      }
      if (field == 0 || field == 1) colorMin = 0.;
    }
    vertices.resize(3 * faces.size());
    for (size_t f = 0; f < faces.size(); ++f) {
      const auto& ids = faces[f].vertices;
      const glm::vec3 p[3] = {position(ids[0]), position(ids[1]),
                              position(ids[2])};
      const auto cross = glm::cross(p[1] - p[0], p[2] - p[0]);
      const glm::vec3 normal =
          glm::length(cross) > 0.f ? glm::normalize(cross) : glm::vec3(0, 0, 1);
      const glm::vec4 color = model.hasSolution
                                  ? heat((scalar(faces[f].element) - colorMin) /
                                         std::max(colorMax - colorMin, 1e-30))
                                  : glm::vec4(.36f, .57f, .77f, 1.f);
      for (size_t j = 0; j < 3; ++j) {
        auto& v = vertices[3 * f + j];
        v.Position = p[j];
        v.Normal = normal;
        v.Color = color;
      }
    }
    if (!solid) {
      std::vector<unsigned int> indices(vertices.size());
      std::iota(indices.begin(), indices.end(), 0u);
      auto meshes =
          std::make_shared<std::vector<std::shared_ptr<OpenGLMesh>>>();
      meshes->push_back(std::make_shared<OpenGLMesh>(vertices, indices,
                                                     std::vector<Texture>{}));
      solid = viewer->addVisualCustomMesh("fem_solid", meshes, glm::vec4(1.f),
                                          false);
      solid->setUseMeshColor(true);
      solid->setDetectable(false);
      raisin::Material appearance;
      appearance.useVertexColorAsAlbedo = true;
      appearance.roughnessFactor = 1.f;
      appearance.unlit = false;
      solid->setMaterialOverride(appearance);
    } else if (!solid->updateSingleMeshVertices(vertices))
      throw std::runtime_error("Cannot update FEM surface buffer");
    visualDirty = false;
#if defined(__EMSCRIPTEN__)
    auto regions = model.faceParts();
    std::sort(regions.begin(), regions.end());
    regions.erase(std::unique(regions.begin(), regions.end()), regions.end());
    EM_ASM(
        {
          const state = globalThis['raisimFemState'] || {};
          state['name'] = UTF8ToString($0);
          state['nodes'] = $1;
          state['tetrahedra'] = $2;
          state['solved'] = !!$3;
          state['residual'] = $4;
          state['energy'] = $5;
          state['regions'] = $6;
          globalThis['raisimFemState'] = state;
        },
        model.name.c_str(), model.selected.size(),
        model.mesh().tetrahedra.size(), model.hasSolution,
        model.hasSolution ? model.solver->statistics().relativeResidual : 0.,
        model.hasSolution ? model.solver->statistics().strainEnergy : 0.,
        regions.size());
#endif
  }
  glm::mat4 viewProjection() const {
    return viewer->getCamera().getProjectionMatrix() *
           viewer->getCamera().getViewMatrix();
  }
  bool projectPoint(const glm::vec3& p, const ExampleApp::Viewport& vp,
                    ImVec2& screen) const {
    const glm::vec4 clip = screenMatrix * glm::vec4(p, 1.f);
    if (clip.w <= 0.f) return false;
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.z < -1.f || ndc.z > 1.f) return false;
    screen = ImVec2(vp.origin.x + (ndc.x + 1.f) * .5f * vp.size.x,
                    vp.origin.y + (1.f - ndc.y) * .5f * vp.size.y);
    return true;
  }
  int pick(const ExampleApp::Viewport& vp) const {
    if (vp.width <= 0 || vp.height <= 0) return -1;
    const glm::mat4 inverse = glm::inverse(viewProjection());
    glm::vec4 a =
        inverse * glm::vec4(float(2. * vp.x / vp.width - 1.),
                            float(1. - 2. * vp.y / vp.height), -1.f, 1.f);
    glm::vec4 b =
        inverse * glm::vec4(float(2. * vp.x / vp.width - 1.),
                            float(1. - 2. * vp.y / vp.height), 1.f, 1.f);
    const glm::vec3 origin = glm::vec3(a) / a.w,
                    direction = glm::normalize(glm::vec3(b) / b.w - origin);
    float nearest = std::numeric_limits<float>::infinity();
    int face = -1;
    for (size_t f = 0; f < vertices.size() / 3; ++f) {
      const auto p = vertices[3 * f].Position;
      const auto e1 = vertices[3 * f + 1].Position - p,
                 e2 = vertices[3 * f + 2].Position - p;
      const auto cross = glm::cross(direction, e2);
      const float determinant = glm::dot(e1, cross);
      if (std::abs(determinant) < 1e-12f) continue;
      const float inv = 1.f / determinant;
      const auto t = origin - p;
      const float u = glm::dot(t, cross) * inv;
      if (u < 0.f || u > 1.f) continue;
      const auto q = glm::cross(t, e1);
      const float v = glm::dot(direction, q) * inv;
      const float distance = glm::dot(e2, q) * inv;
      if (v >= 0.f && u + v <= 1.f && distance > 0.f && distance < nearest) {
        nearest = distance;
        face = int(f);
      }
    }
    return face;
  }
  void selectFace(int f, bool patch, bool add) {
    if (f < 0) return;
    const auto mode = selectionMode == 4
                          ? fem_workbench::SurfaceSelection::Smooth
                      : patch ? fem_workbench::SurfaceSelection::Planar
                              : fem_workbench::SurfaceSelection::Triangle;
    model.selectSurface(static_cast<size_t>(f), mode, add);
  }
  void input(const ExampleApp::Viewport& vp) {
    screenMatrix = viewProjection();
    if (!vp.hovered || ImGui::GetIO().MousePos.x < sidebarWidth + 20.f ||
        !model.hasGeometry())
      return;
    const bool add = append || ImGui::GetIO().KeyShift;
    if (selectionMode == 3) {
      if (ImGui::IsMouseClicked(0)) {
        boxStart = ImGui::GetIO().MousePos;
        selectingBox = true;
      }
      if (selectingBox && ImGui::IsMouseReleased(0)) {
        const auto end = ImGui::GetIO().MousePos;
        if (!add) model.clearSelection();
        model.useNodeSelection();
        for (unsigned int i = 0; i < model.selected.size(); ++i) {
          ImVec2 p;
          if (projectPoint(position(i), vp, p) &&
              p.x >= std::min(boxStart.x, end.x) &&
              p.x <= std::max(boxStart.x, end.x) &&
              p.y >= std::min(boxStart.y, end.y) &&
              p.y <= std::max(boxStart.y, end.y))
            model.selected[i] = 1;
        }
        selectingBox = false;
      }
    } else if (selectionMode != 0 && ImGui::IsMouseClicked(0))
      selectFace(pick(vp), selectionMode == 2, add);
  }
  void overlays(const ExampleApp::Viewport& vp) {
    if (!model.hasGeometry()) return;
    screenMatrix = viewProjection();
    auto* draw = ImGui::GetForegroundDrawList();
    draw->PushClipRect(ImVec2(sidebarWidth + 25.f, 60.f),
                       ImVec2(vp.size.x - 15.f, vp.size.y - 74.f), true);
    const auto& faces = model.boundaryFaces();
    if (edges && faces.size() <= 18000)
      for (size_t f = 0; f < faces.size(); ++f) {
        ImVec2 p[3];
        bool visible = true;
        for (int j = 0; j < 3; ++j)
          visible &= projectPoint(vertices[3 * f + j].Position, vp, p[j]);
        if (visible)
          for (int j = 0; j < 3; ++j)
            draw->AddLine(p[j], p[(j + 1) % 3], IM_COL32(8, 18, 30, 45), .6f);
      }
    if (undeformed && model.hasSolution)
      for (const auto& face : faces) {
        ImVec2 p[3];
        bool visible = true;
        for (int j = 0; j < 3; ++j)
          visible &= projectPoint(position(face.vertices[j], false), vp, p[j]);
        if (visible)
          for (int j = 0; j < 3; ++j)
            draw->AddLine(p[j], p[(j + 1) % 3], IM_COL32(150, 175, 205, 25),
                          .5f);
      }
    for (unsigned int i = 0; i < model.selected.size(); ++i)
      if (model.selected[i]) {
        ImVec2 p;
        if (projectPoint(position(i), vp, p))
          draw->AddCircleFilled(p, 3.f, IM_COL32(255, 214, 80, 255));
      }
    for (const auto& c : model.conditions)
      if (c.enabled && !c.nodes.empty()) {
        const size_t stride = std::max<size_t>(1, c.nodes.size() / 35);
        for (size_t i = 0; i < c.nodes.size(); i += stride) {
          ImVec2 p;
          if (!projectPoint(position(c.nodes[i]), vp, p)) continue;
          if (c.kind != Kind::Force && c.kind != Kind::Traction &&
              c.kind != Kind::Pressure && c.kind != Kind::Gravity) {
            draw->AddTriangleFilled(ImVec2(p.x, p.y), ImVec2(p.x - 5, p.y + 10),
                                    ImVec2(p.x + 5, p.y + 10),
                                    IM_COL32(75, 226, 189, 255));
          } else if (c.kind == Kind::Force || c.kind == Kind::Traction) {
            const auto unit = c.value.e().norm();
            if (unit == 0.) continue;
            const auto extent = float(std::max({boundsMax[0] - boundsMin[0],
                                                boundsMax[1] - boundsMin[1],
                                                boundsMax[2] - boundsMin[2]}));
            ImVec2 end;
            if (projectPoint(position(c.nodes[i]) +
                                 extent * .08f * vector(c.value) / float(unit),
                             vp, end)) {
              draw->AddLine(p, end, IM_COL32(255, 174, 90, 255), 2.f);
              const float dx = end.x - p.x, dy = end.y - p.y,
                          length = std::sqrt(dx * dx + dy * dy);
              if (length > 1.f)
                draw->AddTriangleFilled(
                    end,
                    ImVec2(end.x - 7 * dx / length + 3 * dy / length,
                           end.y - 7 * dy / length - 3 * dx / length),
                    ImVec2(end.x - 7 * dx / length - 3 * dy / length,
                           end.y - 7 * dy / length + 3 * dx / length),
                    IM_COL32(255, 174, 90, 255));
            }
          }
        }
      }
    if (selectingBox)
      draw->AddRect(boxStart, ImGui::GetIO().MousePos,
                    IM_COL32(255, 214, 80, 255), 0.f, 0, 1.5f);
    draw->PopClipRect();
  }
  bool editVector(const char* name, Vec<3>& value, const char* unit = "") {
    bool changed = false;
    for (int a = 0; a < 3; ++a) {
      char label[80];
      std::snprintf(label, sizeof(label), "%s %c%s", name, 'X' + a, unit);
      changed |= ImGui::InputDouble(label, &value[a], 0., 0., "%.6g");
    }
    return changed;
  }
  bool editFixture(Condition& c) {
    bool changed = false;
    if (c.faces.empty() && !c.nodes.empty())
      ImGui::TextWrapped(
          "Point selection: remeshing keeps these locations. Select a surface "
          "for a continuous support.");
    if (c.kind == Kind::Directional || c.kind == Kind::Slider ||
        c.kind == Kind::Bearing || c.kind == Kind::BoltHole ||
        c.kind == Kind::Spring) {
      ImGui::TextDisabled("Fixture direction / cylinder axis");
      if (ImGui::Button("Along X")) {
        c.axis = {1., 0., 0.};
        changed = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Along Y")) {
        c.axis = {0., 1., 0.};
        changed = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Along Z")) {
        c.axis = {0., 0., 1.};
        changed = true;
      }
      changed |= editVector("Axis", c.axis);
    }
    if (c.kind == Kind::Bearing || c.kind == Kind::BoltHole ||
        c.kind == Kind::Remote)
      changed |= editVector("Center", c.center, " (m)");
    if (c.kind == Kind::Bearing || c.kind == Kind::BoltHole) {
      changed |= ImGui::Checkbox("Radial restraint", &c.axes[0]);
      changed |= ImGui::Checkbox(
          c.kind == Kind::Bearing ? "Hold axial center" : "Restrain axial seat",
          &c.axes[1]);
      changed |= ImGui::Checkbox(c.kind == Kind::Bearing
                                     ? "Restrain roll"
                                     : "Restrain shaft tangential motion",
                                 &c.axes[2]);
      if (c.axes[0])
        changed |= ImGui::InputDouble("Radial displacement (m)", &c.value[0],
                                      0., 0., "%.6g");
      if (c.axes[1])
        changed |= ImGui::InputDouble("Axial displacement (m)", &c.value[1], 0.,
                                      0., "%.6g");
      if (c.axes[2])
        changed |= ImGui::InputDouble(c.kind == Kind::Bearing
                                          ? "Reference roll (rad)"
                                          : "Tangential displacement (m)",
                                      &c.value[2], 0., 0., "%.6g");
      if (c.kind == Kind::Bearing && !c.axes[2])
        ImGui::TextWrapped(
            "Roll about the axis is free. Other supports must remove any "
            "remaining rigid rotation for a unique static "
            "solution.");
    }
    if (c.kind == Kind::Directional || c.kind == Kind::Roller ||
        c.kind == Kind::Slider || c.kind == Kind::Spring)
      changed |= ImGui::InputDouble("Displacement / rest (m)", &c.value[0], 0.,
                                    0., "%.6g");
    if (c.kind == Kind::Roller)
      ImGui::TextWrapped(
          "Normal rollers follow the meshed surface. Use a "
          "cylindrical bearing when roll must remain free.");
    if (c.kind == Kind::Slider)
      changed |= ImGui::InputDouble("Second transverse offset (m)", &c.value[1],
                                    0., 0., "%.6g");
    if (c.kind == Kind::Spring)
      changed |= ImGui::InputDouble("Total stiffness (N/m)", &c.stiffness, 0.,
                                    0., "%.6g");
    if (c.kind == Kind::Remote || c.kind == Kind::Ball)
      changed |= editVector("Translation", c.value, " (m)");
    if (c.kind == Kind::Remote) {
      changed |= editVector("Rotation", c.rotation, " (rad)");
      for (int a = 0; a < 3; ++a) {
        if (a) ImGui::SameLine();
        changed |= ImGui::Checkbox(a == 0   ? "Constrain X"
                                   : a == 1 ? "Constrain Y"
                                            : "Constrain Z",
                                   &c.axes[a]);
      }
    }
    if (c.kind == Kind::Tie)
      changed |= ImGui::InputDouble("Projection tolerance (m)", &c.tolerance,
                                    0., 0., "%.6g");
    if (c.kind == Kind::BoltConnector) {
      changed |= ImGui::InputDouble("Bolt diameter (m)", &c.bolt.diameter, 0.,
                                    0., "%.6g");
      double gpa = c.bolt.youngsModulus / 1e9;
      if (ImGui::InputDouble("Bolt modulus (GPa)", &gpa, 0., 0., "%.6g")) {
        c.bolt.youngsModulus = gpa * 1e9;
        changed = true;
      }
      changed |= ImGui::InputDouble("Bolt Poisson ratio", &c.bolt.poissonRatio,
                                    0., 0., "%.6g");
      changed |= ImGui::InputDouble("Grip length, 0 = auto (m)",
                                    &c.bolt.gripLength, 0., 0., "%.6g");
      changed |= ImGui::InputDouble("Reference preload (N)", &c.bolt.preload,
                                    0., 0., "%.6g");
      ImGui::TextWrapped(
          "Distributed circular beam connector: axial, shear, bending and "
          "torsion. Reference preload is the tension at "
          "zero relative head motion; the installed force depends on joint "
          "compliance.");
      if (model.hasSolution)
        for (size_t b = 0; b < model.boltConditions.size(); ++b)
          if (model.boltConditions[b] == size_t(edited)) {
            const auto& r = model.solver->boltResults()[b];
            ImGui::Text("Bolt tension: %.6g N", r.axis.e().dot(r.force.e()));
            ImGui::Text("Head / nut moment: %.4g / %.4g Nm",
                        r.headMoment.e().norm(), r.nutMoment.e().norm());
          }
    }
    if (c.kind == Kind::BoltConnector || c.kind == Kind::BoltHole ||
        c.kind == Kind::Tie) {
      const char* label = c.kind == Kind::BoltConnector
                              ? "Capture second head/nut"
                          : c.kind == Kind::BoltHole ? "Capture seat region"
                                                     : "Capture master surface";
      if (ImGui::Button(label)) {
        c.targetNodes.clear();
        for (unsigned int i = 0; i < model.selected.size(); ++i)
          if (model.selected[i]) c.targetNodes.push_back(i);
        c.targetFaces = model.selectedFaces();
        changed = true;
      }
      ImGui::TextDisabled("Second region: %zu nodes / %zu faces",
                          c.targetNodes.size(), c.targetFaces.size());
      if (ImGui::Button("Show second region")) {
        model.restoreSelection(c.targetNodes, c.targetFaces);
      }
    }
    return changed;
  }
  void panel() {
    const auto display = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(16, 16));
    ImGui::SetNextWindowSize(ImVec2(sidebarWidth, display.y - 32));
    ImGui::Begin("FEM Workbench", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse);
    ImGui::PushItemWidth(115.f);
    ImGui::TextColored(ImVec4(.35f, .85f, .9f, 1.f), "LINEAR ELASTICITY");
    ImGui::TextWrapped("%s", model.name.c_str());
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.07f, .45f, .52f, 1.f));
    if (ImGui::Button("SOLVE", ImVec2(140, 30))) attempt([&] { solve(); });
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::Checkbox("Live solve", &autoSolve);
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Import solid",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::TextWrapped(
          "Drop a solid file onto this window, or enter its path.");
#if !defined(__EMSCRIPTEN__)
      ImGui::SetNextItemWidth(-1);
      ImGui::InputText("##source", file.data(), file.size());
#endif
      ImGui::InputDouble("Metres / file unit", &importScale, 0., 0., "%.6g");
      if (ImGui::InputDouble("Max element edge (m)", &elementSize, 0., 0.,
                             "%.5g")) {
        syncSettings();
        model.changed();
        visualDirty = true;
      }
      if (ImGui::Button("Import", ImVec2(90, 0)))
        attempt([&] {
#if defined(__EMSCRIPTEN__)
          EM_ASM({ globalThis['raisimFem'] ?.['open'](); });
#else
          import(file.data());
#endif
        });
      ImGui::SameLine();
      if (ImGui::Button("Demo beam")) attempt([&] { demo(); });
      if (ImGui::TreeNode("Mesher / file formats")) {
        ImGui::BeginDisabled(bool(model.source.volume));
        if (ImGui::InputDouble("Surface error, 0 = auto (m)", &surfaceTolerance,
                               0., 0., "%.5g")) {
          syncSettings();
          model.changed();
          visualDirty = true;
        }
        ImGui::EndDisabled();
        if (!model.source.volume)
          ImGui::TextDisabled("Surface tolerance: %.4g m",
                              effectiveTolerance());
        else
          ImGui::TextDisabled("Existing volume geometry is retained.");
#if defined(__EMSCRIPTEN__)
        ImGui::TextWrapped(
            "Open MSH, MESH, VTK, RFEM, or watertight STL/OBJ/PLY/glTF/DAE "
            "surfaces. Surface meshing runs locally with "
            "fTetWild (MPL-2.0). Include companion files when importing glTF "
            "or OBJ.");
#else
        ImGui::TextWrapped(
            "Opening a file displays its geometry. Solve generates tetrahedra "
            "at the maximum edge setting. Open CASCADE "
            "reads solid STEP/BREP/IGES files. Each CAD solid becomes a "
            "separate region. Set metres per unit for BREP and surface files. "
            "Imported volumes retain their region interfaces and refine to the "
            "requested maximum edge.");
        if (ImGui::Checkbox("Use external Gmsh", &externalGmsh)) {
          syncSettings();
          model.changed();
          visualDirty = true;
        }
        ImGui::SetNextItemWidth(-1);
        if (externalGmsh)
          ImGui::InputText("##gmsh", gmsh.data(), gmsh.size());
        else
          ImGui::InputText("##nativeMesher", nativeMesher.data(),
                           nativeMesher.size());
#endif
        ImGui::TreePop();
      }
      if (meshJob) {
        ImGui::TextColored(
            ImVec4(1.f, .8f, .3f, 1.f),
            previewJob ? "Reading CAD..." : "Meshing for solve...");
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
          meshJob.reset();
          message =
              previewJob
                  ? "CAD import cancelled."
                  : "Meshing cancelled; geometry and conditions retained.";
        }
      }
    }
    if (!model.hasGeometry()) {
      ImGui::PopItemWidth();
      ImGui::End();
      return;
    }
    ImGui::TextDisabled("%zu nodes   %zu triangles   %zu tetrahedra",
                        model.selected.size(), model.boundaryFaces().size(),
                        model.mesh().tetrahedra.size());
    if (model.solver)
      ImGui::TextDisabled("Mesh maximum edge: %.5g m", actualMaximumEdge);
    if (!model.solver)
      ImGui::TextWrapped(
          "Geometry preview. Volume meshing starts when you press Solve.");
    else if (needsMesh())
      ImGui::TextWrapped("Resolution changed. Solve will rebuild the mesh.");
    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
      double gpa = model.material.youngsModulus / 1e9;
      if (ImGui::InputDouble("Young's modulus (GPa)", &gpa, 0., 0., "%.4g")) {
        model.material.youngsModulus = gpa * 1e9;
        model.changed();
        visualDirty = true;
      }
      if (ImGui::InputDouble("Poisson ratio", &model.material.poissonRatio, 0.,
                             0., "%.4g")) {
        model.changed();
        visualDirty = true;
      }
      if (model.material.poissonRatio > .45)
        ImGui::TextWrapped(
            "Nearly incompressible material: linear tetrahedra "
            "can be too stiff. Check mesh convergence.");
      if (ImGui::InputDouble("Density (kg/m3)", &model.material.density, 0., 0.,
                             "%.5g")) {
        model.changed();
        visualDirty = true;
      }
    }
    if (ImGui::CollapsingHeader("Select region",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Combo("Tool", &selectionMode,
                   "Orbit / pan\0Triangle\0Planar patch\0Box (through "
                   "solid)\0Smooth surface\0");
      ImGui::Checkbox("Add to selection (or Shift)", &append);
      ImGui::TextDisabled(
          "Selected: %zu nodes, %zu faces",
          size_t(std::count(model.selected.begin(), model.selected.end(), 1)),
          model.selectedFaces().size());
      ImGui::Combo("Plane normal", &planeAxis, "X\0Y\0Z\0");
      ImGui::InputDouble("Coordinate (m)", &plane, 0., 0., "%.6g");
      ImGui::InputDouble("Tolerance (m)", &tolerance, 0., 0., "%.4g");
      if (ImGui::Button("Minimum")) {
        plane = boundsMin[planeAxis];
        attempt([&] { model.selectSlab(planeAxis, plane, tolerance, append); });
      }
      ImGui::SameLine();
      if (ImGui::Button("Maximum")) {
        plane = boundsMax[planeAxis];
        attempt([&] { model.selectSlab(planeAxis, plane, tolerance, append); });
      }
      ImGui::SameLine();
      if (ImGui::Button("Select"))
        attempt([&] { model.selectSlab(planeAxis, plane, tolerance, append); });
      ImGui::SameLine();
      if (ImGui::Button("Clear")) model.clearSelection();
      if (ImGui::TreeNode("Cylindrical surface selection")) {
        editVector("Cylinder axis", cylinderAxis);
        editVector("Cylinder center", cylinderCenter, " (m)");
        ImGui::InputDouble("Cylinder radius (m)", &cylinderRadius, 0., 0.,
                           "%.6g");
        ImGui::InputDouble("Axial minimum (m)", &cylinderMinimum, 0., 0.,
                           "%.6g");
        ImGui::InputDouble("Axial maximum (m)", &cylinderMaximum, 0., 0.,
                           "%.6g");
        if (ImGui::Button("Select cylindrical boundary"))
          attempt([&] {
            model.selectCylinder(cylinderCenter, cylinderAxis, cylinderRadius,
                                 tolerance, cylinderMinimum, cylinderMaximum,
                                 append);
          });
        ImGui::TreePop();
      }
    }
    if (ImGui::CollapsingHeader("Supports and loads",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      for (int k = 0; k < 6; ++k) {
        if (k % 2) ImGui::SameLine();
        if (ImGui::Button(fem_workbench::kindName(static_cast<Kind>(k)),
                          ImVec2(145, 0)))
          attempt([&] {
            model.add(static_cast<Kind>(k));
            edited = int(model.conditions.size()) - 1;
            visualDirty = true;
          });
      }
      ImGui::SetNextItemWidth(-1.f);
      if (ImGui::BeginCombo("##fixture-method", "More fixture methods...")) {
        for (int k = 6; k <= static_cast<int>(Kind::Tie); ++k)
          if (ImGui::Selectable(fem_workbench::kindName(static_cast<Kind>(k))))
            attempt([&] {
              auto& c = model.add(static_cast<Kind>(k));
              if (c.kind == Kind::Bearing || c.kind == Kind::BoltHole) {
                c.axis = cylinderAxis;
                c.center = cylinderCenter;
              }
              edited = int(model.conditions.size()) - 1;
              visualDirty = true;
            });
        ImGui::EndCombo();
      }
      ImGui::BeginChild("conditions", ImVec2(0, 105), true);
      for (size_t i = 0; i < model.conditions.size(); ++i) {
        ImGui::PushID(int(i));
        if (ImGui::Checkbox("##enabled", &model.conditions[i].enabled)) {
          model.changed();
          visualDirty = true;
        }
        ImGui::SameLine();
        if (ImGui::Selectable(model.conditions[i].name.c_str(),
                              edited == int(i)))
          edited = int(i);
        ImGui::PopID();
      }
      ImGui::EndChild();
      if (edited >= 0 && size_t(edited) < model.conditions.size()) {
        auto& c = model.conditions[size_t(edited)];
        ImGui::TextDisabled("%zu nodes / %zu faces", c.nodes.size(),
                            c.faces.size());
        bool changed = false;
        if (c.kind == Kind::Displacement)
          for (int a = 0; a < 3; ++a) {
            if (a) ImGui::SameLine();
            changed |= ImGui::Checkbox(a == 0   ? "X"
                                       : a == 1 ? "Y"
                                                : "Z",
                                       &c.axes[a]);
          }
        if (static_cast<int>(c.kind) > static_cast<int>(Kind::Gravity))
          changed |= editFixture(c);
        else if (c.kind == Kind::Pressure)
          changed |=
              ImGui::InputDouble("Pressure (Pa)", &c.value[0], 0., 0., "%.6g");
        else if (c.kind != Kind::Fixed) {
          const char* unit = c.kind == Kind::Displacement ? "m"
                             : c.kind == Kind::Force      ? "N"
                             : c.kind == Kind::Traction   ? "Pa"
                                                          : "m/s2";
          for (int a = 0; a < 3; ++a) {
            char label[24];
            std::snprintf(label, sizeof(label), "%c (%s)", 'X' + a, unit);
            changed |= ImGui::InputDouble(label, &c.value[a], 0., 0., "%.6g");
          }
        }
        if (changed) {
          model.changed();
          visualDirty = true;
        }
        if (ImGui::Button("Show region")) {
          model.restoreSelection(c.nodes, c.faces);
        }
        ImGui::SameLine();
        if (ImGui::Button("Delete")) {
          model.conditions.erase(model.conditions.begin() + edited);
          edited = -1;
          model.changed();
          visualDirty = true;
        }
      }
    }
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Results and files",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::Combo("Field", &field,
                       "Von Mises stress\0Displacement magnitude\0Max "
                       "principal stress\0Min principal stress\0Stress "
                       "XX\0Stress YY\0Stress ZZ\0Stress XY\0Stress YZ\0Stress "
                       "ZX\0Strain XX\0Strain YY\0Strain "
                       "ZZ\0Engineering shear XY\0Engineering shear "
                       "YZ\0Engineering shear ZX\0"))
        visualDirty = true;
      if (ImGui::InputDouble("Deformation scale", &deformation, 0., 0.,
                             "%.4g")) {
        deformation =
            std::isfinite(deformation) ? std::clamp(deformation, 0., 1e6) : 1.;
        visualDirty = true;
      }
      if (ImGui::Checkbox("Automatic color range", &autoRange))
        visualDirty = true;
      if (!autoRange) {
        if (ImGui::InputDouble("Color minimum", &colorMin, 0., 0., "%.5g"))
          visualDirty = true;
        if (ImGui::InputDouble("Color maximum", &colorMax, 0., 0., "%.5g"))
          visualDirty = true;
      }
      ImGui::Checkbox("Mesh edges", &edges);
      ImGui::SameLine();
      ImGui::Checkbox("Rest outline", &undeformed);
      if (ImGui::Button("Fit view")) fit();
      ImGui::InputText("Project", project.data(), project.size());
      if (ImGui::Button("Save project"))
        attempt([&] {
          syncSettings();
          model.save(project.data());
          download(project.data());
          message = "Project saved.";
        });
      ImGui::SameLine();
      if (ImGui::Button("Load project"))
        attempt([&] {
#if defined(__EMSCRIPTEN__)
          EM_ASM({ globalThis['raisimFem'] ?.['open'](); });
#else
          import(project.data());
#endif
        });
      ImGui::InputText("VTK output", exportPath.data(), exportPath.size());
      if (ImGui::Button("Export results"))
        attempt([&] {
          if (!model.hasSolution)
            throw std::runtime_error(
                "Solve the current model before exporting results");
          model.solver->writeVtk(exportPath.data());
          download(exportPath.data());
          message = "VTK results exported.";
        });
    }
    ImGui::PopItemWidth();
    ImGui::End();
    ImGui::SetNextWindowPos(ImVec2(sidebarWidth + 34, 16));
    ImGui::SetNextWindowBgAlpha(.88f);
    ImGui::Begin("Analysis status", nullptr,
                 ImGuiWindowFlags_NoDecoration |
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoMove);
    ImGui::TextColored(ImVec4(.5f, .85f, .9f, 1.f),
                       "RAISIM  /  SOLID MECHANICS");
    ImGui::Text("Small displacement | linear tetrahedra | SI units");
    if (model.hasSolution) {
      const auto& s = model.solver->statistics();
      ImGui::Text("Solver: %s | %zu iterations",
                  s.iterative ? "ICCG" : "Sparse LDLT", s.iterations);
      double umax = 0.;
      for (const auto& u : model.solver->displacements())
        umax = std::max(umax, u.e().norm());
      ImGui::Text("Max displacement %.4g m  |  Energy %.4g J", umax,
                  s.strainEnergy);
      ImGui::Text(
          "Assembly %.2f ms  |  Factor %.2f ms  |  Solve + fields %.2f ms",
          s.assemblyMs, s.factorizationMs, s.solveMs);
    } else
      ImGui::TextColored(ImVec4(1.f, .78f, .35f, 1.f),
                         "Results pending - solve the current model");
    ImGui::End();
    ImGui::SetNextWindowPos(ImVec2(sidebarWidth + 34, display.y - 65));
    ImGui::SetNextWindowSize(ImVec2(display.x - sidebarWidth - 50, 49));
    ImGui::Begin("Status", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove);
    ImGui::TextWrapped("%s", message.c_str());
    ImGui::End();
    if (model.hasSolution) {
      ImGui::SetNextWindowPos(ImVec2(display.x - 240, 100));
      ImGui::Begin("Legend", nullptr,
                   ImGuiWindowFlags_NoDecoration |
                       ImGuiWindowFlags_AlwaysAutoResize |
                       ImGuiWindowFlags_NoMove);
      const char* unit = field == 1 ? "m" : field >= 10 ? "strain" : "Pa";
      ImGui::Text("%.4g - %.4g %s", colorMin, colorMax, unit);
      const auto start = ImGui::GetCursorScreenPos();
      auto* d = ImGui::GetWindowDrawList();
      for (int i = 0; i < 180; ++i)
        d->AddRectFilled(ImVec2(start.x + i, start.y),
                         ImVec2(start.x + i + 1, start.y + 12),
                         ImGui::ColorConvertFloat4ToU32(ImVec4(
                             heat(double(i) / 179).r, heat(double(i) / 179).g,
                             heat(double(i) / 179).b, 1.f)));
      ImGui::Dummy(ImVec2(180, 12));
      ImGui::TextDisabled("Element values | deformation %.3gx", deformation);
      ImGui::End();
    }
  }
  void download(const char* path) {
#if defined(__EMSCRIPTEN__)
    EM_ASM(
        {
          const path = UTF8ToString($0);
          globalThis['raisimFem']
              ?.['download'](path.split('/').pop(), FS.readFile(path));
        },
        path);
#else
    (void)path;
#endif
  }
};

bool selfTest() {
  raisin::Camera camera;
  for (const float scale : {1e-4f, 1.f, 100.f}) {
    const glm::vec3 low =
        scale < .01f ? glm::vec3(0.f) : glm::vec3(3.f, -2.f, 1.f);
    const glm::vec3 high = low + scale * glm::vec3(1.f, 2.f, 3.f);
    const auto center = low + .5f * (high - low);
    fem_workbench::frameMeshCamera(camera, low, high);
    if (!camera.isCameraFixedTarget() || !camera.isCameraFixedDistance() ||
        glm::length(camera.getTargetPosition() - center) > 1e-5f)
      return false;
    const float distance = glm::distance(camera.position, center);
    for (const float angle : {0.f, 30.f, -90.f}) {
      camera.mouseDelta(angle, 5.f);
      camera.update(false);
      const auto projected = camera.getProjectionMatrix() *
                             camera.getViewMatrix() * glm::vec4(center, 1.f);
      if (std::abs(projected.x / projected.w) > 1e-4f ||
          std::abs(projected.y / projected.w) > 1e-4f ||
          std::abs(glm::distance(camera.position, center) - distance) >
              1e-4f * distance)
        return false;
    }
    camera.moveForward(-8.f);
    camera.update(false);
    if (!(glm::distance(camera.position, center) < distance * .3f))
      return false;
    camera.moveForward(30.f);
    camera.update(false);
    fem_workbench::updateMeshCameraClipping(camera);
    const auto clipped = camera.getProjectionMatrix() * camera.getViewMatrix() *
                         glm::vec4(center, 1.f);
    if (clipped.z / clipped.w < -1.f || clipped.z / clipped.w > 1.f ||
        glm::length(camera.getTargetPosition() - center) > 1e-5f)
      return false;
  }
  Model model;
  model.demo();
  const auto first = model.solver->displacements();
  model.conditions[1].value[2] *= 2.;
  model.changed();
  model.solve();
  if (!model.solver->statistics().reusedFactorization) return false;
  for (size_t i = 0; i < first.size(); ++i)
    if ((model.solver->displacements()[i].e() - 2. * first[i].e()).norm() >
        1e-10)
      return false;
  fem_workbench::MeshJob temp;
  const auto path = raisim::pathUtf8(temp.directory / "case.rfem");
  model.save(path);
  Model restored;
  restored.loadProject(path);
  restored.solve();
  if (restored.conditions.size() != 2) return false;
  restored.solver->writeVtk(raisim::pathUtf8(temp.directory / "case.vtk"));
  return TetMesh::load(raisim::pathUtf8(temp.directory / "case.vtk"))
             .tetrahedra.size() == model.mesh().tetrahedra.size();
}
void capture(const std::string& path, SDL_Window* window) {
#if !defined(__EMSCRIPTEN__)
  int width = 0, height = 0;
  SDL_GL_GetDrawableSize(window, &width, &height);
  std::vector<unsigned char> rgba(size_t(width) * size_t(height) * 4);
  gl::glReadPixels(0, 0, width, height, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE,
                   rgba.data());
  for (int y = 0; y < height / 2; ++y)
    for (int x = 0; x < width * 4; ++x)
      std::swap(rgba[size_t(y) * width * 4 + x],
                rgba[size_t(height - 1 - y) * width * 4 + x]);
  if (!stbi_write_png(path.c_str(), width, height, 4, rgba.data(), width * 4))
    throw std::runtime_error("Cannot save workbench capture");
#else
  (void)path;
  (void)window;
#endif
}
}  // namespace

int main(int argc, char** argv) {
  std::string input, capturePath, drop, recording;
  bool showcase = false;
  int maxFrames = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--self-test") {
      try {
        const bool ok = selfTest();
        std::cout << (ok ? "FEM workbench self-test passed\n"
                         : "FEM workbench self-test failed\n");
        return ok ? 0 : 1;
      } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
      }
    }
    if (arg == "--help") {
      std::cout << "rayrai_fem_workbench [solid-file] [--frames N] [--capture "
                   "image.png] [--test-drop solid-file] "
                   "[--self-test] [--showcase --record directory]\n";
      return 0;
    }
    if (arg == "--showcase")
      showcase = true;
    else if (arg == "--record" && i + 1 < argc)
      recording = argv[++i];
    else if (arg == "--frames" && i + 1 < argc)
      maxFrames = std::stoi(argv[++i]);
    else if (arg == "--capture" && i + 1 < argc)
      capturePath = argv[++i];
    else if (arg == "--test-drop" && i + 1 < argc)
      drop = argv[++i];
    else if (arg.rfind("--", 0) != 0)
      input = arg;
  }
  int width = 1600, height = 1000;
#if defined(__EMSCRIPTEN__)
  width = EM_ASM_INT({ return window.innerWidth; });
  height = EM_ASM_INT({ return window.innerHeight; });
#endif
  ExampleApp app;
  if (!app.init("RaiSim FEM Workbench", width, height)) return 1;
  ImGui::GetIO().IniFilename = nullptr;
  const auto fontPath =
      exampleRscPath(argv[0], "fonts/roboto/Roboto-Medium.ttf");
  if (std::filesystem::exists(raisim::pathFromUtf8(fontPath)))
    ImGui::GetIO().Fonts->AddFontFromFileTTF(fontPath.c_str(), 16.f);
  ImGui::StyleColorsDark();
  auto& style = ImGui::GetStyle();
  style.WindowRounding = 7.f;
  style.FrameRounding = 4.f;
  style.ItemSpacing = ImVec2(7, 6);
  int exitCode = 0;
  {
    Workbench workbench;
    const auto nativeMesher =
        fem_workbench::defaultNativeMesherExecutable(argv[0]);
    std::snprintf(workbench.nativeMesher.data(), workbench.nativeMesher.size(),
                  "%s", nativeMesher.c_str());
    workbench.viewer = std::make_shared<raisin::RayraiWindow>(
        std::make_shared<raisim::World>(), width, height);
    auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(
        raisin::RayraiWindow::RenderQualityPreset::Balanced);
    quality.mainLightDirection = glm::normalize(glm::vec3(-.35f, .55f, -1.f));
    quality.mainLightAmbient = glm::vec3(.28f);
    quality.mainLightDiffuse = glm::vec3(.95f);
    quality.mainLightSpecular = glm::vec3(.02f);
    quality.pbrEnvironmentIntensity = .12f;
    quality.addViewerFillLights = false;
    workbench.viewer->setRenderQualitySettings(quality);
    workbench.viewer->setBackgroundColorLinear(
        glm::vec4(.025f, .035f, .055f, 1.f));
    workbench.attempt([&] { workbench.demo(); });
    if (!input.empty()) workbench.attempt([&] { workbench.import(input); });
    if (!drop.empty()) {
      SDL_Event event{};
      event.type = SDL_DROPFILE;
      event.drop.file = SDL_strdup(drop.c_str());
      event.drop.windowID = SDL_GetWindowID(app.window);
      SDL_PushEvent(&event);
    }
    if (!recording.empty())
      std::filesystem::create_directories(raisim::pathFromUtf8(recording));
    int frame = 0;
    while (!app.quit) {
#if defined(__EMSCRIPTEN__)
      const int nextWidth = EM_ASM_INT({ return window.innerWidth; });
      const int nextHeight = EM_ASM_INT({ return window.innerHeight; });
      if (width != nextWidth || height != nextHeight) {
        width = nextWidth;
        height = nextHeight;
        SDL_SetWindowSize(app.window, width, height);
      }
#endif
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if (event.type == SDL_QUIT ||
            event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE)
          app.quit = true;
        if (event.type == SDL_DROPFILE) {
          const std::string path = event.drop.file ? event.drop.file : "";
          SDL_free(event.drop.file);
          workbench.attempt([&] { workbench.import(path); });
#if defined(__EMSCRIPTEN__)
          const auto directory = raisim::pathFromUtf8(path).parent_path();
          if (directory.parent_path() ==
                  std::filesystem::path("/examples/rsc") &&
              directory.filename().string().rfind("fem-import-", 0) == 0) {
            std::error_code error;
            std::filesystem::remove_all(directory, error);
            if (error)
              workbench.message += " Could not release imported source files.";
          }
#endif
        }
      }
      if (app.quit) break;
      workbench.attempt([&] { workbench.pollMesh(); });
      workbench.pollBrowserErrors();
      if (showcase && drop.empty() && input.empty()) {
        workbench.model.conditions[1].value[2] =
            -5000. * (.75 + .25 * std::cos(frame * 6.283185307179586 / 120.));
        workbench.model.changed();
        workbench.attempt([&] { workbench.solve(); });
      }
      app.beginFrame();
      workbench.panel();
      if (workbench.autoSolve && !workbench.meshJob && !workbench.needsMesh() &&
          workbench.model.dirty && !ImGui::IsAnyItemActive())
        workbench.attempt([&] { workbench.solve(); });
      workbench.attempt([&] { workbench.updateVisual(); });
      fem_workbench::updateMeshCameraClipping(workbench.viewer->getCamera());
      auto viewport = app.beginViewport();
      workbench.input(viewport);
      if (ImGui::GetIO().MousePos.x < workbench.sidebarWidth + 25.f)
        viewport.hovered = false;
      app.endViewport(*workbench.viewer, viewport, workbench.selectionMode == 0,
                      false);
      workbench.overlays(viewport);
#if defined(__EMSCRIPTEN__)
      const auto& camera = workbench.viewer->getCamera();
      const auto center = camera.getTargetPosition();
      const auto projection = workbench.viewProjection();
      EM_ASM(
          {
            const state = globalThis['raisimFemState'];
            if (state) {
              state['frame'] = $3;
              state['orbitCenter'] =
                  Array.from(HEAPF32.subarray($0 / 4, $0 / 4 + 3));
              state['cameraPosition'] =
                  Array.from(HEAPF32.subarray($1 / 4, $1 / 4 + 3));
              state['viewProjection'] =
                  Array.from(HEAPF32.subarray($2 / 4, $2 / 4 + 16));
            }
          },
          &center.x, &camera.position.x, &projection[0][0], frame);
#endif
      ++frame;
      const bool finished =
          maxFrames > 0 && frame >= maxFrames && !workbench.meshJob;
      if ((finished && !capturePath.empty()) || !recording.empty()) {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        try {
          if (finished && !capturePath.empty())
            capture(capturePath, app.window);
          if (!recording.empty()) {
            char name[32];
            std::snprintf(name, sizeof(name), "frame-%05d.png", frame);
            capture(raisim::pathUtf8(raisim::pathFromUtf8(recording) / name),
                    app.window);
          }
        } catch (const std::exception& e) {
          std::cerr << e.what() << '\n';
          exitCode = 1;
        }
        SDL_GL_SwapWindow(app.window);
      } else
        app.endFrame();
      if (finished) {
        if (!drop.empty() &&
            workbench.model.name !=
                raisim::pathUtf8(raisim::pathFromUtf8(drop).filename())) {
          std::cerr << workbench.message << '\n';
          exitCode = 1;
        }
        app.quit = true;
      }
    }
  }
  app.shutdown();
  return exitCode;
}
