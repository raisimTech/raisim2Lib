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
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "example_resources.hpp"
#include "fem_workbench_import.hpp"
#include "fem_workbench_model.hpp"
#include "raisim/World.hpp"
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
  std::string importName,
      message = "Select a surface, add supports and loads, then Solve.";
  std::array<char, 2048> file{}, project{}, exportPath{}, gmsh{};
  double importScale = 1., elementSize = .05, pendingScale = 1.,
         deformation = 100., colorMin = 0., colorMax = 1.;
  double boundsMin[3]{}, boundsMax[3]{};
  double plane = 0., tolerance = .0001;
  int planeAxis = 0, selectionMode = 0, field = 0, edited = -1;
  Vec<3> cylinderCenter{0., 0., 0.}, cylinderAxis{0., 0., 1.};
  double cylinderRadius = .05, cylinderMinimum = -1., cylinderMaximum = 1.;
  bool autoSolve = false, autoRange = true, edges = false, undeformed = false,
       visualDirty = true, append = false;
  ImVec2 boxStart{};
  bool selectingBox = false;
  glm::mat4 screenMatrix{1.f};
  float sidebarWidth = 380.f;

  Workbench() {
    std::snprintf(gmsh.data(), gmsh.size(), "%s", "gmsh");
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
    const auto& x = model.solver->mesh().vertices;
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
    cam.target = .5f * (low + high);
    cam.position = cam.target + extent * glm::vec3(1.1f, -1.45f, .85f);
    cam.yaw = 126.f;
    cam.pitch = -25.f;
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
  void replaceVisual() {
    if (solid) {
      viewer->removeVisualObject("fem_solid");
      solid.reset();
    }
    vertices.clear();
    visualDirty = true;
    fit();
  }
  void solve() {
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
  void demo() {
    model.demo();
    edited = 1;
    replaceVisual();
    message = "Demo solved. Change the tip force to reuse the factorization.";
  }
  void import(const std::string& path) {
    if (meshJob)
      throw std::runtime_error(
          "Finish or cancel the current mesh import first");
    if (!std::isfinite(importScale) || importScale <= 0.)
      throw std::runtime_error(
          "Import scale must be positive (metres per file unit)");
    const auto ext = extension(path);
    if (ext == ".rfem") {
      model.loadProject(path);
      edited = -1;
      replaceVisual();
      message = "Project loaded; ready to solve.";
      return;
    }
    if (ext == ".msh" || ext == ".mesh" || ext == ".vtk") {
      model.setMesh(TetMesh::load(path, importScale),
                    raisim::pathUtf8(raisim::pathFromUtf8(path).filename()));
      edited = -1;
      replaceVisual();
      message = "Volume imported. Select supports and loads.";
      return;
    }
    const bool cad = ext == ".step" || ext == ".stp" || ext == ".brep" ||
                     ext == ".iges" || ext == ".igs";
    if (!cad && ext != ".stl" && ext != ".obj" && ext != ".ply" &&
        ext != ".glb" && ext != ".gltf" && ext != ".dae")
      throw std::runtime_error(
          "Supported: .msh .mesh .vtk .rfem; watertight .stl .obj .ply .glb "
          ".gltf .dae; CAD .step .stp .brep .iges "
          ".igs");
    auto job = std::make_unique<fem_workbench::MeshJob>();
    std::string mesherExtension = ext;
    if (cad || ext == ".stl") {
      std::filesystem::copy_file(raisim::pathFromUtf8(path),
                                 job->directory / ("source" + ext));
    } else {
      const auto meshes = raisin::assimp::loadModel(path);
      if (!meshes || meshes->empty())
        throw std::runtime_error("Surface import produced no triangles");
      std::ofstream stl(job->directory / "source.stl");
      stl.imbue(std::locale::classic());
      stl.precision(9);
      stl << "solid model\n";
      for (const auto& mesh : *meshes)
        for (size_t i = 0; i + 2 < mesh->indices.size(); i += 3) {
          stl << "facet normal 0 0 0\nouter loop\n";
          for (size_t j = 0; j < 3; ++j) {
            const auto& p = mesh->vertices.at(mesh->indices[i + j]).Position;
            stl << "vertex " << p.x << ' ' << p.y << ' ' << p.z << '\n';
          }
          stl << "endloop\nendfacet\n";
        }
      stl << "endsolid model\n";
      stl.close();
      if (!stl) throw std::runtime_error("Cannot prepare surface for meshing");
      mesherExtension = ".stl";
    }
    job->start(gmsh.data(), mesherExtension,
               elementSize / (cad ? 1. : importScale));
    meshJob = std::move(job);
    pendingScale = cad ? 1. : importScale;
    importName = raisim::pathUtf8(raisim::pathFromUtf8(path).filename());
#if defined(__EMSCRIPTEN__)
    message = "Generating volume tetrahedra in a browser worker...";
#else
    message = "Generating volume tetrahedra with Gmsh...";
#endif
  }
  void pollMesh() {
    if (!meshJob || !meshJob->poll()) return;
    auto job = std::move(meshJob);
    auto mesh = job->result(pendingScale);
    model.setMesh(std::move(mesh), importName);
    edited = -1;
    replaceVisual();
    message = "Solid meshed. Select supports and loads.";
  }
  void pollBrowserErrors() {
#if defined(__EMSCRIPTEN__)
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
          if (globalThis['raisimFemState'])
            globalThis['raisimFemState']['message'] = UTF8ToString($0);
        },
        message.c_str());
#endif
  }
  double scalar(size_t element) const {
    if (!model.hasSolution) return 0.;
    const auto& r = model.solver->elementResults()[element];
    if (field == 0) return r.vonMises;
    if (field == 1) {
      double sum = 0.;
      for (const auto v : model.solver->mesh().tetrahedra[element])
        sum += .25 * model.solver->displacements()[v].e().norm();
      return sum;
    }
    if (field == 2) return r.principalStress[2];
    if (field == 3) return r.principalStress[0];
    if (field < 10) return r.stress[field - 4];
    return r.strain[field - 10];
  }
  glm::vec3 position(unsigned int node, bool deformed = true) const {
    auto p = vector(model.solver->mesh().vertices[node]);
    if (deformed && model.hasSolution)
      p += float(deformation) * vector(model.solver->displacements()[node]);
    return p;
  }
  void updateVisual() {
    if (!visualDirty || !model.solver) return;
    if (!std::isfinite(deformation) ||
        !autoRange && (!std::isfinite(colorMin) || !std::isfinite(colorMax) ||
                       colorMax <= colorMin))
      throw std::runtime_error(
          "Display scale and color limits must be finite; maximum must exceed "
          "minimum");
    const auto& faces = model.solver->boundaryFaces();
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
      appearance.unlit = true;
      solid->setMaterialOverride(appearance);
    } else if (!solid->updateSingleMeshVertices(vertices))
      throw std::runtime_error("Cannot update FEM surface buffer");
    visualDirty = false;
#if defined(__EMSCRIPTEN__)
    auto regions = model.solver->mesh().regions;
    std::sort(regions.begin(), regions.end());
    regions.erase(std::unique(regions.begin(), regions.end()), regions.end());
    EM_ASM(
        {
          const state = {};
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
        model.solver->mesh().tetrahedra.size(), model.hasSolution,
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
        !model.solver)
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
    if (!model.solver) return;
    screenMatrix = viewProjection();
    auto* draw = ImGui::GetForegroundDrawList();
    draw->PushClipRect(ImVec2(sidebarWidth + 25.f, 60.f),
                       ImVec2(vp.size.x - 15.f, vp.size.y - 74.f), true);
    const auto& faces = model.solver->boundaryFaces();
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
      ImGui::InputDouble("Element size (m)", &elementSize, 0., 0., "%.5g");
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
#if defined(__EMSCRIPTEN__)
        ImGui::TextWrapped(
            "Open MSH, MESH, VTK, RFEM, or watertight STL/OBJ/PLY/glTF/DAE "
            "surfaces. Surface meshing runs locally with "
            "fTetWild (MPL-2.0). Include companion files when importing glTF "
            "or OBJ.");
#else
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##gmsh", gmsh.data(), gmsh.size());
        ImGui::TextWrapped(
            "Gmsh is required for watertight STL/OBJ/PLY/glTF/DAE and "
            "STEP/BREP/IGES. Native tetrahedral MSH, MESH and "
            "VTK files open directly.");
#endif
        ImGui::TreePop();
      }
      if (meshJob) {
        ImGui::TextColored(ImVec4(1.f, .8f, .3f, 1.f), "Meshing solid...");
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
          meshJob.reset();
          message = "Import cancelled.";
        }
      }
    }
    if (!model.solver) {
      ImGui::PopItemWidth();
      ImGui::End();
      return;
    }
    ImGui::TextDisabled("%zu nodes   %zu tetrahedra", model.selected.size(),
                        model.solver->mesh().tetrahedra.size());
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
             .tetrahedra.size() == model.solver->mesh().tetrahedra.size();
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
    workbench.viewer = std::make_shared<raisin::RayraiWindow>(
        std::make_shared<raisim::World>(), width, height);
    workbench.viewer->setRenderQualitySettings(
        raisin::RayraiWindow::defaultRenderQualitySettings(
            raisin::RayraiWindow::RenderQualityPreset::Balanced));
    workbench.viewer->setBackgroundColorLinear(
        glm::vec4(.025f, .035f, .055f, 1.f));
    if (const char* gmsh = std::getenv("RAISIM_FEM_GMSH"))
      std::snprintf(workbench.gmsh.data(), workbench.gmsh.size(), "%s", gmsh);
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
      if (workbench.autoSolve && workbench.model.dirty &&
          !ImGui::IsAnyItemActive())
        workbench.attempt([&] { workbench.solve(); });
      workbench.attempt([&] { workbench.updateVisual(); });
      auto viewport = app.beginViewport();
      workbench.input(viewport);
      if (ImGui::GetIO().MousePos.x < workbench.sidebarWidth + 25.f)
        viewport.hovered = false;
      app.endViewport(*workbench.viewer, viewport, workbench.selectionMode == 0,
                      false);
      workbench.overlays(viewport);
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
