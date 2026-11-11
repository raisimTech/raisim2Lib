// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#if !defined(__EMSCRIPTEN__)
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "example_resources.hpp"
#include "fem_condition_preview.hpp"
#include "fem_material_library.hpp"
#include "fem_workbench_import.hpp"
#include "fem_workbench_model.hpp"
#include "fem_workbench_view.hpp"
#include "raisim/World.hpp"
#include "rayrai/example_common.hpp"

namespace {
using fem_workbench::Condition;
using fem_workbench::Kind;
using fem_workbench::isConnection;
using fem_workbench::Model;
using raisim::Vec;
using fem_workbench::ForceDirection;
using raisim::fem::LinearElasticity;
using raisim::fem::TetMesh;

glm::vec3 vector(const Vec<3>& v) {
  return {float(v[0]), float(v[1]), float(v[2])};
}
glm::vec4 heat(double value) {
  if (!std::isfinite(value)) value = 0.;
  const float t = float(std::clamp(value, 0., 1.));
  return glm::vec4(
      glm::mix(glm::vec3(.08f, .25f, .9f), glm::vec3(.9f, .08f, .06f), t), 1.f);
}
constexpr const char* kFieldNames[] = {
    "Von Mises stress",     "Displacement magnitude", "Max principal stress",
    "Min principal stress", "Stress XX",              "Stress YY",
    "Stress ZZ",            "Stress XY",              "Stress YZ",
    "Stress ZX",            "Strain XX",              "Strain YY",
    "Strain ZZ",            "Engineering shear XY",   "Engineering shear YZ",
    "Engineering shear ZX", "Factor of safety"};
constexpr int kSafetyField = 16;

bool isLoad(Kind kind) {
  return kind == Kind::Force || kind == Kind::Traction ||
         kind == Kind::Pressure || kind == Kind::Gravity;
}
constexpr unsigned char kSupportMark = 1, kLoadMark = 2, kHoverMark = 4,
                        kSelectionMark = 8, kReferenceMark = 16;
const glm::vec4 kSupportColor{.16f, .78f, .42f, 1.f},
    kLoadColor{.86f, .32f, .86f, 1.f}, kSelectionColor{1.f, .74f, .12f, 1.f},
    kReferenceColor{.15f, .8f, 1.f, 1.f};
constexpr const char* kIconPlay = "\xef\xaa\x9e";
constexpr const char* kIconFolderOpen = "\xef\x9a\xb5";
constexpr const char* kIconCube = "\xef\x94\x80";
constexpr const char* kIconSliders = "\xef\xaf\x91";
constexpr const char* kIconCancel = "\xef\x93\xb4";
constexpr const char* kIconEraser = "\xef\x97\xa2";
constexpr const char* kIconRuler = "\xef\xad\xb5";
constexpr const char* kIconTools = "\xef\xb7\x8b";
constexpr const char* kIconAnchor = "\xef\x85\x83";
constexpr const char* kIconTire = "\xef\xb6\xb6";
constexpr const char* kIconArrows = "\xef\x87\x8c";
constexpr const char* kIconPlus = "\xef\xaa\xac";
constexpr const char* kIconArrowDown = "\xef\x86\x8f";
constexpr const char* kIconCompress = "\xef\x93\x88";
constexpr const char* kIconArrowsToLine = "\xef\x87\x8b";
constexpr const char* kIconGlobe = "\xef\x9c\x85";
constexpr const char* kIconLayers = "\xef\xa1\xbc";
constexpr const char* kIconChart = "\xef\x8e\xa2";
constexpr const char* kIconDisk = "\xef\x95\xaa";
constexpr const char* kIconExport = "\xef\x99\xa2";
constexpr const char* kIconTrash = "\xef\xb8\x97";
constexpr const char* kIconCrosshairs = "\xef\xa3\x8e";
constexpr const char* kIconExpand = "\xef\x97\xb0";
constexpr const char* kIconListCheck = "\xef\xa3\x80";
constexpr const char* kIconPencil = "\xef\xa8\xa9";
constexpr const char* kIconLink = "\xef\xa2\xba";
constexpr const char* kIconScrew = "\xef\xae\xa7";
constexpr const char* kIconRotate = "\xef\xad\xaa";
constexpr const char* kIconCircle = "\xef\x90\xac";
constexpr const char* kIconSlide = "\xef\x87\x82";
constexpr const char* kIconLock = "\xef\xa3\x95";
constexpr const char* kIconPuzzle = "\xef\xab\xa9";
constexpr const char* kIconBroom = "\xef\x8b\xb3";
constexpr const char* kIconGauge = "\xef\xb5\x8b";
constexpr const char* kIconDeflection = "\xef\x9c\x92";
constexpr const char* kIconCopy = "\xef\x93\x9d";
constexpr const char* kIconEye = "\xef\x97\xb8";
constexpr const char* kIconEyeCrossed = "\xef\x97\xb5";
constexpr const char* kIconExchange = "\xef\x97\xaa";
constexpr const char* kIconUndo = "\xef\xb9\xab";
constexpr const char* kIconSearch = "\xef\xae\xba";
constexpr const char* kIconThumbtack = "\xef\xb6\x95";
constexpr const char* kIconCursor = "\xef\x94\x8e";
constexpr const char* kIconArrowUp = "\xef\x86\x9f";
constexpr const char* kIconArrowSmallDown = "\xef\x86\x9c";
constexpr const char* kIconBullseye = "\xef\x8c\x89";
constexpr const char* kIconFlip = "\xef\x9a\x9e";
constexpr const char* kIconRightAngle = "\xef\x85\x85";
constexpr const char* kIconTangent = "\xef\xba\xb3";
constexpr const char* kIconProtractor = "\xef\x96\x92";
constexpr const char* kIconWidth = "\xef\x86\xbb";
constexpr const char* kIconCheck = "\xef\x8f\x88";
constexpr const char* kIconParallel = "\xef\xa8\x8c";
constexpr const char* kIconPalette = "\xef\xa7\xad";
constexpr const char* kIconDots = "\xef\xa4\xa7";
std::string withIcon(const char* icon, const std::string& text) {
  return std::string(icon) + "  " + text;
}
ImU32 toImColor(const glm::vec4& color) {
  return ImGui::ColorConvertFloat4ToU32(
      ImVec4(color.r, color.g, color.b, color.a));
}
void leftBar(ImU32 color, float width = 4.f) {
  const auto min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
  ImGui::GetWindowDrawList()->AddRectFilled(
      min, ImVec2(min.x + width, max.y), color, ImGui::GetStyle().FrameRounding,
      ImDrawFlags_RoundCornersLeft);
}
void axisBar(int axis) {
  static const ImU32 kAxisColors[] = {IM_COL32(236, 82, 74, 255),
                                      IM_COL32(106, 202, 90, 255),
                                      IM_COL32(78, 142, 246, 255)};
  leftBar(kAxisColors[axis]);
}
void pushBarPadding() {
  const ImVec2 padding = ImGui::GetStyle().FramePadding;
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                      ImVec2(padding.x + 6.f, padding.y));
}
bool axisInput(const char* label, double* value, int axis,
               ImGuiInputTextFlags flags = 0) {
  pushBarPadding();
  const bool changed = ImGui::InputDouble(label, value, 0., 0., "%.6g", flags);
  ImGui::PopStyleVar();
  axisBar(axis);
  return changed;
}
bool axisButton(const char* label, int axis) {
  pushBarPadding();
  const bool pressed = ImGui::Button(label);
  ImGui::PopStyleVar();
  axisBar(axis);
  return pressed;
}
std::array<double, 3> eulerDegrees(const Eigen::Matrix3d& r) {
  constexpr double kDegrees = 180. / 3.14159265358979323846;
  const double pitch = std::asin(std::clamp(-r(2, 0), -1., 1.));
  std::array<double, 3> angles{0., pitch, std::atan2(-r(0, 1), r(1, 1))};
  if (std::abs(r(2, 0)) <= 1. - 1e-12)
    angles = {std::atan2(r(2, 1), r(2, 2)), pitch, std::atan2(r(1, 0), r(0, 0))};
  for (auto& angle : angles) angle = angle * kDegrees + 0.;
  return angles;
}
Eigen::Matrix3d eulerRotation(const std::array<double, 3>& degrees) {
  constexpr double kRadians = 3.14159265358979323846 / 180.;
  return (Eigen::AngleAxisd(degrees[2] * kRadians, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(degrees[1] * kRadians, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(degrees[0] * kRadians, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}
std::string fitText(const std::string& text, float width) {
  return fem_workbench::fitText(text, width, [](const std::string& value) {
    return ImGui::CalcTextSize(value.c_str()).x;
  });
}
const char* presetName(const LinearElasticity::Material& m, double yield) {
  const auto* grade = fem_workbench::findMaterialGrade(
      m.youngsModulus, m.poissonRatio, m.density, yield);
  return grade ? grade->name : nullptr;
}
bool containsText(const char* text, const std::string& query) {
  const auto lower = [](char c) {
    return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c;
  };
  std::string haystack(text);
  std::transform(haystack.begin(), haystack.end(), haystack.begin(), lower);
  std::string needle(query);
  std::transform(needle.begin(), needle.end(), needle.begin(), lower);
  return haystack.find(needle) != std::string::npos;
}
std::string materialLabel(const LinearElasticity::Material& m, double yield) {
  if (const char* name = presetName(m, yield)) return name;
  char text[48];
  std::snprintf(text, sizeof(text), "Custom, %.3g GPa", m.youngsModulus / 1e9);
  return text;
}
const char* mateIcon(fem_workbench::MateType type) {
  using T = fem_workbench::MateType;
  switch (type) {
    case T::Coincident:
      return kIconArrowsToLine;
    case T::Parallel:
      return kIconParallel;
    case T::Perpendicular:
      return kIconRightAngle;
    case T::Tangent:
      return kIconTangent;
    case T::Concentric:
      return kIconBullseye;
    case T::Distance:
      return kIconRuler;
    case T::Angle:
      return kIconProtractor;
    case T::Lock:
      return kIconLock;
    case T::ProfileCenter:
      return kIconCrosshairs;
    case T::Width:
      return kIconWidth;
  }
  return kIconArrowsToLine;
}
ImU32 kindColor(Kind kind) {
  if (isConnection(kind)) return toImColor(kReferenceColor);
  return toImColor(isLoad(kind) ? kLoadColor : kSupportColor);
}
const char* kindIcon(Kind kind) {
  switch (kind) {
    case Kind::Fixed:
      return kIconAnchor;
    case Kind::Roller:
      return kIconTire;
    case Kind::Displacement:
      return kIconArrows;
    case Kind::Force:
      return kIconArrowDown;
    case Kind::Pressure:
      return kIconCompress;
    case Kind::Traction:
      return kIconArrowsToLine;
    case Kind::Gravity:
      return kIconGlobe;
    case Kind::Tie:
      return kIconLink;
    case Kind::BoltConnector:
      return kIconScrew;
    case Kind::Hinge:
      return kIconRotate;
    case Kind::BallJoint:
      return kIconCircle;
    case Kind::SliderJoint:
      return kIconSlide;
    case Kind::RigidLink:
      return kIconLock;
    default:
      return kIconPlus;
  }
}
std::string extension(const std::string& path) {
  auto e = raisim::pathFromUtf8(path).extension().string();
  std::transform(e.begin(), e.end(), e.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  return e;
}

template <class Position>
int pickTriangle(const glm::mat4& viewProjection, float ndcX, float ndcY,
                 size_t triangles, Position&& position) {
  const glm::mat4 inverse = glm::inverse(viewProjection);
  const glm::vec4 a = inverse * glm::vec4(ndcX, ndcY, -1.f, 1.f),
                  b = inverse * glm::vec4(ndcX, ndcY, 1.f, 1.f);
  const glm::vec3 origin = glm::vec3(a) / a.w,
                  direction = glm::normalize(glm::vec3(b) / b.w - origin);
  float nearest = std::numeric_limits<float>::infinity();
  int face = -1;
  for (size_t f = 0; f < triangles; ++f) {
    const glm::vec3 p = position(3 * f), e1 = position(3 * f + 1) - p,
                    e2 = position(3 * f + 2) - p;
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

void addTriangle(std::vector<Vertex>& out, const glm::vec3& a,
                 const glm::vec3& b, const glm::vec3& c,
                 const glm::vec4& color) {
  const auto cross = glm::cross(b - a, c - a);
  if (!(glm::length(cross) > 0.f)) return;
  const glm::vec3 normal = glm::normalize(cross);
  for (const auto& p : {a, b, c}) {
    Vertex v{};
    v.Position = p;
    v.Normal = normal;
    v.Color = color;
    out.push_back(v);
  }
}
void addCone(std::vector<Vertex>& out, const glm::vec3& tip,
             const glm::vec3& base, float radius, const glm::vec4& color) {
  const glm::vec3 axis = glm::normalize(tip - base);
  const glm::vec3 side = glm::normalize(glm::cross(
      axis, std::abs(axis.z) < .9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0)));
  const glm::vec3 up = glm::cross(axis, side);
  constexpr int kSegments = 12;
  for (int i = 0; i < kSegments; ++i) {
    const float a0 = 6.2831853f * float(i) / kSegments,
                a1 = 6.2831853f * float(i + 1) / kSegments;
    const glm::vec3 p0 = base + radius * (std::cos(a0) * side + std::sin(a0) * up),
                    p1 = base + radius * (std::cos(a1) * side + std::sin(a1) * up);
    addTriangle(out, p0, p1, tip, color);
    addTriangle(out, base, p1, p0, color);
  }
}
void addArrow(std::vector<Vertex>& out, const glm::vec3& from,
              const glm::vec3& to, float radius, const glm::vec4& color) {
  const float length = glm::length(to - from);
  if (!(length > 0.f)) return;
  const glm::vec3 axis = (to - from) / length;
  const float head = std::min(length * .4f, radius * 6.f);
  const glm::vec3 neck = to - head * axis;
  const glm::vec3 side = glm::normalize(glm::cross(
      axis, std::abs(axis.z) < .9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0)));
  const glm::vec3 up = glm::cross(axis, side);
  constexpr int kSegments = 8;
  for (int i = 0; i < kSegments; ++i) {
    const float a0 = 6.2831853f * float(i) / kSegments,
                a1 = 6.2831853f * float(i + 1) / kSegments;
    const glm::vec3 r0 = radius * (std::cos(a0) * side + std::sin(a0) * up),
                    r1 = radius * (std::cos(a1) * side + std::sin(a1) * up);
    addTriangle(out, from + r0, from + r1, neck + r1, color);
    addTriangle(out, from + r0, neck + r1, neck + r0, color);
  }
  addCone(out, to, neck, radius * 2.5f, color);
}

struct Workbench {
  struct SolveTask {
    std::thread worker;
    std::atomic<int> stage{0}, step{0};
    std::atomic<double> progress{0.};
    std::atomic<bool> cancel{false}, done{false};
    std::exception_ptr error;
    Model::AdaptiveResult result;
    size_t startCells = 0;
  };
  Model model;
  std::unique_ptr<SolveTask> solveTask;
  std::chrono::steady_clock::time_point busySince, progressRead;
  fem_workbench::MeshJob::Progress meshProgress;
  double shownProgress = 0.;
  int shownStep = 0;
  int pickingReference = -1;
  bool solveRequested = false;
  std::shared_ptr<raisin::RayraiWindow> viewer;
  std::shared_ptr<raisin::Visuals> solid, glyphs, boundaries, meshLines;
  size_t glyphCount = 0, cadFaceCount = 0;
  bool boundariesBuilt = false;
  std::vector<std::array<unsigned int, 2>> boundaryEdges, surfaceEdges;
  std::vector<std::array<size_t, 2>> boundaryEdgeFaces, surfaceEdgeFaces;
  std::vector<Vertex> vertices, glyphVertices;
  std::unique_ptr<fem_workbench::MeshJob> meshJob;
  struct FileJob {
    std::unique_ptr<fem_workbench::MeshJob> job;
    size_t file = 0;
    bool started = false;
  };
  std::vector<FileJob> fileJobs;
  fem_workbench::CadFile pendingFile;
  bool appendJob = false, appendNextImport = false;
  std::vector<std::string> queuedFiles;
  fem_workbench::GeometrySource pendingSource;
  bool previewJob = false;
  double pendingEdge = 0., pendingTolerance = 0., surfaceTolerance = 0.,
         actualMaximumEdge = 0.;
  int deformationMode = 0, bands = 0;
  double customDeformation = 100., animationFactor = 1.;
  bool smoothContours = true, animate = false, labelExtremes = true,
       labelFieldRange = false, labelConnections = true, probing = false;
  std::vector<std::optional<Model::ConnectionLoad>> connectionLoads;
  const LinearElasticity* loadsSolver = nullptr;
  bool loadsStale = true;
  float highlightAbove = 0.f;
  std::vector<double> nodalValues, yieldCache;
  std::vector<size_t> probes;
  bool publishedPreview = true, shownGeometry = false;
  size_t shownParts = 0;
  double publishedMeshedEdge = 0., publishedTolerance = 0.;
  std::string importName, shownName, errorText,
      message =
          "Open a STEP, IGES or BREP solid. Click a face, apply a support or "
          "load to it, then Solve.";
  std::array<char, 2048> file{}, project{}, exportPath{}, nativeMesher{};
  double importScale = 1., elementSize = .05, pendingScale = 1.,
         deformation = 100., colorMin = 0., colorMax = 1.;
  double boundsMin[3]{}, boundsMax[3]{};
  double plane = 0., tolerance = .0001;
  int planeAxis = 0, field = 0, edited = -1;
  int hoveredFace = -1, hoveredSurface = 0, hoveredPart = 0;
  Vec<3> cylinderCenter{0., 0., 0.}, cylinderAxis{0., 0., 1.};
  double cylinderRadius = .05, cylinderMinimum = -1., cylinderMaximum = 1.;
  bool autoSolve = false, autoRange = true, edges = false, undeformed = false,
       visualDirty = true, append = false, resultView = false,
       adaptive = true;
  double targetError = 10., minimumEdge = 0.;
  int refinementSteps = 3, elementLimit = 400000;
  std::vector<int> hiddenParts;
  /// Parts hidden with a right-click; they come back when a face is picked.
  std::vector<int> peekHidden;
  int rightFace = -1;
  ImVec2 rightPress{};
  bool rightMoved = false;
  std::vector<unsigned char> shownNodes;
  int activePart = -1, requestedTab = -1, shownTab = 0;
  enum class Editor { None, Mate, Connection };
  Editor editor = Editor::None;
  fem_workbench::MateType editorMate = fem_workbench::MateType::Coincident;
  Kind editorKind = Kind::Tie;
  std::array<std::vector<size_t>, 2> editorFaces;
  int editorBox = 0, editingMate = -1, editorMover = -1;
  fem_workbench::MateOptions editorOptions, editorResolved;
  std::vector<fem_workbench::MateType> editorFits;
  bool editorDirty = true, editorFailed = false;
  std::string editorStatus;
  std::vector<fem_workbench::Pose> previewMotions;
  ImVec2 boxStart{}, pressPosition{};
  enum class Drag { None, Pending, Orbit, Pan, Move, Label };
  struct LabelBox {
    int id;
    ImVec2 low, high, point;
  };
  static constexpr int kStressLabel = 0, kDisplacementLabel = 1,
                       kSafetyLabel = 2, kFieldMaxLabel = 3,
                       kFieldMinLabel = 4, kProbeLabel = 16,
                       kConnectionLabel = 4096;
  std::vector<LabelBox> labelBoxes;
  std::vector<std::pair<int, ImVec2>> labelOffsets;
  int draggedLabel = -1;
  ImVec2 labelGrab{}, labelTarget{};
  Drag drag = Drag::None;
  int dragButton = 0, pressFace = -1, dragPart = -1;
  glm::vec3 dragAnchor{0.f}, dragNormal{0.f, 0.f, 1.f};
  fem_workbench::Pose dragMotion;
  Eigen::Vector3d dragCenter = Eigen::Vector3d::Zero();
  bool dragRotate = false;
  std::vector<int> nodePart;
  bool moveParts = true, revealPart = false;
  ImFont* toolbarIcons = nullptr;
  float legendWidth = 240.f;
  std::vector<size_t> quickFaces, quickA, quickB;
  std::vector<double> quickKey;
  fem_workbench::MateSuggestion quickSuggestion;
  std::vector<int> selectionParts;
  ImVec2 quickAnchor{}, quickSize{}, lastClick{};
  bool quickDismissed = false, quickMore = false, quickMoveSecond = false,
       quickMoreMates = false;
  std::vector<fem_workbench::MateType> quickTypes;
  int quickHovered = -1;
  std::string quickHoverError, quickAdded;
  std::vector<fem_workbench::Pose> quickAddedMotions;
  std::vector<double> quickAddedKey, editorKey;
  int materialCategory = 0;
  std::array<char, 64> materialSearch{};
  std::array<double, 3> moveBy{}, rotateBy{};
  bool selectingBox = false, boxSelect = false, pressed = false;
  glm::mat4 screenMatrix{1.f};
  float sidebarWidth = 380.f;

  ~Workbench() {
    if (solveTask) {
      solveTask->cancel = true;
      solveTask->worker.join();
    }
  }
  Workbench() {
    std::snprintf(project.data(), project.size(), "%s", "analysis.rfem");
    std::snprintf(exportPath.data(), exportPath.size(), "%s", "analysis.vtk");
  }
  template <class F>
  void attempt(F&& operation) {
    try {
      operation();
    } catch (const std::exception& e) {
      message = errorText = e.what();
    }
  }
  void updateBounds() {
    const auto& x = model.mesh().vertices;
    for (int a = 0; a < 3; ++a) {
      boundsMin[a] = boundsMax[a] = x[0][a];
      for (const auto& p : x) {
        boundsMin[a] = std::min(boundsMin[a], p[a]);
        boundsMax[a] = std::max(boundsMax[a], p[a]);
      }
    }
  }
  glm::vec3 boundsLow() const {
    return {float(boundsMin[0]), float(boundsMin[1]), float(boundsMin[2])};
  }
  glm::vec3 boundsHigh() const {
    return {float(boundsMax[0]), float(boundsMax[1]), float(boundsMax[2])};
  }
  void fit() {
    updateBounds();
    const glm::vec3 low = boundsLow(), high = boundsHigh();
    auto& cam = viewer->getCamera();
    const float extent = std::max(glm::length(high - low), .001f);
    fem_workbench::frameMeshCamera(cam, low, high);
    cam.setMinimumOrbitDistance(std::max(1e-7f, extent * 1e-4f));
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
    if (glyphs) {
      viewer->removeVisualObject("fem_glyphs");
      glyphs.reset();
    }
    if (boundaries) {
      viewer->removeVisualObject("fem_boundaries");
      boundaries.reset();
    }
    if (meshLines) {
      viewer->removeVisualObject("fem_mesh_edges");
      meshLines.reset();
    }
    boundariesBuilt = false;
    boundaryEdges.clear();
    boundaryEdgeFaces.clear();
    surfaceEdges.clear();
    surfaceEdgeFaces.clear();
    vertices.clear();
    hoveredFace = -1;
    cadFaceCount = 0;
    shownNodes.clear();
    closeEditor();
    probes.clear();
    labelOffsets.clear();
    labelBoxes.clear();
    pickingReference = -1;
    if (!model.hasGeometry()) return;
    if (model.hasCadFaces()) {
      std::vector<size_t> all(model.faceSurfaces().size());
      std::iota(all.begin(), all.end(), size_t(0));
      cadFaceCount = surfaceCount(all);
    }
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
    updateBounds();
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
           model.source.meshedTolerance != effectiveTolerance();
  }
  void syncSettings() {
    model.source.maximumEdge = elementSize;
    model.source.surfaceTolerance = surfaceTolerance;
  }
  Model::Adaptivity adaptivity() const {
    Model::Adaptivity settings;
    settings.relativeError = targetError / 100.;
    settings.maximumSteps = adaptive ? refinementSteps : 0;
    settings.minimumEdge = minimumEdge;
    settings.maximumCells = size_t(std::max(elementLimit, 1));
    if (!(settings.relativeError > 0.) || !(settings.relativeError < 1.) ||
        refinementSteps < 0 || refinementSteps > 10 ||
        !(settings.minimumEdge >= 0.) || !std::isfinite(settings.minimumEdge) ||
        elementLimit < 1000)
      throw std::runtime_error(
          "Adaptive refinement needs a target error between 0 and 100%, 0-10 "
          "steps, a nonnegative minimum edge and an element limit of at least "
          "1000");
    return settings;
  }
  void finishSolve() {
    if (solveTask) throw std::runtime_error("A solve is already running");
    const auto settings = adaptivity();
    updateVisual();
    auto task = std::make_unique<SolveTask>();
    auto* state = task.get();
    state->startCells = model.mesh().tetrahedra.size();
    task->worker = std::thread([this, state, settings] {
      try {
        state->result = model.solveAdaptive(
            settings,
            [state](int step, LinearElasticity::SolveStage stage, double part) {
              if (state->cancel) throw std::runtime_error("Solve cancelled.");
              constexpr double kBegin[] = {0., .05, .2, .95};
              constexpr double kSpan[] = {.05, .15, .75, .05};
              const int index = static_cast<int>(stage);
              state->step = step;
              state->stage = index;
              state->progress = kBegin[index] + kSpan[index] * part;
            });
      } catch (...) {
        state->error = std::current_exception();
      }
      state->done = true;
    });
    solveTask = std::move(task);
    busy();
    message = "Solving...";
  }
  void busy() {
    busySince = std::chrono::steady_clock::now();
    shownProgress = 0.;
    shownStep = 0;
    meshProgress = {};
  }
  void waitForSolve() {
    if (!solveTask) return;
    solveTask->worker.join();
    pollSolve();
  }
  void pollSolve() {
    if (!solveTask || !solveTask->done) return;
    if (solveTask->worker.joinable()) solveTask->worker.join();
    const auto error = solveTask->error;
    const auto result = solveTask->result;
    const size_t startCells = solveTask->startCells;
    solveTask.reset();
    loadsStale = true;
    visualDirty = true;
    if (model.mesh().tetrahedra.size() != startCells) replaceVisual(false);
    if (error) std::rethrow_exception(error);
    const auto& s = model.solver->statistics();
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - busySince)
                               .count();
    char text[320];
    if (result.refinements > 0)
      std::snprintf(text, sizeof(text),
                    "Solved in %.1f s | refined %d time%s, %zu -> %zu "
                    "elements | estimated error %.1f%%%s",
                    seconds, result.refinements,
                    result.refinements == 1 ? "" : "s", startCells,
                    model.mesh().tetrahedra.size(),
                    100. * result.estimatedError,
                    result.converged ? "" : " (refinement limit reached)");
    else
      std::snprintf(
          text, sizeof(text),
          "Solved: %.2f ms%s | residual %.2e | estimated error %.1f%%",
          s.assemblyMs + s.factorizationMs + s.solveMs,
          s.reusedFactorization
              ? (s.iterative ? " (cached preconditioner)" : " (cached factor)")
              : "",
          s.relativeResidual, 100. * result.estimatedError);
    message = text;
  }
  void solve() {
    if (solveTask) throw std::runtime_error("A solve is already running");
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
    const std::string ext =
        pendingSource.volume ? ".rvm" : pendingSource.extension;
    const auto bytes = pendingSource.volume
                           ? fem_workbench::encodeVolume(*pendingSource.volume)
                           : pendingSource.bytes;
    std::ofstream output(job->directory / ("source" + ext), std::ios::binary);
    output.write(bytes.data(), std::streamsize(bytes.size()));
    output.close();
    if (!output) throw std::runtime_error("Cannot prepare source geometry");
    pendingEdge = elementSize;
    pendingTolerance = effectiveTolerance();
    job->start(nativeMesher.data(), ext, pendingEdge / pendingSource.scale,
               pendingTolerance / pendingSource.scale);
    previewJob = appendJob = false;
    meshJob = std::move(job);
    fileJobs.clear();
    for (size_t i = 0; i < pendingSource.assembly.size(); ++i) {
      FileJob file;
      file.job = std::make_unique<fem_workbench::MeshJob>();
      file.file = i;
      fileJobs.push_back(std::move(file));
    }
    busy();
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
    resultView = true;
    replaceVisual();
    model.source.volume = std::make_shared<TetMesh>(model.mesh());
    syncSettings();
    model.source.meshedEdge = elementSize;
    model.source.meshedTolerance = effectiveTolerance();
    message = "Demo solved. Change the tip force to reuse the factorization.";
  }
  double mappingTolerance() const {
    const double diagonal =
        std::hypot(boundsMax[0] - boundsMin[0], boundsMax[1] - boundsMin[1],
                   boundsMax[2] - boundsMin[2]);
    return std::max({diagonal * 1e-8, model.source.previewTolerance * 2.,
                     model.source.meshedTolerance * 4.});
  }
  void clearModel() {
    if (solveTask)
      throw std::runtime_error("Wait for the solve to finish or cancel it");
    meshJob.reset();
    fileJobs.clear();
    appendJob = false;
    model = Model();
    requestedTab = kModelTab;
    hiddenParts.clear();
    edited = activePart = -1;
    closeEditor();
    resultView = false;
    replaceVisual();
    message = "Model cleared. Open or drop a STEP, IGES or BREP solid.";
  }
  /// Opens or adds dropped and command-line files one at a time: CAD files
  /// join an open CAD model, anything else replaces it.
  void importQueued() {
    if (queuedFiles.empty() || meshJob || solveTask) return;
    const auto path = queuedFiles.front();
    queuedFiles.erase(queuedFiles.begin());
    attempt([&] {
      const auto ext = extension(path);
      appendNextImport = canAddCad() && (ext == ".step" || ext == ".stp" ||
                                         ext == ".iges" || ext == ".igs" ||
                                         ext == ".brep");
      import(path);
    });
#if defined(__EMSCRIPTEN__)
    const auto directory = raisim::pathFromUtf8(path).parent_path();
    if (directory.parent_path() == std::filesystem::path("/examples/rsc") &&
        directory.filename().string().rfind("fem-import-", 0) == 0) {
      std::error_code error;
      std::filesystem::remove_all(directory, error);
      if (error) message += " Could not release imported source files.";
    }
#endif
  }
  bool canAddCad() const {
    const auto& e = model.source.extension;
    return model.hasGeometry() && !model.source.bytes.empty() &&
           (e == ".step" || e == ".stp" || e == ".brep" || e == ".iges" ||
            e == ".igs");
  }
  void import(const std::string& path) {
    const bool append = appendNextImport;
    appendNextImport = false;
    if (solveTask)
      throw std::runtime_error("Wait for the solve to finish or cancel it");
    if (!std::isfinite(importScale) || importScale <= 0.)
      throw std::runtime_error(
          "Import scale must be positive (metres per file unit)");
    const auto ext = extension(path);
    const auto title = raisim::pathUtf8(raisim::pathFromUtf8(path).filename());
    if (ext == ".rfem") {
      if (append)
        throw std::runtime_error(
            "Add CAD takes STEP, IGES or BREP files; open projects with Load "
            "project");
      Model next;
      next.loadProject(path);
      meshJob.reset();
      model = std::move(next);
      elementSize = model.source.maximumEdge;
      surfaceTolerance = model.source.surfaceTolerance;
      edited = -1;
      resultView = false;
      replaceVisual();
      if (model.solver && !model.source.volume && model.source.bytes.empty())
        model.source.volume = std::make_shared<TetMesh>(model.mesh());
      message = "Project loaded. Solve uses its current meshing settings.";
      return;
    }
    if (ext != ".step" && ext != ".stp" && ext != ".brep" && ext != ".iges" &&
        ext != ".igs")
      throw std::runtime_error(
          "Only B-rep CAD solids can be opened: STEP (.step .stp), IGES "
          "(.iges .igs) or BREP (.brep). Saved .rfem projects open too.");
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
    if (append && !canAddCad())
      throw std::runtime_error(
          "Open a STEP, IGES or BREP solid before adding another one");
    if (append && meshJob)
      throw std::runtime_error(
          "Wait for meshing to finish or cancel it before adding parts");
    meshJob.reset();
    fileJobs.clear();
    job->start(nativeMesher.data(), ext, 1.,
               source.surfaceTolerance / source.scale, true);
    if (append) {
      pendingFile = {title, ext, std::move(source.bytes), source.scale, 0};
    } else {
      pendingSource = std::move(source);
    }
    appendJob = append;
    previewJob = true;
    meshJob = std::move(job);
    busy();
    importName = title;
    message = "Reading CAD faces for display...";
  }
  size_t meshingSlots() const {
#if defined(__EMSCRIPTEN__)
    return 1;
#else
    return std::max<size_t>(1, std::thread::hardware_concurrency() / 2);
#endif
  }
  void startFileJob(FileJob& file) {
    const auto& cad = pendingSource.assembly.at(file.file);
    std::ofstream out(file.job->directory / ("source" + cad.extension),
                      std::ios::binary);
    out.write(cad.bytes.data(), std::streamsize(cad.bytes.size()));
    out.close();
    if (!out) throw std::runtime_error("Cannot prepare " + cad.name);
    file.job->start(nativeMesher.data(), cad.extension,
                    pendingEdge / cad.scale, pendingTolerance / cad.scale);
    file.started = true;
  }
  void pollMesh() {
    if (!meshJob) return;
    bool done = meshJob->poll();
    size_t running = done ? 0 : 1;
    for (auto& file : fileJobs)
      if (file.started && !file.job->poll()) ++running;
    for (auto& file : fileJobs)
      if (!file.started && running < meshingSlots()) {
        startFileJob(file);
        ++running;
      }
    for (auto& file : fileJobs) done = done && file.started && file.job->poll();
    if (!done) return;
    auto job = std::move(meshJob);
    auto files = std::move(fileJobs);
    fileJobs.clear();
    if (previewJob && appendJob) {
      appendJob = false;
      auto surface = job->surfaceResult(pendingFile.scale);
      pendingFile.regionOffset = model.addSurfacePart(
          std::move(surface), pendingFile.name, mappingTolerance());
      model.source.assembly.push_back(std::move(pendingFile));
      requestedTab = kPartsTab;
      edited = -1;
      resultView = false;
      replaceVisual(false);
      message = "Added " + model.source.assembly.back().name +
                ". Select it under Parts to set its material, and tie or bolt "
                "it to the other parts.";
      return;
    }
    if (previewJob) {
      auto surface = job->surfaceResult(pendingSource.scale);
      model.setSurface(std::move(surface), importName);
      model.source = std::move(pendingSource);
      for (auto& part : model.parts)
        part.name = model.parts.size() == 1
                        ? importName
                        : importName + " / solid " + std::to_string(part.region);
      requestedTab = model.parts.size() > 1 ? kPartsTab : kLoadsTab;
      edited = -1;
      resultView = false;
      replaceVisual();
      model.source.recordPreviewAccuracy();
      syncSettings();
      message = model.hasCadFaces()
                    ? "CAD displayed. Click a face to select it, then apply "
                      "a support or load."
                    : "The mesher reported no B-rep faces; faces are grouped "
                      "by smooth patches. Update the RaiSim FEM mesher.";
      return;
    }
    if (elementSize != pendingEdge ||
        effectiveTolerance() != pendingTolerance) {
      message =
          "Meshing settings changed. Press Solve to generate the new "
          "resolution.";
      return;
    }
    TetMesh volume;
    {
      auto primary = job->result(pendingSource.scale);
      if (pendingSource.regionOffset == 0)
        volume = std::move(primary);
      else
        fem_workbench::appendVolume(volume, std::move(primary),
                                    pendingSource.regionOffset);
    }
    for (auto& file : files) {
      const auto& cad = pendingSource.assembly.at(file.file);
      auto part = file.job->result(cad.scale);
      fem_workbench::appendVolume(volume, std::move(part), cad.regionOffset);
    }
    fem_workbench::removeRegions(volume, pendingSource.excludedRegions);
    if (!pendingSource.volume && pendingSource.extension != ".rfs") {
      std::vector<std::pair<int, fem_workbench::Pose>> placements;
      for (const auto& part : model.parts)
        if (!Model::placement(part).isIdentity())
          placements.emplace_back(part.region, Model::placement(part));
      fem_workbench::transformRegions(volume, placements);
    }
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
      if (configureOnly && solveTask) {
        message = "Wait for the solve to finish or cancel it";
      } else if (configureOnly) {
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
      std::string text(static_cast<size_t>(size), '\0');
      EM_ASM(
          {
            HEAPU8.set(new TextEncoder().encode(globalThis['raisimFemError']),
                       $0);
            delete globalThis['raisimFemError'];
          },
          text.data());
      message = errorText = std::move(text);
    }
    if (!solveTask) {
      publishedPreview = !model.solver;
      publishedMeshedEdge = model.source.meshedEdge;
      publishedTolerance = effectiveTolerance();
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
        message.c_str(), int(bool(meshJob)), int(publishedPreview), elementSize,
        publishedMeshedEdge, publishedTolerance, actualMaximumEdge);
#endif
  }
  bool results() const { return resultView && model.hasSolution; }
  void setResultView(bool show) {
    if (show == resultView) return;
    resultView = show;
    pickingReference = -1;
    pressed = selectingBox = false;
    hoveredFace = -1;
    visualDirty = true;
  }
  const char* fieldUnit() const {
    return field == 1 ? "m" : field >= 10 ? "" : "Pa";
  }
  /// Yield strength over von Mises stress; infinite where unloaded and NaN
  /// where the yield strength is unknown.
  double safety(size_t element) const {
    const double yield =
        element < yieldCache.size() ? yieldCache[element] : 0.;
    if (!(yield > 0.)) return std::numeric_limits<double>::quiet_NaN();
    const double stress = model.solver->elementResults()[element].vonMises;
    return stress > 0. ? yield / stress
                       : std::numeric_limits<double>::infinity();
  }
  double scalar(size_t element) const {
    if (!model.hasSolution) return 0.;
    const auto& r = model.solver->elementResults()[element];
    if (field == kSafetyField) return safety(element);
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
    if (!previewMotions.empty() && node < nodePart.size() &&
        nodePart[node] >= 0 && size_t(nodePart[node]) < previewMotions.size()) {
      const Eigen::Vector3d x = previewMotions[size_t(nodePart[node])](
          model.mesh().vertices[node].e());
      p = glm::vec3(float(x.x()), float(x.y()), float(x.z()));
    }
    if (deformed && results())
      p += float(deformation * animationFactor) *
           vector(model.solver->displacements()[node]);
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
    shownNodes.assign(model.mesh().vertices.size(), 0);
    for (size_t f = 0; f < faces.size(); ++f)
      if (!hidden(model.faceParts()[f]))
        for (const auto v : faces[f].vertices) shownNodes[v] = 1;
    if (results()) {
      yieldCache = model.elementYield();
      double largest = 0.;
      for (const auto& u : model.solver->displacements())
        largest = std::max(largest, u.e().norm());
      const double extent =
          std::hypot(boundsMax[0] - boundsMin[0], boundsMax[1] - boundsMin[1],
                     boundsMax[2] - boundsMin[2]);
      if (deformationMode == 0)
        deformation = largest > 0. ? .05 * extent / largest : 1.;
      else if (deformationMode == 1)
        deformation = 1.;
      else
        deformation = customDeformation;
      nodalValues.clear();
      if (smoothContours) {
        const auto& mesh = model.mesh();
        nodalValues.assign(mesh.vertices.size(), 0.);
        if (field == 1) {
          for (size_t v = 0; v < mesh.vertices.size(); ++v)
            nodalValues[v] = model.solver->displacements()[v].e().norm();
        } else {
          std::vector<double> values(mesh.tetrahedra.size());
          for (size_t t = 0; t < values.size(); ++t) values[t] = scalar(t);
          nodalValues = fem_workbench::nodalAverage(mesh, values);
        }
      }
    }
    if (autoRange && results()) {
      colorMin = std::numeric_limits<double>::infinity();
      colorMax = -colorMin;
      for (size_t f = 0; f < faces.size(); ++f) {
        if (hidden(model.faceParts()[f])) continue;
        const double s = scalar(faces[f].element);
        if (!std::isfinite(s)) continue;
        colorMin = std::min(colorMin, s);
        colorMax = std::max(colorMax, s);
      }
      if (!(colorMin <= colorMax)) {
        colorMin = 0.;
        colorMax = field == kSafetyField ? 4. : 0.;
      }
      if (field == 0 || field == 1) colorMin = 0.;
      if (field == kSafetyField)
        colorMax = std::min(colorMax, std::max(4., 4. * colorMin));
    }
    const auto marks = faceMarks();
    vertices.resize(3 * faces.size());
    for (size_t f = 0; f < faces.size(); ++f) {
      const auto& ids = faces[f].vertices;
      const int part = model.faceParts()[f];
      glm::vec3 p[3] = {position(ids[0]), position(ids[1]), position(ids[2])};
      if (hidden(part)) p[1] = p[2] = p[0];
      const auto cross = glm::cross(p[1] - p[0], p[2] - p[0]);
      const glm::vec3 normal =
          glm::length(cross) > 0.f ? glm::normalize(cross) : glm::vec3(0, 0, 1);
      const double element = results() ? scalar(faces[f].element) : 0.;
      for (size_t j = 0; j < 3; ++j) {
        auto& v = vertices[3 * f + j];
        v.Position = p[j];
        v.Normal = normal;
        v.Color = shade(
            results() ? valueColor(nodalValues.empty() ? element
                                                       : nodalValues[ids[j]])
                      : partColor(part),
            marks[f]);
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
    updateGlyphs();
    updateBoundaries();
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
          state['cadFaces'] = $7;
          state['selectedFaces'] = $8;
          globalThis['raisimFemState'] = state;
        },
        model.name.c_str(), model.selected.size(),
        model.mesh().tetrahedra.size(), model.hasSolution,
        model.hasSolution ? model.solver->statistics().relativeResidual : 0.,
        model.hasSolution ? model.solver->statistics().strainEnergy : 0.,
        regions.size(), cadFaceCount, surfaceCount(model.selectedFaces()));
#endif
  }
  void updateBoundaries() {
    const auto& faces = model.boundaryFaces();
    if (!boundariesBuilt) {
      struct Edge {
        unsigned int a, b;
        size_t face;
      };
      std::vector<Edge> edges;
      edges.reserve(3 * faces.size());
      for (size_t f = 0; f < faces.size(); ++f)
        for (int i = 0; i < 3; ++i) {
          const auto u = faces[f].vertices[i], v = faces[f].vertices[(i + 1) % 3];
          edges.push_back({std::min(u, v), std::max(u, v), f});
        }
      std::sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) {
        return x.a < y.a || (x.a == y.a && (x.b < y.b || (x.b == y.b && x.face < y.face)));
      });
      const auto& surfaces = model.faceSurfaces();
      const auto& parts = model.faceParts();
      for (size_t i = 0; i < edges.size();) {
        size_t end = i + 1;
        while (end < edges.size() && edges[end].a == edges[i].a &&
               edges[end].b == edges[i].b)
          ++end;
        const size_t first = edges[i].face, last = edges[end - 1].face;
        surfaceEdges.push_back({edges[i].a, edges[i].b});
        surfaceEdgeFaces.push_back({first, last});
        if (end - i != 2 || surfaces[first] != surfaces[last] ||
            parts[first] != parts[last]) {
          boundaryEdges.push_back({edges[i].a, edges[i].b});
          boundaryEdgeFaces.push_back({first, last});
        }
        i = end;
      }
      boundariesBuilt = true;
    }
    const float extent = float(std::max({boundsMax[0] - boundsMin[0],
                                         boundsMax[1] - boundsMin[1],
                                         boundsMax[2] - boundsMin[2]}));
    updateLines(boundaries, "fem_boundaries", boundaryEdges, boundaryEdgeFaces,
                extent * 2e-4f, glm::vec4(.06f, .06f, .08f, 1.f), 1.5f);
    if (edges)
      updateLines(meshLines, "fem_mesh_edges", surfaceEdges, surfaceEdgeFaces,
                  extent * 1e-4f, glm::vec4(.16f, .18f, .22f, 1.f), 1.f);
    else if (meshLines) {
      viewer->removeVisualObject("fem_mesh_edges");
      meshLines.reset();
    }
  }
  void updateLines(std::shared_ptr<raisin::Visuals>& visual, const char* name,
                   const std::vector<std::array<unsigned int, 2>>& lineEdges,
                   const std::vector<std::array<size_t, 2>>& lineFaces,
                   float offset, const glm::vec4& color, float width) {
    if (lineEdges.empty()) return;
    const auto& faces = model.boundaryFaces();
    std::vector<Vertex> lines(2 * lineEdges.size());
    for (size_t e = 0; e < lineEdges.size(); ++e) {
      if (hidden(model.faceParts()[lineFaces[e][0]]) &&
          hidden(model.faceParts()[lineFaces[e][1]])) {
        lines[2 * e].Position = lines[2 * e + 1].Position =
            position(lineEdges[e][0]);
        continue;
      }
      glm::vec3 normal(0.f);
      for (const auto f : lineFaces[e]) {
        const auto& ids = faces[f].vertices;
        const auto cross = glm::cross(position(ids[1]) - position(ids[0]),
                                      position(ids[2]) - position(ids[0]));
        if (glm::length(cross) > 0.f) normal += glm::normalize(cross);
      }
      if (glm::length(normal) > 0.f) normal = glm::normalize(normal) * offset;
      for (int j = 0; j < 2; ++j) {
        auto& v = lines[2 * e + size_t(j)];
        v.Position = position(lineEdges[e][size_t(j)]) + normal;
        v.Normal = glm::vec3(0.f, 0.f, 1.f);
      }
    }
    if (visual) {
      if (!visual->updateSingleMeshVertices(lines))
        throw std::runtime_error("Cannot update surface lines");
      return;
    }
    std::vector<unsigned int> indices(lines.size());
    std::iota(indices.begin(), indices.end(), 0u);
    auto mesh = std::make_shared<OpenGLMesh>(
        lines, indices, std::vector<Texture>{}, color,
        OpenGLMesh::PrimitiveTopology::Lines);
    mesh->setLineWidth(width);
    auto meshes = std::make_shared<std::vector<std::shared_ptr<OpenGLMesh>>>();
    meshes->push_back(mesh);
    visual = viewer->addVisualCustomMesh(name, meshes, color, false);
    visual->setDetectable(false);
    raisin::Material appearance;
    appearance.unlit = true;
    visual->setMaterialOverride(appearance);
  }
  void updateGlyphs() {
    glyphVertices.clear();
    const auto& faces = model.boundaryFaces();
    const float extent = float(std::max({boundsMax[0] - boundsMin[0],
                                         boundsMax[1] - boundsMin[1],
                                         boundsMax[2] - boundsMin[2]}));
    const float size = extent * .035f, radius = extent * .0035f;
    const auto outward = [&](size_t f) {
      const auto& ids = faces[f].vertices;
      return glm::cross(position(ids[1]) - position(ids[0]),
                        position(ids[2]) - position(ids[0]));
    };
    std::vector<glm::vec3> boundaryNormals;
    for (const auto& c : model.conditions) {
      if (resultView) break;
      if (!c.enabled || c.kind == Kind::Gravity) continue;
      if (c.kind == Kind::Pressure || (c.kind == Kind::Force &&
                                       c.direction == ForceDirection::FaceNormal)) {
        if (c.value[0] == 0.) continue;
        const size_t stride = std::max<size_t>(1, c.faces.size() / 40);
        for (size_t i = 0; i < c.faces.size(); i += stride) {
          if (hidden(model.faceParts()[c.faces[i]])) continue;
          const auto& ids = faces[c.faces[i]].vertices;
          const glm::vec3 normal = outward(c.faces[i]);
          if (!(glm::length(normal) > 0.f)) continue;
          const glm::vec3 center =
              (position(ids[0]) + position(ids[1]) + position(ids[2])) / 3.f;
          const glm::vec3 inward =
              -glm::normalize(normal) * (c.value[0] > 0. ? 1.f : -1.f);
          addArrow(glyphVertices, center - 2.f * size * inward, center, radius,
                   kLoadColor);
        }
        continue;
      }
      const size_t stride = std::max<size_t>(1, c.nodes.size() / 40);
      if (isLoad(c.kind)) {
        Vec<3> load = c.value;
        if (c.kind == Kind::Force) {
          try {
            load = model.forceVector(c);
          } catch (const std::exception&) {
            continue;
          }
        }
        const auto unit = load.e().norm();
        if (unit == 0.) continue;
        const glm::vec3 direction = vector(load) / float(unit);
        for (size_t i = 0; i < c.nodes.size(); i += stride) {
          if (!shownNodes[c.nodes[i]]) continue;
          const auto at = position(c.nodes[i]);
          addArrow(glyphVertices, at - 2.f * size * direction, at, radius,
                   kLoadColor);
        }
        continue;
      }
      std::vector<glm::vec3> normals(model.selected.size(), glm::vec3(0.f));
      for (const auto f : c.faces)
        for (const auto node : faces[f].vertices) normals[node] += outward(f);
      for (size_t i = 0; i < c.nodes.size(); i += stride) {
        const auto node = c.nodes[i];
        if (!shownNodes[node]) continue;
        glm::vec3 normal = normals[node];
        if (!(glm::length(normal) > 0.f)) {
          if (boundaryNormals.empty()) {
            boundaryNormals.assign(model.selected.size(), glm::vec3(0.f));
            for (size_t f = 0; f < faces.size(); ++f)
              for (const auto v : faces[f].vertices)
                boundaryNormals[v] += outward(f);
          }
          normal = boundaryNormals[node];
        }
        if (!(glm::length(normal) > 0.f)) continue;
        const auto at = position(node);
        addCone(glyphVertices, at, at + size * glm::normalize(normal),
                size * .4f, kSupportColor);
      }
    }
    if (glyphs && glyphCount != glyphVertices.size()) {
      viewer->removeVisualObject("fem_glyphs");
      glyphs.reset();
    }
    if (glyphVertices.empty()) return;
    if (glyphs) {
      if (!glyphs->updateSingleMeshVertices(glyphVertices))
        throw std::runtime_error("Cannot update support and load glyphs");
      return;
    }
    std::vector<unsigned int> indices(glyphVertices.size());
    std::iota(indices.begin(), indices.end(), 0u);
    auto meshes = std::make_shared<std::vector<std::shared_ptr<OpenGLMesh>>>();
    meshes->push_back(std::make_shared<OpenGLMesh>(glyphVertices, indices,
                                                   std::vector<Texture>{}));
    glyphs =
        viewer->addVisualCustomMesh("fem_glyphs", meshes, glm::vec4(1.f), false);
    glyphCount = glyphVertices.size();
    glyphs->setUseMeshColor(true);
    glyphs->setDetectable(false);
    glyphs->setTwoSided(true);
    raisin::Material appearance;
    appearance.useVertexColorAsAlbedo = true;
    appearance.roughnessFactor = .6f;
    glyphs->setMaterialOverride(appearance);
  }
  std::vector<unsigned char> faceMarks() const {
    std::vector<unsigned char> marks(model.boundaryFaces().size(), 0);
    if (resultView) return marks;
    for (size_t i = 0; i < model.conditions.size(); ++i) {
      const auto& c = model.conditions[i];
      if (!c.enabled) continue;
      for (const auto f : c.faces)
        if (f < marks.size())
          marks[f] |= isLoad(c.kind) ? kLoadMark : kSupportMark;
    }
    if (hoveredFace >= 0) {
      const auto& surfaces = model.faceSurfaces();
      const auto& parts = model.faceParts();
      for (size_t f = 0; f < marks.size(); ++f)
        if (surfaces[f] == hoveredSurface && parts[f] == hoveredPart)
          marks[f] |= kHoverMark;
    }
    if (edited >= 0 && size_t(edited) < model.conditions.size())
      for (const auto f : model.conditions[size_t(edited)].targetFaces)
        if (f < marks.size()) marks[f] |= kReferenceMark;
    if (editor != Editor::None) {
      for (const auto f : editorFaces[0])
        if (f < marks.size()) marks[f] |= kSelectionMark;
      for (const auto f : editorFaces[1])
        if (f < marks.size()) marks[f] |= kReferenceMark;
      return marks;
    }
    for (const auto f : model.selectedFaces()) marks[f] |= kSelectionMark;
    return marks;
  }
  /// Contour color: blue (low) to red (high), reversed for the factor of
  /// safety, quantized into bands when requested, and grey below the
  /// highlight threshold or where the value is unknown.
  glm::vec4 valueColor(double value) const {
    if (std::isnan(value)) return glm::vec4(.62f, .64f, .67f, 1.f);
    double t = std::clamp((value - colorMin) / std::max(colorMax - colorMin, 1e-30),
                          0., 1.);
    if (field == kSafetyField) t = 1. - t;
    if (bands > 0) t = std::min(1., (std::floor(t * bands) + .5) / bands);
    if (highlightAbove > 0.f && t < highlightAbove * .01)
      return glm::vec4(.72f, .74f, .77f, 1.f);
    return heat(t);
  }
  glm::vec4 partColor(int region) const {
    static const glm::vec4 kPalette[] = {
        {.36f, .57f, .77f, 1.f}, {.80f, .58f, .36f, 1.f},
        {.38f, .68f, .58f, 1.f}, {.64f, .52f, .78f, 1.f},
        {.78f, .46f, .50f, 1.f}, {.72f, .70f, .40f, 1.f}};
    size_t index = 0;
    while (index < model.parts.size() && model.parts[index].region != region)
      ++index;
    auto color = kPalette[index % std::size(kPalette)];
    if (model.parts.size() > 1 && region == activePart)
      color = glm::mix(color, glm::vec4(1.f), .18f);
    return color;
  }
  glm::vec4 shade(glm::vec4 color, unsigned char mark) const {
    if (mark & kSupportMark) color = glm::mix(color, kSupportColor, .75f);
    if (mark & kLoadMark) color = glm::mix(color, kLoadColor, .75f);
    if (mark & kReferenceMark) color = glm::mix(color, kReferenceColor, .8f);
    if (mark & kSelectionMark) color = glm::mix(color, kSelectionColor, .9f);
    if (mark & kHoverMark) color = glm::mix(color, glm::vec4(1.f), .35f);
    return color;
  }
  ExampleApp::Viewport layoutViewport(SDL_Window* window) const {
    int fbW = 0, fbH = 0, windowW = 0, windowH = 0;
    SDL_GL_GetDrawableSize(window, &fbW, &fbH);
    SDL_GetWindowSize(window, &windowW, &windowH);
    const float displayW = float(windowW > 0 ? windowW : fbW),
                displayH = float(windowH > 0 ? windowH : fbH);
    const float scaleX = windowW > 0 ? float(fbW) / float(windowW) : 1.f,
                scaleY = windowH > 0 ? float(fbH) / float(windowH) : 1.f;
    const float left = std::min(sidebarWidth, std::max(0.f, displayW - 1.f));
    ImGui::SetNextWindowPos(ImVec2(left, 0.f));
    ImGui::SetNextWindowSize(ImVec2(displayW - left, displayH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
    ImGui::Begin("Viewer", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoScrollWithMouse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);
    ExampleApp::Viewport viewport;
    viewport.origin = ImGui::GetCursorScreenPos();
    viewport.size = ImVec2(displayW - left, displayH);
    ImGui::Dummy(viewport.size);
    viewport.hovered = ImGui::IsItemHovered();
    viewport.width = std::max(1, int(std::lround(viewport.size.x * scaleX)));
    viewport.height = std::max(1, fbH);
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    viewport.x = (mouse.x - viewport.origin.x) * scaleX;
    viewport.y = (mouse.y - viewport.origin.y) * scaleY;
    return viewport;
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
    return pickTriangle(viewProjection(), float(2. * vp.x / vp.width - 1.),
                        float(1. - 2. * vp.y / vp.height), vertices.size() / 3,
                        [&](size_t i) { return vertices[i].Position; });
  }
  void hover(int face) {
    int surface = 0, part = 0;
    if (face >= 0) {
      surface = model.faceSurfaces()[size_t(face)];
      part = model.faceParts()[size_t(face)];
    }
    if ((face < 0) != (hoveredFace < 0) || surface != hoveredSurface ||
        part != hoveredPart)
      visualDirty = true;
    hoveredFace = face;
    hoveredSurface = surface;
    hoveredPart = part;
  }
  /// Hides the part under a right-click until a face is picked, to reach the
  /// faces behind it.
  void peekHide(int face) {
    if (face < 0 || model.parts.size() < 2) return;
    const int region = model.faceParts()[size_t(face)];
    if (hidden(region)) return;
    setHidden(region, true);
    peekHidden.push_back(region);
    const auto* part = model.findPart(region);
    message = (part ? part->name : std::string("The part")) +
              " is hidden until you pick a face.";
  }
  void revealPeeked() {
    const auto regions = std::move(peekHidden);
    peekHidden.clear();
    for (const auto region : regions) setHidden(region, false);
  }
  void click(int face, bool toggle) {
    pickFace(face, toggle);
    if (face >= 0) revealPeeked();
  }
  void pickFace(int face, bool toggle) {
    if (editor != Editor::None) {
      if (editingMate < 0) pickIntoEditor(face);
      return;
    }
    if (pickingReference >= 0) {
      const auto index = size_t(pickingReference);
      pickingReference = -1;
      if (face < 0 || index >= model.conditions.size()) {
        message = "Reference pick cancelled.";
        return;
      }
      auto& c = model.conditions[index];
      c.targetFaces = model.surfaceFaces(size_t(face));
      c.targetNodes.clear();
      for (const auto f : c.targetFaces)
        for (const auto v : model.boundaryFaces()[f].vertices)
          c.targetNodes.push_back(v);
      std::sort(c.targetNodes.begin(), c.targetNodes.end());
      c.targetNodes.erase(
          std::unique(c.targetNodes.begin(), c.targetNodes.end()),
          c.targetNodes.end());
      c.targetCadFaces = model.wholeCadFaces(c.targetNodes, c.targetFaces);
      model.changed();
      visualDirty = true;
      message = std::string("Force direction set normal to ") +
                (model.hasCadFaces() ? "CAD face " : "surface ") +
                std::to_string(hoveredSurface) + ".";
      return;
    }
    if (face < 0) {
      if (!toggle) {
        model.clearSelection();
        selectionParts.clear();
      }
    } else {
      const int part = model.faceParts()[size_t(face)];
      activePart = part;
      revealPart = true;
      lastClick = ImGui::GetIO().MousePos;
      const auto faces = model.surfaceFaces(size_t(face));
      if (toggle && model.faceSelected(size_t(face))) {
        model.deselectFaces(faces);
      } else {
        if (!toggle) selectionParts.clear();
        model.selectFaces(faces, toggle);
        if (std::find(selectionParts.begin(), selectionParts.end(), part) ==
            selectionParts.end())
          selectionParts.push_back(part);
      }
    }
    visualDirty = true;
  }
  bool cursorRay(const ExampleApp::Viewport& vp, ImVec2 mouse,
                 glm::vec3& origin, glm::vec3& direction) const {
    if (vp.size.x <= 0.f || vp.size.y <= 0.f) return false;
    const float x = 2.f * (mouse.x - vp.origin.x) / vp.size.x - 1.f,
                y = 1.f - 2.f * (mouse.y - vp.origin.y) / vp.size.y;
    const glm::mat4 inverse = glm::inverse(viewProjection());
    const glm::vec4 a = inverse * glm::vec4(x, y, -1.f, 1.f),
                    b = inverse * glm::vec4(x, y, 1.f, 1.f);
    origin = glm::vec3(a) / a.w;
    direction = glm::normalize(glm::vec3(b) / b.w - origin);
    return std::isfinite(direction.x);
  }
  /// The surface point under the cursor, or the point on the cursor ray at the
  /// depth of the orbit target when the cursor is over empty space.
  glm::vec3 cursorPoint(const ExampleApp::Viewport& vp, ImVec2 mouse) const {
    auto& cam = viewer->getCamera();
    glm::vec3 origin, direction;
    if (!cursorRay(vp, mouse, origin, direction)) return cam.target;
    float nearest = std::numeric_limits<float>::infinity();
    for (size_t f = 0; 3 * f + 2 < vertices.size(); ++f) {
      const glm::vec3 p = vertices[3 * f].Position,
                      e1 = vertices[3 * f + 1].Position - p,
                      e2 = vertices[3 * f + 2].Position - p;
      const auto cross = glm::cross(direction, e2);
      const float determinant = glm::dot(e1, cross);
      if (std::abs(determinant) < 1e-20f) continue;
      const float inv = 1.f / determinant;
      const auto t = origin - p;
      const float u = glm::dot(t, cross) * inv;
      if (u < 0.f || u > 1.f) continue;
      const auto q = glm::cross(t, e1);
      const float v = glm::dot(direction, q) * inv;
      const float distance = glm::dot(e2, q) * inv;
      if (v >= 0.f && u + v <= 1.f && distance > 0.f && distance < nearest)
        nearest = distance;
    }
    if (std::isfinite(nearest)) return origin + nearest * direction;
    const glm::vec3 front = glm::normalize(cam.target - cam.position);
    const float depth = glm::dot(cam.target - origin, front),
                along = glm::dot(direction, front);
    return origin + direction * (along > 1e-6f ? depth / along : depth);
  }
  void zoomAt(const glm::vec3& point, float steps) {
    auto& cam = viewer->getCamera();
    const float factor = std::pow(.85f, steps);
    const float before = glm::distance(cam.position, cam.target);
    cam.moveForward(std::log(factor) / std::log(1.2f));
    cam.update(false);
    const float after = glm::distance(cam.position, cam.target);
    const float scale = before > 0.f ? after / before : 1.f;
    cam.target = point + (cam.target - point) * scale;
    cam.update(false);
    fem_workbench::updateMeshCameraClipping(cam);
  }
  void lookAtCenter(bool fit) {
    if (!shownGeometry) return;
    auto& cam = viewer->getCamera();
    const glm::vec3 low = boundsLow(), high = boundsHigh();
    const float current =
        std::max(glm::distance(cam.position, cam.target), 1e-9f);
    cam.target = .5f * (low + high);
    if (fit) {
      const float wanted = std::max(glm::length(high - low), 1e-6f) * 1.6f;
      cam.moveForward(std::log(wanted / current) / std::log(1.2f));
    }
    cam.update(false);
    fem_workbench::updateMeshCameraClipping(cam);
  }
  bool canMoveParts() const {
    return moveParts && !resultView && !solveTask && !meshJob &&
           model.parts.size() > 1 && !boxSelect;
  }
  /// Mouse handling for the 3D view: orbit, pan, zoom about the cursor, face
  /// selection and moving parts. Runs every frame; touches the model only
  /// when no solve is running.
  void input(const ExampleApp::Viewport& vp) {
    screenMatrix = viewProjection();
    const auto& io = ImGui::GetIO();
    if (drag != Drag::Label) draggedLabel = -1;
    const bool solving = bool(solveTask);
    const bool geometry = !solving && model.hasGeometry();
    const bool over = vp.hovered && (solving || model.hasGeometry());
    auto& cam = viewer->getCamera();
    if (over && io.MouseWheel != 0.f && !vertices.empty())
      zoomAt(cursorPoint(vp, io.MousePos), io.MouseWheel);
    const LabelBox* label =
        !solving && resultView && vp.hovered && drag == Drag::None
            ? labelAt(io.MousePos)
            : nullptr;
    if (label && ImGui::IsMouseDoubleClicked(0)) {
      labelOffsets.erase(
          std::remove_if(labelOffsets.begin(), labelOffsets.end(),
                         [&](const auto& entry) {
                           return entry.first == label->id;
                         }),
          labelOffsets.end());
    } else if (label && ImGui::IsMouseClicked(0)) {
      drag = Drag::Label;
      dragButton = 0;
      draggedLabel = label->id;
      labelGrab = ImVec2(io.MousePos.x - label->low.x,
                         io.MousePos.y - label->low.y);
      labelTarget = label->low;
    }
    if (drag == Drag::Label)
      labelTarget = ImVec2(io.MousePos.x - labelGrab.x,
                           io.MousePos.y - labelGrab.y);
    if (over && drag == Drag::None) {
      if (ImGui::IsMouseClicked(2))
        drag = io.KeyCtrl ? Drag::Pan : Drag::Orbit;
      else if (ImGui::IsMouseClicked(1)) {
        drag = Drag::Pan;
        rightPress = io.MousePos;
        rightMoved = false;
        rightFace = geometry ? pick(vp) : -1;
      }
      else if (ImGui::IsMouseClicked(0) && !label &&
               !(geometry && boxSelect && !resultView)) {
        drag = Drag::Pending;
        pressPosition = io.MousePos;
        pressFace = geometry ? pick(vp) : -1;
      }
      dragButton = ImGui::IsMouseClicked(2) ? 2 : ImGui::IsMouseClicked(1) ? 1 : 0;
    }
    if (drag == Drag::Pan && dragButton == 1) {
      const float dx = io.MousePos.x - rightPress.x,
                  dy = io.MousePos.y - rightPress.y;
      rightMoved = rightMoved || dx * dx + dy * dy > 16.f;
    }
    if (drag == Drag::Pending && ImGui::IsMouseDown(0)) {
      const float dx = io.MousePos.x - pressPosition.x,
                  dy = io.MousePos.y - pressPosition.y;
      if (dx * dx + dy * dy > 16.f) {
        const auto* grabbed =
            canMoveParts() && pressFace >= 0 && editor == Editor::None
                ? model.findPart(model.faceParts()[size_t(pressFace)])
                : nullptr;
        if (grabbed && !grabbed->fixed) {
          drag = Drag::Move;
          dragPart = grabbed->region;
          dragRotate = io.KeyShift;
          dragAnchor = cursorPoint(vp, pressPosition);
          dragNormal = glm::normalize(cam.target - cam.position);
          dragMotion = {};
          dragCenter = model.partCenter(dragPart);
          activePart = dragPart;
        } else {
          drag = Drag::Orbit;
        }
      }
    }
    const ImVec2 delta = io.MouseDelta;
    if ((drag == Drag::Orbit || drag == Drag::Pan) &&
        (delta.x != 0.f || delta.y != 0.f)) {
      if (drag == Drag::Orbit) {
        cam.mouseDelta(delta.x * .3f, -delta.y * .3f);
      } else {
        const ImVec2 previous(io.MousePos.x - delta.x, io.MousePos.y - delta.y);
        const auto depthPoint = [&](ImVec2 mouse) {
          glm::vec3 origin, direction;
          cursorRay(vp, mouse, origin, direction);
          const glm::vec3 front = glm::normalize(cam.target - cam.position);
          const float along = glm::dot(direction, front);
          return origin + direction * (glm::dot(cam.target - origin, front) /
                                       std::max(along, 1e-6f));
        };
        cam.target -= depthPoint(io.MousePos) - depthPoint(previous);
      }
      cam.update(false);
      fem_workbench::updateMeshCameraClipping(cam);
    }
    if (drag == Drag::Move) {
      fem_workbench::Pose motion = dragMotion;
      if (dragRotate) {
        const glm::vec3 front = glm::normalize(cam.target - cam.position);
        const glm::vec3 right = glm::normalize(glm::cross(front, cam.up));
        const glm::vec3 up = glm::cross(right, front);
        const float dx = io.MousePos.x - pressPosition.x,
                    dy = io.MousePos.y - pressPosition.y;
        const glm::vec3 sweep = right * dx - up * dy;
        const glm::vec3 axis = glm::cross(-front, sweep);
        const float length = glm::length(axis);
        motion = {};
        if (length > 0.f)
          motion = fem_workbench::Pose::about(
              Eigen::AngleAxisd(
                  .01 * std::hypot(dx, dy),
                  Eigen::Vector3d(axis.x, axis.y, axis.z) / double(length))
                  .toRotationMatrix(),
              dragCenter);
      } else {
        glm::vec3 origin, direction;
        if (cursorRay(vp, io.MousePos, origin, direction)) {
          const float along = glm::dot(direction, dragNormal);
          if (std::abs(along) > 1e-6f) {
            const float t = glm::dot(dragAnchor - origin, dragNormal) / along;
            const glm::vec3 offset = origin + t * direction - dragAnchor;
            motion = {};
            motion.translation = Eigen::Vector3d(offset.x, offset.y, offset.z);
          }
        }
      }
      if (!(motion == dragMotion)) {
        dragMotion = motion;
        try {
          setPreview(model.previewMove(dragPart, dragMotion));
        } catch (const std::exception&) {
          setPreview({});
        }
      }
    }
    if (drag != Drag::None && !ImGui::IsMouseDown(dragButton)) {
      const auto finished = drag;
      drag = Drag::None;
      if (finished == Drag::Move) {
        const int region = dragPart;
        const auto motion = dragMotion;
        dragPart = -1;
        dragMotion = {};
        setPreview({});
        visualDirty = true;
        attempt([&] {
          if (solveTask || meshJob)
            throw std::runtime_error(
                "The part was not moved: a solve or meshing started while "
                "dragging");
          const auto* part = model.findPart(region);
          if (!part) return;
          const auto before = Model::placement(*part);
          model.movePart(region, motion);
          updateBounds();
          part = model.findPart(region);
          const auto after = Model::placement(*part);
          const Eigen::Vector3d moved = after.translation - before.translation;
          const double turned =
              Eigen::AngleAxisd(after.rotation * before.rotation.transpose())
                  .angle() *
              180. / 3.14159265358979323846;
          char text[320];
          if (turned > 1e-9)
            std::snprintf(text, sizeof(text), "Turned %s by %.4g degrees.",
                          part->name.c_str(), turned);
          else
            std::snprintf(text, sizeof(text),
                          "Moved %s by (%.4g, %.4g, %.4g) m.",
                          part->name.c_str(), moved.x(), moved.y(), moved.z());
          message = text;
        });
      } else if (finished == Drag::Pan && dragButton == 1 && !rightMoved &&
                 geometry) {
        peekHide(rightFace);
      } else if (finished == Drag::Pending && geometry && !resultView &&
                 vp.hovered) {
        click(pressFace, append || io.KeyCtrl || io.KeyShift ||
                             (shownTab == kConnectTab && pressFace >= 0));
      } else if (finished == Drag::Pending && geometry && resultView &&
                 probing && pressFace >= 0 && results()) {
        probes.push_back(model.boundaryFaces()[size_t(pressFace)].element);
      }
    }
    if (!geometry) {
      if (!solving) hoveredFace = -1;
      selectingBox = false;
      return;
    }
    if (resultView) {
      selectingBox = false;
      hoveredFace = !ImGui::IsAnyMouseDown() && over && !labelAt(io.MousePos) &&
                            results()
                        ? pick(vp)
                        : -1;
      return;
    }
    if (!ImGui::IsAnyMouseDown()) hover(over ? pick(vp) : -1);
    if (boxSelect) {
      const bool toggle = append || io.KeyCtrl || io.KeyShift;
      if (over && ImGui::IsMouseClicked(0)) {
        boxStart = io.MousePos;
        selectingBox = true;
      }
      if (selectingBox && ImGui::IsMouseReleased(0)) {
        const auto end = io.MousePos;
        if (!toggle) model.clearSelection();
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
        visualDirty = true;
      }
    }
  }
  void overlays(const ExampleApp::Viewport& vp) {
    labelBoxes.clear();
    if (!model.hasGeometry()) return;
    screenMatrix = viewProjection();
    auto* draw = ImGui::GetForegroundDrawList();
    draw->PushClipRect(vp.origin,
                       ImVec2(vp.origin.x + vp.size.x, vp.origin.y + vp.size.y),
                       true);
    const auto& faces = model.boundaryFaces();
    if (undeformed && results())
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
    if (resultView) {
      if (results()) annotations(vp, draw);
      draw->PopClipRect();
      if (hoveredFace >= 0 && results())
        ImGui::SetTooltip(
            "%s: %s%s", kFieldNames[field],
            formatValue(
                scalar(model.boundaryFaces()[size_t(hoveredFace)].element),
                fieldUnit())
                .c_str(),
            probing ? "\nClick to pin a probe here" : "");
      return;
    }
    std::vector<unsigned char> onFace(model.selected.size(), 0);
    for (const auto f : model.selectedFaces())
      for (const auto node : faces[f].vertices) onFace[node] = 1;
    for (unsigned int i = 0; i < model.selected.size(); ++i)
      if (model.selected[i] && !onFace[i]) {
        ImVec2 p;
        if (projectPoint(position(i), vp, p))
          draw->AddCircleFilled(p, 3.f, IM_COL32(255, 190, 30, 255));
      }
    if (selectingBox)
      draw->AddRect(boxStart, ImGui::GetIO().MousePos,
                    IM_COL32(255, 214, 80, 255), 0.f, 0, 1.5f);
    draw->PopClipRect();
    const auto* hoveredPartInfo =
        hoveredFace >= 0 && model.parts.size() > 1
            ? model.findPart(model.faceParts()[size_t(hoveredFace)])
            : nullptr;
    if (hoveredFace >= 0 && !ImGui::IsAnyMouseDown() && !boxSelect)
      ImGui::SetTooltip(
          "%s%s%s %d%s", hoveredPartInfo ? hoveredPartInfo->name.c_str() : "",
          hoveredPartInfo ? "\n" : "",
          model.hasCadFaces() ? "CAD face" : "Surface", hoveredSurface,
          pickingReference >= 0 ? "\nClick to set the force direction normal to it"
          : model.faceSelected(size_t(hoveredFace))
              ? (model.parts.size() > 1
                     ? " (selected)\nCtrl/Shift+click removes it\nRight-click "
                       "hides the part"
                     : " (selected)\nCtrl/Shift+click removes it")
          : model.parts.size() > 1
              ? "\nClick to select, Ctrl/Shift+click to add\nRight-click hides "
                "the part until you pick a face"
              : "\nClick to select, Ctrl/Shift+click to add");
  }
  glm::vec3 elementCenter(size_t element) const {
    glm::vec3 center(0.f);
    for (const auto v : model.mesh().tetrahedra[element])
      center += .25f * position(v);
    return center;
  }
  static std::string formatValue(double value, const char* unit) {
    char text[64];
    const double magnitude = std::abs(value);
    if (!std::isfinite(value))
      std::snprintf(text, sizeof(text), value > 0. ? "unloaded" : "unknown");
    else if (std::string(unit) == "Pa" && magnitude >= 1e6)
      std::snprintf(text, sizeof(text), "%.4g MPa", value / 1e6);
    else if (std::string(unit) == "Pa" && magnitude >= 1e3)
      std::snprintf(text, sizeof(text), "%.4g kPa", value / 1e3);
    else if (std::string(unit) == "m" && magnitude < 1e-3 && magnitude > 0.)
      std::snprintf(text, sizeof(text), "%.4g um", value * 1e6);
    else if (std::string(unit) == "m" && magnitude < 1.)
      std::snprintf(text, sizeof(text), "%.4g mm", value * 1e3);
    else if (!*unit)
      std::snprintf(text, sizeof(text), "%.4g", value);
    else
      std::snprintf(text, sizeof(text), "%.4g %s", value, unit);
    return text;
  }
  /// A label pinned to a point: marker, leader line and a text box placed
  /// diagonally away from the point.
  const ImVec2* labelOffset(int id) const {
    for (const auto& [label, offset] : labelOffsets)
      if (label == id) return &offset;
    return nullptr;
  }
  const LabelBox* labelAt(const ImVec2& mouse) const {
    for (auto box = labelBoxes.rbegin(); box != labelBoxes.rend(); ++box)
      if (mouse.x >= box->low.x && mouse.x <= box->high.x &&
          mouse.y >= box->low.y && mouse.y <= box->high.y)
        return &*box;
    return nullptr;
  }
  void pin(const ExampleApp::Viewport& vp, ImDrawList* draw, int id,
           const glm::vec3& at, const std::string& text, ImU32 accent,
           int slot) {
    ImVec2 point;
    if (!projectPoint(at, vp, point)) return;
    const float dx = slot % 2 ? -1.f : 1.f, dy = slot % 4 < 2 ? -1.f : 1.f;
    const ImVec2 size = ImGui::CalcTextSize(text.c_str());
    ImVec2 low(dx > 0.f ? point.x + 46.f : point.x - 46.f - size.x - 12.f,
               dy > 0.f ? point.y + 46.f : point.y - 46.f - size.y - 8.f);
    if (draggedLabel == id) {
      const ImVec2 offset(labelTarget.x - point.x, labelTarget.y - point.y);
      auto entry = std::find_if(
          labelOffsets.begin(), labelOffsets.end(),
          [&](const auto& value) { return value.first == id; });
      if (entry == labelOffsets.end())
        labelOffsets.emplace_back(id, offset);
      else
        entry->second = offset;
    }
    if (const auto* offset = labelOffset(id))
      low = ImVec2(point.x + offset->x, point.y + offset->y);
    const ImVec2 high(low.x + size.x + 12.f, low.y + size.y + 8.f);
    const ImVec2 end(std::clamp(point.x, low.x, high.x),
                     std::clamp(point.y, low.y, high.y));
    const auto mouse = ImGui::GetIO().MousePos;
    const bool active =
        draggedLabel == id ||
        (drag == Drag::None && vp.hovered && mouse.x >= low.x &&
         mouse.x <= high.x && mouse.y >= low.y && mouse.y <= high.y);
    if (active) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
    draw->AddLine(point, end, IM_COL32(20, 24, 30, 255), 1.5f);
    draw->AddCircleFilled(point, 5.f, accent);
    draw->AddCircle(point, 5.f, IM_COL32(20, 24, 30, 255), 0, 1.5f);
    draw->AddRectFilled(low, high,
                        active ? IM_COL32(36, 42, 52, 245)
                               : IM_COL32(24, 28, 34, 235),
                        4.f);
    draw->AddRect(low, high, accent, 4.f, 0, active ? 2.5f : 1.5f);
    draw->AddText(ImVec2(low.x + 6.f, low.y + 4.f), IM_COL32(240, 242, 245, 255),
                  text.c_str());
    labelBoxes.push_back({id, low, high, point});
  }
  /// What every connection carries in the current solution, computed once per
  /// solve.
  const std::vector<std::optional<Model::ConnectionLoad>>& loads() {
    if (!model.hasSolution || !model.solver) {
      connectionLoads.clear();
      loadsStale = true;
      return connectionLoads;
    }
    if (loadsStale || loadsSolver != model.solver.get() ||
        connectionLoads.size() != model.conditions.size()) {
      connectionLoads.assign(model.conditions.size(), std::nullopt);
      for (size_t i = 0; i < model.conditions.size(); ++i)
        if (model.conditions[i].enabled && isConnection(model.conditions[i].kind))
          try {
            connectionLoads[i] = model.connectionLoad(i);
          } catch (const std::exception&) {
          }
      loadsSolver = model.solver.get();
      loadsStale = false;
    }
    return connectionLoads;
  }
  /// Factor of safety of a connection from its current strength.
  double connectionSafety(const Condition& c,
                          const Model::ConnectionLoad& load) const {
    return c.strength > 0. && load.equivalent > 0. &&
                   (c.kind == Kind::Tie || c.kind == Kind::BoltConnector)
               ? c.strength / load.equivalent
               : std::numeric_limits<double>::infinity();
  }
  void annotations(const ExampleApp::Viewport& vp, ImDrawList* draw) {
    const auto& results = model.solver->elementResults();
    const auto& u = model.solver->displacements();
    if (labelExtremes) {
      size_t stress = 0, node = 0;
      for (size_t e = 1; e < results.size(); ++e)
        if (results[e].vonMises > results[stress].vonMises) stress = e;
      for (size_t v = 1; v < u.size(); ++v)
        if (u[v].e().squaredNorm() > u[node].e().squaredNorm()) node = v;
      size_t weakest = results.size();
      for (size_t e = 0; e < results.size(); ++e) {
        const double value = safety(e);
        if (std::isfinite(value) &&
            (weakest == results.size() || value < safety(weakest)))
          weakest = e;
      }
      std::string peak =
          "Max von Mises " + formatValue(results[stress].vonMises, "Pa");
      if (weakest == stress) {
        char text[48];
        std::snprintf(text, sizeof(text), " | safety %.3g", safety(stress));
        peak += text;
      }
      pin(vp, draw, kStressLabel, elementCenter(stress), peak,
          IM_COL32(230, 60, 50, 255), 0);
      pin(vp, draw, kDisplacementLabel, position(unsigned(node)),
          "Max displacement " + formatValue(u[node].e().norm(), "m"),
          IM_COL32(60, 120, 230, 255), 3);
      if (weakest < results.size() && weakest != stress) {
        char text[64];
        std::snprintf(text, sizeof(text), "Min factor of safety %.3g",
                      safety(weakest));
        pin(vp, draw, kSafetyLabel, elementCenter(weakest), text,
            safety(weakest) < 1. ? IM_COL32(230, 60, 50, 255)
                                 : IM_COL32(240, 170, 40, 255),
            1);
      }
    }
    if (labelFieldRange && field != 0 && field != 1) {
      size_t low = results.size(), high = results.size();
      for (size_t e = 0; e < results.size(); ++e) {
        const double value = scalar(e);
        if (std::isnan(value)) continue;
        if (low == results.size() || value < scalar(low)) low = e;
        if (high == results.size() || value > scalar(high)) high = e;
      }
      if (high < results.size()) {
        const auto red = IM_COL32(230, 60, 50, 255);
        const auto blue = IM_COL32(60, 120, 230, 255);
        const bool reversed = field == kSafetyField;
        pin(vp, draw, kFieldMaxLabel, elementCenter(high),
            std::string("Max ") + kFieldNames[field] + " " +
                formatValue(scalar(high), fieldUnit()),
            reversed ? blue : red, 2);
        if (low != high)
          pin(vp, draw, kFieldMinLabel, elementCenter(low),
              std::string("Min ") + kFieldNames[field] + " " +
                  formatValue(scalar(low), fieldUnit()),
              reversed ? red : blue, 1);
      }
    }
    if (labelConnections) {
      const auto& carried = loads();
      for (size_t i = 0; i < carried.size(); ++i) {
        if (!carried[i]) continue;
        const auto& c = model.conditions[i];
        if (c.kind != Kind::Tie && c.kind != Kind::BoltConnector) continue;
        const auto& load = *carried[i];
        const double safety = connectionSafety(c, load);
        std::string text = c.name + " " + formatValue(load.equivalent, "Pa");
        if (std::isfinite(safety)) {
          char extra[40];
          std::snprintf(extra, sizeof(extra), " | safety %.3g", safety);
          text += extra;
        }
        pin(vp, draw, kConnectionLabel + int(i), position(load.peakNode), text,
            std::isfinite(safety) && safety < 1. ? IM_COL32(230, 60, 50, 255)
                                                 : IM_COL32(40, 190, 220, 255),
            int(i) + 2);
      }
    }
    for (size_t i = 0; i < probes.size(); ++i)
      if (probes[i] < results.size())
        pin(vp, draw, kProbeLabel + int(i), elementCenter(probes[i]),
            "#" + std::to_string(i + 1) + "  " +
                formatValue(scalar(probes[i]), fieldUnit()),
            IM_COL32(120, 200, 120, 255), int(i));
  }
  bool editVector(const char* name, Vec<3>& value, const char* unit = "") {
    bool changed = false;
    for (int a = 0; a < 3; ++a) {
      char label[80];
      std::snprintf(label, sizeof(label), "%s %c%s", name, 'X' + a, unit);
      changed |= axisInput(label, &value[a], a);
    }
    return changed;
  }
  bool editFixture(Condition& c) {
    bool changed = false;
    if (c.kind == Kind::Directional || c.kind == Kind::Slider ||
        c.kind == Kind::Bearing || c.kind == Kind::BoltHole ||
        c.kind == Kind::Spring) {
      ImGui::TextDisabled("Fixture direction / cylinder axis");
      for (int a = 0; a < 3; ++a) {
        if (a) ImGui::SameLine();
        const char* label[] = {"Along X", "Along Y", "Along Z"};
        const bool pressed = axisButton(label[a], a);
        if (pressed) {
          c.axis = {a == 0 ? 1. : 0., a == 1 ? 1. : 0., a == 2 ? 1. : 0.};
          changed = true;
        }
      }
      changed |= editVector("Axis", c.axis);
    }
    if (c.kind == Kind::Hinge || c.kind == Kind::SliderJoint) {
      ImGui::TextDisabled(c.kind == Kind::Hinge ? "Rotation axis"
                                                : "Sliding direction");
      if (ImGui::Button(withIcon(kIconCrosshairs, "Fit to faces").c_str())) {
        auto faces = c.faces;
        faces.insert(faces.end(), c.targetFaces.begin(), c.targetFaces.end());
        if (fem_workbench::fitFaceAxis(model.mesh(), model.boundaryFaces(),
                                       faces, c.axis))
          changed = true;
        else
          message = "These faces do not share one axis; enter it below.";
      }
      for (int a = 0; a < 3; ++a) {
        ImGui::SameLine();
        const char* label[] = {"X", "Y", "Z"};
        const bool pressed = axisButton(label[a], a);
        if (pressed) {
          c.axis = {a == 0 ? 1. : 0., a == 1 ? 1. : 0., a == 2 ? 1. : 0.};
          changed = true;
        }
      }
      changed |= editVector("Axis", c.axis);
    }
    if (c.kind == Kind::Hinge)
      changed |= ImGui::Checkbox("Hold axial position", &c.axes[2]);
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
    }
    if (c.kind == Kind::Directional || c.kind == Kind::Roller ||
        c.kind == Kind::Slider || c.kind == Kind::Spring)
      changed |= ImGui::InputDouble("Displacement / rest (m)", &c.value[0], 0.,
                                    0., "%.6g");
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
    if (c.kind == Kind::Tie || c.kind == Kind::BoltConnector) {
      struct Grade {
        const char* name;
        double strength;
      };
      static constexpr Grade kAdhesives[] = {
          {"Structural epoxy", 25e6},   {"Toughened epoxy", 30e6},
          {"Methacrylate (MMA)", 20e6}, {"Cyanoacrylate", 15e6},
          {"Polyurethane", 8e6},        {"Silicone", 2e6}};
      static constexpr Grade kBolts[] = {
          {"Property class 4.6", 240e6},  {"Property class 5.8", 400e6},
          {"Property class 8.8", 640e6},  {"Property class 10.9", 940e6},
          {"Property class 12.9", 1100e6}, {"Stainless A2-70", 450e6},
          {"Stainless A4-80", 600e6}};
      const bool glue = c.kind == Kind::Tie;
      double mpa = c.strength / 1e6;
      ImGui::SetNextItemWidth(120.f);
      if (ImGui::InputDouble(glue ? "Adhesive strength (MPa)"
                                  : "Bolt yield strength (MPa)",
                             &mpa, 0., 0., "%.4g") &&
          mpa >= 0. && std::isfinite(mpa)) {
        c.strength = mpa * 1e6;
        loadsStale = true;
      }
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip(glue ? "Lap shear strength of the adhesive, for the "
                                 "glue's factor of safety; 0 if unknown"
                               : "Yield strength of the bolt material, for "
                                 "its factor of safety; 0 if unknown");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-FLT_MIN);
      if (ImGui::BeginCombo("##strength", "Typical")) {
        const Grade* begin = glue ? kAdhesives : kBolts;
        const size_t count = glue ? std::size(kAdhesives) : std::size(kBolts);
        for (size_t g = 0; g < count; ++g) {
          char item[64];
          std::snprintf(item, sizeof(item), "%s, %.4g MPa", begin[g].name,
                        begin[g].strength / 1e6);
          if (ImGui::Selectable(item, c.strength == begin[g].strength)) {
            c.strength = begin[g].strength;
            loadsStale = true;
          }
        }
        ImGui::EndCombo();
      }
    }
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
      if (model.hasSolution)
        for (size_t b = 0; b < model.boltConditions.size(); ++b)
          if (model.boltConditions[b] == size_t(edited)) {
            const auto& r = model.solver->boltResults()[b];
            ImGui::Text("Bolt tension: %.6g N", r.axis.e().dot(r.force.e()));
            ImGui::Text("Head / nut moment: %.4g / %.4g Nm",
                        r.headMoment.e().norm(), r.nutMoment.e().norm());
          }
    }
    if (c.kind == Kind::BoltHole || fem_workbench::isConnection(c.kind)) {
      const char* label = c.kind == Kind::BoltConnector
                              ? "Capture second head/nut"
                          : c.kind == Kind::BoltHole ? "Capture seat region"
                          : c.kind == Kind::Tie      ? "Capture master surface"
                                                : "Capture second part's faces";
      if (ImGui::Button(withIcon(kIconPlus, label).c_str())) {
        c.targetNodes.clear();
        for (unsigned int i = 0; i < model.selected.size(); ++i)
          if (model.selected[i]) c.targetNodes.push_back(i);
        c.targetFaces = model.selectedFaces();
        c.targetCadFaces = model.wholeCadFaces(c.targetNodes, c.targetFaces);
        changed = true;
      }
      ImGui::TextDisabled("Second region: %zu nodes / %zu faces",
                          c.targetNodes.size(), c.targetFaces.size());
      if (ImGui::Button(withIcon(kIconCursor, "Show second region").c_str())) {
        model.restoreSelection(c.targetNodes, c.targetFaces);
        visualDirty = true;
      }
    }
    return changed;
  }
  size_t surfaceCount(const std::vector<size_t>& faces) const {
    std::vector<std::pair<int, int>> surfaces;
    for (const auto f : faces)
      surfaces.emplace_back(model.faceParts()[f], model.faceSurfaces()[f]);
    std::sort(surfaces.begin(), surfaces.end());
    return size_t(std::unique(surfaces.begin(), surfaces.end()) -
                  surfaces.begin());
  }
  double area(const std::vector<size_t>& faces) const {
    double total = 0.;
    const auto& x = model.mesh().vertices;
    for (const auto f : faces) {
      const auto& ids = model.boundaryFaces()[f].vertices;
      total += .5 * (x[ids[1]].e() - x[ids[0]].e())
                        .cross(x[ids[2]].e() - x[ids[0]].e())
                        .norm();
    }
    return total;
  }
  void addCondition(Kind kind) {
    attempt([&] {
      auto& c = model.add(kind);
      if (c.kind == Kind::Bearing || c.kind == Kind::BoltHole) {
        c.axis = cylinderAxis;
        c.center = cylinderCenter;
      }
      if (c.kind == Kind::Force && !c.faces.empty()) {
        c.direction = ForceDirection::FaceNormal;
        c.value = {1000., 0., 0.};
      }
      pickingReference = -1;
      edited = int(model.conditions.size()) - 1;
      const size_t surfaces = surfaceCount(c.faces);
      model.clearSelection();
      visualDirty = true;
      message = std::string(fem_workbench::kindName(kind)) +
                (kind == Kind::Gravity ? " applied to the whole body"
                 : surfaces ? " applied to " + std::to_string(surfaces) +
                                  (surfaces == 1 ? " face" : " faces")
                            : " applied to " + std::to_string(c.nodes.size()) +
                                  " nodes") +
                ". Set its values below, select more faces, or press Solve.";
    });
  }
  void progressBar(float width) {
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - busySince)
                               .count();
    char overlay[192];
    if (solveTask) {
      constexpr const char* kStage[] = {"Assembling stiffness", "Factorizing",
                                        "Iterating", "Computing stresses"};
      const int stage = std::clamp(solveTask->stage.load(), 0, 3);
      const int step = solveTask->step.load();
      if (step != shownStep) {
        shownStep = step;
        shownProgress = 0.;
      }
      shownProgress = std::max(shownProgress, solveTask->progress.load());
      char refinement[48] = "";
      if (step > 0)
        std::snprintf(refinement, sizeof(refinement), "Refinement %d: ", step);
      std::snprintf(overlay, sizeof(overlay), "%s%s%s  %.0f%%  %.0f s",
                    solveTask->cancel ? "Cancelling: " : "", refinement,
                    kStage[stage], 100. * shownProgress, elapsed);
      ImGui::ProgressBar(float(shownProgress), ImVec2(width, 0.f), overlay);
      return;
    }
    if (!meshJob) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - progressRead > std::chrono::milliseconds(200)) {
      progressRead = now;
      meshProgress = meshJob->progress();
      if (!fileJobs.empty()) {
        double total =
            meshJob->running ? std::max(0., meshProgress.fraction) : 1.;
        for (const auto& file : fileJobs)
          total += !file.started       ? 0.
                   : file.job->running ? std::max(0., file.job->progress().fraction)
                                       : 1.;
        meshProgress.fraction = total / double(fileJobs.size() + 1);
        meshProgress.stage =
            "Meshing " + std::to_string(fileJobs.size() + 1) + " CAD files";
      }
    }
    if (meshProgress.fraction >= 0.) {
      shownProgress = std::max(shownProgress, meshProgress.fraction);
      std::snprintf(overlay, sizeof(overlay), "%s  %.0f%%  %.0f s",
                    meshProgress.stage.c_str(), 100. * shownProgress, elapsed);
      ImGui::ProgressBar(float(shownProgress), ImVec2(width, 0.f), overlay);
      return;
    }
    std::snprintf(overlay, sizeof(overlay), "%s  %.0f s",
                  previewJob ? "Reading CAD faces" : "Meshing", elapsed);
    ImGui::ProgressBar(0.f, ImVec2(width, 0.f), "");
    const ImVec2 low = ImGui::GetItemRectMin(), high = ImGui::GetItemRectMax();
    const float span = (high.x - low.x) * .2f;
    const float left = low.x - span + (high.x - low.x + span) *
                                          std::fmod(float(ImGui::GetTime()) * .5f, 1.f);
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(ImVec2(std::max(low.x, left), low.y),
                        ImVec2(std::min(high.x, left + span), high.y),
                        ImGui::GetColorU32(ImGuiCol_PlotHistogram, .7f),
                        ImGui::GetStyle().FrameRounding);
    const ImVec2 text = ImGui::CalcTextSize(overlay);
    draw->AddText(ImVec2(low.x + (high.x - low.x - text.x) * .5f,
                         low.y + (high.y - low.y - text.y) * .5f),
                  ImGui::GetColorU32(ImGuiCol_Text), overlay);
  }
  void cancelBusy() {
    if (solveTask) {
      solveTask->cancel = true;
      message = "Cancelling the solve...";
    } else if (meshJob) {
      meshJob.reset();
      fileJobs.clear();
      appendJob = false;
      message = previewJob
                    ? "CAD import cancelled."
                    : "Meshing cancelled; geometry and conditions retained.";
    }
  }
  void errorNotice() {
    if (errorText.empty()) return;
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, .45f, .4f, 1.f));
    ImGui::TextWrapped("%s", errorText.c_str());
    ImGui::PopStyleColor();
    if (ImGui::SmallButton(withIcon(kIconCopy, "Copy").c_str()))
      ImGui::SetClipboardText(errorText.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton(withIcon(kIconCancel, "Dismiss").c_str())) errorText.clear();
    ImGui::Separator();
  }
  bool editForce(Condition& c) {
    bool changed = false;
    int direction = static_cast<int>(c.direction);
    ImGui::SetNextItemWidth(-90.f);
    if (ImGui::Combo("Direction", &direction,
                     "X/Y/Z components\0Normal to its faces\0Normal to a "
                     "reference face\0")) {
      const auto next = static_cast<ForceDirection>(direction);
      if (next != c.direction) {
        const double magnitude = c.direction == ForceDirection::Vector
                                     ? c.value.e().norm()
                                     : std::abs(c.value[0]);
        c.value = next == ForceDirection::Vector ? Vec<3>{0., 0., -magnitude}
                                                 : Vec<3>{magnitude, 0., 0.};
        c.direction = next;
        changed = true;
      }
    }
    if (c.direction == ForceDirection::Vector) {
      for (int a = 0; a < 3; ++a) {
        char label[16];
        std::snprintf(label, sizeof(label), "%c (N)", 'X' + a);
        changed |= axisInput(label, &c.value[a], a);
      }
      return changed;
    }
    changed |=
        ImGui::InputDouble("Magnitude (N)", &c.value[0], 0., 0., "%.6g");
    ImGui::SameLine();
    if (ImGui::SmallButton(withIcon(kIconFlip, "Reverse").c_str())) {
      c.value[0] = -c.value[0];
      changed = true;
    }
    const bool reference = c.direction == ForceDirection::ReferenceNormal;
    ImGui::TextDisabled("%s %s", c.value[0] >= 0. ? "Pushes into" : "Pulls out of",
                        reference ? "the reference face" : "its faces");
    if (!reference) return changed;
    const bool picking = pickingReference == edited;
    if (ImGui::Button(
            withIcon(kIconCrosshairs,
                     picking ? "Click a face in the view..." : "Pick reference face")
                .c_str())) {
      pickingReference = picking ? -1 : edited;
      message = picking ? "Reference pick cancelled."
                        : "Click the face whose normal sets the force direction.";
    }
    if (c.targetFaces.empty())
      ImGui::TextColored(ImVec4(1.f, .78f, .3f, 1.f), "No reference face picked");
    else
      ImGui::TextDisabled("Reference: %zu face%s (shown in cyan)",
                          surfaceCount(c.targetFaces),
                          surfaceCount(c.targetFaces) == 1 ? "" : "s");
    return changed;
  }
  static void credit() {
    ImGui::Separator();
    ImGui::TextDisabled("Icons: Uicons by Flaticon (flaticon.com/uicons)");
  }
  void panel() {
    const auto display = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(0.f, 0.f));
    ImGui::SetNextWindowSize(ImVec2(sidebarWidth, display.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(.075f, .085f, .1f, 1.f));
    ImGui::Begin("FEM Workbench", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    ImGui::PushItemWidth(115.f);
    ImGui::TextColored(ImVec4(.35f, .85f, .9f, 1.f), "LINEAR ELASTICITY");
    if (!solveTask) {
      shownName = model.name;
      shownGeometry = model.hasGeometry();
      shownParts = model.parts.size();
    }
    ImGui::TextWrapped("%s", shownName.c_str());
    modeTabs();
    ImGui::Separator();
    errorNotice();
    if (solveTask) {
      ImGui::TextColored(ImVec4(1.f, .8f, .3f, 1.f), "Solving the model");
      progressBar(-1.f);
      if (ImGui::Button(withIcon(kIconCancel, "Cancel").c_str())) cancelBusy();
      finishPanel(display, false);
      return;
    }
    if (resultView) {
      if (animate && results()) {
        animationFactor = .5 - .5 * std::cos(ImGui::GetTime() * 3.141592653589793);
        visualDirty = true;
      } else if (animationFactor != 1.) {
        animationFactor = 1.;
        visualDirty = true;
      }
      resultPanel();
      finishPanel(display, model.hasGeometry());
      return;
    }
    if (solveButton()) setResultView(true);
    ImGui::SameLine();
    ImGui::Checkbox("Live solve", &autoSolve);
    if (meshJob) {
      progressBar(-1.f);
      if (ImGui::Button(withIcon(kIconCancel, "Cancel").c_str())) cancelBusy();
    }
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4(.13f, .15f, .18f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(.25f, .38f, .58f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_TabActive, ImVec4(.2f, .32f, .5f, 1.f));
    const bool tabs =
        ImGui::BeginTabBar("steps", ImGuiTabBarFlags_FittingPolicyResizeDown);
    ImGui::PopStyleColor(3);
    if (tabs) {
      const auto tab = [&](int index, const char* icon, const char* label) {
        const bool open = ImGui::BeginTabItem(
            withIcon(icon, label).c_str(), nullptr,
            requestedTab == index ? ImGuiTabItemFlags_SetSelected : 0);
        if (open) shownTab = index;
        return open;
      };
      if (tab(kModelTab, kIconFolderOpen, "Model")) {
        modelTab();
        ImGui::EndTabItem();
      }
      if (tab(kPartsTab, kIconLayers, "Parts")) {
        if (requireGeometry()) partsPanel();
        ImGui::EndTabItem();
      }
      if (tab(kLoadsTab, kIconTools, "Loads")) {
        if (requireGeometry()) loadsTab();
        ImGui::EndTabItem();
      }
      if (tab(kConnectTab, kIconPuzzle, "Connect")) {
        if (requireGeometry()) matePanel();
        ImGui::EndTabItem();
      }
      if (tab(kFileTab, kIconDisk, "File")) {
        fileTab();
        ImGui::EndTabItem();
      }
      requestedTab = -1;
      ImGui::EndTabBar();
    }
    finishPanel(display, model.hasGeometry());
  }
  bool materialPicker(LinearElasticity::Material& m, double& yield) {
    const auto& library = fem_workbench::materialLibrary();
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowSize(ImVec2(std::min(1000.f, display.x - 32.f),
                                    std::min(600.f, display.y - 32.f)),
                             ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(ImVec2(.5f * display.x, .5f * display.y),
                            ImGuiCond_Appearing, ImVec2(.5f, .5f));
    bool open = true;
    if (!ImGui::BeginPopupModal("Material library", &open,
                                ImGuiWindowFlags_NoSavedSettings))
      return false;
    bool chosen = false;
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
    ImGui::InputTextWithHint("##search",
                             withIcon(kIconSearch, "Search all materials").c_str(),
                             materialSearch.data(), materialSearch.size());
    const std::string query = materialSearch.data();
    const float footer = ImGui::GetFrameHeightWithSpacing() +
                         ImGui::GetStyle().ItemSpacing.y;
    materialCategory =
        std::clamp(materialCategory, 0, int(library.size()) - 1);
    const auto categoryLabel = [](const fem_workbench::MaterialCategory& c) {
      return std::string(c.name) + " (" + std::to_string(c.grades.size()) +
             ")";
    };
    float categoryWidth = 0.f;
    for (const auto& category : library)
      categoryWidth = std::max(
          categoryWidth, ImGui::CalcTextSize(categoryLabel(category).c_str()).x);
    ImGui::BeginChild(
        "categories",
        ImVec2(categoryWidth + 2.f * ImGui::GetStyle().WindowPadding.x +
                   ImGui::GetStyle().ScrollbarSize,
               -footer),
        ImGuiChildFlags_Border);
    for (int i = 0; i < int(library.size()); ++i) {
      const auto& category = library[size_t(i)];
      const std::string label = categoryLabel(category);
      if (ImGui::Selectable(label.c_str(),
                            query.empty() && i == materialCategory)) {
        materialCategory = i;
        materialSearch[0] = '\0';
      }
    }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("grades", ImVec2(0.f, -footer), ImGuiChildFlags_Border);
    const auto& shown = library[size_t(materialCategory)];
    if (query.empty())
      ImGui::TextDisabled("%s", shown.note);
    else
      ImGui::TextDisabled("Materials matching \"%s\"", query.c_str());
    constexpr ImGuiTableFlags kTable =
        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("grades", 5, kTable)) {
      ImGui::TableSetupScrollFreeze(0, 1);
      ImGui::TableSetupColumn("Material", ImGuiTableColumnFlags_WidthStretch,
                              3.f);
      ImGui::TableSetupColumn("E (GPa)", ImGuiTableColumnFlags_WidthStretch, 1.f);
      ImGui::TableSetupColumn("Poisson", ImGuiTableColumnFlags_WidthStretch, 1.f);
      ImGui::TableSetupColumn("Density (kg/m3)",
                              ImGuiTableColumnFlags_WidthStretch, 1.3f);
      ImGui::TableSetupColumn("Strength (MPa)",
                              ImGuiTableColumnFlags_WidthStretch, 1.2f);
      ImGui::TableHeadersRow();
      for (const auto& category : library) {
        if (query.empty() && &category != &shown) continue;
        for (const auto& grade : category.grades) {
          if (!query.empty() && !containsText(grade.name, query) &&
              !containsText(category.name, query))
            continue;
          const bool current =
              grade.youngsModulus == m.youngsModulus &&
              grade.poissonRatio == m.poissonRatio &&
              grade.density == m.density && grade.strength == yield;
          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          if (ImGui::Selectable(grade.name, current,
                                ImGuiSelectableFlags_SpanAllColumns)) {
            m = {grade.youngsModulus, grade.poissonRatio, grade.density};
            yield = grade.strength;
            chosen = true;
            ImGui::CloseCurrentPopup();
          }
          if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s\n%s", category.name, category.note);
          ImGui::TableSetColumnIndex(1);
          ImGui::Text("%.4g", grade.youngsModulus / 1e9);
          ImGui::TableSetColumnIndex(2);
          ImGui::Text("%.3g", grade.poissonRatio);
          ImGui::TableSetColumnIndex(3);
          ImGui::Text("%.0f", grade.density);
          ImGui::TableSetColumnIndex(4);
          if (grade.strength > 0.)
            ImGui::Text("%.4g", grade.strength / 1e6);
          else
            ImGui::TextDisabled("-");
        }
      }
      ImGui::EndTable();
    }
    ImGui::EndChild();
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled(
        "Typical room-temperature values; a supplier data sheet takes "
        "precedence.");
    ImGui::SameLine(ImGui::GetContentRegionMax().x -
                    ImGui::CalcTextSize(withIcon(kIconCancel, "Cancel").c_str()).x -
                    2.f * ImGui::GetStyle().FramePadding.x);
    if (ImGui::Button(withIcon(kIconCancel, "Cancel").c_str()) || !open)
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    return chosen;
  }
  bool editMaterial(LinearElasticity::Material& m, double& yield) {
    bool changed = false;
    const char* current = presetName(m, yield);
    if (ImGui::Button(withIcon(kIconPalette, std::string("Set material: ") +
                                                (current ? current : "Custom"))
                          .c_str(),
                      ImVec2(-FLT_MIN, 0.f))) {
      const auto& library = fem_workbench::materialLibrary();
      for (size_t i = 0; i < library.size(); ++i)
        for (const auto& grade : library[i].grades)
          if (current && grade.name == current) materialCategory = int(i);
      materialSearch[0] = '\0';
      ImGui::OpenPopup("Material library");
    }
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Choose from %zu engineering materials by family",
                        [] {
                          size_t count = 0;
                          for (const auto& c : fem_workbench::materialLibrary())
                            count += c.grades.size();
                          return count;
                        }());
    changed |= materialPicker(m, yield);
    double mpa = yield / 1e6;
    if (ImGui::InputDouble("Yield strength (MPa)", &mpa, 0., 0., "%.4g") &&
        mpa >= 0. && std::isfinite(mpa)) {
      yield = mpa * 1e6;
      changed = true;
    }
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip(
          "Used for the factor of safety (yield / von Mises); 0 if unknown");
    double gpa = m.youngsModulus / 1e9;
    if (ImGui::InputDouble("Young's modulus (GPa)", &gpa, 0., 0., "%.4g") &&
        gpa > 0. && std::isfinite(gpa)) {
      m.youngsModulus = gpa * 1e9;
      changed = true;
    }
    double nu = m.poissonRatio;
    if (ImGui::InputDouble("Poisson ratio", &nu, 0., 0., "%.4g") && nu > -1. &&
        nu < .5) {
      m.poissonRatio = nu;
      changed = true;
    }
    double density = m.density;
    if (ImGui::InputDouble("Density (kg/m3)", &density, 0., 0., "%.5g") &&
        density >= 0. && std::isfinite(density)) {
      m.density = density;
      changed = true;
    }
    if (m.poissonRatio > .45)
      ImGui::TextWrapped(
          "Nearly incompressible material: linear tetrahedra can be too stiff. "
          "Check mesh convergence.");
    return changed;
  }
  bool hidden(int region) const {
    return std::binary_search(hiddenParts.begin(), hiddenParts.end(), region);
  }
  /// Shows or hides a part on request, ending a right-click hide of it.
  void setHidden(int region, bool hide) {
    peekHidden.erase(std::remove(peekHidden.begin(), peekHidden.end(), region),
                     peekHidden.end());
    const auto at =
        std::lower_bound(hiddenParts.begin(), hiddenParts.end(), region);
    if (hide && (at == hiddenParts.end() || *at != region))
      hiddenParts.insert(at, region);
    if (!hide && at != hiddenParts.end() && *at == region)
      hiddenParts.erase(at);
    hoveredFace = -1;
    visualDirty = true;
  }
  void focusPart(int region) {
    glm::vec3 low(std::numeric_limits<float>::max()), high(-low);
    const auto& faces = model.boundaryFaces();
    for (size_t f = 0; f < faces.size(); ++f)
      if (model.faceParts()[f] == region)
        for (const auto v : faces[f].vertices) {
          low = glm::min(low, position(v, false));
          high = glm::max(high, position(v, false));
        }
    if (low.x > high.x) return;
    fem_workbench::frameMeshCamera(viewer->getCamera(), low, high);
  }
  void removePart(int region) {
    if (model.parts.size() < 2) {
      clearModel();
      return;
    }
    if (meshJob)
      throw std::runtime_error(
          "Wait for meshing to finish or cancel it before removing parts");
    const auto* removed = model.findPart(region);
    if (!removed) return;
    const std::string name = removed->name;
    const size_t conditions = model.removePart(region, mappingTolerance());
    hiddenParts.erase(
        std::remove(hiddenParts.begin(), hiddenParts.end(), region),
        hiddenParts.end());
    edited = -1;
    activePart = model.parts.front().region;
    replaceVisual(false);
    message = "Removed " + name +
              (conditions ? " and " + std::to_string(conditions) +
                                " support, load or connection" +
                                (conditions == 1 ? "" : "s") + " on it."
                          : ".");
  }
  void partsPanel() {
    ImGui::BeginDisabled(!canAddCad());
    if (ImGui::Button(withIcon(kIconPlus, "Add CAD").c_str()))
      attempt([&] {
        appendNextImport = true;
#if defined(__EMSCRIPTEN__)
        EM_ASM({ globalThis['raisimFem'] ?.['open'](); });
#else
        import(file.data());
#endif
      });
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip(
          "Adds a STEP, IGES or BREP file as more parts (or drop it on the "
          "window). Enter its path under 1. Open CAD solid on the desktop.");
    ImGui::SameLine();
    ImGui::BeginDisabled(hiddenParts.empty());
    if (ImGui::Button(withIcon(kIconEye, "Show all").c_str())) {
      hiddenParts.clear();
      peekHidden.clear();
      visualDirty = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(withIcon(kIconBroom, "Clear all").c_str()))
      attempt([&] { clearModel(); });
    if (ImGui::TreeNodeEx(withIcon(kIconLayers, "Model material").c_str(),
                          ImGuiTreeNodeFlags_Framed |
                              ImGuiTreeNodeFlags_FramePadding)) {
      if (editMaterial(model.material, model.yieldStrength)) {
        model.changed();
        visualDirty = true;
      }
      ImGui::TreePop();
    }
    if (!model.findPart(activePart))
      activePart = model.parts.empty() ? -1 : model.parts[0].region;
    ImGui::SeparatorText("Parts");
    const float frame = ImGui::GetFrameHeight();
    const auto& style = ImGui::GetStyle();
    int removed = -1, moving = -1;
    fem_workbench::Pose motion;
    for (auto& part : model.parts) {
      ImGui::PushID(part.region);
      const bool active = model.parts.size() > 1 && part.region == activePart;
      const bool shown = !hidden(part.region);
      if (revealPart && part.region == activePart) ImGui::SetNextItemOpen(true);
      const std::string material = materialLabel(
          model.materialOf(part), model.yieldOf(part));
      const float left = ImGui::GetCursorPosX();
      const float right = ImGui::GetContentRegionMax().x;
      const float buttons = right - 2.f * frame - style.ItemSpacing.x;
      const float labels = buttons - left - ImGui::GetFontSize() -
                           3.f * (style.FramePadding.x + 6.f) -
                           2.f * style.ItemSpacing.x;
      float materialWidth =
          std::min(ImGui::CalcTextSize(material.c_str()).x,
                   labels - ImGui::CalcTextSize(part.name.c_str()).x -
                       style.ItemSpacing.x);
      if (materialWidth < 3.f * ImGui::GetFontSize()) materialWidth = 0.f;
      const float nameWidth =
          labels - materialWidth - (materialWidth > 0.f ? style.ItemSpacing.x : 0.f);
      if (active)
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(.2f, .3f, .44f, 1.f));
      if (!shown)
        ImGui::PushStyleColor(ImGuiCol_Text,
                              style.Colors[ImGuiCol_TextDisabled]);
      pushBarPadding();
      const bool open = ImGui::TreeNodeEx(
          "##part",
          ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_FramePadding |
              ImGuiTreeNodeFlags_AllowOverlap |
              (model.parts.size() == 1 ? ImGuiTreeNodeFlags_DefaultOpen : 0),
          "%s", fitText(part.name, nameWidth).c_str());
      ImGui::PopStyleVar();
      if (!shown) ImGui::PopStyleColor();
      if (active) ImGui::PopStyleColor();
      if (ImGui::IsItemClicked() && activePart != part.region) {
        activePart = part.region;
        visualDirty = true;
      }
      if (revealPart && part.region == activePart) ImGui::SetScrollHereY(.3f);
      leftBar(toImColor(partColor(part.region)), 5.f);
      if (materialWidth > 0.f) {
        ImGui::SameLine(buttons - materialWidth - style.ItemSpacing.x);
        ImGui::TextDisabled("%s", fitText(material, materialWidth).c_str());
      }
      ImGui::SameLine(buttons);
      if (ImGui::Button(shown ? kIconEye : kIconEyeCrossed, ImVec2(frame, frame)))
        setHidden(part.region, shown);
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip(shown ? "Hide this part (or right-click it in the view)"
                                : "Show this part");
      ImGui::SameLine();
      if (ImGui::Button(kIconTrash, ImVec2(frame, frame))) removed = part.region;
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Remove this part with its supports, loads and connections");
      if (open) {
        std::array<char, 256> name{};
        std::snprintf(name.data(), name.size(), "%s", part.name.c_str());
        ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Name").x -
                                style.ItemInnerSpacing.x);
        if (ImGui::InputText("Name", name.data(), name.size()))
          part.name = name.data();
        ImGui::SeparatorText("Material");
        int source = part.material ? 1 : 0;
        bool sourceChanged = ImGui::RadioButton("Model material", &source, 0);
        ImGui::SameLine();
        sourceChanged |= ImGui::RadioButton("Own material", &source, 1);
        if (sourceChanged && (source == 1) != part.material.has_value()) {
          if (source == 1) {
            part.material = model.material;
            part.yieldStrength = model.yieldStrength;
          } else {
            part.material.reset();
          }
          model.changed();
          visualDirty = true;
        }
        if (part.material) {
          if (editMaterial(*part.material, part.yieldStrength)) {
            model.changed();
            visualDirty = true;
          }
        } else {
          const auto& m = model.material;
          ImGui::PushStyleColor(ImGuiCol_Text,
                                style.Colors[ImGuiCol_TextDisabled]);
          ImGui::TextWrapped(
              "%s: E %.4g GPa, nu %.3g, %.4g kg/m3, yield %.4g MPa",
              materialLabel(m, model.yieldStrength).c_str(),
              m.youngsModulus / 1e9, m.poissonRatio, m.density,
              model.yieldStrength / 1e6);
          ImGui::PopStyleColor();
        }
        ImGui::SeparatorText("Position");
        bool fixed = part.fixed;
        if (ImGui::Checkbox(withIcon(kIconThumbtack, "Fixed").c_str(), &fixed)) {
          const int region = part.region;
          attempt([&] { model.setFixed(region, fixed); });
        }
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip(
              "A fixed part never moves: mates move the other part, and "
              "dragging it turns the view");
        ImGui::SameLine();
        {
          const int free = model.freeMotions(part.region);
          ImGui::TextDisabled(
              "%s", part.fixed ? "never moves"
                    : free == 6 ? "moves freely"
                    : free == 0 ? "fully positioned by its mates"
                                : (std::to_string(free) + " free motion" +
                                   (free == 1 ? "" : "s") + " left")
                                      .c_str());
        }
        std::string mateNames;
        for (const auto& mate : model.mates)
          if (mate.moving == part.region || mate.reference == part.region)
            mateNames += (mateNames.empty() ? "" : ", ") + mate.name;
        if (!mateNames.empty()) {
          ImGui::PushTextWrapPos(0.f);
          ImGui::TextDisabled("Mates: %s", mateNames.c_str());
          ImGui::PopTextWrapPos();
        }
        Vec<3> next = part.offset;
        bool moved = false;
        for (int a = 0; a < 3; ++a) {
          char label[32];
          std::snprintf(label, sizeof(label), "Offset %c (m)", 'X' + a);
          moved |= axisInput(label, &next[a], a,
                             ImGuiInputTextFlags_EnterReturnsTrue);
        }
        if (moved) {
          moving = part.region;
          motion.translation = next.e() - part.offset.e();
        }
        auto angles = eulerDegrees(part.rotation);
        bool turned = false;
        for (int a = 0; a < 3; ++a) {
          char label[32];
          std::snprintf(label, sizeof(label), "Rotation %c (deg)", 'X' + a);
          turned |= axisInput(label, &angles[a], a,
                              ImGuiInputTextFlags_EnterReturnsTrue);
        }
        if (turned)
          attempt([&] {
            moving = part.region;
            motion = fem_workbench::Pose::about(
                eulerRotation(angles) * part.rotation.transpose(),
                model.partCenter(part.region));
          });
        const auto nudge = [&](const char* icon, const char* label,
                               std::array<double, 3>& values, const char* tip) {
          ImGui::PushID(label);
          const float button =
              ImGui::CalcTextSize(withIcon(kIconRotate, "Rotate by").c_str()).x +
              2.f * style.FramePadding.x;
          const float field = (ImGui::GetContentRegionAvail().x - button -
                               3.f * style.ItemSpacing.x) /
                              3.f;
          for (int a = 0; a < 3; ++a) {
            if (a) ImGui::SameLine();
            ImGui::SetNextItemWidth(field);
            ImGui::PushID(a);
            axisInput("##nudge", &values[size_t(a)], a);
            ImGui::PopID();
          }
          ImGui::SameLine();
          const bool apply =
              ImGui::Button(withIcon(icon, label).c_str(), ImVec2(button, 0.f));
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
          ImGui::PopID();
          return apply;
        };
        if (nudge(kIconArrows, "Move by", moveBy,
                  "Move the part by these X, Y and Z distances (m)")) {
          moving = part.region;
          motion = {};
          motion.translation = Eigen::Vector3d(moveBy[0], moveBy[1], moveBy[2]);
        }
        if (nudge(kIconRotate, "Rotate by", rotateBy,
                  "Turn the part about its center by these angles (deg) about "
                  "X, then Y, then Z"))
          attempt([&] {
            moving = part.region;
            motion = fem_workbench::Pose::about(eulerRotation(rotateBy),
                                                model.partCenter(part.region));
          });
        if (ImGui::Button(withIcon(kIconBroom, "Reset").c_str())) {
          moving = part.region;
          motion = Model::placement(part).inverse();
        }
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("Put the part back where its CAD file places it");
        ImGui::SameLine();
        if (ImGui::Button(withIcon(kIconExpand, "Focus").c_str()))
          focusPart(part.region);
        ImGui::SameLine();
        if (ImGui::Button(withIcon(kIconCrosshairs, "Isolate").c_str()))
          for (const auto& other : model.parts)
            setHidden(other.region, other.region != part.region);
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
    revealPart = false;
    if (moving >= 0)
      attempt([&] {
        model.movePart(moving, motion);
        updateBounds();
        visualDirty = true;
      });
    if (removed >= 0) attempt([&] { removePart(removed); });
  }
  void mateList() {
    using fem_workbench::MateType;
    if (model.mates.empty()) {
      ImGui::TextDisabled("No mates yet");
      return;
    }
    const float frame = ImGui::GetFrameHeight();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float right = ImGui::GetContentRegionMax().x;
    std::optional<size_t> removed, flipped, opened;
    for (size_t i = 0; i < model.mates.size(); ++i) {
      const auto& mate = model.mates[i];
      ImGui::PushID(("mate " + std::to_string(i)).c_str());
      const auto* mover = model.findPart(mate.moving);
      const auto* other = model.findPart(mate.reference);
      std::string label = mate.name + ": " + (mover ? mover->name : "?") +
                          " to " + (other ? other->name : "?");
      char value[48] = "";
      const double v = mate.options.value.value_or(0.);
      if (mate.type == MateType::Distance ||
          (mate.type == MateType::ProfileCenter && v != 0.))
        std::snprintf(value, sizeof(value), ", %.4g m", v);
      if (mate.type == MateType::Angle)
        std::snprintf(value, sizeof(value), ", %.4g deg",
                      v * 180. / 3.14159265358979323846);
      label += value;
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(
          fitText(label, right - ImGui::GetCursorPosX() - 3.f * frame -
                             3.f * spacing)
              .c_str());
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s\n%s", label.c_str(),
                          fem_workbench::mateTypeHelp(mate.type));
      ImGui::SameLine(right - 3.f * frame - 2.f * spacing);
      ImGui::BeginDisabled(mate.type == MateType::Lock);
      if (ImGui::Button(kIconFlip, ImVec2(frame, frame))) flipped = i;
      ImGui::EndDisabled();
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Flip: turn %s around",
                          mover ? mover->name.c_str() : "the part");
      ImGui::SameLine();
      if (ImGui::Button(kIconPencil, ImVec2(frame, frame))) opened = i;
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Edit this mate");
      ImGui::SameLine();
      if (ImGui::Button(kIconTrash, ImVec2(frame, frame))) removed = i;
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Remove this mate; the parts stay where they are");
      ImGui::PopID();
    }
    if (removed) attempt([&] { model.removeMate(*removed); });
    if (flipped)
      attempt([&] {
        auto options = model.mates[*flipped].options;
        if (model.mates[*flipped].type == MateType::Angle)
          options.value = 3.14159265358979323846 - options.value.value_or(0.);
        else
          options.aligned = !options.aligned.value_or(false);
        model.editMate(*flipped, options);
        updateBounds();
        visualDirty = true;
      });
    if (opened) editMate(*opened);
  }
  static constexpr int kModelTab = 0, kPartsTab = 1, kLoadsTab = 2,
                       kConnectTab = 3, kFileTab = 4;
  bool requireGeometry() {
    if (model.hasGeometry()) return true;
    ImGui::TextDisabled("No model open");
    return false;
  }
  void selectionStatus() {
    const auto selectedFaces = model.selectedFaces();
    const size_t selectedNodes =
        size_t(std::count(model.selected.begin(), model.selected.end(), 1));
    if (selectedNodes == 0) {
      ImGui::TextColored(ImVec4(1.f, .78f, .3f, 1.f),
                         "Selected: nothing");
      return;
    }
    if (selectedFaces.empty())
      ImGui::Text("Selected: %zu nodes", selectedNodes);
    else
      ImGui::Text("Selected: %zu face%s, %.4g m2", surfaceCount(selectedFaces),
                  surfaceCount(selectedFaces) == 1 ? "" : "s",
                  area(selectedFaces));
    ImGui::SameLine();
    if (ImGui::SmallButton(withIcon(kIconEraser, "Clear").c_str())) {
      model.clearSelection();
      visualDirty = true;
    }
  }
  void modelTab() {
#if !defined(__EMSCRIPTEN__)
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##source", "path/to/part.step", file.data(),
                             file.size());
#endif
    if (ImGui::Button(withIcon(kIconFolderOpen, "Open CAD").c_str(),
                      ImVec2(120, 0)))
      attempt([&] {
        appendNextImport = false;
#if defined(__EMSCRIPTEN__)
        EM_ASM({ globalThis['raisimFem'] ?.['open'](); });
#else
        import(file.data());
#endif
      });
    ImGui::SameLine();
    ImGui::BeginDisabled(!canAddCad());
    if (ImGui::Button(withIcon(kIconPlus, "Add CAD").c_str()))
      attempt([&] {
        appendNextImport = true;
#if defined(__EMSCRIPTEN__)
        EM_ASM({ globalThis['raisimFem'] ?.['open'](); });
#else
        import(file.data());
#endif
      });
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip(
          "Adds another STEP, IGES or BREP file as more parts. Tie or bolt "
          "the parts together under Apply supports and loads.");
    ImGui::SameLine();
    if (ImGui::Button(withIcon(kIconCube, "Demo beam").c_str()))
      attempt([&] { demo(); });
    ImGui::InputDouble("Metres per BREP unit", &importScale, 0., 0., "%.6g");
    ImGui::SeparatorText("Mesh and accuracy");
    if (ImGui::InputDouble("Max element edge (m)", &elementSize, 0., 0.,
                           "%.5g")) {
      syncSettings();
      model.changed();
      visualDirty = true;
    }
    if (ImGui::Checkbox("Refine stress concentrations", &adaptive)) {
      model.changed();
      visualDirty = true;
    }
    if (adaptive &&
        ImGui::TreeNode(withIcon(kIconSliders, "Refinement").c_str())) {
      bool changed =
          ImGui::InputDouble("Target error (%)", &targetError, 0., 0., "%.3g");
      changed |= ImGui::InputInt("Refinement steps", &refinementSteps);
      changed |= ImGui::InputDouble("Minimum edge, 0 = none (m)",
                                    &minimumEdge, 0., 0., "%.5g");
      changed |= ImGui::InputInt("Element limit", &elementLimit, 0);
      if (changed) {
        model.changed();
        visualDirty = true;
      }
      ImGui::TreePop();
    }
    if (ImGui::TreeNode(withIcon(kIconSliders, "Mesher").c_str())) {
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
#if !defined(__EMSCRIPTEN__)
      ImGui::SetNextItemWidth(-1);
      ImGui::InputText("##nativeMesher", nativeMesher.data(),
                       nativeMesher.size());
#endif
      ImGui::TreePop();
    }
    if (!model.hasGeometry()) return;
    ImGui::TextDisabled("%zu nodes   %zu triangles   %zu tetrahedra",
                        model.selected.size(), model.boundaryFaces().size(),
                        model.mesh().tetrahedra.size());
    if (model.solver)
      ImGui::TextDisabled("Mesh maximum edge: %.5g m", actualMaximumEdge);
    if (model.solver && needsMesh())
      ImGui::TextWrapped("Resolution changed. Solve will rebuild the mesh.");
  }
  void loadsTab() {
    selectionStatus();
    if (ImGui::TreeNode(
            withIcon(kIconRuler, "Select by coordinates").c_str())) {
      ImGui::Checkbox("Add to selection", &append);
      if (ImGui::Checkbox("Box-select nodes (drag)", &boxSelect))
        selectingBox = false;
      ImGui::Combo("Plane normal", &planeAxis, "X\0Y\0Z\0");
      ImGui::InputDouble("Coordinate (m)", &plane, 0., 0., "%.6g");
      ImGui::InputDouble("Tolerance (m)", &tolerance, 0., 0., "%.4g");
      const auto slab = [&](double coordinate) {
        plane = coordinate;
        attempt([&] {
          model.selectSlab(planeAxis, plane, tolerance, append);
          visualDirty = true;
        });
      };
      if (ImGui::Button(withIcon(kIconArrowSmallDown, "Minimum").c_str()))
        slab(boundsMin[planeAxis]);
      ImGui::SameLine();
      if (ImGui::Button(withIcon(kIconArrowUp, "Maximum").c_str()))
        slab(boundsMax[planeAxis]);
      ImGui::SameLine();
      if (ImGui::Button(withIcon(kIconCursor, "Select").c_str())) slab(plane);
      if (ImGui::TreeNode("Cylindrical surface")) {
        editVector("Cylinder axis", cylinderAxis);
        editVector("Cylinder center", cylinderCenter, " (m)");
        ImGui::InputDouble("Cylinder radius (m)", &cylinderRadius, 0., 0.,
                           "%.6g");
        ImGui::InputDouble("Axial minimum (m)", &cylinderMinimum, 0., 0.,
                           "%.6g");
        ImGui::InputDouble("Axial maximum (m)", &cylinderMaximum, 0., 0.,
                           "%.6g");
        if (ImGui::Button(
                withIcon(kIconCircle, "Select cylindrical boundary").c_str()))
          attempt([&] {
            model.selectCylinder(cylinderCenter, cylinderAxis,
                                 cylinderRadius, tolerance, cylinderMinimum,
                                 cylinderMaximum, append);
            visualDirty = true;
          });
        ImGui::TreePop();
      }
      ImGui::TreePop();
    }
    ImGui::Separator();
    const auto button = [&](Kind kind, bool usable, bool sameLine) {
      if (sameLine) ImGui::SameLine();
      ImGui::BeginDisabled(!usable);
      pushBarPadding();
      const bool pressed = ImGui::Button(
          withIcon(kindIcon(kind), fem_workbench::kindName(kind)).c_str(),
          ImVec2(168, 0));
      ImGui::PopStyleVar();
      leftBar(kindColor(kind), 3.f);
      if (pressed) addCondition(kind);
      ImGui::EndDisabled();
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        fem_workbench::conditionTooltip(kind);
    };
    const auto selectedFaces = model.selectedFaces();
    const size_t selectedNodes =
        size_t(std::count(model.selected.begin(), model.selected.end(), 1));
    const bool nodes = selectedNodes > 0, faces = !selectedFaces.empty();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.45f, .9f, .6f, 1.f));
    ImGui::SeparatorText("Supports");
    ImGui::PopStyleColor();
    button(Kind::Fixed, nodes, false);
    button(Kind::Roller, faces, true);
    button(Kind::Displacement, nodes, false);
    ImGui::BeginDisabled(!nodes);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(168.f);
    auto* supportList = ImGui::GetWindowDrawList();
    const auto supportCorner = ImGui::GetCursorScreenPos();
    const ImU32 supportText = ImGui::GetColorU32(ImGuiCol_Text);
    const bool moreSupports = ImGui::BeginCombo("##fixture-method", nullptr);
    const auto& frame = ImGui::GetStyle().FramePadding;
    supportList->AddRectFilled(
        supportCorner,
        ImVec2(supportCorner.x + 3.f,
               supportCorner.y + ImGui::GetFrameHeight()),
        toImColor(kSupportColor), ImGui::GetStyle().FrameRounding,
        ImDrawFlags_RoundCornersLeft);
    supportList->AddText(ImVec2(supportCorner.x + frame.x + 6.f,
                                supportCorner.y + frame.y),
                         supportText,
                         withIcon(kIconPlus, "More supports").c_str());
    if (moreSupports) {
      for (int k = static_cast<int>(Kind::Directional);
           k <= static_cast<int>(Kind::Tie); ++k) {
        if (k == static_cast<int>(Kind::Roller)) continue;
        if (ImGui::Selectable(fem_workbench::kindName(static_cast<Kind>(k))))
          addCondition(static_cast<Kind>(k));
        if (ImGui::IsItemHovered())
          fem_workbench::conditionTooltip(static_cast<Kind>(k));
      }
      ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.95f, .55f, .95f, 1.f));
    ImGui::SeparatorText("Loads");
    ImGui::PopStyleColor();
    button(Kind::Force, nodes, false);
    button(Kind::Pressure, faces, true);
    button(Kind::Traction, faces, false);
    button(Kind::Gravity, true, true);
    conditionList(false);
  }
  void fileTab() {
    ImGui::InputText("Project", project.data(), project.size());
    if (ImGui::Button(withIcon(kIconDisk, "Save project").c_str()))
      attempt([&] {
        syncSettings();
        model.save(project.data());
        download(project.data());
        message = "Project saved.";
      });
    ImGui::SameLine();
    if (ImGui::Button(withIcon(kIconFolderOpen, "Load project").c_str()))
      attempt([&] {
#if defined(__EMSCRIPTEN__)
        EM_ASM({ globalThis['raisimFem'] ?.['open'](); });
#else
        import(project.data());
#endif
      });
  }
  void conditionList(bool mates) {
    size_t count = 0;
    for (const auto& c : model.conditions) count += isConnection(c.kind) == mates;
    ImGui::TextDisabled("%s  %s (%zu)", kIconListCheck,
                        mates ? "Connections" : "Applied", count);
    ImGui::BeginChild(mates ? "mates" : "conditions", ImVec2(0, 105), true);
    for (size_t i = 0; i < model.conditions.size(); ++i) {
      if (isConnection(model.conditions[i].kind) != mates) continue;
      ImGui::PushID(int(i));
      if (ImGui::Checkbox("##enabled", &model.conditions[i].enabled)) {
        model.changed();
        visualDirty = true;
      }
      ImGui::SameLine();
      const Kind kind = model.conditions[i].kind;
      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(
                                               kindColor(kind)));
      ImGui::TextUnformatted(kindIcon(kind));
      ImGui::PopStyleColor();
      ImGui::SameLine();
      if (ImGui::Selectable(model.conditions[i].name.c_str(),
                            edited == int(i))) {
        edited = int(i);
        pickingReference = -1;
        visualDirty = true;
      }
      ImGui::PopID();
    }
    ImGui::EndChild();
    if (edited >= 0 && size_t(edited) < model.conditions.size() &&
        isConnection(model.conditions[size_t(edited)].kind) == mates) {
      auto& c = model.conditions[size_t(edited)];
      if (mates)
        ImGui::TextDisabled("First part: %zu faces | second part: %zu faces",
                            c.faces.size(), c.targetFaces.size());
      else
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
          axisBar(a);
        }
      if (static_cast<int>(c.kind) > static_cast<int>(Kind::Gravity))
        changed |= editFixture(c);
      else if (c.kind == Kind::Pressure)
        changed |=
            ImGui::InputDouble("Pressure (Pa)", &c.value[0], 0., 0., "%.6g");
      else if (c.kind == Kind::Force)
        changed |= editForce(c);
      else if (c.kind != Kind::Fixed) {
        const char* unit = c.kind == Kind::Displacement ? "m"
                           : c.kind == Kind::Force      ? "N"
                           : c.kind == Kind::Traction   ? "Pa"
                                                        : "m/s2";
        for (int a = 0; a < 3; ++a) {
          char label[24];
          std::snprintf(label, sizeof(label), "%c (%s)", 'X' + a, unit);
          changed |= axisInput(label, &c.value[a], a);
        }
      }
      if (changed) {
        model.changed();
        visualDirty = true;
      }
      if (ImGui::Button(
              withIcon(kIconCrosshairs, "Select its region").c_str())) {
        model.restoreSelection(c.nodes, c.faces);
        visualDirty = true;
      }
      ImGui::SameLine();
      if (ImGui::Button(withIcon(kIconTrash, "Delete").c_str())) {
        model.conditions.erase(model.conditions.begin() + edited);
        edited = -1;
        pickingReference = -1;
        model.changed();
        visualDirty = true;
      }
    }
  }
  static const char* mateSide(Kind kind, bool first) {
    switch (kind) {
      case Kind::Tie:
        return first ? "faces to glue" : "faces they touch";
      case Kind::BoltConnector:
        return first ? "bolt head seat around the hole"
                     : "nut seat on the other part";
      case Kind::Hinge:
        return first ? "pin or shaft surface" : "hole surface";
      case Kind::BallJoint:
        return first ? "ball surface" : "socket surface";
      case Kind::SliderJoint:
        return first ? "sliding faces" : "guide faces";
      default:
        return first ? "faces to connect" : "faces on the other part";
    }
  }
  double mateTolerance() const {
    return std::max(1e-4, 4. * effectiveTolerance());
  }
  std::vector<unsigned int> faceNodes(const std::vector<size_t>& faces) const {
    std::vector<unsigned int> nodes;
    for (const auto f : faces)
      for (const auto v : model.boundaryFaces()[f].vertices) nodes.push_back(v);
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
  }
  static const char* connectionName(Kind kind) {
    switch (kind) {
      case Kind::Tie:
        return "Glue";
      case Kind::BoltConnector:
        return "Bolt";
      case Kind::Hinge:
        return "Pin";
      case Kind::BallJoint:
        return "Ball joint";
      case Kind::SliderJoint:
        return "Slider";
      default:
        return "Rigid link";
    }
  }
  bool connectionButton(Kind kind, bool recommended) {
    ImGui::PushStyleColor(ImGuiCol_Button, recommended
                                               ? ImVec4(.2f, .32f, .5f, 1.f)
                                               : ImVec4(.16f, .18f, .22f, 1.f));
    const bool pressed = ImGui::Button(
        (withIcon(kindIcon(kind), connectionName(kind)) + "##connect").c_str());
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) fem_workbench::conditionTooltip(kind);
    return pressed;
  }
  /// Index of every mesh node's part, for drawing parts at preview poses.
  void indexNodeParts() {
    nodePart.assign(model.mesh().vertices.size(), -1);
    const auto& faces = model.boundaryFaces();
    for (size_t f = 0; f < faces.size(); ++f) {
      const auto* part = model.findPart(model.faceParts()[f]);
      if (!part) continue;
      for (const auto v : faces[f].vertices)
        nodePart[v] = int(part - model.parts.data());
    }
  }
  /// Draws the parts moved by @p motions (one per part) until cleared.
  void setPreview(std::vector<fem_workbench::Pose> motions) {
    if (std::all_of(motions.begin(), motions.end(),
                    [](const auto& m) { return m.isIdentity(); }))
      motions.clear();
    if (motions.empty() && previewMotions.empty()) return;
    previewMotions = std::move(motions);
    if (!previewMotions.empty()) indexNodeParts();
    visualDirty = true;
  }
  std::string motionStatus(const std::string& name, int free) const {
    if (free == 0) return name + " is fully positioned";
    if (free == 6) return name + " moves freely";
    return name + " keeps " + std::to_string(free) + " free motion" +
           (free == 1 ? "" : "s");
  }
  Model::MateRequest mateRequest(fem_workbench::MateType type,
                                 const std::vector<size_t>& moving,
                                 const std::vector<size_t>& reference,
                                 const fem_workbench::MateOptions& options)
      const {
    Model::MateRequest request;
    request.type = type;
    request.moving = moving;
    request.reference = reference;
    request.options = options;
    request.tolerance = mateTolerance();
    return request;
  }
  void openEditor(Editor kind, Kind connection,
                  std::vector<size_t> first = {},
                  std::vector<size_t> second = {}) {
    if (first.empty() && second.empty()) {
      auto order = selectionParts;
      for (const auto f : model.selectedFaces())
        if (std::find(order.begin(), order.end(), model.faceParts()[f]) ==
            order.end())
          order.push_back(model.faceParts()[f]);
      for (const auto f : model.selectedFaces()) {
        const int part = model.faceParts()[f];
        if (order.size() > 0 && part == order[0]) first.push_back(f);
        if (order.size() > 1 && part == order[1]) second.push_back(f);
      }
    }
    const auto fixedPart = [&](const std::vector<size_t>& faces) {
      const auto* part =
          faces.empty() ? nullptr : model.findPart(model.faceParts()[faces[0]]);
      return part && part->fixed;
    };
    if (kind == Editor::Mate && fixedPart(first) && !fixedPart(second) &&
        !second.empty())
      std::swap(first, second);
    editor = kind;
    editorKind = connection;
    editorFaces = {std::move(first), std::move(second)};
    editorBox = editorFaces[0].empty() ? 0 : 1;
    editingMate = -1;
    editorOptions = {};
    editorDirty = true;
    model.clearSelection();
    selectionParts.clear();
    requestedTab = kConnectTab;
    visualDirty = true;
  }
  void editMate(size_t index) {
    closeEditor();
    editor = Editor::Mate;
    editingMate = int(index);
    editorMate = model.mates[index].type;
    editorOptions = model.mates[index].options;
    editorDirty = true;
    requestedTab = kConnectTab;
  }
  void closeEditor() {
    editor = Editor::None;
    editorFaces = {};
    editingMate = -1;
    editorOptions = {};
    editorFits.clear();
    editorStatus.clear();
    editorFailed = false;
    setPreview({});
    visualDirty = true;
  }
  /// Picks a face into the active selection box of the editor. A box holds
  /// the faces of one part; clicking another part fills the other box.
  void pickIntoEditor(int face) {
    if (face < 0) return;
    const int part = model.faceParts()[size_t(face)];
    const auto faces = model.surfaceFaces(size_t(face));
    const auto partOf = [&](const std::vector<size_t>& box) {
      return box.empty() ? std::numeric_limits<int>::min()
                         : model.faceParts()[box.front()];
    };
    auto* box = &editorFaces[size_t(editorBox)];
    auto* other = &editorFaces[size_t(1 - editorBox)];
    if (!box->empty() && partOf(*box) != part) {
      if (other->empty() || partOf(*other) == part) {
        editorBox = 1 - editorBox;
        std::swap(box, other);
      } else {
        box->clear();
      }
    }
    const bool present = std::all_of(faces.begin(), faces.end(), [&](size_t f) {
      return std::find(box->begin(), box->end(), f) != box->end();
    });
    if (present) {
      box->erase(std::remove_if(box->begin(), box->end(),
                                [&](size_t f) {
                                  return std::find(faces.begin(), faces.end(),
                                                   f) != faces.end();
                                }),
                 box->end());
    } else {
      for (const auto f : faces)
        if (std::find(box->begin(), box->end(), f) == box->end())
          box->push_back(f);
      if (editorBox == 0 && editorFaces[1].empty()) editorBox = 1;
    }
    activePart = part;
    editorDirty = true;
    visualDirty = true;
  }
  /// Placements, fixed flags and mates, to notice changes made elsewhere.
  std::vector<double> placementKey() const {
    std::vector<double> key{double(model.mates.size())};
    for (const auto& part : model.parts) {
      key.push_back(part.fixed);
      for (int a = 0; a < 3; ++a) key.push_back(part.offset[a]);
      for (int i = 0; i < 9; ++i) key.push_back(part.rotation(i / 3, i % 3));
    }
    return key;
  }
  void updateEditor() {
    if (editor != Editor::Mate) return;
    auto key = placementKey();
    if (key != editorKey) {
      editorKey = std::move(key);
      editorDirty = true;
    }
    if (!editorDirty) return;
    editorDirty = false;
    editorFailed = false;
    try {
      Model::MatePreview preview;
      if (editingMate >= 0) {
        preview = model.previewMateEdit(size_t(editingMate), editorOptions);
      } else {
        editorFits = model.mateTypes(editorFaces[0], editorFaces[1],
                                     mateTolerance());
        if (editorFits.empty()) {
          editorMover = -1;
          setPreview({});
          editorStatus = editorFaces[0].empty() || editorFaces[1].empty()
                             ? "Pick faces on both parts"
                             : "Pick faces on two different parts";
          return;
        }
        if (std::find(editorFits.begin(), editorFits.end(), editorMate) ==
            editorFits.end())
          editorMate = editorFits.front();
        preview = model.previewMate(mateRequest(editorMate, editorFaces[0],
                                                editorFaces[1], editorOptions));
      }
      setPreview(preview.motions);
      editorResolved = preview.mate.options;
      editorMover = preview.mate.moving;
      const auto* part = model.findPart(preview.mate.moving);
      editorStatus = motionStatus(part ? part->name : "The part",
                                  preview.freeMotions);
    } catch (const std::exception& error) {
      setPreview({});
      editorMover = -1;
      editorStatus = error.what();
      editorFailed = true;
    }
  }
  std::string boxSummary(const std::vector<size_t>& faces) const {
    if (faces.empty()) return "Click faces in the view";
    const auto* part = model.findPart(model.faceParts()[faces.front()]);
    std::vector<int> surfaces;
    for (const auto f : faces)
      if (std::find(surfaces.begin(), surfaces.end(),
                    model.faceSurfaces()[f]) == surfaces.end())
        surfaces.push_back(model.faceSurfaces()[f]);
    constexpr const char* kShapes[] = {"", " (plane)", " (cylinder)",
                                       " (sphere)"};
    const auto shape = model.faceShape(faces, mateTolerance()).type;
    return (part ? part->name : std::string("Part")) + ": " +
           std::to_string(surfaces.size()) + " face" +
           (surfaces.size() == 1 ? "" : "s") +
           kShapes[static_cast<int>(shape)];
  }
  /// The selection boxes, mate types, options and buttons of the editor, like
  /// a SolidWorks PropertyManager.
  void editorPanel() {
    using fem_workbench::MateType;
    if (editingMate >= int(model.mates.size())) {
      closeEditor();
      return;
    }
    updateEditor();
    const bool mate = editor == Editor::Mate;
    const bool editing = editingMate >= 0;
    std::string title =
        mate ? (editing ? "Edit " + model.mates[size_t(editingMate)].name
                        : std::string("New mate"))
             : std::string("New ") + fem_workbench::kindName(editorKind);
    ImGui::TextColored(ImVec4(.35f, .85f, .9f, 1.f), "%s",
                       withIcon(mate ? kIconArrowsToLine : kindIcon(editorKind),
                                title)
                           .c_str());
    const float width = ImGui::GetContentRegionAvail().x;
    if (!editing) {
      for (int b = 0; b < 2; ++b) {
        ImGui::PushID(b);
        const bool active = editorBox == b;
        const glm::vec4 tint = b == 0 ? kSelectionColor : kReferenceColor;
        ImGui::PushStyleColor(ImGuiCol_Border,
                              active ? ImVec4(tint.r, tint.g, tint.b, 1.f)
                                     : ImGui::GetStyle().Colors[ImGuiCol_Border]);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, active ? 2.f : 1.f);
        ImGui::BeginChild("box", ImVec2(width, 2.4f * ImGui::GetFrameHeight()),
                          ImGuiChildFlags_Border,
                          ImGuiWindowFlags_NoScrollbar);
        const auto& faces = editorFaces[size_t(b)];
        const bool moves =
            !faces.empty() && editorMover >= 0 && !editorFailed
                ? model.faceParts()[faces.front()] == editorMover
                : b == 0;
        ImGui::TextDisabled("%s", mate ? (moves ? "Moves: faces of the part to place"
                                                : "Stays: faces to place it on")
                                       : mateSide(editorKind, b == 0));
        const auto summary = boxSummary(editorFaces[size_t(b)]);
        ImGui::TextUnformatted(fitText(summary, width - 60.f).c_str());
        if (!editorFaces[size_t(b)].empty()) {
          ImGui::SameLine(width - 34.f);
          if (ImGui::SmallButton(kIconEraser)) {
            editorFaces[size_t(b)].clear();
            editorBox = b;
            editorDirty = visualDirty = true;
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clear these faces");
        }
        ImGui::EndChild();
        if (ImGui::IsItemClicked()) editorBox = b;
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("Click here, then click faces in the view to add "
                            "or remove them");
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        ImGui::PopID();
      }
      if (ImGui::Button(withIcon(kIconExchange, "Swap").c_str())) {
        std::swap(editorFaces[0], editorFaces[1]);
        editorDirty = visualDirty = true;
      }
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip(mate ? "Move the other part instead"
                               : "Exchange the first and second faces");
    }
    if (mate) {
      const auto group = [&](const char* label,
                             std::initializer_list<MateType> types) {
        ImGui::SeparatorText(label);
        int column = 0;
        for (const auto type : types) {
          const bool fits = editing ? type == editorMate
                                    : std::find(editorFits.begin(),
                                                editorFits.end(),
                                                type) != editorFits.end();
          if (column++ % 3) ImGui::SameLine();
          ImGui::BeginDisabled(!fits);
          const bool chosen = type == editorMate;
          if (chosen)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.2f, .42f, .6f, 1.f));
          if (ImGui::Button(withIcon(mateIcon(type),
                                     fem_workbench::mateTypeName(type))
                                .c_str(),
                            ImVec2((width - 2.f * ImGui::GetStyle().ItemSpacing.x) /
                                       3.f,
                                   0.f)) &&
              !chosen) {
            editorMate = type;
            editorOptions = {};
            editorDirty = true;
          }
          if (chosen) ImGui::PopStyleColor();
          ImGui::EndDisabled();
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s%s", fem_workbench::mateTypeHelp(type),
                              fits ? "" : "\n(does not fit these faces)");
        }
      };
      group("Standard mates",
            {MateType::Coincident, MateType::Parallel, MateType::Perpendicular,
             MateType::Tangent, MateType::Concentric, MateType::Distance,
             MateType::Angle, MateType::Lock});
      group("Advanced mates", {MateType::ProfileCenter, MateType::Width});
      ImGui::SeparatorText("Options");
      const bool directed = editorMate != MateType::Lock &&
                            editorMate != MateType::Angle;
      if (directed) {
        const bool aligned = editorOptions.aligned.value_or(
            editorResolved.aligned.value_or(false));
        if (ImGui::RadioButton("Anti-aligned", !aligned)) {
          editorOptions.aligned = false;
          editorDirty = true;
        }
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("The faces or axes point against each other");
        ImGui::SameLine();
        if (ImGui::RadioButton("Aligned", aligned)) {
          editorOptions.aligned = true;
          editorDirty = true;
        }
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("The faces or axes point the same way");
      }
      const bool distance = editorMate == MateType::Distance ||
                            editorMate == MateType::ProfileCenter;
      if (distance || editorMate == MateType::Angle) {
        const double resolved =
            editorOptions.value.value_or(editorResolved.value.value_or(0.));
        double shown = editorMate == MateType::Angle
                           ? resolved * 180. / 3.14159265358979323846
                           : resolved;
        ImGui::SetNextItemWidth(width * .45f);
        if (ImGui::InputDouble(editorMate == MateType::Angle ? "Angle (deg)"
                               : editorMate == MateType::Distance
                                   ? "Distance (m)"
                                   : "Offset (m)",
                               &shown, 0., 0., "%.6g",
                               ImGuiInputTextFlags_EnterReturnsTrue)) {
          editorOptions.value = editorMate == MateType::Angle
                                    ? shown * 3.14159265358979323846 / 180.
                                    : shown;
          editorDirty = true;
        }
      }
      if (distance || editorMate == MateType::Tangent) {
        bool flip = editorOptions.flip;
        if (!editorOptions.value && !editing) flip = editorResolved.flip;
        if (ImGui::Checkbox("Other side", &flip)) {
          editorOptions.flip = flip;
          if (!editorOptions.value) editorOptions.value = editorResolved.value;
          editorDirty = true;
        }
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("Put the moving face on the other side");
      }
      ImGui::PushTextWrapPos(0.f);
      if (editorFailed)
        ImGui::TextColored(ImVec4(1.f, .45f, .4f, 1.f), "%s",
                           editorStatus.c_str());
      else
        ImGui::TextDisabled("%s", editorStatus.c_str());
      ImGui::PopTextWrapPos();
    }
    const bool ready = mate ? !editorFailed && (editing || !editorFits.empty())
                            : !editorFaces[0].empty() && !editorFaces[1].empty();
    ImGui::BeginDisabled(!ready);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.07f, .45f, .52f, 1.f));
    const bool ok = ImGui::Button(withIcon(kIconCheck, "OK").c_str(),
                                  ImVec2(110.f, 0.f));
    ImGui::PopStyleColor();
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(withIcon(kIconCancel, "Cancel").c_str())) {
      closeEditor();
      return;
    }
    if (!ok) return;
    attempt([&] {
      if (meshJob)
        throw std::runtime_error(
            "Wait for meshing to finish or cancel it before moving parts");
      if (mate && editing) {
        model.editMate(size_t(editingMate), editorOptions);
        message = model.mates[size_t(editingMate)].name + " changed.";
      } else if (mate) {
        const auto& added = model.addMate(mateRequest(
            editorMate, editorFaces[0], editorFaces[1], editorOptions));
        message = added.name + " added.";
      } else {
        model.addConnection(editorKind, editorFaces[0], editorFaces[1],
                            mateTolerance());
        edited = int(model.conditions.size()) - 1;
        const auto& c = model.conditions.back();
        message = std::string(fem_workbench::kindName(editorKind)) +
                  " created." +
                  ((editorKind == Kind::Hinge ||
                    editorKind == Kind::SliderJoint) &&
                           !model.connectionAxisFitted(c)
                       ? " The faces do not define one axis; set it below."
                       : "");
      }
      updateBounds();
      closeEditor();
    });
  }
  void quickMates(const ImVec2& display) {
    const auto clearHover = [&]() {
      if (quickHovered >= 0) setPreview({});
      quickHovered = -1;
    };
    if (resultView || solveTask || editor != Editor::None ||
        pickingReference >= 0 || !model.hasGeometry()) {
      quickFaces.clear();
      quickA.clear();
      clearHover();
      return;
    }
    auto faces = model.selectedFaces();
    std::vector<int> parts;
    for (const auto f : faces) {
      const int part = model.faceParts()[f];
      if (std::find(parts.begin(), parts.end(), part) == parts.end())
        parts.push_back(part);
    }
    selectionParts.erase(
        std::remove_if(selectionParts.begin(), selectionParts.end(),
                       [&](int part) {
                         return std::find(parts.begin(), parts.end(), part) ==
                                parts.end();
                       }),
        selectionParts.end());
    std::vector<double> key{mateTolerance(), double(quickMoveSecond),
                            double(model.mates.size())};
    for (const auto part : selectionParts) key.push_back(part);
    for (const auto f : faces)
      for (const auto v : model.boundaryFaces()[f].vertices)
        for (int a = 0; a < 3; ++a) key.push_back(model.mesh().vertices[v][a]);
    if (!quickAdded.empty()) {
      auto placed = placementKey();
      if (quickAddedKey.empty())
        quickAddedKey = std::move(placed);
      else if (placed != quickAddedKey)
        quickAdded.clear();
    }
    if (faces != quickFaces || key != quickKey) {
      const bool reselected = faces != quickFaces;
      quickFaces = std::move(faces);
      quickKey = std::move(key);
      quickA.clear();
      quickB.clear();
      quickTypes.clear();
      clearHover();
      if (reselected) {
        quickDismissed = quickMore = quickMoreMates = quickMoveSecond = false;
        quickAnchor = lastClick;
        quickAdded.clear();
      }
      if (parts.size() == 2) {
        const auto rank = [&](int part) {
          return std::find(selectionParts.begin(), selectionParts.end(),
                           part) -
                 selectionParts.begin();
        };
        if (rank(parts[1]) < rank(parts[0])) std::swap(parts[0], parts[1]);
        for (const auto f : quickFaces)
          (model.faceParts()[f] == parts[0] ? quickA : quickB).push_back(f);
        quickSuggestion = fem_workbench::suggestMates(
            model.mesh(), model.boundaryFaces(), quickA, quickB,
            mateTolerance());
        quickTypes = model.mateTypes(quickMoveSecond ? quickB : quickA,
                                     quickMoveSecond ? quickA : quickB,
                                     mateTolerance());
      }
    }
    if (quickA.empty() || quickB.empty() || quickDismissed) {
      clearHover();
      return;
    }
    const auto& moving = quickMoveSecond ? quickB : quickA;
    const auto& staying = quickMoveSecond ? quickA : quickB;
    ImGui::SetNextWindowPos(
        ImVec2(std::max(sidebarWidth + 8.f,
                        std::min(quickAnchor.x + 18.f,
                                 display.x - quickSize.x - 8.f)),
               std::max(8.f, std::min(quickAnchor.y + 18.f,
                                      display.y - quickSize.y - 8.f))));
    ImGui::SetNextWindowBgAlpha(.95f);
    ImGui::Begin("Quick connection", nullptr,
                 ImGuiWindowFlags_NoDecoration |
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    const auto* mover = model.findPart(model.faceParts()[moving.front()]);
    const auto* other = model.findPart(model.faceParts()[staying.front()]);
    const std::string moverName = mover ? mover->name : "the part",
                      otherName = other ? other->name : "the other part";
    ImGui::TextDisabled("%s", quickSuggestion.relation.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton(kIconCancel)) quickDismissed = true;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Close");
    int hovered = -1;
    std::optional<fem_workbench::MateType> chosenMate;
    if (!quickTypes.empty()) {
      ImGui::AlignTextToFramePadding();
      ImGui::Text("Place %s", fitText(moverName, 140.f).c_str());
      ImGui::SameLine();
      if (ImGui::SmallButton((std::string(kIconExchange) + "##swap").c_str()))
        quickMoveSecond = !quickMoveSecond;
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Place %s instead", otherName.c_str());
      const size_t shown =
          quickMoreMates ? quickTypes.size()
                         : std::min<size_t>(quickTypes.size(), 4);
      for (size_t i = 0; i < shown; ++i) {
        if (i) ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button,
                              i == 0 ? ImVec4(.2f, .32f, .5f, 1.f)
                                     : ImVec4(.16f, .18f, .22f, 1.f));
        if (ImGui::Button((withIcon(mateIcon(quickTypes[i]),
                                    fem_workbench::mateTypeName(quickTypes[i])) +
                           "##mate")
                              .c_str()))
          chosenMate = quickTypes[i];
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) {
          hovered = int(quickTypes[i]);
          ImGui::SetTooltip("%s\n%s%s",
                            fem_workbench::mateTypeHelp(quickTypes[i]),
                            quickHoverError.empty()
                                ? ("Moves " + moverName + "; " + otherName +
                                   " stays")
                                      .c_str()
                                : quickHoverError.c_str(),
                            quickTypes[i] == fem_workbench::MateType::Distance ||
                                    quickTypes[i] ==
                                        fem_workbench::MateType::Angle
                                ? "\nOpens in the panel to set the value"
                                : "");
        }
      }
      if (shown < quickTypes.size()) {
        ImGui::SameLine();
        if (ImGui::Button((std::string(kIconDots) + "##mates").c_str()))
          quickMoreMates = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("All mates");
      }
    }
    if (!quickAdded.empty() && !model.mates.empty() &&
        model.mates.back().name == quickAdded) {
      ImGui::TextDisabled("%s added", quickAdded.c_str());
      ImGui::SameLine();
      if (ImGui::SmallButton(withIcon(kIconFlip, "Flip").c_str()))
        attempt([&] {
          auto options = model.mates.back().options;
          options.aligned = !options.aligned.value_or(false);
          const size_t last = model.mates.size() - 1;
          const auto preview = model.previewMateEdit(last, options);
          model.editMate(last, options);
          if (quickAddedMotions.size() == preview.motions.size())
            for (size_t i = 0; i < preview.motions.size(); ++i)
              quickAddedMotions[i] = preview.motions[i] * quickAddedMotions[i];
          quickAddedKey.clear();
          updateBounds();
          visualDirty = true;
        });
      ImGui::SameLine();
      if (ImGui::SmallButton(withIcon(kIconPencil, "Edit").c_str()))
        editMate(model.mates.size() - 1);
      ImGui::SameLine();
      if (ImGui::SmallButton(withIcon(kIconUndo, "Undo").c_str()))
        attempt([&] {
          std::vector<fem_workbench::Pose> undo;
          for (const auto& motion : quickAddedMotions)
            undo.push_back(motion.inverse());
          model.removeMate(model.mates.size() - 1);
          if (undo.size() == model.parts.size()) model.moveParts(undo);
          quickAdded.clear();
          updateBounds();
          visualDirty = true;
        });
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("Connect");
    auto kinds = quickSuggestion.kinds;
    if (quickMore)
      for (const auto kind : {Kind::Tie, Kind::BoltConnector, Kind::Hinge,
                              Kind::BallJoint, Kind::SliderJoint,
                              Kind::RigidLink})
        if (std::find(kinds.begin(), kinds.end(), kind) == kinds.end())
          kinds.push_back(kind);
    std::optional<Kind> chosen;
    for (size_t i = 0; i < kinds.size(); ++i) {
      ImGui::SameLine();
      if (connectionButton(kinds[i], i == 0)) chosen = kinds[i];
    }
    if (!quickMore && kinds.size() < 6) {
      ImGui::SameLine();
      if (ImGui::Button((std::string(kIconDots) + "##connections").c_str()))
        quickMore = true;
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("All connections");
    }
    quickSize = ImGui::GetWindowSize();
    ImGui::End();
    if (hovered != quickHovered) {
      quickHovered = hovered;
      quickHoverError.clear();
      if (hovered < 0) {
        setPreview({});
      } else {
        try {
          const auto preview = model.previewMate(mateRequest(
              static_cast<fem_workbench::MateType>(hovered), moving, staying,
              {}));
          setPreview(preview.motions);
        } catch (const std::exception& error) {
          setPreview({});
          quickHoverError = error.what();
        }
      }
    }
    if (chosenMate) {
      const auto type = *chosenMate;
      if (type == fem_workbench::MateType::Distance ||
          type == fem_workbench::MateType::Angle ||
          type == fem_workbench::MateType::Width) {
        const auto first = moving, second = staying;
        clearHover();
        openEditor(Editor::Mate, Kind::Tie, first, second);
        editorMate = type;
        editorDirty = true;
        return;
      }
      attempt([&] {
        if (meshJob)
          throw std::runtime_error(
              "Wait for meshing to finish or cancel it before moving parts");
        const auto preview =
            model.previewMate(mateRequest(type, moving, staying, {}));
        model.addMate(mateRequest(type, moving, staying, {}));
        quickAdded = model.mates.back().name;
        quickAddedMotions = preview.motions;
        quickAddedKey.clear();
        clearHover();
        updateBounds();
        visualDirty = true;
        message = quickAdded + " added.";
      });
      return;
    }
    if (!chosen) return;
    const Kind kind = *chosen;
    attempt([&] {
      const auto* firstPart =
          model.findPart(model.faceParts()[quickA.front()]);
      const auto* secondPart =
          model.findPart(model.faceParts()[quickB.front()]);
      const std::string names =
          firstPart && secondPart
              ? " between " + firstPart->name + " and " + secondPart->name
              : "";
      const bool swap = kind == Kind::Tie && quickSuggestion.tieReversed;
      const auto& first = swap ? quickB : quickA;
      const auto& second = swap ? quickA : quickB;
      const auto& c =
          model.addConnection(kind, first, second, mateTolerance());
      const bool fitted = model.connectionAxisFitted(c);
      edited = int(model.conditions.size()) - 1;
      requestedTab = kConnectTab;
      model.clearSelection();
      selectionParts.clear();
      clearHover();
      visualDirty = true;
      message = std::string(fem_workbench::kindName(kind)) + " created" +
                names + "." +
                ((kind == Kind::Hinge || kind == Kind::SliderJoint) && !fitted
                     ? " The faces do not define one axis; set it below."
                     : "");
    });
  }
  void matePanel() {
    if (editor != Editor::None) {
      editorPanel();
      return;
    }
    selectionStatus();
    ImGui::SeparatorText("Mates");
    pushBarPadding();
    if (ImGui::Button(withIcon(kIconArrowsToLine, "Add mate").c_str(),
                      ImVec2(168, 0)))
      openEditor(Editor::Mate, Kind::Tie);
    ImGui::PopStyleVar();
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip(
          "Place a part against another: coincident, parallel, "
          "perpendicular, tangent, concentric, distance, angle, lock, "
          "profile center or width");
    mateList();
    ImGui::SeparatorText("Connections");
    constexpr Kind kKinds[] = {Kind::Tie,         Kind::BoltConnector,
                               Kind::Hinge,       Kind::BallJoint,
                               Kind::SliderJoint, Kind::RigidLink};
    for (size_t i = 0; i < std::size(kKinds); ++i) {
      if (i % 2) ImGui::SameLine();
      pushBarPadding();
      const bool pressed = ImGui::Button(
          withIcon(kindIcon(kKinds[i]), connectionName(kKinds[i])).c_str(),
          ImVec2(168, 0));
      ImGui::PopStyleVar();
      leftBar(kindColor(kKinds[i]), 3.f);
      if (pressed) openEditor(Editor::Connection, kKinds[i]);
      if (ImGui::IsItemHovered()) fem_workbench::conditionTooltip(kKinds[i]);
    }
    if (ImGui::Button(withIcon(kIconLink, "Glue touching faces").c_str()))
      attempt([&] {
        const size_t added = model.bondTouchingFaces(mateTolerance());
        visualDirty = true;
        message = added ? "Glued " + std::to_string(added) +
                              " pair" + (added == 1 ? "" : "s") +
                              " of touching faces."
                        : "No new touching faces to glue.";
      });
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip(
          "Glue every pair of faces where two parts touch, like the bonded "
          "contact sets SolidWorks finds automatically");
    conditionList(true);
  }
  bool solveButton() {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.07f, .45f, .52f, 1.f));
    const bool pressed =
        ImGui::Button(withIcon(kIconPlay, "SOLVE").c_str(), ImVec2(140, 30));
    ImGui::PopStyleColor();
    if (pressed) solveRequested = true;
    return pressed;
  }
  void modeTabs() {
    const float width =
        (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) *
        .5f;
    const auto tab = [&](bool active, const char* icon, const char* label) {
      ImGui::PushStyleColor(ImGuiCol_Button, active
                                                 ? ImVec4(.2f, .32f, .5f, 1.f)
                                                 : ImVec4(.13f, .15f, .18f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                            active ? ImVec4(.25f, .38f, .58f, 1.f)
                                   : ImVec4(.2f, .23f, .28f, 1.f));
      ImGui::PushStyleColor(
          ImGuiCol_Text, active ? ImVec4(1.f, 1.f, 1.f, 1.f)
                                : ImVec4(.65f, .68f, .72f, 1.f));
      const bool pressed =
          ImGui::Button(withIcon(icon, label).c_str(), ImVec2(width, 30.f));
      ImGui::PopStyleColor(3);
      return pressed;
    };
    if (tab(!resultView, kIconPencil, "Edit")) setResultView(false);
    ImGui::SameLine();
    if (tab(resultView, kIconChart, "Result view")) setResultView(true);
  }
  void resultPanel() {
    if (meshJob && !previewJob) {
      ImGui::TextColored(ImVec4(1.f, .8f, .3f, 1.f), "Meshing for the solve");
      progressBar(-1.f);
      if (ImGui::Button(withIcon(kIconCancel, "Cancel").c_str())) cancelBusy();
      return;
    }
    if (!model.hasSolution) {
      ImGui::TextColored(ImVec4(1.f, .78f, .35f, 1.f),
                         "No results for the current model");
      if (!model.hasGeometry()) {
        ImGui::TextDisabled("No model open");
        return;
      }
      solveButton();
      return;
    }
    if (ImGui::CollapsingHeader(withIcon(kIconChart, "Display").c_str(),
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::SetNextItemWidth(-1.f);
      if (ImGui::Combo("##field", &field, kFieldNames,
                       IM_ARRAYSIZE(kFieldNames)))
        visualDirty = true;
      ImGui::SeparatorText("Labels");
      if (ImGui::Checkbox("Label max stress and displacement", &labelExtremes))
        visualDirty = true;
      ImGui::Checkbox("Label min and max of the shown field", &labelFieldRange);
      ImGui::Checkbox("Label glue and bolt stresses", &labelConnections);
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Pin the peak stress of every glue and bolt, with its factor of "
            "safety when its strength is set");
      const bool wasProbing = probing;
      if (wasProbing)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.2f, .32f, .5f, 1.f));
      if (ImGui::Button(withIcon(kIconCrosshairs,
                                 wasProbing ? "Probing: click the part"
                                            : "Probe values")
                            .c_str(),
                        ImVec2(220.f, 0.f)))
        probing = !probing;
      if (wasProbing) ImGui::PopStyleColor();
      if (!probes.empty()) {
        ImGui::SameLine();
        if (ImGui::SmallButton(withIcon(kIconEraser, "Clear probes").c_str())) {
          probes.clear();
          labelOffsets.erase(
              std::remove_if(labelOffsets.begin(), labelOffsets.end(),
                             [](const auto& entry) {
                               return entry.first >= kProbeLabel;
                             }),
              labelOffsets.end());
        }
        for (size_t i = 0; i < probes.size(); ++i)
          if (probes[i] < model.mesh().tetrahedra.size())
            ImGui::TextDisabled("#%zu  %s  (element %zu)", i + 1,
                                formatValue(scalar(probes[i]), fieldUnit()).c_str(),
                                probes[i]);
      }
      ImGui::SeparatorText("Deformation");
      if (ImGui::Combo("Scale", &deformationMode,
                       "Automatic\0True scale\0Custom\0"))
        visualDirty = true;
      if (deformationMode == 2 &&
          ImGui::InputDouble("Custom scale", &customDeformation, 0., 0.,
                             "%.4g")) {
        customDeformation = std::isfinite(customDeformation)
                                ? std::clamp(customDeformation, 0., 1e6)
                                : 1.;
        visualDirty = true;
      }
      if (deformationMode != 2)
        ImGui::TextDisabled("Displayed at %.3gx", deformation);
      if (ImGui::Checkbox("Animate", &animate)) visualDirty = true;
      ImGui::SeparatorText("Contours");
      if (ImGui::Checkbox("Smooth (node-averaged)", &smoothContours))
        visualDirty = true;
      if (ImGui::InputInt("Color bands, 0 = continuous", &bands)) {
        bands = std::clamp(bands, 0, 64);
        visualDirty = true;
      }
      if (ImGui::SliderFloat("Grey below (%)", &highlightAbove, 0.f, 95.f,
                             "%.0f"))
        visualDirty = true;
      if (ImGui::Checkbox("Automatic color range", &autoRange))
        visualDirty = true;
      if (!autoRange) {
        if (ImGui::InputDouble("Color minimum", &colorMin, 0., 0., "%.5g"))
          visualDirty = true;
        if (ImGui::InputDouble("Color maximum", &colorMax, 0., 0., "%.5g"))
          visualDirty = true;
      }
      if (ImGui::Checkbox("Mesh edges", &edges)) visualDirty = true;
      ImGui::SameLine();
      ImGui::Checkbox("Rest outline", &undeformed);
      if (ImGui::Button(withIcon(kIconExpand, "Fit view").c_str())) fit();
    }
    if (ImGui::CollapsingHeader(withIcon(kIconListCheck, "Summary").c_str(),
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      double stress = 0., displacement = 0.;
      for (const auto& r : model.solver->elementResults())
        stress = std::max(stress, r.vonMises);
      for (const auto& u : model.solver->displacements())
        displacement = std::max(displacement, u.e().norm());
      ImGui::Text("Max von Mises stress: %.4g Pa", stress);
      ImGui::Text("Max displacement: %.4g m", displacement);
      ImGui::Text("Strain energy: %.4g J",
                  model.solver->statistics().strainEnergy);
      double weakest = std::numeric_limits<double>::infinity();
      for (size_t e = 0; e < model.solver->elementResults().size(); ++e)
        if (std::isfinite(safety(e))) weakest = std::min(weakest, safety(e));
      if (std::isfinite(weakest))
        ImGui::TextColored(weakest < 1. ? ImVec4(1.f, .45f, .4f, 1.f)
                                        : ImVec4(.85f, .87f, .9f, 1.f),
                           "Min factor of safety: %.3g%s", weakest,
                           weakest < 1. ? " (yields)" : "");
      bool header = false;
      for (const auto& c : model.conditions) {
        if (!c.enabled || isConnection(c.kind) ||
            static_cast<int>(c.kind) == static_cast<int>(Kind::Force) ||
            c.kind == Kind::Pressure || c.kind == Kind::Traction ||
            c.kind == Kind::Gravity)
          continue;
        Vec<3> force{0., 0., 0.};
        for (const auto v : c.nodes)
          if (v < model.solver->reactions().size())
            force += model.solver->reactions()[v];
        if (!header) {
          ImGui::SeparatorText("Reaction forces");
          header = true;
        }
        ImGui::Text("%s: %.4g N", c.name.c_str(), force.e().norm());
        ImGui::TextDisabled("   (%.4g, %.4g, %.4g) N", force[0], force[1],
                            force[2]);
      }
      connectionSummary();
      ImGui::Text("Elements: %zu", model.mesh().tetrahedra.size());
      if (model.estimatedError >= 0.)
        ImGui::Text("Estimated error: %.1f%% (energy norm)",
                    100. * model.estimatedError);
      if (model.refinementLevel() > 0)
        ImGui::TextDisabled("Refined %d time%s at stress concentrations",
                            model.refinementLevel(),
                            model.refinementLevel() == 1 ? "" : "s");
    }
    if (ImGui::CollapsingHeader(withIcon(kIconExport, "Export").c_str())) {
      ImGui::InputText("VTK output", exportPath.data(), exportPath.size());
      if (ImGui::Button(withIcon(kIconExport, "Export results").c_str()))
        attempt([&] {
          model.solver->writeVtk(exportPath.data());
          download(exportPath.data());
          message = "VTK results exported.";
        });
    }
  }
  /// Forces and stresses of the connections, like the connector and contact
  /// results of a simulation package.
  void connectionSummary() {
    const auto& carried = loads();
    bool header = false;
    for (size_t i = 0; i < carried.size(); ++i) {
      if (!carried[i]) continue;
      const auto& c = model.conditions[i];
      const auto& load = *carried[i];
      if (!header) {
        ImGui::SeparatorText("Connections");
        header = true;
      }
      ImGui::PushID(int(i));
      const double safety = connectionSafety(c, load);
      const bool open = ImGui::TreeNodeEx(
          "##connection", ImGuiTreeNodeFlags_DefaultOpen, "%s",
          withIcon(kindIcon(c.kind), c.name + "  " +
                                         formatValue(load.force.norm(), "N"))
              .c_str());
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Force on the first part: (%.4g, %.4g, %.4g) N\n"
                          "Moment about its center: (%.4g, %.4g, %.4g) Nm",
                          load.force.x(), load.force.y(), load.force.z(),
                          load.moment.x(), load.moment.y(), load.moment.z());
      if (open) {
        const auto row = [&](const char* label, const std::string& value,
                             const char* tip) {
          ImGui::TextDisabled("%s", label);
          ImGui::SameLine(150.f);
          ImGui::TextUnformatted(value.c_str());
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
        };
        if (c.kind == Kind::Tie) {
          row("Peel, average", formatValue(load.averageNormal, "Pa"),
              "Force across the glue per area; positive pulls the faces apart");
          row("Shear, average", formatValue(load.averageShear, "Pa"),
              "Force along the glue per area");
          row("Peak peel", formatValue(load.peel, "Pa"),
              "Largest peel from the force and moment the glue carries, "
              "spread over the bond as in the elastic method for bonded "
              "joints");
          if (load.compression < 0.)
            row("Peak compression", formatValue(-load.compression, "Pa"),
                "Largest stress pressing the faces together");
          row("Peak shear", formatValue(load.shear, "Pa"),
              "Largest shear from the force and the twisting moment");
          row("Equivalent shear", formatValue(load.equivalent, "Pa"),
              "sqrt(shear^2 + peel^2 / 3) where it is largest, compared "
              "with the lap shear strength of the adhesive");
          ImGui::TextDisabled("Node peaks: peel %s, shear %s",
                              formatValue(load.nodePeel, "Pa").c_str(),
                              formatValue(load.nodeShear, "Pa").c_str());
          if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Stresses at single nodes from the glue's reactions. At the "
                "edges of a bonded joint they grow as the mesh is refined, so "
                "the safety factor uses the peaks above.");
        } else if (c.kind == Kind::BoltConnector) {
          row("Axial stress", formatValue(load.averageNormal, "Pa"),
              "Bolt tension over the shank area; negative is compression");
          row("Shear stress", formatValue(load.averageShear, "Pa"),
              "Shear force over the shank area");
          row("Von Mises", formatValue(load.equivalent, "Pa"),
              "Axial plus bending stress at the shank surface, with shear");
        } else {
          row("Contact, average",
              formatValue(load.force.norm() / load.area, "Pa"),
              "Force over the area of the first faces");
          row("Moment", formatValue(load.moment.norm(), "Nm"),
              "Moment about the center of the first faces");
        }
        if (std::isfinite(safety))
          ImGui::TextColored(safety < 1. ? ImVec4(1.f, .45f, .4f, 1.f)
                                         : ImVec4(.85f, .87f, .9f, 1.f),
                             "Factor of safety %.3g%s", safety,
                             safety < 1. ? " (fails)" : "");
        else if (c.kind == Kind::Tie || c.kind == Kind::BoltConnector)
          ImGui::TextDisabled("Set its strength in Connect for a safety "
                              "factor");
        if (load.sharedNodes)
          ImGui::TextDisabled(
              "Includes supports or connections on shared nodes");
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
  }
  bool overlayButton(const char* icon, const char* name, const char* help,
                     bool active) {
    const ImVec2 size =
        toolbarIcons
            ? ImVec2(38.f, 34.f)
            : ImVec2(ImGui::CalcTextSize(name).x +
                         2.f * ImGui::GetStyle().FramePadding.x,
                     34.f);
    const auto at = ImGui::GetCursorScreenPos();
    const bool hovered =
        ImGui::IsMouseHoveringRect(at, ImVec2(at.x + size.x, at.y + size.y));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, hovered ? 1.f : .5f);
    ImGui::PushStyleColor(ImGuiCol_Button, active
                                               ? ImVec4(.2f, .32f, .5f, 1.f)
                                               : ImVec4(.1f, .11f, .13f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          active ? ImVec4(.25f, .38f, .58f, 1.f)
                                 : ImVec4(.2f, .23f, .28f, 1.f));
    if (toolbarIcons) ImGui::PushFont(toolbarIcons);
    const bool pressed = ImGui::Button(
        (std::string(toolbarIcons ? icon : name) + "##" + name).c_str(), size);
    if (toolbarIcons) ImGui::PopFont();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(name);
      ImGui::PushTextWrapPos(ImGui::GetFontSize() * 22.f);
      ImGui::TextDisabled("%s", help);
      ImGui::PopTextWrapPos();
      ImGui::EndTooltip();
    }
    return pressed;
  }
  void viewToolbar(const ImVec2& display, bool fields) {
    ImGui::SetNextWindowPos(
        ImVec2(sidebarWidth + .5f * (display.x - sidebarWidth), display.y),
        ImGuiCond_Always, ImVec2(.5f, 1.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4.f, 0.f));
    ImGui::Begin("View tools", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::PopStyleVar();
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(.5f, .5f));
    if (overlayButton(kIconCrosshairs, "Center",
                      "Look at the geometric center of all parts", false))
      lookAtCenter(false);
    ImGui::SameLine();
    if (overlayButton(kIconExpand, "Fit all",
                      "Look at the center and zoom to show every part", false))
      lookAtCenter(true);
    if (!resultView && shownParts > 1) {
      ImGui::SameLine();
      if (overlayButton(kIconArrows, "Move parts",
                        moveParts
                            ? "Dragging a part moves it; drag empty space or "
                              "use the middle button to rotate. Click to lock "
                              "parts."
                            : "Parts are locked; dragging rotates the view. "
                              "Click to move parts by dragging.",
                        moveParts))
        moveParts = !moveParts;
    }
    if (fields) {
      constexpr int kFields[] = {0, 1};
      constexpr const char* kIcons[] = {kIconGauge, kIconDeflection};
      constexpr const char* kNames[] = {"Von Mises stress",
                                        "Displacement magnitude"};
      constexpr const char* kHelp[] = {
          "Show the equivalent stress that predicts yielding",
          "Show how far each point moves"};
      for (int i = 0; i < 2; ++i) {
        ImGui::SameLine(0.f, i ? -1.f : 18.f);
        if (overlayButton(kIcons[i], kNames[i], kHelp[i],
                          field == kFields[i]) &&
            field != kFields[i]) {
          field = kFields[i];
          visualDirty = true;
        }
      }
    }
    ImGui::PopStyleVar();
    ImGui::End();
  }
  /// "Show all parts" at the right edge of the view while parts are hidden.
  void showPartsButton(const ImVec2& display) {
    ImGui::SetNextWindowPos(ImVec2(display.x - 12.f, .5f * display.y),
                            ImGuiCond_Always, ImVec2(1.f, .5f));
    ImGui::SetNextWindowBgAlpha(.9f);
    ImGui::Begin("Hidden parts", nullptr,
                 ImGuiWindowFlags_NoDecoration |
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.2f, .32f, .5f, 1.f));
    if (ImGui::Button(withIcon(kIconEye, "Show all parts").c_str(),
                      ImVec2(0.f, 30.f))) {
      hiddenParts.clear();
      peekHidden.clear();
      hoveredFace = -1;
      visualDirty = true;
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("%zu part%s hidden", hiddenParts.size(),
                        hiddenParts.size() == 1 ? " is" : "s are");
    ImGui::End();
  }
  void finishPanel(const ImVec2& display, bool viewOverlays) {
    credit();
    ImGui::PopItemWidth();
    ImGui::End();
    if (shownGeometry) viewToolbar(display, viewOverlays && results());
    if (shownGeometry && !hiddenParts.empty()) showPartsButton(display);
    quickMates(display);
    if (!viewOverlays && !resultView) return;
    const float statusWidth =
        std::max(0.f, display.x - 16.f - (sidebarWidth + 16.f));
    ImGui::SetNextWindowPos(ImVec2(sidebarWidth + 16, 16));
    ImGui::SetNextWindowSizeConstraints(ImVec2(0.f, 0.f),
                                        ImVec2(statusWidth, FLT_MAX));
    ImGui::SetNextWindowBgAlpha(.88f);
    ImGui::Begin("Analysis status", nullptr,
                 ImGuiWindowFlags_NoDecoration |
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoMove);
    const auto& style = ImGui::GetStyle();
    const float wrap = std::max(1.f, statusWidth - style.WindowPadding.x);
    ImGui::PushTextWrapPos(wrap);
    ImGui::TextColored(ImVec4(.5f, .85f, .9f, 1.f),
                       "RAISIM  /  SOLID MECHANICS");
    ImGui::Text("Small displacement | linear tetrahedra | SI units");
    const bool busyNow = solveTask || meshJob;
    if (resultView && busyNow) {
      const std::string cancel = withIcon(kIconCancel, "Cancel");
      const float room = statusWidth - 2.f * style.WindowPadding.x;
      const float button =
          ImGui::CalcTextSize(cancel.c_str()).x + 2.f * style.FramePadding.x;
      const bool sameRow = room - button - style.ItemSpacing.x >= 120.f;
      progressBar(std::max(1.f, std::min(330.f, sameRow ? room - button -
                                                              style.ItemSpacing.x
                                                        : room)));
      if (sameRow) ImGui::SameLine();
      if (ImGui::Button(cancel.c_str())) cancelBusy();
    } else if (!viewOverlays) {
    } else if (model.hasSolution) {
      const auto& s = model.solver->statistics();
      ImGui::Text("Solver: %s | %zu iterations",
                  s.multigrid   ? "Multigrid CG"
                  : s.iterative ? "ICCG"
                                : "Sparse LDLT",
                  s.iterations);
      ImGui::Text(
          "Assembly %.2f ms  |  %s %.2f ms  |  Solve + fields %.2f ms",
          s.assemblyMs, s.iterative ? "Preconditioner" : "Factor",
          s.factorizationMs, s.solveMs);
    } else
      ImGui::TextColored(ImVec4(1.f, .78f, .35f, 1.f),
                         "Results pending - solve the current model");
    if (resultView && !busyNow && !message.empty()) {
      ImGui::PushTextWrapPos(std::min(ImGui::GetCursorPosX() + 560.f, wrap));
      ImGui::TextUnformatted(message.c_str());
      ImGui::PopTextWrapPos();
    }
    ImGui::PopTextWrapPos();
    const ImVec2 statusEnd(ImGui::GetWindowPos().x + ImGui::GetWindowSize().x,
                           ImGui::GetWindowPos().y + ImGui::GetWindowSize().y);
    ImGui::End();
    if (!viewOverlays || !results()) return;
    const bool besideStatus =
        display.x - 16.f - legendWidth > statusEnd.x + 8.f;
    ImGui::SetNextWindowPos(
        ImVec2(display.x - 16.f, besideStatus ? 16.f : statusEnd.y + 8.f),
        ImGuiCond_Always, ImVec2(1.f, 0.f));
    ImGui::Begin("Legend", nullptr,
                 ImGuiWindowFlags_NoDecoration |
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoMove);
    if (*fieldUnit())
      ImGui::Text("%s (%s)", kFieldNames[field], fieldUnit());
    else
      ImGui::Text("%s", kFieldNames[field]);
    ImGui::Text("%.4g - %.4g", colorMin, colorMax);
    const auto start = ImGui::GetCursorScreenPos();
    auto* d = ImGui::GetWindowDrawList();
    for (int i = 0; i < 180; ++i) {
      const auto color =
          valueColor(colorMin + (colorMax - colorMin) * double(i) / 179.);
      d->AddRectFilled(ImVec2(start.x + i, start.y),
                       ImVec2(start.x + i + 1, start.y + 12),
                       ImGui::ColorConvertFloat4ToU32(
                           ImVec4(color.r, color.g, color.b, 1.f)));
    }
    ImGui::Dummy(ImVec2(180, 12));
    ImGui::TextDisabled("%s | deformation %.3gx",
                        smoothContours ? "Node-averaged" : "Element values",
                        deformation);
    legendWidth = ImGui::GetWindowSize().x;
    ImGui::End();
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
  {
    const auto& x = model.mesh().vertices;
    const auto& faces = model.boundaryFaces();
    raisin::Camera view;
    fem_workbench::frameMeshCamera(view, glm::vec3(0.f),
                                   glm::vec3(1.f, .16f, .24f));
    const glm::mat4 matrix = view.getProjectionMatrix() * view.getViewMatrix();
    const auto corner = [&](size_t i) {
      return vector(x[faces[i / 3].vertices[i % 3]]);
    };
    size_t top = 0;
    while (top < faces.size() &&
           glm::cross(corner(3 * top + 1) - corner(3 * top),
                      corner(3 * top + 2) - corner(3 * top))
                   .z <= 0.f)
      ++top;
    if (top == faces.size()) return false;
    const glm::vec4 clip =
        matrix * glm::vec4((corner(3 * top) + corner(3 * top + 1) +
                            corner(3 * top + 2)) / 3.f,
                           1.f);
    const int picked = pickTriangle(matrix, clip.x / clip.w, clip.y / clip.w,
                                    faces.size(), corner);
    if (picked < 0 || model.faceSurfaces()[size_t(picked)] !=
                          model.faceSurfaces()[top] ||
        model.surfaceFaces(size_t(picked)).size() != 32u * 6u * 2u ||
        pickTriangle(matrix, .99f, .99f, faces.size(), corner) != -1)
      return false;
  }
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
  std::string capturePath, drop, recording;
  std::vector<std::string> inputs;

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
      std::cout << "rayrai_fem_workbench [cad-file...] [--frames N] [--capture "
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
    else if (arg.rfind("--", 0) != 0) {
      inputs.push_back(arg);
    }
  }
  int width = 1600, height = 1000;
#if defined(__EMSCRIPTEN__)
  width = EM_ASM_INT({ return window.innerWidth; });
  height = EM_ASM_INT({ return window.innerHeight; });
#endif
  ExampleApp app;
  if (!app.init("RaiSim FEM Workbench", width, height)) return 1;
  ImGui::GetIO().IniFilename = nullptr;
  auto& fonts = *ImGui::GetIO().Fonts;
  const auto fontPath =
      exampleRscPath(argv[0], "fonts/roboto/Roboto-Medium.ttf");
  if (std::filesystem::exists(raisim::pathFromUtf8(fontPath)))
    fonts.AddFontFromFileTTF(fontPath.c_str(), 16.f);
  else
    fonts.AddFontDefault();
  ImFont* toolbarIcons = nullptr;
  const auto iconPath = exampleRscPath(
      argv[0], "fonts/uicons/uicons-regular-rounded-fem.ttf");
  if (std::filesystem::exists(raisim::pathFromUtf8(iconPath))) {
    ImFontConfig icons;
    icons.MergeMode = true;
    icons.PixelSnapH = true;
    icons.GlyphOffset.y = 2.f;
    static const ImWchar kIconRange[] = {0xf000, 0xffff, 0};
    fonts.AddFontFromFileTTF(iconPath.c_str(), 14.f, &icons, kIconRange);
    ImFontConfig large;
    large.PixelSnapH = true;
    large.GlyphOffset.y = 1.f;
    toolbarIcons =
        fonts.AddFontFromFileTTF(iconPath.c_str(), 19.f, &large, kIconRange);
  }
  ImGui::StyleColorsDark();
  auto& style = ImGui::GetStyle();
  style.WindowRounding = 7.f;
  style.FrameRounding = 4.f;
  style.ItemSpacing = ImVec2(7, 6);
  style.ButtonTextAlign = ImVec2(0.f, .5f);
  int exitCode = 0;
  {
    Workbench workbench;
    workbench.toolbarIcons = toolbarIcons;
    const auto nativeMesher =
        fem_workbench::defaultNativeMesherExecutable(argv[0]);
    std::snprintf(workbench.nativeMesher.data(), workbench.nativeMesher.size(),
                  "%s", nativeMesher.c_str());
    workbench.viewer = std::make_shared<raisin::RayraiWindow>(
        std::make_shared<raisim::World>(), width, height);
    auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(
        raisin::RayraiWindow::RenderQualityPreset::Balanced);
    quality.mainLightDirection = glm::normalize(glm::vec3(-.35f, .55f, -1.f));
    quality.mainLightAmbient = glm::vec3(.5f);
    quality.mainLightDiffuse = glm::vec3(.95f);
    quality.mainLightSpecular = glm::vec3(.02f);
    quality.pbrEnvironmentIntensity = .12f;
    quality.addViewerFillLights = false;
    quality.proceduralSkyBackgroundEnabled = false;
    quality.backgroundColorRgb255 = glm::vec4(222.f, 223.f, 226.f, 255.f);
    quality.fogDensity = 0.f;
    workbench.viewer->setRenderQualitySettings(quality);
    workbench.attempt([&] { workbench.demo(); });
    workbench.queuedFiles = inputs;
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
          workbench.queuedFiles.push_back(path);
        }
      }
      if (app.quit) break;
      workbench.attempt([&] { workbench.pollSolve(); });
      if (workbench.solveRequested) {
        workbench.solveRequested = false;
        workbench.attempt([&] { workbench.solve(); });
      }
      if (!workbench.solveTask)
        workbench.attempt([&] { workbench.pollMesh(); });
      workbench.importQueued();
      workbench.pollBrowserErrors();
      if (showcase && drop.empty() && inputs.empty() && !workbench.solveTask) {
        workbench.model.conditions[1].value[2] =
            -5000. * (.75 + .25 * std::cos(frame * 6.283185307179586 / 120.));
        workbench.model.changed();
        workbench.attempt([&] {
          workbench.solve();
          workbench.waitForSolve();
        });
      }
      app.beginFrame();
      workbench.panel();
      const bool solving = bool(workbench.solveTask);
      if (!solving && workbench.autoSolve && !workbench.meshJob &&
          !workbench.needsMesh() && workbench.model.dirty &&
          !ImGui::IsAnyItemActive())
        workbench.solveRequested = true;
      if (!solving) workbench.attempt([&] { workbench.updateVisual(); });
      fem_workbench::updateMeshCameraClipping(workbench.viewer->getCamera());
      auto viewport = workbench.layoutViewport(app.window);
      workbench.input(viewport);
      app.endViewport(*workbench.viewer, viewport, false, false);
      if (!solving) workbench.overlays(viewport);
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
              state['viewport'] = Array.of($4, $5, $6, $7);
            }
          },
          &center.x, &camera.position.x, &projection[0][0], frame,
          viewport.origin.x, viewport.origin.y, viewport.size.x,
          viewport.size.y);
#endif
      ++frame;
      const bool finished =
          maxFrames > 0 && frame >= maxFrames && !workbench.meshJob &&
          !workbench.solveTask && workbench.queuedFiles.empty();
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
