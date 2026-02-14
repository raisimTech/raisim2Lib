// Copyright (c) 2025 Raion Robotics Inc.
// All rights reserved.

#include <SDL.h>

#include <glbinding/glbinding.h>
#include <glbinding/gl/gl.h>
#include <imgui/imgui.h>

#include "imgui/backend/imgui_impl_opengl3.h"
#include "imgui/backend/imgui_impl_sdl2.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "rayrai/RayraiWindow.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/OpenGLMesh.hpp"
#include "rayrai/CoordinateFrame.hpp"
#include "rayrai/RaisimTcpCommon.hpp"
#include "rayrai/raisin_imgui_style.h"
#include "raisim/configure.hpp"
#include "raisim/sensors/Sensors.hpp"

#if defined(__linux__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace
{

// SDL installs signal handlers by default (SIGINT/SIGTERM) that may call SDL_Quit()
// from inside the handler. That can conflict with our normal shutdown sequence and
// lead to double-free/corruption. We disable SDL signal handlers and handle SIGINT
// ourselves by requesting a graceful shutdown.
std::atomic<bool> gSignalQuit{false};

void handleSignalQuit(int /*sig*/) {
  gSignalQuit.store(true, std::memory_order_relaxed);
}

constexpr int kDefaultPort = raisin::tcp_viewer::kDefaultPort;
constexpr int kConnectTimeoutMs = 2000;
constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
constexpr auto kAutoConnectInterval = std::chrono::seconds(3);
constexpr float kBaseFontSize = 24.0f;
constexpr float kFontScale = 0.75f;
constexpr float kUiScaleEpsilon = 0.01f;

using raisin::tcp_viewer::BufferReader;
using raisin::tcp_viewer::ObjectListItem;
using raisin::tcp_viewer::PendingSensorUpdate;
using raisin::tcp_viewer::RemoteScene;
using raisin::tcp_viewer::SelectedObjectInfo;
using raisin::tcp_viewer::sendSensorUpdate;
using raisin::tcp_viewer::sendUpdateRequest;
using raisin::tcp_viewer::TcpClient;
using raisin::tcp_viewer::VisualEntry;

struct ConnectionEntry {
  std::string host;
  int port = kDefaultPort;
};

std::string shortenPathLabel(const std::string& value, size_t maxLen) {
  if (value.size() <= maxLen) {
    return value;
  }
  if (maxLen <= 3) {
    return value.substr(0, maxLen);
  }
  return value.substr(0, maxLen - 3) + "...";
}

std::string formatConnectionLabel(const ConnectionEntry& entry) {
  if (entry.host.empty()) {
    return {};
  }
  return entry.host + ":" + std::to_string(entry.port);
}

std::string toLowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool isContactLabel(const std::string& label) {
  if (label.empty()) {
    return false;
  }
  const std::string lower = toLowerAscii(label);
  if (lower.find("contact") == std::string::npos) {
    return false;
  }
  return lower.find("point") != std::string::npos || lower.find("force") != std::string::npos ||
         lower.find("contacts") != std::string::npos;
}

bool isContactEntry(const VisualEntry* entry) {
  if (!entry) {
    return false;
  }
  return isContactLabel(entry->name) || isContactLabel(entry->objectName) ||
         isContactLabel(entry->meshFile);
}

bool isContactItem(const ObjectListItem& item) {
  return isContactLabel(item.name);
}

void recordConnection(
  std::vector<ConnectionEntry>& connections, const std::string& host, int port) {
  if (host.empty()) {
    return;
  }
  connections.erase(
    std::remove_if(connections.begin(), connections.end(),
      [&](const ConnectionEntry& entry) { return entry.host == host && entry.port == port; }),
    connections.end());
  connections.insert(connections.begin(), ConnectionEntry{host, port});
  if (connections.size() > 8) {
    connections.resize(8);
  }
}

void recordResourceDir(std::vector<std::string>& dirs, const std::string& path) {
  if (path.empty()) {
    return;
  }
  dirs.erase(std::remove(dirs.begin(), dirs.end(), path), dirs.end());
  dirs.insert(dirs.begin(), path);
  if (dirs.size() > 24) {
    dirs.resize(24);
  }
}

glm::vec3 lightDirectionFromYawPitch(float yawDeg, float pitchDeg) {
  const float yawRad = yawDeg * kDegToRad;
  const float pitchRad = pitchDeg * kDegToRad;
  glm::vec3 dir(std::cos(pitchRad) * std::cos(yawRad), std::cos(pitchRad) * std::sin(yawRad),
    std::sin(pitchRad));
  return glm::normalize(dir);
}

void yawPitchFromDirection(const glm::vec3& dir, float& yawDeg, float& pitchDeg) {
  const glm::vec3 n = glm::normalize(dir);
  const float yaw = std::atan2(n.y, n.x);
  const float pitch = std::asin(std::clamp(n.z, -1.0f, 1.0f));
  yawDeg = yaw * kRadToDeg;
  pitchDeg = pitch * kRadToDeg;
}

double computeWorldFrameSize(const raisin::Camera& cam) {
  constexpr double kMinDistance = 0.05;
  double visibleHeight = 0.0;
  if (cam.getProjectionMode() == raisin::Camera::ProjectionMode::ORTHOGRAPHIC) {
    visibleHeight = cam.orthoScale;
  } else {
    const glm::vec3 camPos = cam.getPosition();
    const double distance =
      std::max(kMinDistance, static_cast<double>(glm::distance(camPos, glm::vec3(0.0f))));
    const double fovy = glm::radians(static_cast<double>(cam.zoom));
    visibleHeight = 2.0 * distance * std::tan(0.5 * fovy);
  }
  if (visibleHeight <= 0.0) {
    return 0.1;
  }
  return visibleHeight * 0.1;
}

const char* objectTypeLabel(int objectTypeRaw) {
  if (objectTypeRaw == -1) {
    return "visual";
  }
  if (objectTypeRaw < 0) {
    return "unknown";
  }
  switch (static_cast<raisim::ObjectType>(objectTypeRaw)) {
    case raisim::ObjectType::SPHERE:
      return "sphere";
    case raisim::ObjectType::BOX:
      return "box";
    case raisim::ObjectType::CYLINDER:
      return "cylinder";
    case raisim::ObjectType::CAPSULE:
      return "capsule";
    case raisim::ObjectType::MESH:
      return "mesh";
    case raisim::ObjectType::HALFSPACE:
      return "halfspace";
    case raisim::ObjectType::HEIGHTMAP:
      return "heightmap";
    case raisim::ObjectType::ARTICULATED_SYSTEM:
      return "articulated system";
    case raisim::ObjectType::COMPOUND:
      return "compound";
    default:
      return "unknown";
  }
}

void drawOverlaySlider(const char* id, const char* label, float* value, float min, float max,
  const char* format, float valueWidth, float itemWidth, bool disabled) {
  ImGui::PushItemWidth(itemWidth);
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
  ImGui::SliderFloat(id, value, min, max, format);
  ImGui::PopStyleColor();
  ImGui::PopItemWidth();

  char valueBuf[32];
  std::snprintf(valueBuf, sizeof(valueBuf), format, *value);

  const ImVec2 itemMin = ImGui::GetItemRectMin();
  const ImVec2 itemMax = ImGui::GetItemRectMax();
  const float textHeight = ImGui::GetFontSize();
  const float textY = itemMin.y + (itemMax.y - itemMin.y - textHeight) * 0.5f;
  const float padding = ImGui::GetStyle().FramePadding.x;
  const float labelX = itemMin.x + padding;
  const float valueX = itemMax.x - padding - valueWidth;
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  drawList->PushClipRect(itemMin, itemMax, true);
  const ImU32 textColor = ImGui::GetColorU32(disabled ? ImGuiCol_TextDisabled : ImGuiCol_Text);
  drawList->AddText(ImVec2(labelX, textY), textColor, label);
  drawList->AddText(ImVec2(valueX, textY), textColor, valueBuf);
  drawList->PopClipRect();
}

void renderViewer(raisin::RayraiWindow& viewer, SDL_Window* window) {
  int fbW = 0;
  int fbH = 0;
  SDL_GL_GetDrawableSize(window, &fbW, &fbH);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2((float)fbW, (float)fbH));
  ImGui::Begin("Viewer", nullptr,
    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
      ImGuiWindowFlags_NoFocusOnAppearing);

  ImTextureID tex = (ImTextureID)(intptr_t)viewer.getImageTexture();
  ImVec2 windowPos = ImGui::GetCursorScreenPos();
  ImGuiIO& io = ImGui::GetIO();
  ImGui::Image(tex, ImVec2((float)fbW, (float)fbH), ImVec2(0, 1), ImVec2(1, 0));

  const bool isHovered = ImGui::IsItemHovered();
  const int cursorX = static_cast<int>(io.MousePos.x - windowPos.x);
  const int cursorY = static_cast<int>(io.MousePos.y - windowPos.y);
  viewer.update(fbW, fbH, isHovered, cursorX, cursorY, true);

  ImGui::End();
  ImGui::PopStyleVar(2);
}

} // namespace

int main(int /*argc*/, char* /*argv*/[]) {
  SDL_SetHint(SDL_HINT_VIDEO_HIGHDPI_DISABLED, "permonitorv2");
  SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
  std::signal(SIGINT, handleSignalQuit);
#if defined(SIGTERM)
  std::signal(SIGTERM, handleSignalQuit);
#endif
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
    std::cerr << "FATAL ERROR: Failed to initialize SDL: " << SDL_GetError() << "\n";
    return -1;
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  SDL_Window* window = SDL_CreateWindow("Rayrai Raisim TCP Viewer", SDL_WINDOWPOS_CENTERED,
    SDL_WINDOWPOS_CENTERED, 1280, 720,
    SDL_WindowFlags(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI));

  if (!window) {
    std::cerr << "FATAL ERROR: Failed to create SDL window: " << SDL_GetError() << "\n";
    SDL_Quit();
    return -1;
  }

  SDL_GLContext context = SDL_GL_CreateContext(window);
  if (!context) {
    std::cerr << "FATAL ERROR: Failed to create OpenGL context: " << SDL_GetError() << "\n";
    SDL_DestroyWindow(window);
    SDL_Quit();
    return -1;
  }

  SDL_GL_MakeCurrent(window, context);
  SDL_GL_SetSwapInterval(1);

  glbinding::initialize(
    [](const char* name) {
      return reinterpret_cast<glbinding::ProcAddress>(SDL_GL_GetProcAddress(name));
    },
    false);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplSDL2_InitForOpenGL(window, context);
  ImGui_ImplOpenGL3_Init("#version 330");
  raionrobotics_imgui_theme();

  auto world = std::make_shared<raisim::World>();
  auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 720);
  viewer->setBackgroundColor({20, 20, 30, 255});
  // viewer->setGroundPatternResourcePath(
  //   raisin::getResourceDirectory("raisin_gui") + "material/checkerboard/checker_gray-01.png");
  viewer->setShowCollisionBodies(false);
  auto& camera = viewer->getCamera();
  camera.nearPlane = 0.01f;
  camera.farPlane = 1000.0f;
  camera.zNear = 0.01f;
  camera.zFar = 1000.0f;
  auto& light = viewer->getLight();
  light.type = raisin::LightType::DIRECTIONAL;
  light.ambient = glm::vec3(0.31f, 0.31f, 0.31f);
  light.diffuse = glm::vec3(0.9f, 0.9f, 0.9f);
  light.specular = glm::vec3(0.2f, 0.2f, 0.2f);
  light.setShadowsEnabled(true);

  TcpClient client;
  RemoteScene scene(viewer);
  scene.setShowCollisionBodies(false);
  scene.setForceTransparent(false);

  char host[128] = "127.0.0.1";
  int port = kDefaultPort;
  char portBuf[16];
  std::snprintf(portBuf, sizeof(portBuf), "%d", port);
  std::vector<ConnectionEntry> recentConnections;
  std::vector<std::string> resourceDirs;
  char searchPathBuf[256] = "";
  bool quit = false;
  std::string lastStatus = "disconnected";
  bool verboseParsing = false;
  bool showCollisionBodies = false;
  bool showWorldFrame = false;
  bool showContactPoints = false;
  bool showContactForces = false;
  bool forceTransparent = false;
  float contactPointSize = 0.05f;
  float contactForceSize = 0.3f;
  float cameraSpeed = viewer->getCamera().movementSpeed;
  float lightYawDeg = 0.0f;
  float lightPitchDeg = -30.0f;
  float lightStrength = 0.9f;
  float ambientStrength = 1.0f;
  ImVec2 overlayOffset(0.0f, 0.0f);
  ImVec2 detailOffset(0.0f, 0.0f);
  ImVec2 detailSize(260.0f, 200.0f);
  bool overlayMinimized = false;
  std::shared_ptr<raisin::CoordinateFrame> worldFrame;
  bool awaitingResponse = false;
  bool awaitingSensorAck = false;
  bool autoConnect = false;
  float uiScale = 1.0f;
  float defaultUiScale = 1.0f;
  bool uiScaleInitialized = false;
  bool uiScaleUserSet = false;
  float appliedUiScale = 0.0f;
  bool baseStyleCaptured = false;
  ImGuiStyle baseStyle;
  ImVec2 lastDisplaySize(0.0f, 0.0f);
  auto nextAutoConnectAttempt = std::chrono::steady_clock::now();
  light.direction = lightDirectionFromYawPitch(lightYawDeg, lightPitchDeg);

  while (!quit && !gSignalQuit.load(std::memory_order_relaxed)) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL2_ProcessEvent(&event);
      if (event.type == SDL_QUIT)
        quit = true;
      if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
          event.window.windowID == SDL_GetWindowID(window))
        quit = true;
    }

    int fbW = 0;
    int fbH = 0;
    SDL_GL_GetDrawableSize(window, &fbW, &fbH);
    const ImVec2 displaySize(static_cast<float>(fbW), static_cast<float>(fbH));
    const float scaleX = displaySize.x / 1920.0f;
    const float scaleY = displaySize.y / 1080.0f;
    defaultUiScale = std::clamp(std::min(scaleX, scaleY) * 1.25f, 1.1f, 2.6f);
    if (!uiScaleInitialized || (!uiScaleUserSet && (displaySize.x != lastDisplaySize.x ||
                                                     displaySize.y != lastDisplaySize.y))) {
      uiScale = defaultUiScale;
      uiScaleInitialized = true;
    }
    lastDisplaySize = displaySize;
    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.0f;
    if (!baseStyleCaptured) {
      baseStyle = ImGui::GetStyle();
      baseStyleCaptured = true;
    }
    if (std::abs(appliedUiScale - uiScale) > kUiScaleEpsilon) {
      ImFontConfig fontConfig;
      fontConfig.SizePixels = std::max(1.0f, std::round(kBaseFontSize * uiScale * kFontScale));
      fontConfig.OversampleH = 2;
      fontConfig.OversampleV = 2;
      fontConfig.PixelSnapH = true;
      fontConfig.RasterizerMultiply = 1.0f;
      io.Fonts->Clear();
      io.FontDefault = io.Fonts->AddFontDefault(&fontConfig);
      ImGui_ImplOpenGL3_DestroyFontsTexture();
      ImGui_ImplOpenGL3_CreateFontsTexture();
      ImGuiStyle scaledStyle = baseStyle;
      scaledStyle.ScaleAllSizes(uiScale);
      ImGui::GetStyle() = scaledStyle;
      appliedUiScale = uiScale;
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    float menuBarHeight = 0.0f;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    if (ImGui::BeginMainMenuBar()) {
      menuBarHeight = ImGui::GetWindowHeight();
      if (ImGui::BeginMenu("View")) {
        ImGui::SetNextItemWidth(160.0f);
        if (ImGui::SliderFloat("UI Scale", &uiScale, 0.8f, 2.6f, "%.2f")) {
          uiScaleUserSet = true;
        }
        if (ImGui::MenuItem("Reset to Screen Scale")) {
          uiScale = defaultUiScale;
          uiScaleUserSet = false;
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }
    ImGui::PopStyleVar(3);

    const auto now = std::chrono::steady_clock::now();
    if (autoConnect && !client.isConnected() && now >= nextAutoConnectAttempt) {
      lastStatus = "auto-connecting";
      if (client.connectTo(host, port, false, kConnectTimeoutMs)) {
        lastStatus = "connected";
        awaitingResponse = false;
        awaitingSensorAck = false;
        recordConnection(recentConnections, host, port);
      } else {
        lastStatus = "auto-connect failed";
      }
      nextAutoConnectAttempt = now + kAutoConnectInterval;
    }

    uint32_t requestedTag = 0;
    int requestedIndex = 0;
    const VisualEntry* requestedEntry = nullptr;
    scene.getVisualInfo(viewer->getTargetVisual(), requestedTag, requestedIndex, requestedEntry);
    static_cast<void>(requestedIndex);
    static_cast<void>(requestedEntry);
    if (requestedEntry && (requestedEntry->shape == raisim::Shape::Ground ||
                            requestedEntry->shape == raisim::Shape::HeightMap)) {
      viewer->setTargetVisual(nullptr);
      requestedTag = 0;
    }
    scene.setSelectionTag(requestedTag);

    if (client.isConnected()) {
      std::vector<char> payload;
      bool networkFailed = false;

      if (awaitingSensorAck) {
        if (!client.recvMessage(payload)) {
          if (!client.lastIoWouldBlock()) {
            lastStatus = "sensor ack failed";
            networkFailed = true;
          }
        } else {
          awaitingSensorAck = false;
        }
      }

      if (!networkFailed && !awaitingSensorAck) {
        if (!awaitingResponse) {
          if (!sendUpdateRequest(client, requestedTag)) {
            if (!client.lastIoWouldBlock()) {
              lastStatus = "connection lost";
              networkFailed = true;
            }
          } else {
            awaitingResponse = true;
          }
        }

        if (!networkFailed && awaitingResponse) {
          if (!client.recvMessage(payload)) {
            if (!client.lastIoWouldBlock()) {
              lastStatus = "connection lost";
              networkFailed = true;
            }
          } else {
            awaitingResponse = false;
            BufferReader reader(payload);
            std::vector<PendingSensorUpdate> pending;
            scene.setVerbose(verboseParsing);
            const bool parsedOk = scene.applyResponse(reader, pending);
            const bool disconnectRequested = scene.consumeDisconnectRequested();
            if (disconnectRequested) {
              lastStatus = "protocol error (disconnect)";
              networkFailed = true;
            } else if (!parsedOk) {
              lastStatus = "parse error (dropped update)";
            } else if (!pending.empty()) {
              if (!sendSensorUpdate(client, pending)) {
                if (!client.lastIoWouldBlock()) {
                  lastStatus = "sensor update failed";
                  networkFailed = true;
                }
              } else {
                if (!client.recvMessage(payload)) {
                  if (client.lastIoWouldBlock()) {
                    awaitingSensorAck = true;
                  } else {
                    lastStatus = "sensor ack failed";
                    networkFailed = true;
                  }
                }
              }
            }
          }
        }
      }

      if (networkFailed) {
        awaitingResponse = false;
        awaitingSensorAck = false;
        client.disconnect();
        scene.clear();
      }
    }

    if (showWorldFrame) {
      if (!worldFrame) {
        worldFrame = viewer->addCoordinateFrame("world_frame");
      }
      if (worldFrame) {
        worldFrame->poses.resize(1);
        worldFrame->poses[0].position = Eigen::Vector3d(0.0, 0.0, 0.0);
        worldFrame->poses[0].quaternion = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
        worldFrame->frameSize = computeWorldFrameSize(viewer->getCamera());
      }
    } else if (worldFrame) {
      viewer->removeCoordinateFrame("world_frame");
      worldFrame.reset();
    }
    if (showWorldFrame && worldFrame) {
      worldFrame->frameSize = computeWorldFrameSize(viewer->getCamera());
    }

    auto& cam = viewer->getCamera();
    cam.nearPlane = 0.01f;
    cam.farPlane = 1000.0f;
    cam.zNear = 0.01f;
    cam.zFar = 1000.0f;
    cam.movementSpeed = cameraSpeed;
    auto& lightRef = viewer->getLight();
    const glm::vec3 baseAmbient(0.15f, 0.15f, 0.15f);
    const glm::vec3 baseDiffuse(0.9f, 0.9f, 0.9f);
    const glm::vec3 baseSpecular(0.2f, 0.2f, 0.2f);
    lightRef.type = raisin::LightType::DIRECTIONAL;
    lightRef.ambient = baseAmbient * ambientStrength;
    lightRef.diffuse = baseDiffuse * lightStrength;
    lightRef.specular = baseSpecular * lightStrength;
    lightRef.direction = lightDirectionFromYawPitch(lightYawDeg, lightPitchDeg);

    scene.updateContactVisuals(
      showContactPoints, contactPointSize, showContactForces, contactForceSize);

    renderViewer(*viewer, window);

    const ImVec2 overlayBase(12.0f, 12.0f + menuBarHeight);
    const ImVec2 overlayPos(overlayBase.x + overlayOffset.x, overlayBase.y + overlayOffset.y);
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::SetNextWindowPos(overlayPos, ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    const ImGuiWindowFlags overlayFlags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNavFocus;
    if (ImGui::Begin("Raisim TCP##Overlay", nullptr, overlayFlags)) {
      const ImGuiStyle& style = ImGui::GetStyle();
      const float toggleWidth =
        ImGui::CalcTextSize("-").x + style.FramePadding.x * 2.0f;
      const float toggleButtonWidth = toggleWidth * 1.6f;
      const float headerWidth = ImGui::GetContentRegionAvail().x;
      const float dragWidth =
        std::max(0.0f, headerWidth - toggleButtonWidth - style.ItemSpacing.x);

      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.35f, 0.8f, 1.0f, 1.0f));
      ImGui::Selectable("Raisim TCP", false, ImGuiSelectableFlags_None, ImVec2(dragWidth, 0.0f));
      ImGui::PopStyleColor();
      if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 delta = ImGui::GetIO().MouseDelta;
        overlayOffset.x = std::max(0.0f, overlayOffset.x + delta.x);
        overlayOffset.y = std::max(0.0f, overlayOffset.y + delta.y);
      }
      ImGui::SameLine();
      const float toggleHeight = ImGui::GetFrameHeight();
      if (ImGui::Button("-##overlay_toggle", ImVec2(toggleButtonWidth, toggleHeight))) {
        overlayMinimized = !overlayMinimized;
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", overlayMinimized ? "Expand" : "Minimize");
      }

      if (!overlayMinimized && ImGui::BeginTabBar("##LeftTabs")) {
        if (ImGui::BeginTabItem("Control")) {
          ConnectionEntry current;
          current.host = host;
          current.port = port;
          std::string preview = formatConnectionLabel(current);
          if (preview.empty()) {
            preview = "set host:port";
          }
          float comboLabelWidth = ImGui::CalcTextSize(preview.c_str()).x;
          for (const auto& entry : recentConnections) {
            const std::string label = formatConnectionLabel(entry);
            comboLabelWidth = std::max(comboLabelWidth, ImGui::CalcTextSize(label.c_str()).x);
          }
          const float minHostTextWidth = ImGui::CalcTextSize("255.255.255.255").x;
          const float minPortTextWidth = ImGui::CalcTextSize("65535").x;
          const float hostTextWidth =
            std::max(minHostTextWidth, ImGui::CalcTextSize(host).x);
          const float portTextWidth =
            std::max(minPortTextWidth, ImGui::CalcTextSize(portBuf).x);
          const float hostInputWidth = hostTextWidth + style.FramePadding.x * 2.0f;
          const float portInputWidth = portTextWidth + style.FramePadding.x * 2.0f;
          const float hostLabelWidth = ImGui::CalcTextSize("Host").x;
          const float portLabelWidth = ImGui::CalcTextSize("Port").x;
          const float labelSpacing = style.ItemInnerSpacing.x;
          const float segmentSpacing = style.ItemSpacing.x;
          const float hostSegmentWidth = hostLabelWidth + labelSpacing + hostInputWidth;
          const float portSegmentWidth = portLabelWidth + labelSpacing + portInputWidth;
          const float hostRowWidth = hostSegmentWidth + segmentSpacing + portSegmentWidth;
          const float comboWidth =
            comboLabelWidth + style.FramePadding.x * 2.0f + ImGui::GetFrameHeight();
          const float popupMinWidth = std::max(comboWidth, hostRowWidth) + style.WindowPadding.x * 2.0f;
          ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
          ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.35f, 0.35f, 0.35f, 0.9f));
          ImGui::SetNextItemWidth(comboWidth);
          ImGui::SetNextWindowSizeConstraints(
            ImVec2(popupMinWidth, 0.0f), ImVec2(FLT_MAX, FLT_MAX));
          if (ImGui::BeginCombo("##Connection", preview.c_str(), ImGuiComboFlags_HeightSmall)) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Host");
            ImGui::SameLine(0.0f, labelSpacing);
            ImGui::SetNextItemWidth(hostInputWidth);
            ImGui::InputText("##Host", host, sizeof(host));
            ImGui::SameLine(0.0f, segmentSpacing);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Port");
            ImGui::SameLine(0.0f, labelSpacing);
            ImGui::SetNextItemWidth(portInputWidth);
            ImGui::InputText("##Port", portBuf, sizeof(portBuf),
              ImGuiInputTextFlags_CharsDecimal);
            if (portBuf[0] != '\0') {
              int parsed = std::atoi(portBuf);
              if (parsed < 0) {
                parsed = 0;
              } else if (parsed > 65535) {
                parsed = 65535;
              }
              port = parsed;
            }
            ImGui::SeparatorText("Recent");
            if (recentConnections.empty()) {
              ImGui::TextDisabled("No recent connections");
            } else {
              for (const auto& entry : recentConnections) {
                const std::string label = formatConnectionLabel(entry);
                if (ImGui::Selectable(label.c_str())) {
                  std::snprintf(host, sizeof(host), "%s", entry.host.c_str());
                  port = entry.port;
                  std::snprintf(portBuf, sizeof(portBuf), "%d", port);
                }
              }
            }
            ImGui::EndCombo();
          }
          ImGui::PopStyleColor();
          ImGui::PopStyleVar();

          ImGui::SameLine();
          if (!client.isConnected()) {
            if (ImGui::Button("Connect")) {
              lastStatus = "connecting";
              if (client.connectTo(host, port, true, kConnectTimeoutMs)) {
                lastStatus = "connected";
                awaitingResponse = false;
                awaitingSensorAck = false;
                recordConnection(recentConnections, host, port);
              } else {
                lastStatus = "connect failed";
              }
            }
          } else {
            if (ImGui::Button("Disconnect")) {
              client.disconnect();
              awaitingResponse = false;
              awaitingSensorAck = false;
              lastStatus = "disconnected";
              scene.clear();
            }
          }

          ImGui::SameLine();
          if (ImGui::Checkbox("Auto-connect", &autoConnect)) {
            if (autoConnect) {
              nextAutoConnectAttempt = now;
            }
          }

          char worldText[32];
          if (scene.hasServerWorldTime()) {
            std::snprintf(worldText, sizeof(worldText), "World %.3f s", scene.getServerWorldTime());
          } else {
            std::snprintf(worldText, sizeof(worldText), "World --");
          }
          const ImVec4 statusColor =
            client.isConnected() ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(0.9f, 0.2f, 0.2f, 1.0f);
          const float statusWidth =
            ImGui::CalcTextSize("Status: parse error (dropped update)").x;
          const float worldTextWidth = ImGui::CalcTextSize("World 12345678.901 s").x;
          const float rowWidth = statusWidth + style.ItemSpacing.x + worldTextWidth;
          const float rowStartX = ImGui::GetCursorPosX();
          ImGui::TextColored(statusColor, "Status: %s", lastStatus.c_str());
          ImGui::SameLine();
          const float worldX = rowStartX + rowWidth - worldTextWidth;
          if (worldX > ImGui::GetCursorPosX()) {
            ImGui::SetCursorPosX(worldX);
          }
          ImGui::TextUnformatted(worldText);
          ImGui::TextDisabled("Heightmap colors: server color map");

          if (ImGui::BeginTable("##viewer_checkboxes", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableNextColumn();
            ImGui::Checkbox("Verbose parsing", &verboseParsing);

            ImGui::TableNextColumn();
            if (ImGui::Checkbox("Show Collision Bodies", &showCollisionBodies)) {
              viewer->setShowCollisionBodies(showCollisionBodies);
              scene.setShowCollisionBodies(showCollisionBodies);
            }

            ImGui::TableNextColumn();
            if (ImGui::Checkbox("All Transparent", &forceTransparent)) {
              scene.setForceTransparent(forceTransparent);
            }

            ImGui::TableNextColumn();
            if (ImGui::Checkbox("Show World Frame", &showWorldFrame)) {
              if (showWorldFrame) {
                if (!worldFrame) {
                  worldFrame = viewer->addCoordinateFrame("world_frame");
                }
                if (worldFrame) {
                  worldFrame->poses.resize(1);
                  worldFrame->poses[0].position = Eigen::Vector3d(0.0, 0.0, 0.0);
                  worldFrame->poses[0].quaternion = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
                  worldFrame->frameSize = computeWorldFrameSize(viewer->getCamera());
                }
              } else if (worldFrame) {
                viewer->removeCoordinateFrame("world_frame");
                worldFrame.reset();
              }
            }

            ImGui::TableNextColumn();
            ImGui::Checkbox("Show Contact Points", &showContactPoints);

            ImGui::TableNextColumn();
            ImGui::Checkbox("Show Contact Forces", &showContactForces);
            ImGui::EndTable();
          }

          const bool contactPointsEnabled = showContactPoints;
          const bool contactForcesEnabled = showContactForces;
          auto measureValueWidth = [](float value, const char* format) {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), format, value);
            return ImGui::CalcTextSize(buffer).x;
          };
          const float leftValueWidth = std::max({measureValueWidth(contactPointSize, "%.3f"),
            measureValueWidth(cameraSpeed, "%.1f"), measureValueWidth(lightPitchDeg, "%.1f"),
            measureValueWidth(ambientStrength, "%.2f")});
          const float rightValueWidth = std::max({measureValueWidth(contactForceSize, "%.2f"),
            measureValueWidth(lightYawDeg, "%.1f"), measureValueWidth(lightStrength, "%.2f")});
          const float innerSpacing = ImGui::GetStyle().ItemInnerSpacing.x;
          const float padding = ImGui::GetStyle().FramePadding.x;
          const float leftLabelWidth = std::max({ImGui::CalcTextSize("Contact Pt (m)").x,
            ImGui::CalcTextSize("Camera Speed").x, ImGui::CalcTextSize("Light Pitch (deg)").x,
            ImGui::CalcTextSize("Ambient Strength").x});
          const float rightLabelWidth = std::max({ImGui::CalcTextSize("Contact Force (m)").x,
            ImGui::CalcTextSize("Light Yaw (deg)").x, ImGui::CalcTextSize("Light Strength").x});
          const float leftItemWidth = leftLabelWidth + leftValueWidth + innerSpacing + padding * 2;
          const float rightItemWidth =
            rightLabelWidth + rightValueWidth + innerSpacing + padding * 2;
          const float cellPadX = ImGui::GetStyle().CellPadding.x;
          const float sliderRowWidth =
            leftItemWidth + rightItemWidth + ImGui::GetStyle().ItemSpacing.x + cellPadX * 4.0f;
          ImGui::PushStyleVar(
            ImGuiStyleVar_CellPadding, ImVec2(8.0f, ImGui::GetStyle().CellPadding.y));
          if (ImGui::BeginTable("##viewer_sliders", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("left", ImGuiTableColumnFlags_WidthFixed, leftItemWidth);
            ImGui::TableSetupColumn("right", ImGuiTableColumnFlags_WidthFixed, rightItemWidth);
            ImGui::TableNextColumn();
            ImGui::BeginDisabled(!contactPointsEnabled);
            drawOverlaySlider("##contact_point_size", "Contact Pt (m)", &contactPointSize, 0.001f,
              0.4f, "%.3f", leftValueWidth, leftItemWidth, !contactPointsEnabled);
            ImGui::EndDisabled();

            ImGui::TableNextColumn();
            ImGui::BeginDisabled(!contactForcesEnabled);
            drawOverlaySlider("##contact_force_size", "Contact Force (m)", &contactForceSize, 0.01f,
              2.0f, "%.2f", rightValueWidth, rightItemWidth, !contactForcesEnabled);
            ImGui::EndDisabled();

            ImGui::TableNextColumn();
            drawOverlaySlider("##camera_speed", "Camera Speed", &cameraSpeed, 0.1f, 30.0f, "%.1f",
              leftValueWidth, leftItemWidth, false);

            ImGui::TableNextColumn();
            drawOverlaySlider("##light_yaw", "Light Yaw (deg)", &lightYawDeg, -180.0f, 180.0f,
              "%.1f", rightValueWidth, rightItemWidth, false);

            ImGui::TableNextColumn();
            drawOverlaySlider("##light_pitch", "Light Pitch (deg)", &lightPitchDeg, -89.0f, 89.0f,
              "%.1f", leftValueWidth, leftItemWidth, false);

            ImGui::TableNextColumn();
            drawOverlaySlider("##light_strength", "Light Strength", &lightStrength, 0.1f, 1.2f,
              "%.2f", rightValueWidth, rightItemWidth, false);

            ImGui::TableNextColumn();
            drawOverlaySlider("##ambient_strength", "Ambient Strength", &ambientStrength, 0.0f,
              2.0f, "%.2f", leftValueWidth, leftItemWidth, false);

            ImGui::TableNextColumn();
            ImGui::Dummy(ImVec2(rightItemWidth, 0.0f));
            ImGui::EndTable();
          }
          ImGui::PopStyleVar();

          ImGui::SeparatorText("Resource dirs");
          ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.15f, 0.55f));
          ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
          ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
          const ImGuiChildFlags resourceFlags = ImGuiChildFlags_Border |
                                                ImGuiChildFlags_AutoResizeY |
                                                ImGuiChildFlags_AlwaysUseWindowPadding;
          if (ImGui::BeginChild(
                "##resource_dirs_box", ImVec2(sliderRowWidth, 0.0f), resourceFlags)) {
            const float rowWidth = ImGui::GetContentRegionAvail().x;
            const ImVec2 buttonTextSize = ImGui::CalcTextSize("Add");
            const float buttonWidth = buttonTextSize.x + ImGui::GetStyle().FramePadding.x * 2.0f;
            const float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
            const float inputWidth = std::max(0.0f, rowWidth - buttonWidth - spacing);
            ImGui::SetNextItemWidth(inputWidth);
            ImGui::InputTextWithHint(
              "##resource_dir_input", "path/to/resources", searchPathBuf, sizeof(searchPathBuf));
            ImGui::SameLine(0.0f, spacing);
            if (ImGui::Button("Add", ImVec2(buttonWidth, 0.0f))) {
              std::string path(searchPathBuf);
              if (!path.empty()) {
                scene.addSearchPath(path);
                recordResourceDir(resourceDirs, path);
                searchPathBuf[0] = '\0';
              }
            }

            if (!resourceDirs.empty()) {
              const float removeSize = ImGui::GetFontSize() * 1.1f;
              if (ImGui::BeginTable("##resource_dirs", 2, ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableSetupColumn("path", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("remove", ImGuiTableColumnFlags_WidthFixed, removeSize);
                for (size_t i = 0; i < resourceDirs.size(); ++i) {
                  const auto& entry = resourceDirs[i];
                  const std::string label = shortenPathLabel(entry, 50);
                  ImGui::TableNextRow();
                  ImGui::TableSetColumnIndex(0);
                  ImGui::TextUnformatted(label.c_str());
                  ImGui::TableSetColumnIndex(1);
                  ImGui::PushID(static_cast<int>(i));
                  if (ImGui::Button("x", ImVec2(removeSize, removeSize))) {
                    resourceDirs.erase(resourceDirs.begin() + static_cast<long>(i));
                    ImGui::PopID();
                    break;
                  }
                  ImGui::PopID();
                }
                ImGui::EndTable();
              }
            }
          }
          ImGui::EndChild();
          ImGui::PopStyleVar(2);
          ImGui::PopStyleColor();
          ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Objects")) {
          uint32_t selectedTag = 0;
          int selectedIndex = 0;
          const VisualEntry* selectedEntry = nullptr;
          const raisin::Visuals* selectedVisual = viewer->getTargetVisual();
          const bool hasSelectedRaw =
            scene.getVisualInfo(selectedVisual, selectedTag, selectedIndex, selectedEntry);
          const bool selectedIsContact = hasSelectedRaw && isContactEntry(selectedEntry);
          if (selectedIsContact && viewer) {
            viewer->setTargetVisual(nullptr);
          }
          const bool hasSelected = hasSelectedRaw && !selectedIsContact;
          static_cast<void>(selectedIndex);
          static_cast<void>(selectedEntry);
          const auto items = scene.getSelectableObjects();

          if (items.empty()) {
            ImGui::TextDisabled("No objects available");
          } else {
            const ImU32 typeColor = ImGui::GetColorU32(ImVec4(0.35f, 0.8f, 1.0f, 1.0f));
            const float listHeight = ImGui::GetFontSize() * 12.0f;
            if (ImGui::BeginChild("##ObjectList", ImVec2(0.0f, listHeight), true)) {
              for (const auto& item : items) {
                if (isContactItem(item)) {
                  continue;
                }
                const char* typeName = objectTypeLabel(item.objectTypeRaw);
                const bool selected = hasSelected && item.tag == selectedTag;
                const std::string id =
                  "##obj_" + std::to_string(item.tag) + "_" + std::to_string(item.index);
                if (ImGui::Selectable(id.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
                  if (viewer) {
                    viewer->setTargetVisual(item.visual.get());
                  }
                }
                const ImVec2 itemMin = ImGui::GetItemRectMin();
                const ImVec2 textPos(itemMin.x + ImGui::GetStyle().FramePadding.x,
                  itemMin.y + ImGui::GetStyle().FramePadding.y);
                const std::string nameText =
                  item.name.empty() ? ("tag " + std::to_string(item.tag)) : item.name;
                const char* sep = ": ";
                const ImVec2 typeSize = ImGui::CalcTextSize(typeName);
                const ImVec2 sepSize = ImGui::CalcTextSize(sep);
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                drawList->AddText(
                  ImGui::GetFont(), ImGui::GetFontSize(), textPos, typeColor, typeName);
                drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
                  ImVec2(textPos.x + typeSize.x, textPos.y), ImGui::GetColorU32(ImGuiCol_Text),
                  sep);
                drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
                  ImVec2(textPos.x + typeSize.x + sepSize.x, textPos.y),
                  ImGui::GetColorU32(ImGuiCol_Text), nameText.c_str());
              }
              ImGui::EndChild();
            }
          }
          ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
      }
    }
    ImGui::End();
    ImGui::PopStyleVar();

    const raisin::Visuals* selectedVisual = viewer->getTargetVisual();
    const VisualEntry* selectedEntry = nullptr;
    uint32_t selectedTag = 0;
    int selectedIndex = 0;
    scene.getVisualInfo(selectedVisual, selectedTag, selectedIndex, selectedEntry);

    if (selectedEntry && isContactEntry(selectedEntry)) {
      selectedEntry = nullptr;
    }

    if (selectedEntry) {
      const ImVec2 detailsBasePos(displaySize.x - detailSize.x - 12.0f, 12.0f + menuBarHeight);
      ImVec2 detailsPos = detailsBasePos;
      detailsPos.x += detailOffset.x;
      detailsPos.y += detailOffset.y;
      ImGui::SetNextWindowBgAlpha(0.5f);
      ImGui::SetNextWindowPos(detailsPos, ImGuiCond_Always);
      ImGui::SetNextWindowSizeConstraints(detailSize, ImVec2(FLT_MAX, FLT_MAX));
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
      const ImGuiWindowFlags detailFlags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNavFocus;
      ImVec2 detailWindowSize(0.0f, 0.0f);
      if (ImGui::Begin("Selected Object##Overlay", nullptr, detailFlags)) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.35f, 0.8f, 1.0f, 1.0f));
        ImGui::Selectable("Selected Object", false, ImGuiSelectableFlags_SpanAllColumns);
        ImGui::PopStyleColor();
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
          ImVec2 delta = ImGui::GetIO().MouseDelta;
          detailOffset.x = std::min(0.0f, detailOffset.x + delta.x);
          detailOffset.y = std::max(0.0f, detailOffset.y + delta.y);
        }
        ImGui::Separator();
        const ImVec4 tagColor(0.95f, 0.72f, 0.2f, 1.0f);
        const ImVec4 indexColor(0.25f, 0.85f, 0.7f, 1.0f);
        const ImVec4 shapeColor(0.35f, 0.6f, 1.0f, 1.0f);
        const ImVec4 nameColor(0.95f, 0.95f, 0.95f, 1.0f);
        const ImVec4 metaColor(0.85f, 0.85f, 0.85f, 1.0f);
        const SelectedObjectInfo& selectedInfo = scene.getSelectedInfo();

        std::string objectName = selectedEntry->objectName;
        if (objectName.empty()) {
          objectName = scene.getObjectName(selectedTag);
        }
        if (objectName.empty()) {
          objectName = "unnamed";
        }

        if (ImGui::BeginTable("##selected_props", 2, ImGuiTableFlags_SizingFixedFit)) {
          ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed);
          ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted("Name");
          ImGui::TableSetColumnIndex(1);
          ImGui::TextColored(nameColor, "%s", objectName.c_str());

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted("Tag");
          ImGui::TableSetColumnIndex(1);
          ImGui::TextColored(tagColor, "%u", selectedTag);

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted("Index");
          ImGui::TableSetColumnIndex(1);
          ImGui::TextColored(indexColor, "%d", selectedIndex);

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted("Type");
          ImGui::TableSetColumnIndex(1);
          ImGui::TextColored(shapeColor, "%s", objectTypeLabel(selectedEntry->objectTypeRaw));

          if (!selectedEntry->meshFile.empty()) {
            std::string meshLabel = selectedEntry->meshFile;
            const size_t slashPos = meshLabel.find_last_of("/\\");
            if (slashPos != std::string::npos && slashPos + 1 < meshLabel.size()) {
              meshLabel = meshLabel.substr(slashPos + 1);
            }
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Mesh");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(metaColor, "%s", meshLabel.c_str());
          }

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted("Articulated");
          ImGui::TableSetColumnIndex(1);
          ImGui::TextColored(metaColor, "%s", selectedEntry->isArticulated ? "yes" : "no");

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted("Collision");
          ImGui::TableSetColumnIndex(1);
          ImGui::TextColored(metaColor, "%s", selectedEntry->isCollision ? "yes" : "no");

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted("Pos");
          ImGui::TableSetColumnIndex(1);
          ImGui::TextColored(metaColor, "%.3f %.3f %.3f", selectedEntry->lastPos.x,
            selectedEntry->lastPos.y, selectedEntry->lastPos.z);

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted("Quat");
          ImGui::TableSetColumnIndex(1);
          ImGui::TextColored(metaColor, "%.3f %.3f %.3f %.3f", selectedEntry->lastQuat.w,
            selectedEntry->lastQuat.x, selectedEntry->lastQuat.y, selectedEntry->lastQuat.z);

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted("Size");
          ImGui::TableSetColumnIndex(1);
          ImGui::TextColored(metaColor, "%.3f %.3f %.3f", selectedEntry->lastSize.x,
            selectedEntry->lastSize.y, selectedEntry->lastSize.z);

          ImGui::EndTable();
        }

        if (selectedEntry->isArticulated) {
          ImGui::SeparatorText("Joints");
          if (selectedInfo.valid && selectedInfo.isArticulated && selectedInfo.tag == selectedTag &&
              !selectedInfo.jointNames.empty()) {
            if (ImGui::BeginTable("##selected_joints", 2,
                  ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                    ImGuiTableFlags_SizingFixedFit)) {
              ImGui::TableSetupColumn("Joint", ImGuiTableColumnFlags_WidthStretch);
              ImGui::TableSetupColumn("Angle", ImGuiTableColumnFlags_WidthFixed);
              ImGui::TableHeadersRow();
              for (size_t i = 0; i < selectedInfo.jointNames.size(); ++i) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(selectedInfo.jointNames[i].c_str());
                ImGui::TableSetColumnIndex(1);
                const float angle =
                  (i < selectedInfo.jointAngles.size()) ? selectedInfo.jointAngles[i] : 0.0f;
                ImGui::TextColored(ImVec4(0.85f, 0.9f, 1.0f, 1.0f), "%.4f", angle);
              }
              ImGui::EndTable();
            }
          } else {
            ImGui::TextDisabled("Joint data not available");
          }
        }
        detailWindowSize = ImGui::GetWindowSize();
      }
      ImGui::End();
      ImGui::PopStyleVar();
      detailSize.x = std::max(detailSize.x, detailWindowSize.x);
      detailSize.y = std::max(detailSize.y, detailWindowSize.y);
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window);
  }

  client.disconnect();
  scene.shutdown();
  viewer.reset();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(context);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
