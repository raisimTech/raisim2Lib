// Copyright (c) 2025 Raion Robotics Inc.
// All rights reserved.

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <SDL.h>

#include <glbinding/glbinding.h>
#include <glbinding/gl/gl.h>
#include <imgui/imgui.h>

#include "imgui/backend/imgui_impl_opengl3.h"
#include "imgui/backend/imgui_impl_sdl2.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <cerrno>
#include <chrono>
#include <clocale>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <locale>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "stb/stb_image.h"

#include "TcpViewerDiscovery.hpp"
#include "TcpViewerPaneLayout.hpp"
#include "TcpViewerPaneChrome.hpp"
#include "TcpViewerScreenshot.hpp"
#include "TcpViewerSession.hpp"
#include "TcpViewerActuators.hpp"
#include "TcpViewerSensors.hpp"
#include "TcpViewerSettings.hpp"
#include "TcpViewerSceneFiles.hpp"
#include "TcpViewerSignals.hpp"
#include "TcpViewerVideo.hpp"
#include "TcpViewerSimulation.hpp"
#include "TcpViewerConnection.hpp"

#include "rayrai/RayraiWindow.hpp"
#include "rayrai/TextureBindingCache.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/OpenGLMesh.hpp"
#include "rayrai/CoordinateFrame.hpp"
#include "rayrai/RaisimTcpCommon.hpp"
#include "rayrai/sdl_hints.hpp"
#include "rayrai/raisin_imgui_style.h"
#include "raisim/configure.hpp"
#include "raisim/sensors/Sensors.hpp"
#include "raisim/World.hpp"
#include "raisim/object/ArticulatedSystem/ArticulatedSystem.hpp"
#include "raisim/object/ArticulatedSystem/JointAndBodies.hpp"

#if defined(__linux__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <fcntl.h>
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
constexpr int kManualConnectTimeoutMs = 2000;
constexpr int kAutoConnectTimeoutMs = 100;
constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
constexpr auto kAutoConnectInterval = std::chrono::seconds(3);
constexpr auto kLoopbackAutoConnectInterval = std::chrono::milliseconds(250);
// How long an update request may go unanswered before the connection is treated
// as dead. Generous next to the 60 Hz request rate, short enough that a server
// which will never answer does not hold the pane hostage.
constexpr auto kServerResponseTimeout = std::chrono::milliseconds(5000);
// Server discovery re-lists itself on this interval, so there is no rescan button.
constexpr auto kDiscoveryRefreshInterval = std::chrono::seconds(2);
constexpr auto kOverlayAutoCollapseDelay = std::chrono::milliseconds(3500);
constexpr auto kSettingsSaveDebounce = std::chrono::milliseconds(750);
constexpr int kTransferRateGraphBuckets = 60;
constexpr double kTransferRateGraphWindowSeconds = 30.0;
constexpr double kDiagnosticsPresentationRateHz = 5.0;
constexpr double kDiagnosticsPresentationIntervalSeconds =
  1.0 / kDiagnosticsPresentationRateHz;
constexpr float kBaseFontSize = 24.0f;
constexpr float kFontScale = 0.75f;
constexpr float kDefaultFontRasterizerDensity = 1.75f;
constexpr float kUiScaleEpsilon = 0.01f;
constexpr float kCollapsedLogoSizeInFontHeights = 2.75f;
// The collapsed logo's wordmark is dark ink on a light chip, so that chip must
// stay opaque: any translucency lets the 3D scene show through the lettering.
constexpr float kCollapsedLogoBackdropAlpha = 1.0f;
constexpr const char* kRobotoFontRelativePath = "rsc/fonts/roboto/Roboto-Medium.ttf";
constexpr float kDefaultMouseForceAccelPerPixel = 0.10f;
constexpr float kMinMouseForceAccelPerPixel = 0.01f;
constexpr float kMaxMouseForceAccelPerPixel = 5.0f;
// Interaction-wire spring constant, in newtons per metre per kilogram of the
// grabbed body: RaisimServer multiplies it by the body mass, so the resulting
// acceleration is mass-independent and the same value works for a marble and a
// quadruped. 60 pulls firmly without fighting the contact solver.
constexpr float kDefaultWireDragStiffness = 60.0f;
constexpr float kMinWireDragStiffness = 1.0f;
constexpr float kMaxWireDragStiffness = 600.0f;
#if defined(__APPLE__)
constexpr const char* kWireDragGestureLabel = "Cmd-drag wire";
#else
constexpr const char* kWireDragGestureLabel = "Ctrl-drag wire";
#endif

// Sentinel for "no queued request" in the gestures' pendingRequestIndex.
constexpr size_t kNoPendingRequest = std::numeric_limits<size_t>::max();

float resolveUiScaleForDisplay(float configuredScale, float automaticScale,
                               bool userSet, bool initialized,
                               bool displaySizeChanged) {
  if (!userSet && (!initialized || displaySizeChanged)) {
    return automaticScale;
  }
  return configuredScale;
}

using raisin::tcp_viewer::BufferReader;
using raisin::tcp_viewer::ConnectionEntry;
using raisin::tcp_viewer::DiscoveredServer;
using raisin::tcp_viewer::DiscoveryBeaconReceiver;
using raisin::tcp_viewer::ObjectListItem;
using raisin::tcp_viewer::PendingSensorUpdate;
using raisin::tcp_viewer::RecordedFrame;
using raisin::tcp_viewer::RemoteScene;
using raisin::tcp_viewer::SelectedObjectInfo;
using raisin::tcp_viewer::SessionRecorder;
using raisin::tcp_viewer::SensorInfo;
using raisin::tcp_viewer::SensorPreviewInfo;
using raisin::tcp_viewer::SensorRenderer;
using raisin::tcp_viewer::ViewerSettings;
using raisin::tcp_viewer::cloudQualityName;
using raisin::tcp_viewer::consumeTcpUpdateSlot;
using raisin::tcp_viewer::colorModeName;
using raisin::tcp_viewer::formatConnectionLabel;
using raisin::tcp_viewer::formatEndpointHost;
using raisin::tcp_viewer::findSessionFrameAtOrAfter;
using raisin::tcp_viewer::highFidelityPbrAllowedForQuality;
using raisin::tcp_viewer::kSessionMagic;
using raisin::tcp_viewer::kTcpUpdateRateDefaultHz;
using raisin::tcp_viewer::kTcpUpdateRateMaxHz;
using raisin::tcp_viewer::kTcpUpdateRateMinHz;
using raisin::tcp_viewer::isCompatibleDiscoveryVersion;
using raisin::tcp_viewer::loadViewerSettings;
using raisin::tcp_viewer::loadSessionFile;
using raisin::tcp_viewer::normalizeConnectionEndpoint;
using raisin::tcp_viewer::parseConnectionLabel;
using raisin::tcp_viewer::parseDiscoveryBeacon;
using raisin::tcp_viewer::parsePortStrict;
using raisin::tcp_viewer::qualityName;
using raisin::tcp_viewer::recordConnection;
using raisin::tcp_viewer::recordResourceDir;
using raisin::tcp_viewer::sanitizeViewerSettings;
using raisin::tcp_viewer::captureViewerRgba;
using raisin::tcp_viewer::saveRgbaPng;
using raisin::tcp_viewer::saveViewerTexturePng;
using raisin::tcp_viewer::saveViewerSettings;
using raisin::tcp_viewer::sendSensorUpdate;
using raisin::tcp_viewer::sendUpdateRequest;
using raisin::tcp_viewer::sanitizeTcpUpdateRateHz;
using raisin::tcp_viewer::TcpClient;
using raisin::tcp_viewer::tcpUpdatePeriodForHz;
using raisin::tcp_viewer::timestampedCapturePath;
using raisin::tcp_viewer::VisualEntry;
using raisin::tcp_viewer::weatherDefaultEnabledForQuality;

struct ServerEntry {
  ConnectionEntry endpoint;
  std::string bindHost;
  std::string process;
  std::string protocol;
  std::unordered_map<std::string, std::string> metadata;
  bool remoteBeacon = false;
  std::chrono::steady_clock::time_point lastSeen;
};

struct ViewerViewportState {
  ImVec2 origin{0.0f, 0.0f};
  ImVec2 size{0.0f, 0.0f};
  bool hovered = false;
  int cursorX = 0;
  int cursorY = 0;
};

struct MouseForceGesture {
  bool active = false;
  uint32_t tag = 0;
  int index = 0;
  int localBodyIdx = 0;
  glm::vec3 applicationPoint{0.0f};
  glm::vec3 localApplicationPoint{0.0f};
  glm::vec3 force{0.0f};
  ImVec2 pressMouse{0.0f, 0.0f};
  ImVec2 currentMouse{0.0f, 0.0f};
  size_t pendingRequestIndex = kNoPendingRequest;
};

// Interaction wire (CR_ATTACH_WIRE + CR_DRAG_OBJECT). Unlike the pose grabber,
// which teleports a body with CR_SET_POSE, this pulls it with a mass-scaled
// spring, so constraints and contacts stay satisfied while it moves. The server
// clears `wireStiffness_` on any request frame that omits CR_DRAG_OBJECT, so a
// live drag has to resend the target every update; releasing simply stops
// sending and the wire goes slack on the next frame.
struct WireDragGesture {
  bool active = false;
  bool attachQueued = false;
  uint32_t tag = 0;
  int index = 0;
  int localBodyIdx = 0;
  glm::vec3 localAttachPoint{0.0f};  // grab point in the body frame
  glm::vec3 attachPoint{0.0f};       // grab point in world coordinates, this frame
  glm::vec3 target{0.0f};            // where the wire pulls, in world coordinates
  size_t pendingRequestIndex = kNoPendingRequest;
};

// One recorded object in the signal workbench. The weak visual pointer is how a
// pinned object is re-found each frame without rescanning the scene, and it goes
// empty by itself when the object is removed server-side.
struct SignalObjectRecord {
  uint32_t tag = 0;
  int index = 0;
  std::string label;
  std::weak_ptr<raisin::Visuals> visual;
  raisin::tcp_viewer::SignalHistory history;
};

struct RulerToolState {
  bool enabled = false;
  bool hasA = false;
  bool hasB = false;
  int nextPoint = 0;
  glm::vec3 a{0.0f};
  glm::vec3 b{0.0f};
  std::string aLabel;
  std::string bLabel;
};

// Pose gizmo: translate handles (3 colored axis lines) plus rotate handles
// (3 colored rings, each in a plane perpendicular to one world axis). While
// the gizmo is enabled, the selected body's visual pose is "held" — the
// server's pose stream is shadowed each frame so the body stays where the
// user put it. CR_SET_POSE is queued during drag to push the new pose to the
// server. Disabling the gizmo releases the hold and the server takes over again.
struct PoseGrabberGesture {
  enum class Mode { Translate, Rotate };
  bool enabled = false;
  bool dragging = false;
  Mode mode = Mode::Translate;
  int axis = -1;                 // 0=X, 1=Y, 2=Z while dragging
  uint32_t tag = 0;
  int index = 0;
  glm::vec3 anchorWorld{0.0f};   // body origin captured on drag start
  // Quaternion vec4s here are wxyz (x holds w), like VisualEntry::lastQuat,
  // ClientRequest::quat and Visuals::setOrientation.
  glm::vec4 anchorQuat{1.0f, 0.0f, 0.0f, 0.0f};
  ImVec2 anchorMouse{0.0f, 0.0f};
  float anchorScreenAngle = 0.0f;                  // mouse angle around body center at start
  glm::vec3 currentTarget{0.0f};                   // proposed position during drag
  glm::vec4 currentQuat{1.0f, 0.0f, 0.0f, 0.0f};   // proposed orientation during drag

  // Held pose: while `heldActive` is true the body identified by `heldTag` has
  // its visual pose forced to (heldPos, heldQuat) each frame. Survives across
  // frames so the body stays put between drags. `heldDirty` flags whether the
  // user has actually changed the pose during this hold; we only emit a single
  // CR_SET_POSE to the server on deactivation when this flag is set.
  bool heldActive = false;
  bool heldDirty = false;
  uint32_t heldTag = 0;
  glm::vec3 heldPos{0.0f};
  glm::vec4 heldQuat{1.0f, 0.0f, 0.0f, 0.0f};
};

// 3-point angle measurement. Picks three points; the angle is measured at the
// middle (B) between rays BA and BC. UI parallel to RulerToolState.
struct AngleToolState {
  bool enabled = false;
  int picked = 0;            // 0, 1, 2, or 3 (= complete)
  glm::vec3 a{0.0f};         // first arm endpoint
  glm::vec3 b{0.0f};         // vertex
  glm::vec3 c{0.0f};         // second arm endpoint
  std::string aLabel;
  std::string bLabel;
  std::string cLabel;
};

struct ProgramOptions {
  std::filesystem::path simulationPath;
  std::filesystem::path activationKey;
  bool noSaveSettings = false;
  std::string host = "127.0.0.1";
  int port = kDefaultPort;
  // Whether --host / --port / --connect named an endpoint. A saved pane
  // placement is only overridden when the command line actually asked for one.
  bool endpointSet = false;
  bool autoConnect = true;
  bool autoConnectSet = false;
  bool minimizePanels = false;
  bool minimizePanelsSet = false;
  bool keepOverlayOpen = false;  // --keep-overlay-open: never auto-collapse the left panel
  bool autoFrame = false;
  bool autoFrameSet = false;
  bool fullscreen = false;
  int windowWidth = 1280;
  int windowHeight = 720;
  bool hasCameraLookAt = false;
  bool forceCameraLookAt = false;
  glm::vec3 cameraPos{0.0f};
  glm::vec3 cameraTarget{0.0f};
  bool hasTargetOffset = false;
  glm::vec3 targetOffset{0.0f};
  std::vector<std::string> resourceDirs;
  std::filesystem::path screenshotPath;
  std::filesystem::path screenshotDir = std::filesystem::current_path();
  std::filesystem::path recordSessionPath;
  std::filesystem::path replaySessionPath;
  std::filesystem::path exportScenePath;
  std::filesystem::path trajectoryCsvPath;
  std::filesystem::path endpointListPath;
  std::filesystem::path inspectorPath; // --inspect FILE: load URDF/MJCF as if drag-dropped
  // --inspect-after-frames N: defer the --inspect load until the main loop has
  // ticked N frames. Lets the headless harness measure drag-drop latency after
  // the background shader warmup has had time to run.
  int inspectAfterFrames = -1;
  bool preWarmShaders = true;          // --no-pre-warm to skip the targeted shader pre-compile
  // --warm-at-startup: drive one content render at startup with a transient AS to
  // warm the renderer's non-shader lazy init (IBL convolution / FBOs / texture pools).
  // Costs ~13 s at startup; makes every drag-drop after that ~30 ms. Default off so
  // empty-viewer launches stay fast.
  bool warmAtStartup = false;
  // --inspect-close-after-frames N: drive load → render N frames → close → render N → exit.
  // Headless reproducer for "close inspector segfaults" bug reports.
  int inspectCloseAfterFrames = -1;
  // --inspect-reload PATH: after the close phase, also reload this second path so we
  // can measure first-frame shader-compile cost across the two loads (proves that the
  // World/Renderer + compiled shaders are reused between drops).
  std::filesystem::path inspectReloadPath;
  bool exitAfterScreenshot = false;
  bool replayLoop = false;
  float replaySpeed = 1.0f;
  float updateRateHz = -1.0f;
  double waitForServerSeconds = 0.0;
  double exitAfterSeconds = 0.0;
  bool printHelp = false;
};

struct CameraBookmark {
  bool valid = false;
  glm::vec3 position{0.0f};
  glm::vec3 target{0.0f};
};

// Local "AS inspector" mode: activated when the user drags a URDF/MJCF onto the viewer
// while there is no TCP connection. The robot is loaded into the viewer's own
// raisim::World and posed with the joint sliders — no physics, no server, just kinematics.
struct InspectorJoint {
  std::string name;
  raisim::Joint::Type type = raisim::Joint::Type::FIXED;
  int gcOffset = 0;       // start index into the ArticulatedSystem GC vector
  int gcDim = 0;          // 0 (fixed) / 1 (rev/prismatic) / 4 (spherical) / 7 (floating)
  double minLimit = 0.0;  // for revolute/prismatic
  double maxLimit = 0.0;
  bool hasLimits = false;
};

struct InspectorState {
  bool active = false;
  std::string sourceFile;
  std::string lastError;
  raisim::ArticulatedSystem* as = nullptr;   // owned by the local raisim::World
  // Other raisim objects added by an MJCF load (ground plane, extra bodies, etc.) that
  // need to be removed when we close the inspector. Does not include `as`.
  std::vector<raisim::Object*> sideObjects;
  std::vector<double> gc;                    // mirror of the current generalized coordinates
  std::vector<InspectorJoint> joints;
  // True while the user is actively dragging a slider; suppresses auto-connect attempts.
  bool dragging = false;
};

// Identify MJCF vs URDF by scanning the file for the first non-comment, non-declaration
// root element. The 1 KB short-read used to misclassify files like
// half_cheetah.xml whose <mujoco> tag lives after a long license/header comment
// (the comment + whitespace pushed it past byte 1024). We now read up to 64 KB —
// big enough for any realistic preamble — and skip past <?xml...?> declarations and
// <!-- ... --> comments before sniffing the root tag.
inline bool looksLikeMjcf(const std::string& path) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) return false;
  std::string buf(65536, '\0');
  fs.read(buf.data(), static_cast<std::streamsize>(buf.size()));
  buf.resize(static_cast<size_t>(fs.gcount()));
  size_t i = 0;
  const size_t n = buf.size();
  while (i < n) {
    while (i < n && std::isspace(static_cast<unsigned char>(buf[i]))) ++i;
    if (i + 1 < n && buf[i] == '<' && buf[i + 1] == '?') {
      const auto end = buf.find("?>", i + 2);
      if (end == std::string::npos) return false;
      i = end + 2;
      continue;
    }
    if (i + 3 < n && buf.compare(i, 4, "<!--") == 0) {
      const auto end = buf.find("-->", i + 4);
      if (end == std::string::npos) return false;
      i = end + 3;
      continue;
    }
    if (i + 8 < n && buf.compare(i, 9, "<!DOCTYPE") == 0) {
      const auto end = buf.find('>', i + 9);
      if (end == std::string::npos) return false;
      i = end + 1;
      continue;
    }
    if (i < n && buf[i] == '<') {
      return buf.compare(i, 7, "<mujoco") == 0;
    }
    ++i;
  }
  return false;
}

struct MotionEstimate {
  bool hasPrevious = false;
  bool valid = false;
  double previousTime = 0.0;
  glm::vec3 previousPosition{0.0f};
  glm::vec4 previousQuat{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec3 linearVelocity{0.0f};
  float angularSpeed = 0.0f;
};

struct ViewerStats {
  int frames = 0;
  int updates = 0;
  int parseErrors = 0;
  int reconnects = 0;
  int lastPayloadBytes = 0;
  size_t bytes = 0;
  double fps = 0.0;
  double updateHz = 0.0;
  double rxKbps = 0.0;
  double lastRoundTripMs = 0.0;
  int pendingSensorRequests = 0;
  size_t unresolvedAssets = 0;
  std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
};

struct PacketSample {
  double timeSeconds = 0.0;
  int bytes = 0;
  bool parsed = false;
  bool replay = false;
  int pendingSensors = 0;
  size_t objects = 0;
  size_t visuals = 0;
  size_t instanced = 0;
  size_t pointClouds = 0;
  size_t unresolvedAssets = 0;
  double roundTripMs = 0.0;
};

struct NetworkTimingSummary {
  double currentMs = 0.0;
  double averageMs = 0.0;
  double jitterMs = 0.0;
  double maximumMs = 0.0;
};

struct DiagnosticsPresentationState {
  bool initialized = false;
  double lastRefreshSeconds = 0.0;
  std::deque<PacketSample> packetSamples;
  std::array<float, kTransferRateGraphBuckets> transferRates{};
  NetworkTimingSummary timing;
  std::vector<float> roundTripTimes;
  int parseErrors = 0;
  double rxKbps = 0.0;
  size_t unresolvedAssets = 0;
};

NetworkTimingSummary summarizePacketTimings(const std::deque<PacketSample>& samples) {
  NetworkTimingSummary result;
  double sum = 0.0;
  double jitterSum = 0.0;
  double previous = 0.0;
  size_t count = 0;
  for (const auto& sample : samples) {
    if (sample.replay || !sample.parsed || sample.roundTripMs <= 0.0) continue;
    result.currentMs = sample.roundTripMs;
    result.maximumMs = std::max(result.maximumMs, sample.roundTripMs);
    sum += sample.roundTripMs;
    if (count > 0) jitterSum += std::abs(sample.roundTripMs - previous);
    previous = sample.roundTripMs;
    ++count;
  }
  if (count > 0) result.averageMs = sum / static_cast<double>(count);
  if (count > 1) result.jitterMs = jitterSum / static_cast<double>(count - 1);
  return result;
}

/** Stand-in channel list for an object with no recorded history yet. */
const std::vector<raisin::tcp_viewer::SignalChannelDesc> kEmptySignalChannels;

struct AssetDiagnostic {
  uint32_t tag = 0;
  int index = 0;
  std::string name;
  std::string meshFile;
  std::string meshPath;
  std::string resourceDir;
  bool resolved = false;
};

std::string toLowerAscii(std::string value);
std::string trimAscii(const std::string& value);

std::string shortenPlainLabel(const std::string& value, size_t maxLen) {
  if (value.size() <= maxLen) {
    return value;
  }
  if (maxLen <= 3) {
    return value.substr(0, maxLen);
  }
  return value.substr(0, maxLen - 3) + "...";
}

bool isPathSeparator(char c) {
  return c == '/' || c == '\\';
}

std::vector<std::string> splitPathComponents(const std::string& value) {
  std::vector<std::string> components;
  size_t pos = 0;
  while (pos < value.size()) {
    while (pos < value.size() && isPathSeparator(value[pos])) {
      ++pos;
    }
    const size_t start = pos;
    while (pos < value.size() && !isPathSeparator(value[pos])) {
      ++pos;
    }
    if (start < pos) {
      components.emplace_back(value.substr(start, pos - start));
    }
  }
  return components;
}

std::string joinPathComponentsTail(const std::vector<std::string>& components, size_t tailCount) {
  const size_t first = components.size() - tailCount;
  std::string label;
  for (size_t i = first; i < components.size(); ++i) {
    if (!label.empty()) {
      label += '/';
    }
    label += components[i];
  }
  return label;
}

std::string shortenPurePathLabel(const std::string& value, size_t maxLen) {
  constexpr size_t kMaxVisibleParentDirs = 3;
  const auto components = splitPathComponents(value);
  if (components.empty()) {
    return shortenPlainLabel(value, maxLen);
  }

  const size_t maxTailComponents = kMaxVisibleParentDirs + 1;
  size_t tailCount = std::min(maxTailComponents, components.size());
  const bool hasHiddenParents = components.size() > tailCount;
  auto makeLabel = [&](size_t count) {
    std::string tail = joinPathComponentsTail(components, count);
    if (hasHiddenParents || count < components.size()) {
      return std::string(".../") + tail;
    }
    return value;
  };

  std::string label = makeLabel(tailCount);
  while (label.size() > maxLen && tailCount > 1) {
    --tailCount;
    label = makeLabel(tailCount);
  }
  return shortenPlainLabel(label, maxLen);
}

std::string shortenPathLabel(const std::string& value, size_t maxLen) {
  const size_t firstSep = value.find_first_of("/\\");
  if (firstSep == std::string::npos) {
    return shortenPlainLabel(value, maxLen);
  }

  const bool startsWithPath = firstSep == 0 ||
                              (firstSep == 2 && value.size() > 2 && value[1] == ':') ||
                              value.find_first_of(" \t", 0) > firstSep;
  if (startsWithPath) {
    return shortenPurePathLabel(value, maxLen);
  }

  if (firstSep > 0 && std::isspace(static_cast<unsigned char>(value[firstSep - 1]))) {
    const std::string prefix = value.substr(0, firstSep);
    const size_t pathMaxLen = maxLen > prefix.size() ? maxLen - prefix.size() : 0;
    return prefix + shortenPurePathLabel(value.substr(firstSep), pathMaxLen);
  }

  return shortenPlainLabel(value, maxLen);
}

// Shared body of the strict number parsers: reject null/blank input, a partial
// parse, ERANGE, non-finite floating-point results and trailing garbage.
// `parse` wraps a single strto* call.
template <typename T, typename Parse>
bool parseNumberStrict(const char* value, T& out, Parse parse) {
  if (!value) {
    return false;
  }
  while (std::isspace(static_cast<unsigned char>(*value))) {
    ++value;
  }
  if (*value == '\0') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  out = parse(value, &end);
  if (end == value || errno == ERANGE) {
    return false;
  }
  if constexpr (std::is_floating_point_v<T>) {
    if (!std::isfinite(out)) {
      return false;
    }
  }
  while (std::isspace(static_cast<unsigned char>(*end))) {
    ++end;
  }
  return *end == '\0';
}

bool parseLongStrict(const char* value, int base, long& out) {
  return parseNumberStrict(value, out,
    [base](const char* text, char** end) { return std::strtol(text, end, base); });
}

bool parseFloatStrict(const char* value, float& out) {
  return parseNumberStrict(value, out,
    [](const char* text, char** end) { return std::strtof(text, end); });
}

bool parseDoubleStrict(const char* value, double& out) {
  return parseNumberStrict(value, out,
    [](const char* text, char** end) { return std::strtod(text, end); });
}

bool parseFloatListStrict(const char* value, float* values, size_t count) {
  if (!value || !values || count == 0) {
    return false;
  }

  const char* cursor = value;
  for (size_t i = 0; i < count; ++i) {
    while (std::isspace(static_cast<unsigned char>(*cursor))) {
      ++cursor;
    }
    if (*cursor == '\0') {
      return false;
    }

    errno = 0;
    char* end = nullptr;
    values[i] = std::strtof(cursor, &end);
    if (end == cursor || errno == ERANGE || !std::isfinite(values[i])) {
      return false;
    }
    cursor = end;

    if (i + 1 < count) {
      bool sawSeparator = false;
      while (std::isspace(static_cast<unsigned char>(*cursor))) {
        sawSeparator = true;
        ++cursor;
      }
      if (*cursor == ',' || *cursor == ';') {
        sawSeparator = true;
        ++cursor;
        while (std::isspace(static_cast<unsigned char>(*cursor))) {
          ++cursor;
        }
      }
      if (!sawSeparator) {
        return false;
      }
    }
  }

  while (std::isspace(static_cast<unsigned char>(*cursor))) {
    ++cursor;
  }
  return *cursor == '\0';
}

bool parseWindowSize(const std::string& value, int& width, int& height) {
  const auto sep = value.find_first_of("xX,");
  if (sep == std::string::npos) {
    return false;
  }
  long parsedW = 0;
  long parsedH = 0;
  if (!parseLongStrict(value.substr(0, sep).c_str(), 10, parsedW) ||
      !parseLongStrict(value.substr(sep + 1).c_str(), 10, parsedH)) {
    return false;
  }
  if (parsedW < 320 || parsedH < 240 || parsedW > 16384 || parsedH > 16384) {
    return false;
  }
  width = static_cast<int>(parsedW);
  height = static_cast<int>(parsedH);
  return true;
}

bool parseVec3Text(const std::string& value, glm::vec3& out) {
  float values[3]{};
  if (!parseFloatListStrict(value.c_str(), values, 3)) {
    return false;
  }
  out = glm::vec3(values[0], values[1], values[2]);
  return true;
}

bool parseCameraLookAtText(const std::string& value, glm::vec3& pos, glm::vec3& target) {
  float values[6]{};
  if (!parseFloatListStrict(value.c_str(), values, 6)) {
    return false;
  }
  pos = glm::vec3(values[0], values[1], values[2]);
  target = glm::vec3(values[3], values[4], values[5]);
  return glm::length(target - pos) > 1e-4f;
}

void printUsage(const char* argv0) {
  std::cout
    << "Usage: " << (argv0 ? argv0 : "rayrai_tcp_viewer") << " [options]\n\n"
    << "Options:\n"
    << "  --simulate FILE             Launch a RaiSim world XML in a child simulation process\n"
    << "  --activation-key FILE       Use this RaiSim activation key (also for local simulation)\n"
    << "  --no-save-settings          Keep this session's preferences temporary\n"
    << "  --host HOST                 Server host (default: 127.0.0.1)\n"
    << "  --port PORT                 Server port (default: 8080)\n"
    << "  --connect HOST:PORT         Server endpoint shortcut; use [IPv6]:PORT for IPv6\n"
    << "  --auto-connect              Enable automatic reconnect attempts\n"
    << "  --no-auto-connect           Disable automatic reconnect attempts\n"
    << "  --no-pre-warm               Skip shader pre-warm (faster startup, slower first content)\n"
    << "  --warm-at-startup           Also warm renderer content-frame init (~13s) so the first\n"
    << "                              drag-drop is instant. Off by default; empty-viewer startup\n"
    << "                              stays fast unless this flag is passed.\n"
    << "  --resource-dir PATH         Add a mesh/resource search directory; repeatable\n"
    << "  --window-size WxH           Initial window size, e.g. 1600x900\n"
    << "  --fullscreen                Start fullscreen desktop\n"
    << "  --minimize-panels           Start with overlay panels minimized\n"
    << "  --keep-overlay-open         Disable auto-collapse of the left overlay (doc screenshots)\n"
    << "  --auto-frame                Frame the scene after the first valid update\n"
    << "  --camera-lookat p,t         Six floats: px,py,pz,tx,ty,tz\n"
    << "  --camera-offset x,y,z       Follow target from this offset\n"
    << "  --force-camera-lookat       Reapply --camera-lookat every frame\n"
    << "  --screenshot PATH           Save one PNG after the first rendered frame and exit\n"
    << "  --screenshot-dir PATH       Directory used by F12 and PNG sequence recording\n"
    << "  --record-session PATH       Record raw TCP scene updates for offline replay\n"
    << "  --update-rate HZ            Target TCP scene update rate (15-120, default 60)\n"
    << "  --replay-session PATH       Replay a previously recorded TCP session\n"
    << "  --replay-speed SCALE        Replay speed multiplier (default: 1.0)\n"
    << "  --replay-loop               Loop replay sessions\n"
    << "  --export-scene PATH         Export current scene/object diagnostics as JSON\n"
    << "  --trajectory-csv PATH       Log object poses to CSV while updates arrive\n"
    << "  --server-list PATH          Load extra host:port endpoints from a text file\n"
    << "  --wait-for-server SECONDS   Batch wait limit for initial connection; exits with\n"
    << "                              status 1 if no server connected in time\n"
    << "  --exit-after SECONDS        Exit after the given wall-clock duration\n"
    << "  --help                      Show this help\n"
    << "\nDiscovery:\n"
    << "  The connection dropdown lists compatible LAN RaisimServer beacons only.\n";
}

bool parseProgramOptions(int argc, char** argv, ProgramOptions& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    const auto requireValue = [&](const char* name) -> const char* {
      if (i + 1 >= argc || !argv[i + 1]) {
        std::cerr << "ERROR: " << name << " requires a value\n";
        return nullptr;
      }
      return argv[++i];
    };
    // Options whose value is stored verbatim as a path.
    const std::pair<const char*, std::filesystem::path*> pathOptions[] = {
      {"--screenshot-dir", &options.screenshotDir},
      {"--record-session", &options.recordSessionPath},
      {"--export-scene", &options.exportScenePath},
      {"--trajectory-csv", &options.trajectoryCsvPath},
      {"--server-list", &options.endpointListPath},
      {"--inspect-reload", &options.inspectReloadPath}};
    std::filesystem::path* pathOption = nullptr;
    for (const auto& [name, target] : pathOptions) {
      if (arg == name) pathOption = target;
    }

    if (pathOption) {
      const char* value = requireValue(arg.c_str());
      if (!value) return false;
      *pathOption = value;
    } else if (arg == "--help" || arg == "-h") {
      options.printHelp = true;
      return true;
    } else if (arg == "--no-save-settings") {
      options.noSaveSettings = true;
    } else if (arg == "--simulate" || arg == "--activation-key") {
      const char* value = requireValue(arg.c_str());
      if (!value || !*value) return false;
      if (arg == "--simulate") options.simulationPath = value;
      else options.activationKey = value;
    } else if (arg == "--host") {
      const char* value = requireValue("--host");
      if (!value) return false;
      options.host = value;
      options.endpointSet = true;
    } else if (arg == "--port") {
      const char* value = requireValue("--port");
      if (!value) return false;
      if (!parsePortStrict(value, options.port)) {
        std::cerr << "ERROR: invalid --port value: " << value << "\n";
        return false;
      }
      options.endpointSet = true;
    } else if (arg == "--connect") {
      const char* value = requireValue("--connect");
      if (!value) return false;
      ConnectionEntry entry;
      if (!parseConnectionLabel(value, entry)) {
        std::cerr << "ERROR: invalid --connect value: " << value << "\n";
        return false;
      }
      options.host = entry.host;
      options.port = entry.port;
      options.endpointSet = true;
    } else if (arg == "--auto-connect") {
      options.autoConnect = true;
      options.autoConnectSet = true;
    } else if (arg == "--no-auto-connect") {
      options.autoConnect = false;
      options.autoConnectSet = true;
    } else if (arg == "--resource-dir") {
      const char* value = requireValue("--resource-dir");
      if (!value) return false;
      options.resourceDirs.emplace_back(value);
    } else if (arg == "--window-size") {
      const char* value = requireValue("--window-size");
      if (!value) return false;
      if (!parseWindowSize(value, options.windowWidth, options.windowHeight)) {
        std::cerr << "ERROR: invalid --window-size value: " << value << "\n";
        return false;
      }
    } else if (arg == "--fullscreen") {
      options.fullscreen = true;
    } else if (arg == "--minimize-panels") {
      options.minimizePanels = true;
      options.minimizePanelsSet = true;
    } else if (arg == "--keep-overlay-open") {
      options.keepOverlayOpen = true;
    } else if (arg == "--auto-frame") {
      options.autoFrame = true;
      options.autoFrameSet = true;
    } else if (arg == "--camera-lookat") {
      const char* value = requireValue("--camera-lookat");
      if (!value) return false;
      if (!parseCameraLookAtText(value, options.cameraPos, options.cameraTarget)) {
        std::cerr << "ERROR: invalid --camera-lookat value: " << value << "\n";
        return false;
      }
      options.hasCameraLookAt = true;
    } else if (arg == "--camera-offset") {
      const char* value = requireValue("--camera-offset");
      if (!value) return false;
      if (!parseVec3Text(value, options.targetOffset)) {
        std::cerr << "ERROR: invalid --camera-offset value: " << value << "\n";
        return false;
      }
      options.hasTargetOffset = true;
    } else if (arg == "--force-camera-lookat") {
      options.forceCameraLookAt = true;
    } else if (arg == "--screenshot") {
      const char* value = requireValue("--screenshot");
      if (!value) return false;
      options.screenshotPath = value;
      options.exitAfterScreenshot = true;
    } else if (arg == "--update-rate") {
      const char* value = requireValue("--update-rate");
      if (!value) return false;
      float rateHz = 0.0f;
      if (!parseFloatStrict(value, rateHz) || rateHz < kTcpUpdateRateMinHz ||
          rateHz > kTcpUpdateRateMaxHz) {
        std::cerr << "ERROR: invalid --update-rate value: " << value
                  << " (expected " << kTcpUpdateRateMinHz << ".."
                  << kTcpUpdateRateMaxHz << " Hz)\n";
        return false;
      }
      options.updateRateHz = rateHz;
    } else if (arg == "--replay-session") {
      const char* value = requireValue("--replay-session");
      if (!value) return false;
      options.replaySessionPath = value;
      options.autoConnect = false;
      options.autoConnectSet = true;
    } else if (arg == "--replay-speed") {
      const char* value = requireValue("--replay-speed");
      if (!value) return false;
      float speed = 0.0f;
      if (!parseFloatStrict(value, speed) || speed <= 0.0f || speed > 100.0f) {
        std::cerr << "ERROR: invalid --replay-speed value: " << value << "\n";
        return false;
      }
      options.replaySpeed = speed;
    } else if (arg == "--replay-loop") {
      options.replayLoop = true;
    } else if (arg == "--no-pre-warm") {
      options.preWarmShaders = false;
    } else if (arg == "--warm-at-startup") {
      options.warmAtStartup = true;
    } else if (arg == "--inspect") {
      const char* value = requireValue("--inspect");
      if (!value) return false;
      options.inspectorPath = value;
      options.autoConnect = false;
      options.autoConnectSet = true;
    } else if (arg == "--inspect-after-frames" || arg == "--inspect-close-after-frames") {
      const char* value = requireValue(arg.c_str());
      if (!value) return false;
      int n = 0;
      try { n = std::stoi(value); }
      catch (...) { std::cerr << "ERROR: invalid " << arg << "\n"; return false; }
      (arg == "--inspect-after-frames" ? options.inspectAfterFrames
                                       : options.inspectCloseAfterFrames) = std::max(0, n);
    } else if (arg == "--wait-for-server" || arg == "--exit-after") {
      double& seconds = arg == "--wait-for-server" ? options.waitForServerSeconds
                                                   : options.exitAfterSeconds;
      const char* value = requireValue(arg.c_str());
      if (!value) return false;
      if (!parseDoubleStrict(value, seconds) || seconds < 0.0) {
        std::cerr << "ERROR: invalid " << arg << " value: " << value << "\n";
        return false;
      }
    } else {
      std::cerr << "ERROR: unknown option: " << arg << "\n";
      return false;
    }
  }
  return true;
}

#include "TcpViewerUiHelpers.inl"

bool directoryExists(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_directory(path, ec);
}

std::filesystem::path absolutePathNoThrow(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::path absolute = std::filesystem::absolute(path, ec);
  if (ec) {
    return path;
  }
  return absolute.lexically_normal();
}

std::filesystem::path findTcpViewerIconDir(const std::filesystem::path& binaryDir) {
  std::vector<std::filesystem::path> candidates;
  const std::filesystem::path sourceDir = std::filesystem::path(__FILE__).parent_path();
  if (!sourceDir.empty()) {
    candidates.push_back(sourceDir / "../assets/flaticon/tcp_viewer");
  }
  if (!binaryDir.empty()) {
    candidates.push_back(binaryDir / "assets/flaticon/tcp_viewer");
    candidates.push_back(binaryDir / "../assets/flaticon/tcp_viewer");
    candidates.push_back(binaryDir / "../share/rayrai/assets/flaticon/tcp_viewer");
  }
  std::error_code ec;
  const std::filesystem::path cwd = std::filesystem::current_path(ec);
  if (!ec) {
    candidates.push_back(cwd / "visualizer/rayrai/assets/flaticon/tcp_viewer");
    candidates.push_back(cwd / "../visualizer/rayrai/assets/flaticon/tcp_viewer");
    candidates.push_back(cwd / "examples/src/rayrai/assets/flaticon/tcp_viewer");
    candidates.push_back(cwd / "../examples/src/rayrai/assets/flaticon/tcp_viewer");
  }
  for (const auto& candidate : candidates) {
    const std::filesystem::path absolute = absolutePathNoThrow(candidate);
    if (directoryExists(absolute)) {
      return absolute;
    }
  }
  return {};
}

std::filesystem::path findTcpViewerIconDir() {
  return findTcpViewerIconDir({});
}

std::filesystem::path findRaisimLogoPath(const std::filesystem::path& binaryDir) {
  std::vector<std::filesystem::path> candidates;
  const std::filesystem::path sourceDir = std::filesystem::path(__FILE__).parent_path();
  if (!sourceDir.empty()) {
    candidates.push_back(sourceDir / "../../../logo.png");
    candidates.push_back(sourceDir / "../../../../logo.png");
    candidates.push_back(sourceDir / "../../../../docs/logo.png");
    // raisim2Lib examples: synced next to the viewer icons.
    candidates.push_back(sourceDir / "../assets/logo.png");
  }
  if (!binaryDir.empty()) {
    candidates.push_back(binaryDir / "logo.png");
    candidates.push_back(binaryDir / "../logo.png");
    candidates.push_back(binaryDir / "../../logo.png");
    candidates.push_back(binaryDir / "../share/rayrai/logo.png");
  }
  std::error_code ec;
  const std::filesystem::path cwd = std::filesystem::current_path(ec);
  if (!ec) {
    candidates.push_back(cwd / "logo.png");
    candidates.push_back(cwd / "docs/logo.png");
    candidates.push_back(cwd / "../logo.png");
    candidates.push_back(cwd / "../docs/logo.png");
  }
  for (const auto& candidate : candidates) {
    const std::filesystem::path absolute = absolutePathNoThrow(candidate);
    if (std::filesystem::is_regular_file(absolute, ec) && !ec) {
      return absolute;
    }
    ec.clear();
  }
  return {};
}

// Decodes an image file to tightly packed RGBA8. Returns null (with nothing to
// free) when the file is unreadable or empty; otherwise the caller owns the
// pixels and must stbi_image_free() them.
unsigned char* loadRgba8Pixels(const std::filesystem::path& path, int& width, int& height) {
  int channels = 0;
  unsigned char* pixels = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
  if (!pixels || width <= 0 || height <= 0) {
    if (pixels) {
      stbi_image_free(pixels);
    }
    return nullptr;
  }
  return pixels;
}

// Uploads RGBA8 pixels as a linear-filtered, edge-clamped 2D texture. Returns 0
// when no texture name could be allocated. Leaves GL_TEXTURE_2D unbound and the
// unpack alignment as it found it.
unsigned int uploadRgba8Texture(const unsigned char* pixels, int width, int height) {
  unsigned int texture = 0;
  gl::glGenTextures(1, &texture);
  if (texture == 0) {
    return 0;
  }

  gl::GLint previousAlignment = 4;
  gl::glGetIntegerv(gl::GL_UNPACK_ALIGNMENT, &previousAlignment);
  gl::glBindTexture(gl::GL_TEXTURE_2D, texture);
  gl::glPixelStorei(gl::GL_UNPACK_ALIGNMENT, 1);
  gl::glTexImage2D(gl::GL_TEXTURE_2D, 0, static_cast<gl::GLint>(gl::GL_RGBA8), width, height, 0,
    gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, pixels);
  gl::glPixelStorei(gl::GL_UNPACK_ALIGNMENT, previousAlignment);
  gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MIN_FILTER,
    static_cast<gl::GLint>(gl::GL_LINEAR));
  gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MAG_FILTER,
    static_cast<gl::GLint>(gl::GL_LINEAR));
  gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_S,
    static_cast<gl::GLint>(gl::GL_CLAMP_TO_EDGE));
  gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_T,
    static_cast<gl::GLint>(gl::GL_CLAMP_TO_EDGE));
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
  return texture;
}

bool loadTcpViewerImageTexture(const std::filesystem::path& path, TcpViewerImageTexture& image) {
  if (path.empty()) {
    return false;
  }
  int width = 0;
  int height = 0;
  unsigned char* pixels = loadRgba8Pixels(path, width, height);
  if (!pixels) {
    return false;
  }

  int alphaMinX = width;
  int alphaMinY = height;
  int alphaMaxX = -1;
  int alphaMaxY = -1;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const unsigned char alpha = pixels[(y * width + x) * 4 + 3];
      if (alpha <= 8) {
        continue;
      }
      alphaMinX = std::min(alphaMinX, x);
      alphaMinY = std::min(alphaMinY, y);
      alphaMaxX = std::max(alphaMaxX, x);
      alphaMaxY = std::max(alphaMaxY, y);
    }
  }
  ImVec2 uvMin(0.0f, 0.0f);
  ImVec2 uvMax(1.0f, 1.0f);
  if (alphaMaxX >= alphaMinX && alphaMaxY >= alphaMinY) {
    constexpr int cropMarginPixels = 2;
    alphaMinX = std::max(0, alphaMinX - cropMarginPixels);
    alphaMinY = std::max(0, alphaMinY - cropMarginPixels);
    alphaMaxX = std::min(width - 1, alphaMaxX + cropMarginPixels);
    alphaMaxY = std::min(height - 1, alphaMaxY + cropMarginPixels);
    uvMin = ImVec2(static_cast<float>(alphaMinX) / static_cast<float>(width),
      static_cast<float>(alphaMinY) / static_cast<float>(height));
    uvMax = ImVec2(static_cast<float>(alphaMaxX + 1) / static_cast<float>(width),
      static_cast<float>(alphaMaxY + 1) / static_cast<float>(height));
  }

  const unsigned int texture = uploadRgba8Texture(pixels, width, height);
  stbi_image_free(pixels);
  if (texture == 0) {
    return false;
  }

  image.release();
  image.texture = texture;
  image.width = width;
  image.height = height;
  image.uvMin = uvMin;
  image.uvMax = uvMax;
  return true;
}

bool loadTcpViewerIconTexture(const std::filesystem::path& path, TcpViewerIcon& icon) {
  int width = 0;
  int height = 0;
  unsigned char* pixels = loadRgba8Pixels(path, width, height);
  if (!pixels) {
    return false;
  }

  for (int i = 0; i < width * height; ++i) {
    unsigned char* pixel = pixels + i * 4;
    if (pixel[3] == 0) {
      continue;
    }
    // Treat icon artwork as an alpha mask so button colors stay consistent
    // across Flaticon styles and remain legible on dark TCP viewer panels.
    pixel[0] = 255;
    pixel[1] = 255;
    pixel[2] = 255;
  }

  const unsigned int texture = uploadRgba8Texture(pixels, width, height);
  stbi_image_free(pixels);
  if (texture == 0) {
    return false;
  }

  icon.texture = texture;
  icon.width = width;
  icon.height = height;
  return true;
}

struct TcpViewerIcons {
  std::array<TcpViewerIcon, static_cast<size_t>(TcpViewerIconKind::Count)> icons{};

  bool load(const std::filesystem::path& iconDir) {
    if (iconDir.empty()) {
      return false;
    }
    bool loadedAny = false;
    for (size_t i = 0; i < icons.size(); ++i) {
      const auto kind = static_cast<TcpViewerIconKind>(i);
      const std::filesystem::path path = iconDir / tcpViewerIconFileName(kind);
      loadedAny |= loadTcpViewerIconTexture(path, icons[i]);
    }
    return loadedAny;
  }

  void release() {
    for (auto& icon : icons) {
      raisin::deleteTextureAndInvalidateCache(icon.texture);
      icon = {};
    }
  }

  [[nodiscard]] const TcpViewerIcon* get(TcpViewerIconKind kind) const {
    const auto index = static_cast<size_t>(kind);
    if (index >= icons.size() || !icons[index].valid()) {
      return nullptr;
    }
    return &icons[index];
  }
};

ImVec4 tcpViewerIconTint(TcpViewerIconKind kind, bool hovered, bool active) {
  ImVec4 color;
  switch (kind) {
    case TcpViewerIconKind::Connect: color = ImVec4(0.30f, 0.95f, 0.58f, 1.0f); break;
    case TcpViewerIconKind::Disconnect: color = ImVec4(1.00f, 0.44f, 0.38f, 1.0f); break;
    case TcpViewerIconKind::Refresh: color = ImVec4(0.40f, 0.72f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::Save: color = ImVec4(1.00f, 0.72f, 0.26f, 1.0f); break;
    case TcpViewerIconKind::Home: color = ImVec4(0.46f, 0.92f, 0.96f, 1.0f); break;
    case TcpViewerIconKind::Focus: color = ImVec4(0.76f, 0.66f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::Camera: color = ImVec4(0.55f, 0.86f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::Folder: color = ImVec4(1.00f, 0.78f, 0.36f, 1.0f); break;
    case TcpViewerIconKind::Export: color = ImVec4(0.38f, 0.96f, 0.78f, 1.0f); break;
    case TcpViewerIconKind::Options: color = ImVec4(0.93f, 0.72f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::Robot: color = ImVec4(0.55f, 0.85f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::Reset: color = ImVec4(0.40f, 0.72f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::Exit: color = ImVec4(1.00f, 0.55f, 0.42f, 1.0f); break;
    case TcpViewerIconKind::Pause: color = ImVec4(1.00f, 0.78f, 0.36f, 1.0f); break;
    case TcpViewerIconKind::Play: color = ImVec4(0.30f, 0.95f, 0.58f, 1.0f); break;
    // Recording controls read as a warm "armed" pair, distinct from the green
    // sim-playback icons they used to borrow.
    case TcpViewerIconKind::Stop: color = ImVec4(1.00f, 0.52f, 0.44f, 1.0f); break;
    case TcpViewerIconKind::Force: color = ImVec4(1.00f, 0.84f, 0.36f, 1.0f); break;
    case TcpViewerIconKind::Torque: color = ImVec4(0.72f, 0.86f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::Video: color = ImVec4(1.00f, 0.62f, 0.52f, 1.0f); break;
    case TcpViewerIconKind::Add: color = ImVec4(0.44f, 0.94f, 0.66f, 1.0f); break;
    case TcpViewerIconKind::Delete: color = ImVec4(1.00f, 0.46f, 0.40f, 1.0f); break;
    case TcpViewerIconKind::Render: color = ImVec4(0.98f, 0.76f, 0.44f, 1.0f); break;
    case TcpViewerIconKind::Diagnostics: color = ImVec4(0.52f, 0.90f, 0.86f, 1.0f); break;
    case TcpViewerIconKind::Objects: color = ImVec4(0.62f, 0.78f, 1.00f, 1.0f); break;
    // File-navigator rows: folders warm, file kinds cool, so a listing scans as
    // "containers vs contents" before the glyph is even read.
    case TcpViewerIconKind::Help: color = ImVec4(0.86f, 0.88f, 0.94f, 1.0f); break;
    // Axis-aligned camera views share one cool tint so they read as a set,
    // with the perspective toggle warmer to set it apart from the six ortho faces.
    case TcpViewerIconKind::ViewTop:
    case TcpViewerIconKind::ViewBottom:
    case TcpViewerIconKind::ViewLeft:
    case TcpViewerIconKind::ViewRight:
    case TcpViewerIconKind::ViewFront:
    case TcpViewerIconKind::ViewBack: color = ImVec4(0.62f, 0.84f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::ViewPerspective: color = ImVec4(0.98f, 0.82f, 0.52f, 1.0f); break;
    case TcpViewerIconKind::FolderClosed: color = ImVec4(1.00f, 0.80f, 0.42f, 1.0f); break;
    case TcpViewerIconKind::FolderUp: color = ImVec4(0.98f, 0.86f, 0.58f, 1.0f); break;
    case TcpViewerIconKind::File: color = ImVec4(0.78f, 0.82f, 0.90f, 1.0f); break;
    case TcpViewerIconKind::FileCode: color = ImVec4(0.62f, 0.90f, 0.78f, 1.0f); break;
    case TcpViewerIconKind::FileImage: color = ImVec4(0.72f, 0.80f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::FileCsv: color = ImVec4(0.56f, 0.92f, 0.72f, 1.0f); break;
    case TcpViewerIconKind::FileArchive: color = ImVec4(0.88f, 0.76f, 0.96f, 1.0f); break;
    case TcpViewerIconKind::Step: color = ImVec4(0.55f, 0.86f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::StepFast: color = ImVec4(0.55f, 0.86f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::SensorDepth: color = ImVec4(0.43f, 0.87f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::SensorImu: color = ImVec4(0.55f, 0.95f, 0.68f, 1.0f); break;
    case TcpViewerIconKind::SensorLidar: color = ImVec4(1.00f, 0.70f, 0.30f, 1.0f); break;
    case TcpViewerIconKind::SensorUnknown: color = ImVec4(0.72f, 0.76f, 0.84f, 1.0f); break;
    case TcpViewerIconKind::ObjectVisual: color = ImVec4(0.88f, 0.68f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::ObjectSphere: color = ImVec4(0.42f, 0.86f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::ObjectBox: color = ImVec4(0.48f, 0.78f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::ObjectCylinder: color = ImVec4(0.42f, 0.91f, 0.78f, 1.0f); break;
    case TcpViewerIconKind::ObjectCapsule: color = ImVec4(0.66f, 0.88f, 0.56f, 1.0f); break;
    case TcpViewerIconKind::ObjectMesh: color = ImVec4(0.78f, 0.70f, 1.00f, 1.0f); break;
    case TcpViewerIconKind::ObjectGround: color = ImVec4(0.68f, 0.82f, 0.45f, 1.0f); break;
    case TcpViewerIconKind::ObjectHeightmap: color = ImVec4(0.78f, 0.72f, 0.42f, 1.0f); break;
    case TcpViewerIconKind::ObjectCompound: color = ImVec4(0.48f, 0.82f, 0.96f, 1.0f); break;
    case TcpViewerIconKind::ObjectDeformable: color = ImVec4(0.96f, 0.58f, 0.78f, 1.0f); break;
    case TcpViewerIconKind::ObjectGranular: color = ImVec4(0.94f, 0.68f, 0.40f, 1.0f); break;
    case TcpViewerIconKind::Count: color = ImVec4(0.92f, 0.94f, 0.98f, 1.0f); break;
  }
  const float boost = active ? 1.18f : (hovered ? 1.08f : 1.0f);
  color.x = std::min(color.x * boost, 1.0f);
  color.y = std::min(color.y * boost, 1.0f);
  color.z = std::min(color.z * boost, 1.0f);
  return color;
}

TcpViewerIconKind sensorTypeIconKind(raisim::Sensor::Type type) {
  switch (type) {
    case raisim::Sensor::Type::RGB: return TcpViewerIconKind::Camera;
    case raisim::Sensor::Type::DEPTH: return TcpViewerIconKind::SensorDepth;
    case raisim::Sensor::Type::IMU: return TcpViewerIconKind::SensorImu;
    case raisim::Sensor::Type::SPINNING_LIDAR: return TcpViewerIconKind::SensorLidar;
    default: return TcpViewerIconKind::SensorUnknown;
  }
}

TcpViewerIconKind jointTypeIconKind(int32_t rawType) {
  switch (static_cast<raisim::Joint::Type>(rawType)) {
    case raisim::Joint::Type::REVOLUTE: return TcpViewerIconKind::Reset;
    case raisim::Joint::Type::PRISMATIC: return TcpViewerIconKind::Step;
    case raisim::Joint::Type::SPHERICAL: return TcpViewerIconKind::ObjectSphere;
    case raisim::Joint::Type::FLOATING: return TcpViewerIconKind::ObjectVisual;
    case raisim::Joint::Type::FIXED: return TcpViewerIconKind::Connect;
    default: return TcpViewerIconKind::Options;
  }
}

void drawJointTypeIcon(const TcpViewerIcons& icons, int32_t rawType) {
  const TcpViewerIconKind kind = jointTypeIconKind(rawType);
  const TcpViewerIcon* icon = icons.get(kind);
  if (!icon) return;

  const float iconSize = std::round(ImGui::GetFontSize() * 0.9f);
  ImGui::Image(reinterpret_cast<ImTextureID>(uint64_t(icon->texture)),
    ImVec2(iconSize, iconSize), ImVec2(0, 0), ImVec2(1, 1), tcpViewerIconTint(kind, false, false));
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("%s joint", tcpViewerJointTypeName(rawType));
  }
  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
}

TcpViewerIconKind objectTypeIconKind(int objectTypeRaw) {
  if (objectTypeRaw == -1) return TcpViewerIconKind::ObjectVisual;
  if (objectTypeRaw == 10) return TcpViewerIconKind::ObjectDeformable;
  if (objectTypeRaw == 11) return TcpViewerIconKind::ObjectGranular;
  if (objectTypeRaw < 0) return TcpViewerIconKind::Options;
  switch (static_cast<raisim::ObjectType>(objectTypeRaw)) {
    case raisim::ObjectType::SPHERE: return TcpViewerIconKind::ObjectSphere;
    case raisim::ObjectType::BOX: return TcpViewerIconKind::ObjectBox;
    case raisim::ObjectType::CYLINDER: return TcpViewerIconKind::ObjectCylinder;
    case raisim::ObjectType::CAPSULE: return TcpViewerIconKind::ObjectCapsule;
    case raisim::ObjectType::MESH: return TcpViewerIconKind::ObjectMesh;
    case raisim::ObjectType::HALFSPACE: return TcpViewerIconKind::ObjectGround;
    case raisim::ObjectType::HEIGHTMAP: return TcpViewerIconKind::ObjectHeightmap;
    case raisim::ObjectType::ARTICULATED_SYSTEM: return TcpViewerIconKind::Robot;
    case raisim::ObjectType::COMPOUND: return TcpViewerIconKind::ObjectCompound;
    default: return TcpViewerIconKind::Options;
  }
}

// Tab glyphs are drawn at this multiple of the font size. The tab bar is the
// panel's primary navigation, so its icons are deliberately larger than the
// inline button icons (which sit at ~0.95 of the font size).
constexpr float kTabIconGlyphScale = 1.55f;
// Fraction of the tab's short side the glyph fills, leaving a little breathing
// room inside the tab's highlight.
constexpr float kTabIconFillFactor = 0.86f;

/** Tab side length needed to draw a glyph at kTabIconGlyphScale. */
float iconTabExtent(float fontSize) {
  if (!std::isfinite(fontSize) || fontSize <= 0.0f) {
    return 0.0f;
  }
  return fontSize * kTabIconGlyphScale / kTabIconFillFactor;
}

/**
 * @brief Vertical frame padding that makes a tab tall enough for its glyph.
 *
 * ImGui computes tab height as font size + 2 * FramePadding.y, so the glyph
 * cannot grow past the font size without more padding. Never shrinks the
 * caller's padding.
 */
float iconTabFramePaddingY(float fontSize, float stylePaddingY) {
  const float extent = iconTabExtent(fontSize);
  if (extent <= 0.0f) {
    return stylePaddingY;
  }
  return std::max(stylePaddingY, (extent - fontSize) * 0.5f);
}

/**
 * @brief Spaces needed to reserve an icon tab wide enough for its glyph.
 *
 * ImGui derives a tab's width from its label, and the label here is whitespace,
 * so this is what keeps icon tabs from collapsing as the UI scale changes.
 * Clamped to a sane range so a degenerate font metric cannot produce a tab that
 * is invisible or absurdly wide.
 */
int iconTabLabelSpaceCount(float fontSize, float spaceWidth) {
  if (!std::isfinite(fontSize) || !std::isfinite(spaceWidth) || spaceWidth <= 0.0f ||
      fontSize <= 0.0f) {
    return 4;
  }
  // Clamp before the cast: a tiny space width makes the ratio large enough to
  // overflow int, and that conversion is undefined behaviour.
  const float needed = std::clamp(iconTabExtent(fontSize) / spaceWidth, 2.0f, 24.0f);
  return static_cast<int>(std::ceil(needed));
}

/**
 * @brief Open a tab bar sized for icon tabs.
 *
 * Pairs with endIconTabBar(). The extra frame padding has to be in style before
 * the first tab is submitted, which is why it lives here rather than in
 * beginIconTabItem().
 */
bool beginIconTabBar(const char* id) {
  const ImGuiStyle& style = ImGui::GetStyle();
  ImVec2 padding = style.FramePadding;
  padding.y = iconTabFramePaddingY(ImGui::GetFontSize(), padding.y);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, padding);
  if (ImGui::BeginTabBar(id)) {
    return true;
  }
  ImGui::PopStyleVar();
  return false;
}

void endIconTabBar() {
  ImGui::EndTabBar();
  ImGui::PopStyleVar();
}

// An icon over a soft drop shadow, the shared look of every icon glyph.
void drawIconGlyph(ImDrawList* drawList, ImTextureID textureId, const ImVec2& iconMin,
                   const ImVec2& iconMax, const ImVec4& iconTint) {
  drawList->AddImage(textureId, ImVec2(iconMin.x + 1.0f, iconMin.y + 1.0f),
    ImVec2(iconMax.x + 1.0f, iconMax.y + 1.0f), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f),
    ImGui::GetColorU32(ImVec4(0.0f, 0.0f, 0.0f, 0.34f)));
  drawList->AddImage(textureId, iconMin, iconMax, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f),
    ImGui::GetColorU32(iconTint));
}

// The tinted rounded chip behind an icon button's glyph, then the glyph itself.
void drawIconChip(ImDrawList* drawList, ImTextureID textureId, const ImVec2& iconMin,
                  const ImVec2& iconMax, float chipPad, const ImVec4& iconTint, bool hovered,
                  bool active) {
  const ImVec4 chipFill(iconTint.x, iconTint.y, iconTint.z,
                        active ? 0.26f : (hovered ? 0.22f : 0.16f));
  const ImVec4 chipBorder(iconTint.x, iconTint.y, iconTint.z,
                          active ? 0.72f : (hovered ? 0.58f : 0.42f));
  drawList->AddRectFilled(ImVec2(iconMin.x - chipPad, iconMin.y - chipPad),
    ImVec2(iconMax.x + chipPad, iconMax.y + chipPad),
    ImGui::GetColorU32(chipFill), 3.0f);
  drawList->AddRect(ImVec2(iconMin.x - chipPad, iconMin.y - chipPad),
    ImVec2(iconMax.x + chipPad, iconMax.y + chipPad),
    ImGui::GetColorU32(chipBorder), 3.0f);
  drawIconGlyph(drawList, textureId, iconMin, iconMax, iconTint);
}

/**
 * @brief An icon-only tab, styled like the icon buttons.
 *
 * Five text tabs no longer fit the left panel at larger UI scales, so the label
 * is reduced to a glyph. ImGui sizes a tab from its label, so the visible label
 * is a run of spaces wide enough for the icon and the glyph is drawn over the
 * resulting rect -- the same overlay approach drawIconOnlyButton() and
 * drawSensorTreeNode() use. The tab name moves to a tooltip so it stays
 * discoverable.
 *
 * @return True when the tab is selected; the caller must then call EndTabItem().
 */
bool beginIconTabItem(const TcpViewerIcons& icons, TcpViewerIconKind kind, const char* tooltip,
                      const char* id) {
  const TcpViewerIcon* icon = icons.get(kind);
  if (!icon) {
    // No texture: fall back to the text label rather than an unclickable blank.
    return ImGui::BeginTabItem(tooltip ? tooltip : id);
  }

  const int spaceCount = iconTabLabelSpaceCount(ImGui::GetFontSize(), ImGui::CalcTextSize(" ").x);
  // "###" (not "##") so ImHashStr reseeds and the id alone determines the tab's
  // ImGui ID. With "##" the leading spaces are hashed in too, and since their
  // count tracks the font size, dragging the UI Scale slider silently renamed
  // every tab -- the bar then lost its selection and fell back to the first one.
  const std::string label = std::string(static_cast<size_t>(spaceCount), ' ') + "###" + id;

  const bool selected = ImGui::BeginTabItem(label.c_str());
  const bool hovered = ImGui::IsItemHovered();
  const ImVec2 itemMin = ImGui::GetItemRectMin();
  const ImVec2 itemMax = ImGui::GetItemRectMax();
  if (hovered && tooltip && tooltip[0] != '\0') {
    ImGui::SetTooltip("%s", tooltip);
  }

  // ImGui registers a zero-size item for a tab on the frame it first appears,
  // and for one scrolled fully out of the bar. There is no rect to draw into
  // yet, and forcing a minimum size would paint the glyph at the window origin.
  const float rectWidth = itemMax.x - itemMin.x;
  const float rectHeight = itemMax.y - itemMin.y;
  if (rectWidth < 2.0f || rectHeight < 2.0f) {
    return selected;
  }

  // Fit the glyph inside whatever rect the tab bar handed us so a compressed
  // tab clips nothing.
  const float iconSize =
    std::max(1.0f, std::min(rectWidth, rectHeight) * kTabIconFillFactor);
  const ImVec2 centre((itemMin.x + itemMax.x) * 0.5f, (itemMin.y + itemMax.y) * 0.5f);
  const ImVec2 iconMin(std::round(centre.x - iconSize * 0.5f),
                       std::round(centre.y - iconSize * 0.5f));
  const ImVec2 iconMax(iconMin.x + iconSize, iconMin.y + iconSize);

  ImDrawList* drawList = ImGui::GetWindowDrawList();
  const ImTextureID textureId = (ImTextureID)(intptr_t)icon->texture;
  // Selected tabs read as "active" so the tint matches the button palette.
  const ImVec4 iconTint = tcpViewerIconTint(kind, hovered, selected);
  drawList->PushClipRect(itemMin, itemMax, true);
  drawIconGlyph(drawList, textureId, iconMin, iconMax, iconTint);
  drawList->PopClipRect();
  return selected;
}

TcpViewerIconKind spawnShapeIconKind(raisin::tcp_viewer::ClientRequestType type) {
  using raisin::tcp_viewer::ClientRequestType;
  switch (type) {
    case ClientRequestType::CR_SPAWN_BOX: return TcpViewerIconKind::ObjectBox;
    case ClientRequestType::CR_SPAWN_SPHERE: return TcpViewerIconKind::ObjectSphere;
    case ClientRequestType::CR_SPAWN_CYLINDER: return TcpViewerIconKind::ObjectCylinder;
    case ClientRequestType::CR_SPAWN_CAPSULE: return TcpViewerIconKind::ObjectCapsule;
    case ClientRequestType::CR_SPAWN_MESH: return TcpViewerIconKind::ObjectMesh;
    case ClientRequestType::CR_SPAWN_AS: return TcpViewerIconKind::Robot;
    case ClientRequestType::CR_SPAWN_PLANE: return TcpViewerIconKind::ObjectGround;
    case ClientRequestType::CR_SPAWN_HEIGHT_MAP: return TcpViewerIconKind::ObjectHeightmap;
    default: return TcpViewerIconKind::Options;
  }
}

bool drawSensorTreeNode(const TcpViewerIcons& icons, raisim::Sensor::Type type,
                        const char* label) {
  const TcpViewerIconKind kind = sensorTypeIconKind(type);
  const TcpViewerIcon* icon = icons.get(kind);
  if (!icon) return ImGui::TreeNode(label);

  const bool open = ImGui::TreeNodeEx("##sensor", ImGuiTreeNodeFlags_SpanAvailWidth);
  const ImVec2 rowMin = ImGui::GetItemRectMin();
  const ImVec2 rowMax = ImGui::GetItemRectMax();
  const float iconSize = std::round(ImGui::GetFontSize() * 0.9f);
  const SensorTreeRowLayout layout = sensorTreeRowLayout(
    rowMin.x, ImGui::GetTreeNodeToLabelSpacing(), iconSize,
    ImGui::GetStyle().ItemInnerSpacing.x);
  const float iconY = rowMin.y + std::max(0.0f, (rowMax.y - rowMin.y - iconSize) * 0.5f);
  const float textY = rowMin.y + std::max(0.0f, (rowMax.y - rowMin.y - ImGui::GetFontSize()) * 0.5f);
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  drawList->AddImage(reinterpret_cast<ImTextureID>(uint64_t(icon->texture)),
    ImVec2(layout.iconX, iconY), ImVec2(layout.iconX + iconSize, iconY + iconSize),
    ImVec2(0, 0), ImVec2(1, 1),
    ImGui::GetColorU32(tcpViewerIconTint(kind, ImGui::IsItemHovered(), open)));
  drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
    ImVec2(layout.labelX, textY), ImGui::GetColorU32(ImGuiCol_Text), label);
  return open;
}

ImVec2 iconTextButtonSize(const char* label, ImVec2 requestedSize = ImVec2(0.0f, 0.0f)) {
  const ImGuiStyle& style = ImGui::GetStyle();
  const ImVec2 textSize = ImGui::CalcTextSize(label);
  const float iconSize = std::round(ImGui::GetFontSize() * 0.95f);
  const ImVec2 minSize(
    textSize.x + iconSize + style.ItemInnerSpacing.x + style.FramePadding.x * 2.0f,
    std::max(textSize.y, iconSize) + style.FramePadding.y * 2.0f);
  return ImVec2(
    requestedSize.x > 0.0f ? std::max(requestedSize.x, minSize.x) : minSize.x,
    requestedSize.y > 0.0f ? std::max(requestedSize.y, minSize.y) : minSize.y);
}

float fontScaledTextControlWidth(float visibleChars) {
  const ImGuiStyle& style = ImGui::GetStyle();
  const float charWidth = std::max(1.0f, ImGui::CalcTextSize("M").x);
  return charWidth * std::max(1.0f, visibleChars) + style.FramePadding.x * 2.0f;
}

float comboWidthForTextItems(const char* const* items, int itemCount) {
  const ImGuiStyle& style = ImGui::GetStyle();
  float maxTextWidth = 0.0f;
  for (int i = 0; i < itemCount; ++i) {
    maxTextWidth = std::max(maxTextWidth, ImGui::CalcTextSize(items[i]).x);
  }
  return maxTextWidth + style.FramePadding.x * 2.0f + ImGui::GetFrameHeight();
}

bool drawIconTextButton(const TcpViewerIcons& icons, TcpViewerIconKind kind, const char* label,
                        const char* id, ImVec2 requestedSize = ImVec2(0.0f, 0.0f)) {
  const ImVec2 buttonSize = iconTextButtonSize(label, requestedSize);
  const TcpViewerIcon* icon = icons.get(kind);
  ImGui::PushID(id ? id : label);
  if (!icon) {
    const bool pressed = ImGui::Button(label, buttonSize);
    ImGui::PopID();
    return pressed;
  }

  const bool pressed = ImGui::Button("##icon_text_button", buttonSize);
  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();
  const ImVec2 itemMin = ImGui::GetItemRectMin();
  const ImVec2 itemMax = ImGui::GetItemRectMax();
  const ImVec2 textSize = ImGui::CalcTextSize(label);
  const ImGuiStyle& style = ImGui::GetStyle();
  const float iconSize = std::round(ImGui::GetFontSize() * 0.95f);
  const float contentWidth = iconSize + style.ItemInnerSpacing.x + textSize.x;
  const float centerY = itemMin.y + (itemMax.y - itemMin.y) * 0.5f;
  const float startX = itemMin.x + std::max(style.FramePadding.x,
    ((itemMax.x - itemMin.x) - contentWidth) * 0.5f);
  const ImVec2 iconMin(startX, centerY - iconSize * 0.5f);
  const ImVec2 iconMax(iconMin.x + iconSize, iconMin.y + iconSize);
  const float chipPad = std::max(1.0f, std::round(ImGui::GetFontSize() * 0.10f));
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  const ImTextureID textureId = (ImTextureID)(intptr_t)icon->texture;
  const ImVec4 iconTint = tcpViewerIconTint(kind, hovered, active);
  drawList->PushClipRect(itemMin, itemMax, true);
  drawIconChip(drawList, textureId, iconMin, iconMax, chipPad, iconTint, hovered, active);
  drawList->AddText(ImVec2(iconMax.x + style.ItemInnerSpacing.x, centerY - textSize.y * 0.5f),
    ImGui::GetColorU32(ImGuiCol_Text), label);
  drawList->PopClipRect();
  ImGui::PopID();
  return pressed;
}

// Icon-only variant of drawIconTextButton. The button is a square (button height) with the
// same coloured chip + icon draw as the text variant, no label. Pass a tooltip so users
// can still discover the action on hover.
bool drawIconOnlyButton(const TcpViewerIcons& icons, TcpViewerIconKind kind,
                        const char* tooltip, const char* id) {
  const ImGuiStyle& style = ImGui::GetStyle();
  const float buttonExtent =
      ImGui::GetFontSize() + style.FramePadding.y * 2.0f;
  const ImVec2 buttonSize(buttonExtent, buttonExtent);
  const TcpViewerIcon* icon = icons.get(kind);
  ImGui::PushID(id ? id : tooltip);
  if (!icon) {
    const bool pressed = ImGui::Button(tooltip ? tooltip : "?", buttonSize);
    ImGui::PopID();
    return pressed;
  }

  const bool pressed = ImGui::Button("##icon_only_button", buttonSize);
  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();
  if (hovered && tooltip && tooltip[0] != '\0') {
    ImGui::SetTooltip("%s", tooltip);
  }
  const ImVec2 itemMin = ImGui::GetItemRectMin();
  const ImVec2 itemMax = ImGui::GetItemRectMax();
  const float iconSize = std::round(ImGui::GetFontSize() * 0.95f);
  const ImVec2 centre((itemMin.x + itemMax.x) * 0.5f, (itemMin.y + itemMax.y) * 0.5f);
  const ImVec2 iconMin(centre.x - iconSize * 0.5f, centre.y - iconSize * 0.5f);
  const ImVec2 iconMax(iconMin.x + iconSize, iconMin.y + iconSize);
  const float chipPad = std::max(1.0f, std::round(ImGui::GetFontSize() * 0.10f));
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  const ImTextureID textureId = (ImTextureID)(intptr_t)icon->texture;
  const ImVec4 iconTint = tcpViewerIconTint(kind, hovered, active);
  drawList->PushClipRect(itemMin, itemMax, true);
  drawIconChip(drawList, textureId, iconMin, iconMax, chipPad, iconTint, hovered, active);
  drawList->PopClipRect();
  ImGui::PopID();
  return pressed;
}

/**
 * @brief Draw the spawn palette.
 *
 * @param canSpawn Whether a spawn request could be delivered right now.
 * @param dropPoint World point used when "place at camera target" is enabled.
 * @param serverIsLocal True when the server shares this filesystem, which lets
 *   us check a geometry path before a bad one costs the connection.
 * @param request Receives the built request when this returns true.
 * @param status Receives a human-readable validation or progress message.
 * @return True when the user asked to spawn and the form validated.
 */
bool drawSpawnForm(const TcpViewerIcons& icons, SpawnFormState& form, bool canSpawn,
                   const glm::vec3& dropPoint, bool serverIsLocal,
                   raisin::tcp_viewer::ClientRequest& request, std::string& status,
                   FileBrowserState* browser = nullptr) {
  using raisin::tcp_viewer::ClientRequestType;
  const float vecWidth = std::round(ImGui::GetFontSize() * 12.5f);
  const float textWidth = fontScaledTextControlWidth(26.0f);
  // Every field is labelled after the control, in the disabled colour.
  const auto trailingLabel = [](const char* label) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", label);
  };

  ImGui::TextUnformatted("Shape");
  for (int i = 0; i < static_cast<int>(kSpawnShapes.size()); ++i) {
    const SpawnShapeInfo& shape = kSpawnShapes[static_cast<size_t>(i)];
    if (i % 4 != 0) {
      ImGui::SameLine();
    }
    ImGui::PushID(i);
    const bool selected = form.shapeIndex == i;
    if (selected) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    if (drawIconOnlyButton(icons, spawnShapeIconKind(shape.type), shape.label, "spawn_shape")) {
      form.shapeIndex = i;
    }
    if (selected) {
      ImGui::PopStyleColor();
    }
    ImGui::PopID();
  }
  const SpawnShapeInfo& shape = spawnShapeAt(form.shapeIndex);
  ImGui::TextDisabled("%s", shape.label);

  ImGui::SetNextItemWidth(textWidth);
  ImGui::InputText("##spawn_name", form.name, sizeof(form.name));
  trailingLabel("Name");

  if (shape.needsFile) {
    ImGui::SetNextItemWidth(textWidth);
    ImGui::InputText("##spawn_file", form.file, sizeof(form.file));
    // Only offer the picker when the server shares this filesystem; browsing
    // locally for a remote server's path would just produce a wrong path.
    if (browser != nullptr && serverIsLocal) {
      ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
      if (drawIconOnlyButton(icons, TcpViewerIconKind::Folder, "Browse...", "browse_spawn_file")) {
        std::vector<std::string> extensions;
        std::istringstream extensionStream(shape.fileExtensions ? shape.fileExtensions : "");
        std::string extension;
        while (extensionStream >> extension) {
          extensions.push_back(extension);
        }
        if (extensions.empty()) {
          // Mesh spawns accept any format Assimp can read; list the common ones.
          extensions = {"obj", "stl", "dae", "ply", "gltf", "glb", "fbx"};
        }
        openFileBrowser(*browser, FileBrowserMode::OpenFile, std::string("Select ") + shape.label,
          std::filesystem::path(form.file), std::move(extensions),
          [&form](const std::filesystem::path& chosen) {
            std::snprintf(form.file, sizeof(form.file), "%s", chosen.string().c_str());
          });
      }
    }
    trailingLabel("File (server-side path)");
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + textWidth);
    ImGui::TextDisabled(
      "RaiSim resolves this path on the simulation host. A path the server cannot open "
      "is a protocol error and drops the viewer connection.");
    ImGui::PopTextWrapPos();
  }

  switch (shape.type) {
    case ClientRequestType::CR_SPAWN_BOX:
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat3("##spawn_box", form.boxExtent, 0.01f, 0.001f, 1000.0f, "%.3f");
      trailingLabel("Extents (m)");
      break;
    case ClientRequestType::CR_SPAWN_SPHERE:
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat("##spawn_radius", &form.radius, 0.005f, 0.001f, 1000.0f, "%.3f");
      trailingLabel("Radius (m)");
      break;
    case ClientRequestType::CR_SPAWN_CYLINDER:
    case ClientRequestType::CR_SPAWN_CAPSULE:
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat("##spawn_radius", &form.radius, 0.005f, 0.001f, 1000.0f, "%.3f");
      trailingLabel("Radius (m)");
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat("##spawn_height", &form.height,
        0.005f, shape.type == ClientRequestType::CR_SPAWN_CAPSULE ? 0.0f : 0.001f, 1000.0f, "%.3f");
      trailingLabel("Height (m)");
      break;
    case ClientRequestType::CR_SPAWN_PLANE:
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat("##spawn_ground", &form.groundHeight, 0.01f, -1000.0f, 1000.0f, "%.3f");
      trailingLabel("Height (m)");
      break;
    case ClientRequestType::CR_SPAWN_HEIGHT_MAP:
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat2("##spawn_hm_center", form.heightMapCenter, 0.05f, -10000.0f, 10000.0f,
        "%.3f");
      trailingLabel("Center X/Y (m)");
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat2("##spawn_hm_size", form.heightMapSize, 0.05f, 0.001f, 10000.0f, "%.3f");
      trailingLabel("Size X/Y (m)");
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat("##spawn_hm_scale", &form.heightMapHeightScale, 0.01f, -1000.0f, 1000.0f,
        "%.4f");
      trailingLabel("Height scale");
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat("##spawn_hm_offset", &form.heightMapHeightOffset, 0.01f, -1000.0f, 1000.0f,
        "%.4f");
      trailingLabel("Height offset (m)");
      break;
    default:
      break;
  }

  if (shape.needsMass) {
    ImGui::SetNextItemWidth(vecWidth);
    compactDragFloat("##spawn_mass", &form.mass, 0.05f, 0.001f, 100000.0f, "%.4g");
    trailingLabel("Mass (kg)");
    ImGui::SetNextItemWidth(vecWidth);
    ImGui::Combo("##spawn_body_type", &form.bodyType, "dynamic\0kinematic\0static\0");
    trailingLabel("Body type");
    ImGui::SetNextItemWidth(textWidth);
    ImGui::InputText("##spawn_appearance", form.appearance, sizeof(form.appearance));
    trailingLabel("Appearance");
  }

  const bool placeable = shape.type != ClientRequestType::CR_SPAWN_PLANE &&
                         shape.type != ClientRequestType::CR_SPAWN_HEIGHT_MAP;
  if (placeable) {
    drawCompactCheckbox("Place at camera target", &form.useCameraPlacement);
    ImGui::BeginDisabled(form.useCameraPlacement);
    ImGui::SetNextItemWidth(vecWidth);
    compactDragFloat3("##spawn_position", form.position, 0.02f, -100000.0f, 100000.0f, "%.3f");
    trailingLabel("Position (m)");
    ImGui::EndDisabled();
    if (form.useCameraPlacement) {
      form.position[0] = dropPoint.x;
      form.position[1] = dropPoint.y;
      form.position[2] = dropPoint.z;
    }
    if (ImGui::TreeNode("Initial state")) {
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat3("##spawn_lin_vel", form.linearVelocity, 0.05f, -1000.0f, 1000.0f, "%.3f");
      trailingLabel("Linear velocity (m/s)");
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat3("##spawn_ang_vel", form.angularVelocity, 0.05f, -1000.0f, 1000.0f, "%.3f");
      trailingLabel("Angular velocity (rad/s)");
      ImGui::SetNextItemWidth(vecWidth);
      compactDragFloat4("##spawn_quat", form.quatWxyz, 0.005f, -1.0f, 1.0f, "%.3f");
      trailingLabel("Quaternion WXYZ");
      ImGui::TreePop();
    }
  }

  // Preflight the request so the reason is visible before the button is pressed.
  raisin::tcp_viewer::ClientRequest candidate;
  const std::string validationError = buildSpawnRequest(form, candidate);
  std::string localFileError;
  if (validationError.empty() && shape.needsFile && serverIsLocal) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(candidate.file, ec)) {
      localFileError = "file not found on this host: " + candidate.file;
    }
  }
  const std::string blockingError =
    validationError.empty() ? localFileError : validationError;

  ImGui::BeginDisabled(!canSpawn || !blockingError.empty());
  const bool pressed = drawIconTextButton(icons, TcpViewerIconKind::Add, "Spawn",
                                          "spawn_object");
  ImGui::EndDisabled();
  if (!blockingError.empty()) {
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + textWidth);
    ImGui::TextDisabled("%s", blockingError.c_str());
    ImGui::PopTextWrapPos();
  }
  if (!pressed) {
    return false;
  }
  if (!blockingError.empty()) {
    status = blockingError;
    return false;
  }
  request = std::move(candidate);
  status = std::string("spawn queued: ") + shape.label;
  return true;
}

void drawCollapsedLeftPanelLogo(const TcpViewerImageTexture& logo) {
  const bool hasLogo = logo.valid();
  const float fontSize = ImGui::GetFontSize();
  const float outerSize = std::round(fontSize * kCollapsedLogoSizeInFontHeights);
  const ImVec2 handleSize = hasLogo
    ? ImVec2(outerSize, outerSize)
    : ImVec2(std::max(8.0f, std::round(fontSize * 0.48f)),
        std::max(28.0f, std::round(fontSize * 1.65f)));
  ImGui::InvisibleButton("##CollapsedLeftPanelHover", handleSize);

  const bool hovered = ImGui::IsItemHovered();
  const ImVec2 itemMin = ImGui::GetItemRectMin();
  const ImVec2 itemMax = ImGui::GetItemRectMax();
  ImDrawList* drawList = ImGui::GetWindowDrawList();

  if (!hasLogo) {
    const float barWidth = std::max(3.0f, std::round(fontSize * 0.16f));
    const float verticalPad = std::max(3.0f, std::round(fontSize * 0.18f));
    const float centerX = (itemMin.x + itemMax.x) * 0.5f;
    const ImVec2 barMin(centerX - barWidth * 0.5f, itemMin.y + verticalPad);
    const ImVec2 barMax(centerX + barWidth * 0.5f, itemMax.y - verticalPad);
    drawList->AddRectFilled(barMin, barMax,
      ImGui::GetColorU32(ImVec4(0.35f, 0.80f, 1.0f, hovered ? 0.70f : 0.48f)),
      barWidth * 0.5f);
    return;
  }

  const float uvWidth = std::max(1.0f, (logo.uvMax.x - logo.uvMin.x) * static_cast<float>(logo.width));
  const float uvHeight = std::max(1.0f, (logo.uvMax.y - logo.uvMin.y) * static_cast<float>(logo.height));
  const float aspect = uvWidth / uvHeight;
  const float maxImageW = outerSize;
  const float maxImageH = outerSize;
  float imageW = maxImageW;
  float imageH = maxImageH;
  if (aspect >= 1.0f) {
    imageH = imageW / aspect;
  } else {
    imageW = imageH * aspect;
  }

  const ImVec2 center((itemMin.x + itemMax.x) * 0.5f, (itemMin.y + itemMax.y) * 0.5f);
  const ImVec2 imageMin(center.x - imageW * 0.5f, center.y - imageH * 0.5f);
  const ImVec2 imageMax(center.x + imageW * 0.5f, center.y + imageH * 0.5f);
  const ImTextureID textureId = (ImTextureID)(intptr_t)logo.texture;
  // Untinted: ImGui multiplies this colour into the texture, so anything below
  // opaque white washes the logo out against its backing.
  drawList->AddImage(textureId, imageMin, imageMax, logo.uvMin, logo.uvMax,
    ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)));
}

// Flags shared by the floating panels (left overlay, object inspector, AS
// inspector). NoDecoration minus NoScrollbar: a panel's height cap can make it
// shorter than its content in a small pane, and a scrollbar is the only way to
// reach the rest of it.
constexpr ImGuiWindowFlags kOverlayPanelWindowFlags =
  ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
  ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
  ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
  ImGuiWindowFlags_NoNavFocus;

/**
 * @brief Draw one pane's rendered texture and feed that pane's viewer its input.
 * @param viewer Renderer owning the offscreen texture for this pane.
 * @param window Host SDL window, used for the logical-to-physical pixel scale.
 * @param rect Pane rectangle in ImGui logical points.
 * @param paneId Pane id, which makes the backing ImGui window name unique.
 * @param allowViewportInput Whether camera drags reach the renderer this frame.
 * @param allowClickSelection Whether a click may change the picked object.
 * @param viewportState Receives the pane's screen rect and cursor, for overlays.
 *
 * Each pane renders into its own texture at its own resolution, so the image is
 * blitted 1:1 rather than stretched: a 4-way split renders four quarter-sized
 * frames instead of four full-resolution ones.
 */
constexpr ImGuiWindowFlags kPaneViewportWindowFlags =
  ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
  ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
  ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
  ImGuiWindowFlags_NoFocusOnAppearing;

void renderViewer(raisin::RayraiWindow& viewer, SDL_Window* window,
                  const raisin::tcp_viewer::PaneRect& rect, uint32_t paneId,
                  bool allowViewportInput = true, bool allowClickSelection = true,
                  ViewerViewportState* viewportState = nullptr) {
  int fbW = 0;
  int fbH = 0;
  int windowW = 0;
  int windowH = 0;
  SDL_GL_GetDrawableSize(window, &fbW, &fbH);
  SDL_GetWindowSize(window, &windowW, &windowH);
  const int displayW = windowW > 0 ? windowW : fbW;
  const int displayH = windowH > 0 ? windowH : fbH;
  const float scaleX = displayW > 0 ? static_cast<float>(fbW) / static_cast<float>(displayW) : 1.0f;
  const float scaleY = displayH > 0 ? static_cast<float>(fbH) / static_cast<float>(displayH) : 1.0f;

  const float paneW = std::max(1.0f, rect.width);
  const float paneH = std::max(1.0f, rect.height);
  const int paneFbW = std::max(1, static_cast<int>(std::lround(paneW * scaleX)));
  const int paneFbH = std::max(1, static_cast<int>(std::lround(paneH * scaleY)));

  char windowName[64];
  std::snprintf(windowName, sizeof(windowName), "Viewer##pane%u", paneId);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::SetNextWindowPos(ImVec2(rect.x, rect.y));
  ImGui::SetNextWindowSize(ImVec2(paneW, paneH));
  ImGui::Begin(windowName, nullptr, kPaneViewportWindowFlags);

  ImTextureID tex = (ImTextureID)(intptr_t)viewer.getImageTexture();
  // The frame loop already opened this window once to fix its draw order, so the
  // cursor is placed explicitly rather than inherited from that earlier pass.
  const ImVec2 windowPos(rect.x, rect.y);
  ImGui::SetCursorScreenPos(windowPos);
  ImGuiIO& io = ImGui::GetIO();
  ImGui::Image(tex, ImVec2(paneW, paneH), ImVec2(0, 1), ImVec2(1, 0));

  const bool isHovered = ImGui::IsItemHovered();
  int cursorX = static_cast<int>((io.MousePos.x - windowPos.x) * scaleX);
  int cursorY = static_cast<int>((io.MousePos.y - windowPos.y) * scaleY);
  if (isHovered) {
    cursorX = std::clamp(cursorX, 0, std::max(0, paneFbW - 1));
    cursorY = std::clamp(cursorY, 0, std::max(0, paneFbH - 1));
  }
  if (viewportState) {
    viewportState->origin = windowPos;
    viewportState->size = ImVec2(paneW, paneH);
    viewportState->hovered = isHovered;
    viewportState->cursorX = cursorX;
    viewportState->cursorY = cursorY;
  }
  if (!allowViewportInput) {
    viewer.cancelViewportMouseDrag();
  }
  viewer.update(paneFbW, paneFbH, allowViewportInput ? isHovered : false, cursorX, cursorY,
    allowClickSelection);

  ImGui::End();
  ImGui::PopStyleVar(2);
}

/**
 * @brief One split pane: a self-contained viewer session.
 *
 * A pane owns everything that belongs to a single connection — its renderer and
 * world, its TCP client, the remote scene mirror, the tools and gestures that
 * act on that scene, and the state of its own copies of the overlay panels.
 * Anything deliberately common to every pane (render quality settings, UI scale,
 * the recent-connection list, resource directories and server discovery) stays
 * in main() and is shared by reference.
 *
 * Panes are keyed by @ref id, which PaneLayout keeps stable for the pane's whole
 * lifetime so ImGui window names and saved per-pane endpoints stay valid across
 * splits and closes.
 */
struct ViewerPane {
  explicit ViewerPane(uint32_t paneId) : id(paneId) {}
  ViewerPane(const ViewerPane&) = delete;
  ViewerPane& operator=(const ViewerPane&) = delete;

  uint32_t id = 0;
  /** Deferred one-shot startup work (initial connect, --inspect, replay). */
  bool startupPending = true;
  /** Serial of the shared settings this pane's renderer was last configured with. */
  uint64_t appliedSettingsSerial = 0;
  /** Serial of the shared resource-directory list already given to this scene. */
  uint64_t appliedResourceDirSerial = 0;

  // ----- renderer and connection -----
  std::shared_ptr<raisim::World> world;
  std::shared_ptr<raisin::RayraiWindow> viewer;
  std::unique_ptr<RemoteScene> scene;
  /** The server's RaiSim Engine scene: its files, the download prompt and its visuals. */
  std::unique_ptr<raisin::tcp_viewer::ViewerSceneFiles> sceneFiles;
  TcpClient client;
  SensorRenderer sensorRenderer;
  raisin::tcp_viewer::ActuatorTraces actuatorTraces;
  raisin::tcp_viewer::LocalSimulation localSimulation;
  bool connectingLocalSimulation = false;
  bool autoConnect = false;
  bool everConnected = false;
  // True once a scene payload has actually arrived on the current connection. A
  // socket that connected but has said nothing is not a session: RaisimServer
  // serves one client, so a second viewer's TCP connect is accepted by the
  // kernel backlog and then ignored, which would otherwise look live for ever.
  bool sceneReceived = false;
  bool awaitingResponse = false;
  bool awaitingSensorAck = false;
  /** Rotation state of this pane's sensor-update batches (see sendSensorUpdate()). */
  size_t nextSensorUpdateIndex = 0;
  std::chrono::steady_clock::time_point updateRequestSentAt = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point nextAutoConnectAttempt = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point nextTcpUpdateRequestTime = std::chrono::steady_clock::now();
  std::vector<char> tcpPayload;
  char host[256] = "127.0.0.1";
  int port = 0;
  char portBuf[16] = "";
  std::string lastStatus = "disconnected";
  bool verboseParsing = false;
  ViewerStats stats;

  // ----- camera -----
  glm::vec3 defaultCameraPos{0.0f};
  glm::vec3 defaultCameraTarget{0.0f};
  bool requestFrameScene = false;
  bool requestFrameSelected = false;
  bool requestResetCamera = false;
  bool autoFrameApplied = false;
  std::array<CameraBookmark, 4> cameraBookmarks;
  CameraFrustumUiStates cameraFrustums;

  // ----- scene display toggles -----
  bool showCollisionBodies = false;
  bool showWorldFrame = false;
  bool showContactPoints = false;
  bool showContactForces = false;
  bool contactForceAbsolute = false;
  bool forceTransparent = false;
  bool showBodyFrames = false;
  bool showComMarkers = readEnvBool("RAYRAI_TCP_VIEWER_SHOW_COM_MARKERS", false);
  float contactPointSize = 0.05f;
  float contactForceSize = 0.3f;
  float bodyFrameSize = 0.15f;
  float comMarkerSize = 0.03f;
  std::shared_ptr<raisin::CoordinateFrame> worldFrame;
  std::shared_ptr<raisin::CoordinateFrame> bodyFramesNode;
  std::vector<std::shared_ptr<raisin::Visuals>> comMarkers;

  // ----- selection -----
  uint32_t requestedTag = 0;
  int requestedIndex = 0;
  const VisualEntry* requestedEntry = nullptr;
  std::unordered_map<uint64_t, MotionEstimate> motionEstimates;

  // ----- sim control -----
  std::vector<raisin::tcp_viewer::ClientRequest> pendingControlRequests;
  bool simPaused = false;
  glm::vec3 controlForce{0.0f, 0.0f, 20.0f};
  glm::vec3 controlTorque{0.0f, 0.0f, 1.0f};
  glm::vec3 controlPointOffset{0.0f};
  int controlBodyIdx = 0;
  bool controlBodyFollowsSelection = true;
  uint32_t controlSelectionTag = 0;
  int controlSelectionIndex = -1;
  glm::vec3 controlPosePosition{0.0f};
  glm::vec4 controlPoseQuat{1.0f, 0.0f, 0.0f, 0.0f};
  uint32_t controlPoseTag = 0;
  bool controlPoseInitialized = false;
  std::vector<float> controlGc;
  uint32_t controlGcTag = 0;
  bool controlGcDirty = false;

  // ----- viewport gestures and tools -----
  ViewerViewportState viewportState;
  bool mouseForceEnabled = true;
  float mouseForceScale = kDefaultMouseForceAccelPerPixel;
  MouseForceGesture mouseForce;
  bool wireDragEnabled = true;
  float wireDragStiffness = kDefaultWireDragStiffness;
  WireDragGesture wireDrag;
  RulerToolState ruler;
  AngleToolState angle;
  PoseGrabberGesture poseGrabber;
  SpawnFormState spawnForm;
  std::string spawnStatus;
  char worldExportPathBuf[512] = "";

  // ----- inspector -----
  InspectorState inspector;

  // ----- panels -----
  FileBrowserState fileBrowser;
  ImGuiID pathFieldEditing = 0;
  ImVec2 overlayOffset{0.0f, 0.0f};
  ImVec2 detailOffset{0.0f, 0.0f};
  bool overlayMinimized = false;
  bool detailMinimized = false;
  bool overlayCollapsedHoveredLastFrame = false;
  std::chrono::steady_clock::time_point overlayLastInteractionTime =
    std::chrono::steady_clock::now();
  bool groupObjectsByType = false;
  bool hideCollisionObjects = false;
  int objectSortMode = kDefaultObjectSortMode;
  char objectFilterBuf[160] = "";

  // ----- capture -----
  char screenshotDirBuf[512] = "";
  std::filesystem::path pendingScreenshotPath;
  bool screenshotAfterFirstScene = false;
  bool screenshotRequested = false;
  bool recordPngSequence = false;
  int recordEveryNFrames = 1;
  int recordFrameIndex = 0;
  std::string recordFramePrefix = "rayrai_tcp_viewer_frame";
  std::filesystem::path serverRecordFrameDirectory;
  bool serverRequestedRecording = false;
  std::string captureStatus;
  std::vector<unsigned char> captureRgba;
  raisin::tcp_viewer::VideoEncoder videoEncoder;
  double videoFramesPerSecond = 30.0;
  int videoQuality = 20;
  char videoPathBuf[512] = "";
  std::string videoStatus;

  // ----- session record / replay -----
  char sessionPathBuf[512] = "";
  std::string sessionStatus;
  SessionRecorder sessionRecorder;
  std::vector<RecordedFrame> replayFrames;
  bool replayMode = false;
  bool replayPaused = false;
  bool replayStep = false;
  size_t replayIndex = 0;
  size_t replaySeekIndex = std::numeric_limits<size_t>::max();
  float replaySpeed = 1.0f;
  std::chrono::steady_clock::time_point replayStart = std::chrono::steady_clock::now();
  uint64_t replayBaseMicros = 0;
  std::ofstream trajectoryCsv;

  // ----- diagnostics and signals -----
  std::deque<PacketSample> packetSamples;
  DiagnosticsPresentationState diagnosticsPresentation;
  std::vector<AssetDiagnostic> assetDiagnostics;
  bool assetDiagnosticsDirty = true;
  bool exportScenePending = false;
  std::string sensorStatus;
  std::unordered_map<uint64_t, SignalObjectRecord> signalRecords;
  std::vector<std::string> signalPlotChannels;
  bool signalPlotChannelsInitialized = false;
  std::unordered_set<uint64_t> pinnedSignalObjects;
  uint64_t selectedSignalKey = 0;
  std::string signalExportStatus;

  /**
   * @brief Release GL and network resources in the order the renderer expects.
   *
   * The scene mirror holds raw pointers into the renderer's object list, so it
   * has to be torn down before the renderer itself goes away.
   */
  void shutdown() {
    client.disconnect();
    localSimulation.stop();
    if (viewer) clearCameraFrustums(*viewer, cameraFrustums);
    if (scene) scene->shutdown();
    scene.reset();
    viewer.reset();
    world.reset();
  }
};

} // namespace

static void reportClosedServerConnection(ViewerPane& pane) {
  if (pane.sceneReceived) {
    pane.lastStatus = "connection lost";
  } else {
    pane.lastStatus = "server closed before its first reply; requested protocol features may be unsupported";
    // A matching discovery version does not guarantee support for our feature bits.
    // Retrying this rejected first frame without user action would loop forever.
    pane.autoConnect = false;
  }
}

// Point the pane's host/port fields (and the port text box) at an endpoint.
static void setPaneEndpoint(ViewerPane& pane, const std::string& host, int port) {
  std::snprintf(pane.host, sizeof(pane.host), "%s", host.c_str());
  pane.port = port;
  std::snprintf(pane.portBuf, sizeof(pane.portBuf), "%d", port);
}

// Restored panes resume their own endpoint through the normal retry loop.
static void restorePaneConnection(ViewerPane& pane, const ConnectionEntry& endpoint,
                                  bool autoConnect) {
  setPaneEndpoint(pane, endpoint.host, endpoint.port);
  pane.autoConnect = autoConnect;
}

static void scheduleAutoConnectRetry(ViewerPane& pane, const ConnectionEntry& endpoint,
                                     std::chrono::steady_clock::time_point attemptTime) {
  pane.nextAutoConnectAttempt = attemptTime +
      (isLoopbackHostName(endpoint.host) ? kLoopbackAutoConnectInterval : kAutoConnectInterval);
}

#ifndef RAYRAI_TCP_VIEWER_NO_MAIN
int main(int argc, char* argv[]) {
  std::setlocale(LC_ALL, "C");
  std::locale::global(std::locale::classic());
  if (argc > 1 && std::string(argv[1]) == "--xml-simulation-worker")
    return raisin::tcp_viewer::runXmlSimulationWorker(argc, argv);

  ProgramOptions options;
  if (!parseProgramOptions(argc, argv, options)) {
    printUsage(argc > 0 ? argv[0] : "rayrai_tcp_viewer");
    return 2;
  }
  if (options.printHelp) {
    printUsage(argc > 0 ? argv[0] : "rayrai_tcp_viewer");
    return 0;
  }
  if (!options.activationKey.empty()) raisim::World::setActivationKey(options.activationKey.string());

  std::error_code pathEc;
  const std::filesystem::path argvPath = raisin::tcp_viewer::viewerExecutablePath(argc > 0 ? argv[0] : nullptr);
  std::filesystem::path binaryPath = std::filesystem::absolute(argvPath, pathEc);
  if (pathEc) {
    pathEc.clear();
    binaryPath = std::filesystem::current_path(pathEc);
  }
  const std::filesystem::path binaryDir = pathEc ? std::filesystem::path{} : binaryPath.parent_path();
  const std::string robotoFontPath = findRobotoFontPath(binaryDir);
  const float fontRasterizerDensity = readEnvFloatClamped(
    "RAYRAI_TCP_VIEWER_FONT_DENSITY", kDefaultFontRasterizerDensity, 1.0f, 3.0f);

  raisin::rayrai::sdl::setWindowsDpiAwarenessHint();
  SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
  std::signal(SIGINT, handleSignalQuit);
#if defined(SIGTERM)
  std::signal(SIGTERM, handleSignalQuit);
#endif
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
    std::cerr << "FATAL ERROR: Failed to initialize SDL: " << SDL_GetError() << "\n";
    return -1;
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  auto setContextVersion = [](int major, int minor) {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, major);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minor);
  };
  setContextVersion(4, 3);

  SDL_Window* window = SDL_CreateWindow("Rayrai Raisim TCP Viewer", SDL_WINDOWPOS_CENTERED,
    SDL_WINDOWPOS_CENTERED, options.windowWidth, options.windowHeight,
    SDL_WindowFlags(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI |
                    (options.fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0)));

  if (!window) {
    std::cerr << "FATAL ERROR: Failed to create SDL window: " << SDL_GetError() << "\n";
    SDL_Quit();
    return -1;
  }
  if (!options.fullscreen) {
    SDL_SetWindowBordered(window, SDL_TRUE);
  }

  SDL_GLContext context = SDL_GL_CreateContext(window);
  if (!context) {
    SDL_ClearError();
    setContextVersion(3, 3);
    context = SDL_GL_CreateContext(window);
  }
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

  gl::GLint fragmentTextureUnits = 0;
  gl::glGetIntegerv(gl::GL_MAX_TEXTURE_IMAGE_UNITS, &fragmentTextureUnits);
  const std::string capabilityError =
    raisin::rayrai_internal::rendererCapabilityError(fragmentTextureUnits);
  if (!capabilityError.empty()) {
    std::cerr << "ERROR: " << capabilityError << '\n';
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplSDL2_InitForOpenGL(window, context);
  ImGui_ImplOpenGL3_Init("#version 330");
  raionrobotics_imgui_theme();

  TcpViewerIcons uiIcons;
  const std::filesystem::path tcpViewerIconDir = findTcpViewerIconDir(binaryDir);
  if (!uiIcons.load(tcpViewerIconDir)) {
    std::cerr << "WARN: TCP viewer icons were not found; falling back to text-only buttons\n";
  }
  TcpViewerImageTexture raisimLogo;
  const std::filesystem::path raisimLogoPath = findRaisimLogoPath(binaryDir);
  if (!loadTcpViewerImageTexture(raisimLogoPath, raisimLogo)) {
    std::cerr << "WARN: Raisim logo was not found; using collapsed-panel fallback handle\n";
  }

  // ----- shared, pane-independent state ----------------------------------
  // A split gives you another connection, not another set of render settings:
  // quality, lighting, UI scale, the recent-connection list and the resource
  // search paths are shared by every pane and edited from whichever pane's
  // panel has focus. `settingsSerial` / `resourceDirSerial` let a pane notice
  // that the shared state changed — including a pane created long after the
  // last edit — and re-apply it to its own renderer and scene.
  ViewerSettings settings;
  uint64_t settingsSerial = 1;
  uint64_t resourceDirSerial = 1;

  raisin::tcp_viewer::PaneLayout paneLayout;
  std::unordered_map<uint32_t, std::unique_ptr<ViewerPane>> panes;

  auto createPaneRenderer = [&](uint32_t paneId) {
    auto pane = std::make_unique<ViewerPane>(paneId);
    pane->world = std::make_shared<raisim::World>();
    // Construct the renderer with the shader binary cache turned on and verbose so
    // we can see whether the slow "pbrMeshHigh" compile is a cache miss or a
    // genuine driver recompile. ThreadingMode default, 1 compile thread, cache
    // enabled, default cache dir ($HOME/.raisim/rayrai), log hits/misses.
    pane->viewer = std::make_shared<raisin::RayraiWindow>(
        pane->world, options.windowWidth, options.windowHeight,
        raisin::RayraiWindow::ThreadingMode::SingleThread,
        /*shaderCompileThreadCount=*/1u,
        /*shaderBinaryCacheEnabled=*/true,
        /*shaderBinaryCacheDirectory=*/std::string{},
        /*logShaderBinaryCache=*/true);
    // TCP scene updates need mesh assets available on the first render/export pass.
    pane->viewer->setAsyncMeshLoadingEnabled(false);
    return pane;
  };

  // The stored camera speed and field of view default to the renderer's own, so
  // the first renderer has to exist before the settings file is merged over them.
  // Its pane id is assigned afterwards, because the saved layout decides it.
  auto firstPane = createPaneRenderer(1);
  copyRenderDefaultsToSettings(settings, settings.renderQuality);
  settings.cameraSpeed = firstPane->viewer->getCamera().movementSpeed;
  settings.cameraFovDeg = firstPane->viewer->getCamera().zoom;
  loadViewerSettings(settings);
  if (!settings.paneLayout.empty() && !paneLayout.deserialize(settings.paneLayout)) {
    std::cerr << "WARN: saved pane layout '" << settings.paneLayout
              << "' is unreadable; opening a single pane\n";
    settings.paneLayout.clear();
    settings.panePlacements.clear();
  }
  const uint32_t primaryPaneId = paneLayout.panes().front();
  firstPane->id = primaryPaneId;
  panes.emplace(primaryPaneId, std::move(firstPane));
  ViewerPane& primaryPane = *panes[primaryPaneId];
  if (options.updateRateHz > 0.0f) {
    settings.tcpUpdateRateHz = options.updateRateHz;
    sanitizeViewerSettings(settings);
  }
  const GpuQualityRecommendation gpuQuality = recommendRenderQualityForCurrentGpu();
  if (applyAutomaticRenderQualityIfUnset(settings, gpuQuality.quality)) {
    std::cerr << "INFO: Auto render quality selected " << qualityName(settings.renderQuality)
              << " for GPU '" << gpuQuality.gpu.renderer << "'\n";
  }

  const char* cameraEnv = std::getenv("RAYRAI_TCP_VIEWER_CAMERA_LOOKAT");
  const bool forceCameraEnv = options.forceCameraLookAt ||
                              std::getenv("RAYRAI_TCP_VIEWER_FORCE_CAMERA_LOOKAT") != nullptr;
  glm::vec3 forcedCameraPos{0.0f};
  glm::vec3 forcedCameraTarget{0.0f};
  bool hasForcedCamera = false;
  if (options.hasCameraLookAt) {
    forcedCameraPos = options.cameraPos;
    forcedCameraTarget = options.cameraTarget;
    hasForcedCamera = true;
  } else {
    hasForcedCamera = parseCameraLookAtEnv(cameraEnv, forcedCameraPos, forcedCameraTarget);
  }
  glm::vec3 forcedTargetOffset{0.0f};
  bool hasForcedTargetOffset = false;
  if (options.hasTargetOffset) {
    forcedTargetOffset = options.targetOffset;
    hasForcedTargetOffset = true;
  } else {
    hasForcedTargetOffset = parseVec3Env(
      std::getenv("RAYRAI_TCP_VIEWER_CAMERA_OFFSET_FROM_TARGET"), forcedTargetOffset);
  }

  float cameraSpeed = settings.cameraSpeed;
  float lightYawDeg = settings.lightYawDeg;
  float lightPitchDeg = settings.lightPitchDeg;
  float lightStrength = settings.lightStrength;
  float ambientStrength = settings.ambientStrength;
  std::vector<ConnectionEntry> recentConnections = settings.recentConnections;
  if (!options.endpointListPath.empty()) {
    loadEndpointList(options.endpointListPath, recentConnections);
  }
  std::vector<std::string> resourceDirs = settings.resourceDirs;
  for (const auto& dir : options.resourceDirs) {
    recordResourceDir(resourceDirs, dir);
  }
  const bool defaultAutoConnect = options.autoConnectSet ? options.autoConnect :
                                  readEnvBool("RAYRAI_TCP_VIEWER_AUTO_CONNECT", true);
  const bool envMinimizePanels = options.minimizePanelsSet ? options.minimizePanels :
                                 std::getenv("RAYRAI_TCP_VIEWER_MINIMIZE_PANELS") != nullptr;
  const bool envAutoFrame = options.autoFrameSet ? options.autoFrame :
                            std::getenv("RAYRAI_TCP_VIEWER_AUTO_FRAME") != nullptr;
  // ffmpeg is resolved once at startup: a GUI process's PATH does not change
  // while it runs, and probing the filesystem every frame to decide whether a
  // button is greyed out would be wasteful.
  const bool ffmpegAvailable = raisin::tcp_viewer::videoEncodingAvailable();

  // Bring a freshly created pane up to the shared configuration. `primary` marks
  // the pane that owns the command line: only it inherits --connect / --screenshot
  // style options, so a pane opened by a split starts idle instead of silently
  // opening a second connection to the same server.
  auto configurePane = [&](ViewerPane& pane, bool primary) {
    auto& paneViewer = *pane.viewer;
    paneViewer.setBackgroundColorRgb255({20, 20, 30, 255});
    paneViewer.setShowCollisionBodies(false);
    auto& camera = paneViewer.getCamera();
    camera.nearPlane = 0.01f;
    camera.farPlane = 1000.0f;
    camera.zNear = 0.01f;
    camera.zFar = 1000.0f;
    if (hasForcedCamera) {
      applyCameraLookAt(camera, forcedCameraPos, forcedCameraTarget);
    } else {
      const glm::vec3 horizonCameraPos(6.0f, -7.0f, 1.6f);
      const glm::vec3 horizonCameraTarget(0.0f, 0.0f, 1.6f);
      applyCameraLookAt(camera, horizonCameraPos, horizonCameraTarget);
    }
    pane.defaultCameraPos = camera.getPosition();
    pane.defaultCameraTarget = camera.target;

    auto& light = paneViewer.getLight();
    light.type = raisin::LightType::DIRECTIONAL;
    light.ambient = glm::vec3(0.42f, 0.42f, 0.42f);
    light.diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
    light.specular = glm::vec3(0.22f, 0.22f, 0.22f);
    light.setShadowParams(0.0008f, 0.6f, 1.25f);
    light.setShadowsEnabled(true);
    light.direction = lightDirectionFromYawPitch(lightYawDeg, lightPitchDeg);
    applyViewerSettings(paneViewer, settings);
    pane.appliedSettingsSerial = settingsSerial;

    pane.scene = std::make_unique<RemoteScene>(pane.viewer);
    pane.scene->setShowCollisionBodies(false);
    pane.scene->setForceTransparent(false);
    pane.scene->setResourceSearchPaths(resourceDirs);
    pane.sceneFiles = std::make_unique<raisin::tcp_viewer::ViewerSceneFiles>();
    pane.sceneFiles->setResourceDirs(resourceDirs);
    pane.appliedResourceDirSerial = resourceDirSerial;

    setPaneEndpoint(pane, options.host, options.port);
    std::snprintf(pane.screenshotDirBuf, sizeof(pane.screenshotDirBuf), "%s",
                  options.screenshotDir.string().c_str());
    const std::filesystem::path defaultSessionPath =
      options.recordSessionPath.empty()
        ? timestampedDataPath(options.screenshotDir,
            ("rayrai_tcp_viewer_session_pane" + std::to_string(pane.id)).c_str(),
            ".rrtcs")
        : options.recordSessionPath;
    std::snprintf(pane.sessionPathBuf, sizeof(pane.sessionPathBuf), "%s",
                  defaultSessionPath.string().c_str());
    std::snprintf(pane.videoPathBuf, sizeof(pane.videoPathBuf), "%s",
                  timestampedDataPath(options.screenshotDir,
                    ("rayrai_tcp_viewer_video_pane" + std::to_string(pane.id)).c_str(),
                    ".mp4")
                    .string().c_str());
    pane.videoStatus = ffmpegAvailable
      ? std::string()
      : std::string("video recording needs ffmpeg on PATH (or $RAYRAI_FFMPEG)");
    pane.replaySpeed = options.replaySpeed;
    pane.overlayMinimized = envMinimizePanels;
    pane.detailMinimized = envMinimizePanels;
    pane.autoConnect = primary && defaultAutoConnect;
    pane.screenshotAfterFirstScene = primary && !options.screenshotPath.empty();
    pane.pendingScreenshotPath = primary ? options.screenshotPath : std::filesystem::path();
    pane.exportScenePending = primary && !options.exportScenePath.empty();
    pane.startupPending = primary;
  };

  auto createPane = [&](uint32_t paneId) {
    auto pane = createPaneRenderer(paneId);
    configurePane(*pane, false);
    return pane;
  };

  // Snapshot the split arrangement into the settings about to be written. Called
  // from every save path rather than only when the tree changes, so re-pointing a
  // pane at another server is persisted too.
  auto capturePaneSettings = [&]() {
    settings.paneLayout = paneLayout.serialize();
    settings.panePlacements.clear();
    for (uint32_t paneId : paneLayout.panes()) {
      const auto paneIt = panes.find(paneId);
      if (paneIt == panes.end()) continue;
      raisin::tcp_viewer::PanePlacement placement;
      placement.pane = paneId;
      placement.endpoint.host = paneIt->second->host;
      placement.endpoint.port = paneIt->second->port;
      settings.panePlacements.push_back(placement);
    }
  };

  configurePane(primaryPane, true);

  // Restore the other panes of a saved layout, then point each pane at the
  // endpoint it was last on. The primary pane keeps whatever --connect asked for.
  for (uint32_t paneId : paneLayout.panes()) {
    if (paneId == primaryPaneId) continue;
    panes.emplace(paneId, createPane(paneId));
  }
  for (const auto& placement : settings.panePlacements) {
    const auto restored = panes.find(placement.pane);
    if (restored == panes.end()) continue;
    // The command line wins over a saved placement, but only when it named an
    // endpoint; otherwise the first pane comes back where it was left too.
    if (placement.pane == primaryPaneId && options.endpointSet) continue;
    restorePaneConnection(*restored->second, placement.endpoint, defaultAutoConnect);
  }

  // --warm-at-startup: pay the ~13 s of non-shader lazy init up front so that any
  // later drag-drop completes in <50 ms. Off by default — empty-viewer launches
  // (which never need PBR mesh rendering) stay fast.
  if (options.warmAtStartup) {
    static const char* kWarmupUrdf = R"(<?xml version="1.0"?>
<robot name="rayrai_warmup">
  <link name="base">
    <visual><geometry><box size="0.001 0.001 0.001"/></geometry></visual>
    <collision><geometry><box size="0.001 0.001 0.001"/></geometry></collision>
    <inertial><mass value="0.001"/>
      <inertia ixx="1e-9" iyy="1e-9" izz="1e-9" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>)";
    std::cerr << "[rayrai] --warm-at-startup: warming content-frame init (~13s)\n";
    auto* warmupGround = primaryPane.world->addGround();
    raisim::ArticulatedSystem* warmupAs = nullptr;
    try {
      warmupAs = primaryPane.world->addArticulatedSystem(kWarmupUrdf);
      if (warmupAs) warmupAs->setBasePos(raisim::Vec<3>{0.0, 0.0, -1000.0});
    } catch (...) { warmupAs = nullptr; }
    const auto t0 = std::chrono::steady_clock::now();
    primaryPane.viewer->update(options.windowWidth, options.windowHeight, false, 0, 0, true);
    const auto warmMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
    if (warmupAs) primaryPane.world->removeObject(warmupAs);
    if (warmupGround) primaryPane.world->removeObject(warmupGround);
    primaryPane.viewer->updateObjectLists();
    std::cerr << "[rayrai] content-frame warmup: " << warmMs << " ms\n";
  }

  // Renderer warmup is now scoped to *shader* pre-compile only, which is cheap
  // (~40 ms with the binary cache hit) and fits comfortably in startup. The other
  // content-frame init (IBL convolution / texture pool / FBOs / etc.) is NOT
  // covered by the shader binary cache — it's per-launch state that takes ~13 s.
  // We don't want to pay that at every startup; instead it fires inside the first
  // load (drag-drop OR --inspect FILE) as part of that explicit user action, with
  // a clearly-visible status message so the wait is attributed to the load.
  static const std::vector<std::string> kViewerWarmupShaders = {
    "pbrMeshHigh",
  };
  size_t shaderWarmupIdx = 0;
  bool shaderWarmupActive = options.preWarmShaders;
  const auto shaderWarmupStart = std::chrono::steady_clock::now();
  if (shaderWarmupActive) {
    std::cerr << "[rayrai] background shader warmup started ("
              << kViewerWarmupShaders.size() << " targeted shader"
              << (kViewerWarmupShaders.size() == 1 ? "" : "s") << ")\n";
  }


  int frameSerial = 0;
  bool quit = false;
  int viewerExitCode = 0;
  // Path of a file dropped on the window, handed to the focused pane below.
  std::string pendingDropPath;
  float uiScale = settings.uiScale;
  float defaultUiScale = 1.0f;
  bool uiScaleInitialized = false;
  bool uiScaleUserSet = settings.uiScaleUserSet;
  float appliedUiScale = 0.0f;
  bool baseStyleCaptured = false;
  ImGuiStyle baseStyle;
  ImVec2 lastDisplaySize(0.0f, 0.0f);
  bool settingsDirty = !options.resourceDirs.empty() || options.updateRateHz > 0.0f;
  bool settingsSavePending = false;
  auto lastSettingsDirtyTime = std::chrono::steady_clock::now();
  // One discovery listener serves every pane: the beacons describe the network,
  // not the pane, and a second socket on the same port would just fight for them.
  DiscoveryBeaconReceiver beaconReceiver;
  std::string discoveryStatus;
  beaconReceiver.start(discoveryStatus);
  std::vector<ServerEntry> discoveredServers =
    serverEntriesFromDiscovered(beaconReceiver.servers());
  auto nextDiscoveryRefresh = std::chrono::steady_clock::now() + kDiscoveryRefreshInterval;
  const auto steadyStart = std::chrono::steady_clock::now();
  // Set when the split tree changes, so the arrangement is written back to the
  // settings file along with each pane's endpoint.
  bool paneLayoutDirty = false;
  // Work that is per-process rather than per-pane (frame counter, discovery poll,
  // background shader warmup) runs on the first pane in layout order. It is not
  // pinned to the pane that owned the command line, because that pane can be
  // closed like any other.
  uint32_t leadPaneId = primaryPaneId;
  const bool logExitFps = readEnvBool("RAYRAI_TCP_VIEWER_LOG_EXIT_FPS", false);
  const auto toggleFullscreenDesktop = [window]() {
    const Uint32 flags = SDL_GetWindowFlags(window);
    const bool isFullscreen = (flags & SDL_WINDOW_FULLSCREEN_DESKTOP) != 0;
    SDL_SetWindowFullscreen(window, isFullscreen ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP);
  };
  const auto fpsMeasureStart = std::chrono::steady_clock::now();
  uint64_t fpsMeasureFrames = 0;

  // Run one pane for one frame: connection upkeep, scene mirroring, rendering
  // into the pane rectangle, and that pane's own copy of the overlay panels.
  auto paneFrame = [&](ViewerPane& pane, const raisin::tcp_viewer::PaneRect& paneRect,
                       bool focused) {
    // The pane's own state, bound to the names the frame body has always used.
    // Everything not aliased here (settings, uiScale, recent connections,
    // discovery, quit) is shared across panes and captured by reference.
    auto& scene = *pane.scene;
    auto& sceneFiles = *pane.sceneFiles;
    auto& world = pane.world;
    auto& viewer = pane.viewer;
    auto& client = pane.client;
    auto& sensorRenderer = pane.sensorRenderer;
    auto& actuatorTraces = pane.actuatorTraces;
    auto& localSimulation = pane.localSimulation;
    auto& connectingLocalSimulation = pane.connectingLocalSimulation;
    auto& autoConnect = pane.autoConnect;
    auto& everConnected = pane.everConnected;
    auto& sceneReceived = pane.sceneReceived;
    auto& awaitingResponse = pane.awaitingResponse;
    auto& awaitingSensorAck = pane.awaitingSensorAck;
    auto& nextSensorUpdateIndex = pane.nextSensorUpdateIndex;
    auto& updateRequestSentAt = pane.updateRequestSentAt;
    auto& nextAutoConnectAttempt = pane.nextAutoConnectAttempt;
    auto& nextTcpUpdateRequestTime = pane.nextTcpUpdateRequestTime;
    auto& tcpPayload = pane.tcpPayload;
    auto& host = pane.host;
    auto& port = pane.port;
    auto& portBuf = pane.portBuf;
    auto& lastStatus = pane.lastStatus;
    auto& verboseParsing = pane.verboseParsing;
    auto& stats = pane.stats;
    auto& defaultCameraPos = pane.defaultCameraPos;
    auto& defaultCameraTarget = pane.defaultCameraTarget;
    auto& requestFrameScene = pane.requestFrameScene;
    auto& requestFrameSelected = pane.requestFrameSelected;
    auto& requestResetCamera = pane.requestResetCamera;
    auto& autoFrameApplied = pane.autoFrameApplied;
    auto& cameraBookmarks = pane.cameraBookmarks;
    auto& cameraFrustums = pane.cameraFrustums;
    auto& showCollisionBodies = pane.showCollisionBodies;
    auto& showWorldFrame = pane.showWorldFrame;
    auto& showContactPoints = pane.showContactPoints;
    auto& showContactForces = pane.showContactForces;
    auto& contactForceAbsolute = pane.contactForceAbsolute;
    auto& forceTransparent = pane.forceTransparent;
    auto& showBodyFrames = pane.showBodyFrames;
    auto& showComMarkers = pane.showComMarkers;
    auto& contactPointSize = pane.contactPointSize;
    auto& contactForceSize = pane.contactForceSize;
    auto& bodyFrameSize = pane.bodyFrameSize;
    auto& comMarkerSize = pane.comMarkerSize;
    auto& worldFrame = pane.worldFrame;
    auto& bodyFramesNode = pane.bodyFramesNode;
    auto& comMarkers = pane.comMarkers;
    auto& requestedTag = pane.requestedTag;
    auto& requestedIndex = pane.requestedIndex;
    auto& requestedEntry = pane.requestedEntry;
    auto& motionEstimates = pane.motionEstimates;
    auto& pendingControlRequests = pane.pendingControlRequests;
    auto& simPaused = pane.simPaused;
    auto& controlForce = pane.controlForce;
    auto& controlTorque = pane.controlTorque;
    auto& controlPointOffset = pane.controlPointOffset;
    auto& controlBodyIdx = pane.controlBodyIdx;
    auto& controlBodyFollowsSelection = pane.controlBodyFollowsSelection;
    auto& controlSelectionTag = pane.controlSelectionTag;
    auto& controlSelectionIndex = pane.controlSelectionIndex;
    auto& controlPosePosition = pane.controlPosePosition;
    auto& controlPoseQuat = pane.controlPoseQuat;
    auto& controlPoseTag = pane.controlPoseTag;
    auto& controlPoseInitialized = pane.controlPoseInitialized;
    auto& controlGc = pane.controlGc;
    auto& controlGcTag = pane.controlGcTag;
    auto& controlGcDirty = pane.controlGcDirty;
    auto& viewportState = pane.viewportState;
    auto& mouseForceEnabled = pane.mouseForceEnabled;
    auto& mouseForceScale = pane.mouseForceScale;
    auto& mouseForce = pane.mouseForce;
    auto& wireDragEnabled = pane.wireDragEnabled;
    auto& wireDragStiffness = pane.wireDragStiffness;
    auto& wireDrag = pane.wireDrag;
    auto& ruler = pane.ruler;
    auto& angle = pane.angle;
    auto& poseGrabber = pane.poseGrabber;
    auto& spawnForm = pane.spawnForm;
    auto& spawnStatus = pane.spawnStatus;
    auto& worldExportPathBuf = pane.worldExportPathBuf;
    auto& inspector = pane.inspector;
    auto& fileBrowser = pane.fileBrowser;
    auto& pathFieldEditing = pane.pathFieldEditing;
    auto& overlayOffset = pane.overlayOffset;
    auto& detailOffset = pane.detailOffset;
    auto& overlayMinimized = pane.overlayMinimized;
    auto& detailMinimized = pane.detailMinimized;
    auto& overlayCollapsedHoveredLastFrame = pane.overlayCollapsedHoveredLastFrame;
    auto& overlayLastInteractionTime = pane.overlayLastInteractionTime;
    auto& groupObjectsByType = pane.groupObjectsByType;
    auto& hideCollisionObjects = pane.hideCollisionObjects;
    auto& objectSortMode = pane.objectSortMode;
    auto& objectFilterBuf = pane.objectFilterBuf;
    auto& screenshotDirBuf = pane.screenshotDirBuf;
    auto& pendingScreenshotPath = pane.pendingScreenshotPath;
    auto& screenshotAfterFirstScene = pane.screenshotAfterFirstScene;
    auto& screenshotRequested = pane.screenshotRequested;
    auto& recordPngSequence = pane.recordPngSequence;
    auto& recordEveryNFrames = pane.recordEveryNFrames;
    auto& recordFrameIndex = pane.recordFrameIndex;
    auto& recordFramePrefix = pane.recordFramePrefix;
    auto& serverRecordFrameDirectory = pane.serverRecordFrameDirectory;
    auto& serverRequestedRecording = pane.serverRequestedRecording;
    auto& captureStatus = pane.captureStatus;
    auto& captureRgba = pane.captureRgba;
    auto& videoEncoder = pane.videoEncoder;
    auto& videoFramesPerSecond = pane.videoFramesPerSecond;
    auto& videoQuality = pane.videoQuality;
    auto& videoPathBuf = pane.videoPathBuf;
    auto& videoStatus = pane.videoStatus;
    auto& sessionPathBuf = pane.sessionPathBuf;
    auto& sessionStatus = pane.sessionStatus;
    auto& sessionRecorder = pane.sessionRecorder;
    auto& replayFrames = pane.replayFrames;
    auto& replayMode = pane.replayMode;
    auto& replayPaused = pane.replayPaused;
    auto& replayStep = pane.replayStep;
    auto& replayIndex = pane.replayIndex;
    auto& replaySeekIndex = pane.replaySeekIndex;
    auto& replaySpeed = pane.replaySpeed;
    auto& replayStart = pane.replayStart;
    auto& replayBaseMicros = pane.replayBaseMicros;
    auto& trajectoryCsv = pane.trajectoryCsv;
    auto& packetSamples = pane.packetSamples;
    auto& diagnosticsPresentation = pane.diagnosticsPresentation;
    auto& assetDiagnostics = pane.assetDiagnostics;
    auto& assetDiagnosticsDirty = pane.assetDiagnosticsDirty;
    auto& exportScenePending = pane.exportScenePending;
    auto& sensorStatus = pane.sensorStatus;
    auto& signalRecords = pane.signalRecords;
    auto& signalPlotChannels = pane.signalPlotChannels;
    auto& signalPlotChannelsInitialized = pane.signalPlotChannelsInitialized;
    auto& pinnedSignalObjects = pane.pinnedSignalObjects;
    auto& selectedSignalKey = pane.selectedSignalKey;
    auto& signalExportStatus = pane.signalExportStatus;
    auto& light = viewer->getLight();
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 paneOrigin(paneRect.x, paneRect.y);
    const ImVec2 uiSize(paneRect.width, paneRect.height);
    // ImGui keys window state by name, so every pane needs its own names or all
    // panes would share one collapsed/scrolled/positioned panel.
    const std::string paneWindowSuffix = "##pane" + std::to_string(pane.id);
    const std::string overlayWindowName = "Raisim TCP##Overlay" + paneWindowSuffix;
    const std::string detailWindowName = "Selected Object##Overlay" + paneWindowSuffix;
    const std::string inspectorWindowName = "Raisim Inspector##Overlay" + paneWindowSuffix;

  auto closeInspector = [&]() {
    if (!inspector.active) return;
    // Drop any selection that points at our AS first, so the renderer doesn't keep
    // a dangling pointer when it next iterates targetVisual_ against the world list.
    viewer->setTargetVisual(nullptr);
    // Remove the AS first so any contact/constraint references on side objects clear.
    if (inspector.as) {
      world->removeObject(inspector.as);
    }
    // MJCF can also add ground planes, lights, mocap bodies, etc. Tear them down too.
    for (raisim::Object* ob : inspector.sideObjects) {
      if (ob) world->removeObject(ob);
    }
    // Critical: tell the renderer its objectList_ cache is now stale. updateWeather()
    // and other per-frame helpers walk that cache directly and would dereference the
    // freed raisim::Object pointers we just deleted. Without this, the next frame
    // segfaults inside RaisimObject::configureGroundMaterial (or any other place that
    // touches the wrapper's stored raisim::Object*).
    viewer->updateObjectLists();
    inspector = InspectorState{};
    lastStatus = "disconnected";
  };
  auto loadAsInspector = [&](const std::string& path) -> bool {
    namespace fs = std::filesystem;
    inspector.lastError.clear();
    if (path.empty() || !fs::exists(path)) {
      inspector.lastError = "file not found: " + path;
      return false;
    }

    // Close any previously-loaded inspector model before replacing it.
    closeInspector();

    const auto tStart = std::chrono::steady_clock::now();
    const bool isMjcf = looksLikeMjcf(path);

    // Snapshot the object list so we can attribute any new objects to this load.
    std::vector<raisim::Object*> before(world->getObjList().begin(), world->getObjList().end());

    raisim::ArticulatedSystem* as = nullptr;
    try {
      if (isMjcf) {
        // loadMjcfFile populates the world; we pick the first ArticulatedSystem it added.
        world->loadMjcfFile(path);
      } else {
        as = world->addArticulatedSystem(path);
      }
    } catch (const std::exception& e) {
      inspector.lastError = std::string("load failed: ") + e.what();
      return false;
    } catch (...) {
      inspector.lastError = "load failed (unknown exception)";
      return false;
    }

    // Diff before/after to collect the new objects.
    std::vector<raisim::Object*> newObjects;
    {
      std::unordered_set<raisim::Object*> beforeSet(before.begin(), before.end());
      for (raisim::Object* ob : world->getObjList()) {
        if (beforeSet.find(ob) == beforeSet.end()) newObjects.push_back(ob);
      }
    }
    if (isMjcf) {
      for (raisim::Object* ob : newObjects) {
        if (!as && ob->getObjectType() == raisim::ObjectType::ARTICULATED_SYSTEM) {
          as = dynamic_cast<raisim::ArticulatedSystem*>(ob);
        }
      }
    }
    if (!as) {
      // No AS produced — tear down whatever was added so the world stays clean.
      for (raisim::Object* ob : newObjects) {
        if (ob) world->removeObject(ob);
      }
      inspector.lastError = isMjcf ? "MJCF contained no articulated system"
                                   : "URDF parse failed (see console)";
      return false;
    }

    inspector = InspectorState{};
    inspector.active = true;
    inspector.sourceFile = path;
    inspector.as = as;
    for (raisim::Object* ob : newObjects) {
      if (ob != as) inspector.sideObjects.push_back(ob);
    }
    inspector.as->setName("inspector_" + fs::path(path).stem().string());

    const int gcDim = inspector.as->getGeneralizedCoordinateDim();
    inspector.gc.assign(gcDim, 0.0);
    {
      const raisim::VecDyn& q = inspector.as->getGeneralizedCoordinate();
      const int copyN = std::min<int>(gcDim, static_cast<int>(q.size()));
      for (int i = 0; i < copyN; ++i) inspector.gc[i] = q[i];
    }

    // Walk all joints (including FIXED ones at the root) so we get the correct GC offset
    // for each movable joint. movableJointNames is in joint order but skips FIXED joints;
    // we re-pair them with their owning joint index using the same trick the server uses
    // when the first joint is a FIXED ground attachment.
    const auto& movableNames = inspector.as->getMovableJointNames();
    const auto& jointLimits = inspector.as->getJointLimits();
    const size_t numJoints = inspector.as->getNumberOfJoints();
    int gcCursor = 0;
    size_t movableNameIdx = 0;
    for (size_t j = 0; j < numJoints; ++j) {
      const raisim::Joint::Type type = inspector.as->getJointType(j);
      int dofThisJoint = 0;
      switch (type) {
        case raisim::Joint::Type::FIXED: dofThisJoint = 0; break;
        case raisim::Joint::Type::REVOLUTE:
        case raisim::Joint::Type::PRISMATIC: dofThisJoint = 1; break;
        case raisim::Joint::Type::SPHERICAL: dofThisJoint = 4; break;
        case raisim::Joint::Type::FLOATING: dofThisJoint = 7; break;
      }
      if (type == raisim::Joint::Type::FIXED) {
        continue; // FIXED joints don't appear in movableJointNames and consume no GC.
      }
      InspectorJoint info;
      info.type = type;
      info.gcOffset = gcCursor;
      info.gcDim = dofThisJoint;
      info.name = (movableNameIdx < movableNames.size())
                      ? movableNames[movableNameIdx++]
                      : ("joint_" + std::to_string(j));
      if (j < jointLimits.size() && info.gcDim == 1) {
        info.minLimit = jointLimits[j][0];
        info.maxLimit = jointLimits[j][1];
        info.hasLimits = (info.maxLimit > info.minLimit);
      }
      inspector.joints.push_back(std::move(info));
      gcCursor += dofThisJoint;
    }

    const auto tEnd = std::chrono::steady_clock::now();
    const auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
    awaitingResponse = false;
    awaitingSensorAck = false;
    lastStatus = "inspector: " + fs::path(path).filename().string() +
                 " (" + std::to_string(loadMs) + " ms)";
    // Frame the camera on the loaded AS directly. We can't rely on requestFrameScene
    // here because that path goes through RemoteScene::computeSceneBounds, which is
    // empty in inspector mode (the AS lives in the local raisim::World, not RemoteScene).
    {
      raisim::Vec<3> centerVec;
      try {
        centerVec = inspector.as->getCOM();
      } catch (...) {
        centerVec = {0.0, 0.0, 0.5};
      }
      const glm::vec3 center(static_cast<float>(centerVec[0]),
                             static_cast<float>(centerVec[1]),
                             static_cast<float>(centerVec[2]));
      const glm::vec3 halfExtent(1.0f, 1.0f, 0.8f);
      frameBounds(*viewer, center - halfExtent, center + halfExtent);
    }
    std::cerr << "[inspector] loaded " << path
              << " in " << loadMs << " ms"
              << " (gcDim=" << inspector.gc.size()
              << " joints=" << inspector.joints.size() << ")\n";
    return true;
  };

  auto clearSceneState = [&]() {
    if (sessionRecorder.active() && !replayMode) {
      sessionRecorder.recordReset(std::chrono::steady_clock::now(), sessionStatus);
    }
    viewer->setTargetVisual(nullptr);
    requestedTag = 0;
    requestedIndex = 0;
    requestedEntry = nullptr;
    mouseForce = {};
    poseGrabber = {};
    pendingControlRequests.clear();
    controlSelectionTag = 0;
    controlSelectionIndex = -1;
    controlPoseTag = 0;
    controlPoseInitialized = false;
    controlGc.clear();
    controlGcTag = 0;
    controlGcDirty = false;
    comMarkers.clear();
    if (bodyFramesNode) {
      bodyFramesNode->poses.clear();
      bodyFramesNode->enable(false);
    }
    sceneFiles.reset(*viewer, scene);
    clearRemoteSceneAndFrustums(*viewer, scene, cameraFrustums);
    sensorRenderer.clear();
    sceneReceived = false;
    motionEstimates.clear();
    signalRecords.clear();
    pinnedSignalObjects.clear();
    selectedSignalKey = 0;
    assetDiagnostics.clear();
    assetDiagnosticsDirty = true;
    stats.pendingSensorRequests = 0;
    stats.unresolvedAssets = 0;
  };
  auto connectToEndpoint = [&](const ConnectionEntry& endpoint, bool verbose,
                               const char* connectingStatus, const char* failureStatus,
                               int timeoutMs = kManualConnectTimeoutMs,
                               raisin::tcp_viewer::ConnectionTarget target =
                                   raisin::tcp_viewer::ConnectionTarget::ExternalServer) -> bool {
    if (inspector.active) {
      lastStatus = "close inspector to connect";
      return false;
    }
    lastStatus = connectingStatus;
    if (target == raisin::tcp_viewer::ConnectionTarget::ExternalServer && localSimulation.active()) {
      clearSceneState();
      awaitingResponse = awaitingSensorAck = false;
    }
    if (raisin::tcp_viewer::connectViewerEndpoint(client, localSimulation, connectingLocalSimulation,
        target, endpoint.host, endpoint.port, verbose, timeoutMs)) {
      lastStatus = "waiting for scene";
      awaitingResponse = false;
      awaitingSensorAck = false;
      everConnected = true;
      setPaneEndpoint(pane, endpoint.host, endpoint.port);
      if (!localSimulation.active()) {
        recordConnection(recentConnections, endpoint.host, endpoint.port);
        settingsDirty = true;
      }
      nextTcpUpdateRequestTime = std::chrono::steady_clock::now();
      stats.reconnects++;
      return true;
    }
    lastStatus = failureStatus;
    return false;
  };

  auto loadDroppedScene = [&](const std::string& path) -> bool {
    std::string error;
    const auto kind = raisin::tcp_viewer::classifyDroppedScene(path, error);
    if (kind == raisin::tcp_viewer::DroppedSceneKind::Invalid) {
      lastStatus = error;
      std::cerr << "Drop failed: " << error << '\n';
      return false;
    }
    if (kind == raisin::tcp_viewer::DroppedSceneKind::World) {
      if (!localSimulation.start(binaryPath, path, options.activationKey, error)) {
        lastStatus = error; return false;
      }
      client.disconnect();
      awaitingResponse = awaitingSensorAck = false;
      closeInspector();
      clearSceneState();
      replayMode = false;
      autoConnect = false;
      connectingLocalSimulation = true;
      const auto directory = std::filesystem::absolute(path).parent_path();
      scene.addSearchPath(directory.string());
      scene.addSearchPath((directory / "assets").string());
      autoFrameApplied = false;
      nextAutoConnectAttempt = std::chrono::steady_clock::now();
      lastStatus = "Starting simulation: " + std::filesystem::path(path).filename().string();
      return true;
    }
    if (client.isConnected() && !localSimulation.active()) {
      lastStatus = "Disconnect before opening a robot inspector"; return false;
    }
    if (localSimulation.active()) {
      client.disconnect(); localSimulation.stop(); clearSceneState();
      awaitingResponse = awaitingSensorAck = false;
    }
    connectingLocalSimulation = false;
    autoConnect = false;
    const bool loaded = loadAsInspector(path);
    if (!loaded) lastStatus = "Drop failed: " + inspector.lastError;
    return loaded;
  };

  auto applyScenePayload = [&](const std::vector<char>& payload, bool fromReplay,
                              std::chrono::steady_clock::time_point sampleNow,
                              std::vector<PendingSensorUpdate>& pending) -> bool {
    BufferReader reader(payload);
    scene.setVerbose(verboseParsing);
    const bool parsedOk = scene.applyResponse(reader, pending);
    if (!fromReplay) {
      for (const auto& command : scene.takeViewerCommands()) {
        switch (command.type) {
          case raisin::tcp_viewer::ViewerCommandType::StartRecording: {
            if (!raisin::tcp_viewer::canStartServerRecording(
                    videoEncoder.isOpen(), recordPngSequence,
                    serverRequestedRecording)) {
              captureStatus = "server recording ignored while a user recording is active";
              break;
            }
            const std::filesystem::path requested(command.path);
            recordFramePrefix = requested.stem().empty()
              ? "rayrai_tcp_viewer_video"
              : requested.stem().string();
            recordFrameIndex = 0;
            serverRequestedRecording = true;
            // startRecordingVideo() names a video file, so honour that when an
            // encoder is available and fall back to the PNG sequence otherwise.
            raisin::tcp_viewer::VideoEncoderSettings encoderSettings;
            encoderSettings.width = viewer->getCamera().rtWidth();
            encoderSettings.height = viewer->getCamera().rtHeight();
            encoderSettings.framesPerSecond = videoFramesPerSecond;
            encoderSettings.quality = videoQuality;
            const std::filesystem::path videoOutput =
              std::filesystem::path(screenshotDirBuf) /
              (recordFramePrefix + (requested.extension().empty()
                                      ? std::string(".mp4")
                                      : requested.extension().string()));
            if (ffmpegAvailable &&
                videoEncoder.open(videoOutput, encoderSettings, videoStatus)) {
              recordPngSequence = false;
              serverRecordFrameDirectory.clear();
              captureStatus = "server recording video: " + videoOutput.string();
            } else {
              serverRecordFrameDirectory = std::filesystem::path(screenshotDirBuf) /
                                           (recordFramePrefix + "_frames");
              recordPngSequence = true;
              captureStatus = "server recording PNG sequence: " +
                              serverRecordFrameDirectory.string();
            }
          } break;
          case raisin::tcp_viewer::ViewerCommandType::StopRecording:
            if (serverRequestedRecording) {
              const bool wasVideo = videoEncoder.isOpen();
              if (wasVideo) {
                videoEncoder.close(videoStatus);
              }
              recordPngSequence = false;
              serverRequestedRecording = false;
              captureStatus = wasVideo
                ? "server recording stopped: " + videoStatus
                : "server recording stopped after " + std::to_string(recordFrameIndex) +
                    " frame(s)";
            }
            break;
          case raisin::tcp_viewer::ViewerCommandType::Screenshot:
            screenshotRequested = true;
            break;
          case raisin::tcp_viewer::ViewerCommandType::SetWindowSize:
            if (command.width > 0 && command.height > 0) {
              SDL_SetWindowSize(window, std::clamp(command.width, 160, 8192),
                                std::clamp(command.height, 120, 8192));
            }
            break;
        }
      }
      sceneFiles.receive(scene.takeSceneFileBlock());
    } else {
      static_cast<void>(scene.takeViewerCommands());
    }
    const bool disconnectRequested = scene.consumeDisconnectRequested();
    bool ok = parsedOk && !disconnectRequested;
    if (disconnectRequested) {
      lastStatus = fromReplay ? "replay protocol disconnect" : "server disconnected";
    } else if (!parsedOk) {
      lastStatus = fromReplay ? "replay parse error" : "parse error (dropped update)";
      stats.parseErrors++;
    } else {
      lastStatus = fromReplay ? "replay" : "connected";
      // A scene the server shares is shown first.
      if (screenshotAfterFirstScene && sceneFiles.settled(*viewer)) {
        screenshotRequested = true;
        screenshotAfterFirstScene = false;
      }
      if ((envAutoFrame || localSimulation.active()) && !autoFrameApplied && frameScene(*viewer, scene)) {
        autoFrameApplied = true;
      }
    }

    if (ok) {
      sceneReceived = true;
      if (hasForcedTargetOffset && viewer->getTargetVisual()) {
        const glm::vec3 target = viewer->getTargetVisual()->getPosition();
        applyCameraLookAt(viewer->getCamera(), target + forcedTargetOffset, target);
      } else if (hasForcedTargetOffset) {
        if (raisin::Visuals* followTarget = chooseDefaultFollowTarget(scene)) {
          viewer->setTargetVisual(followTarget);
          const glm::vec3 target = followTarget->getPosition();
          applyCameraLookAt(viewer->getCamera(), target + forcedTargetOffset, target);
        }
      } else if (forceCameraEnv && hasForcedCamera) {
        viewer->setTargetVisual(nullptr);
        applyCameraLookAt(viewer->getCamera(), forcedCameraPos, forcedCameraTarget);
      }
    }

    assetDiagnosticsDirty = true;
    stats.pendingSensorRequests = static_cast<int>(pending.size());
    stats.unresolvedAssets = unresolvedAssetCount(scene);
    const auto selectableObjects = scene.getSelectableObjects();
    const double sampleTime = std::chrono::duration<double>(sampleNow - steadyStart).count();
    PacketSample sample;
    sample.timeSeconds = sampleTime;
    sample.bytes = static_cast<int>(payload.size());
    sample.parsed = ok;
    sample.replay = fromReplay;
    sample.pendingSensors = static_cast<int>(pending.size());
    sample.objects = selectableObjects.size();
    sample.visuals = scene.visualCount();
    sample.instanced = scene.instancedCount();
    sample.pointClouds = scene.pointCloudCount();
    sample.unresolvedAssets = stats.unresolvedAssets;
    sample.roundTripMs = fromReplay ? 0.0 : stats.lastRoundTripMs;
    pushPacketSample(packetSamples, sample);
    if (ok) {
      const double worldTime = scene.hasServerWorldTime() ? scene.getServerWorldTime() : sampleTime;
      if (trajectoryCsv) {
        writeTrajectoryRows(trajectoryCsv, scene, worldTime);
      }
      for (const auto& item : selectableObjects) {
        if (isContactItem(item) || !item.visual) {
          continue;
        }
        uint32_t motionTag = 0;
        int motionIndex = 0;
        const VisualEntry* motionEntry = nullptr;
        if (scene.getVisualInfo(item.visual.get(), motionTag, motionIndex, motionEntry) && motionEntry) {
          updateMotionEstimate(motionEstimates[visualMotionKey(motionTag, motionIndex)], *motionEntry, worldTime);
        }
      }
    }
    return ok;
  };

    // Command-line startup work (replay, recording, --simulation, --inspect) runs
    // once, on the pane that owns the command line. It lives here because it
    // drives the same per-pane lambdas the UI does.
    if (pane.startupPending) {
      pane.startupPending = false;
      if (!options.replaySessionPath.empty()) {
        replayMode = loadSessionFile(options.replaySessionPath, replayFrames, sessionStatus);
        replayPaused = false;
        replayStart = std::chrono::steady_clock::now();
        replayBaseMicros = replayFrames.empty() ? 0 : replayFrames.front().timeMicros;
        autoConnect = false;
        lastStatus = replayMode ? "replay" : "replay load failed";
      }
      if (!options.recordSessionPath.empty()) {
        sessionRecorder.open(options.recordSessionPath, sessionStatus);
      }
      if (!options.trajectoryCsvPath.empty()) {
        std::error_code ec;
        if (!options.trajectoryCsvPath.parent_path().empty()) {
          std::filesystem::create_directories(options.trajectoryCsvPath.parent_path(), ec);
        }
        trajectoryCsv.open(options.trajectoryCsvPath);
        if (trajectoryCsv) {
          trajectoryCsv << "time,tag,index,name,type,x,y,z,qw,qx,qy,qz\n";
        } else {
          std::cerr << "WARN: failed to open trajectory CSV " << options.trajectoryCsvPath << "\n";
        }
      }
      light.direction = lightDirectionFromYawPitch(lightYawDeg, lightPitchDeg);
      if (!options.simulationPath.empty()) {
        std::string error;
        if (raisin::tcp_viewer::classifyDroppedScene(options.simulationPath, error) !=
            raisin::tcp_viewer::DroppedSceneKind::World) {
          std::cerr << "Simulation launch failed: " << (error.empty() ? "Expected a RaiSim world XML" : error) << '\n';
          quit = true;
          viewerExitCode = 1;
        } else {
          // Share the drop handler, not SDL's platform-owned drop-event transport.
          // SDL2 compatibility layers can retain/convert that transport's payload;
          // synthesizing it here breaks ownership during later event filtering.
          if (!loadDroppedScene(options.simulationPath.string())) {
            quit = true; viewerExitCode = 1;
          }
        }
      }

      // --inspect FILE was passed on the CLI: load the model as if drag-dropped. When
      // --inspect-after-frames N is also set, defer the load until the main loop has
      // ticked N frames (lets the headless harness measure drag-drop latency after
      // the background shader warmup has run).
      if (!options.inspectorPath.empty() && options.inspectAfterFrames < 0) {
        if (!loadAsInspector(options.inspectorPath.string())) {
          std::cerr << "ERROR: --inspect failed: " << inspector.lastError << "\n";
        }
      }
    }

    // A file dropped on the window is handed to the focused pane, which owns the
    // scene it would load.
    if (focused && !pendingDropPath.empty()) {
      const std::string droppedPath = pendingDropPath;
      pendingDropPath.clear();
      if (!loadDroppedScene(droppedPath) && !options.simulationPath.empty() && !everConnected) {
        quit = true;
        viewerExitCode = 1;
      }
    }

    // Viewport shortcuts belong to the pane the pointer last selected; without
    // the focus test every pane would reset its camera on one press of R.
    if (focused && !io.WantTextInput) {
      if (ImGui::IsKeyPressed(ImGuiKey_F, false)) requestFrameScene = true;
      if (ImGui::IsKeyPressed(ImGuiKey_C, false)) requestFrameSelected = true;
      if (ImGui::IsKeyPressed(ImGuiKey_R, false)) requestResetCamera = true;
      if (ImGui::IsKeyPressed(ImGuiKey_M, false)) {
        // Cycle: off -> ruler -> angle -> off. 'A' would collide with the
        // camera's WASD strafe, so the angle tool reuses M's second press.
        if (!ruler.enabled && !angle.enabled) {
          ruler.enabled = true;
          lastStatus = "ruler enabled (M again for angle)";
        } else if (ruler.enabled) {
          ruler.enabled = false;
          angle.enabled = true;
          angle.picked = 0;
          lastStatus = "angle tool: pick A";
        } else {
          angle.enabled = false;
          angle.picked = 0;
          lastStatus = "measure tool disabled";
        }
      }
      if (ImGui::IsKeyPressed(ImGuiKey_G, false)) {
        poseGrabber.enabled = !poseGrabber.enabled;
        if (!poseGrabber.enabled) {
          poseGrabber.dragging = false;
          poseGrabber.axis = -1;
        }
        lastStatus = poseGrabber.enabled ? "pose grabber enabled" : "pose grabber disabled";
      }
      if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        const Uint32 flags = SDL_GetWindowFlags(window);
        const bool isFullscreen = (flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP)) != 0;
        if (isFullscreen) {
          SDL_SetWindowFullscreen(window, 0);
        }
        if (ruler.enabled) {
          ruler.enabled = false;
          lastStatus = "ruler disabled";
        }
        if (angle.enabled) {
          angle.enabled = false;
          angle.picked = 0;
          lastStatus = "angle tool disabled";
        }
        if (poseGrabber.dragging || poseGrabber.enabled) {
          poseGrabber.enabled = false;
          poseGrabber.dragging = false;
          poseGrabber.axis = -1;
        }
      }
      if (ImGui::IsKeyPressed(ImGuiKey_F12, false)) screenshotRequested = true;
      if (ImGui::IsKeyPressed(ImGuiKey_F11, false)) toggleFullscreenDesktop();
    }

    constexpr float menuBarHeight = 0.0f;

    const auto now = std::chrono::steady_clock::now();
    const bool localWasActive = localSimulation.active();
    localSimulation.poll();
    if (localWasActive && !localSimulation.active()) {
      client.disconnect(); clearSceneState();
      awaitingResponse = awaitingSensorAck = false;
      connectingLocalSimulation = false;
      lastStatus = "Simulation stopped: " + localSimulation.error();
      std::cerr << lastStatus << '\n';
      if (!options.simulationPath.empty() && !everConnected) { quit = true; viewerExitCode = 1; }
    }
    if (connectingLocalSimulation && localSimulation.state() == raisin::tcp_viewer::LocalSimulation::State::Running &&
        now >= nextAutoConnectAttempt) {
      connectToEndpoint({"127.0.0.1", localSimulation.port()}, false,
          "Connecting to local simulation", "Waiting for local simulation", kAutoConnectTimeoutMs,
          raisin::tcp_viewer::ConnectionTarget::LocalSimulation);
      nextAutoConnectAttempt = now + std::chrono::milliseconds(100);
    }
    const double wallElapsed = std::chrono::duration<double>(now - steadyStart).count();
    if (options.exitAfterSeconds > 0.0 && wallElapsed >= options.exitAfterSeconds) {
      quit = true;
    }
    // --wait-for-server is about the session the command line asked for. Judging
    // every pane by it would let an idle pane of a restored split layout, which
    // is never expected to connect on its own, close the whole viewer.
    if (pane.id == primaryPaneId &&
        shouldQuitForInitialServerWait(options.waitForServerSeconds, replayMode,
        client.isConnected(), everConnected, wallElapsed)) {
      lastStatus = "wait-for-server timed out";
      std::cerr << "rayrai_tcp_viewer: no server connected within " << options.waitForServerSeconds
                << " s (--wait-for-server)\n";
      quit = true;
      viewerExitCode = 1;
    }
    if (pane.id == leadPaneId) {
      // poll() drains newly arrived beacons and drops ones that stopped
      // announcing; the timer re-lists on a fixed beat so the panel keeps up
      // without anyone asking it to.
      const bool beaconsChanged = beaconReceiver.poll();
      if (beaconsChanged || now >= nextDiscoveryRefresh) {
        discoveredServers = serverEntriesFromDiscovered(beaconReceiver.servers());
        nextDiscoveryRefresh = now + kDiscoveryRefreshInterval;
      }
    }
    if (replayMode && !replayFrames.empty() &&
        replaySeekIndex != std::numeric_limits<size_t>::max()) {
      const size_t target = std::min(replaySeekIndex, replayFrames.size() - 1u);
      replaySeekIndex = std::numeric_limits<size_t>::max();
      clearSceneState();
      size_t appliedCount = 0;
      for (size_t i = 0; i <= target; ++i) {
        if (!raisin::tcp_viewer::applyRecordedFrame(replayFrames[i], clearSceneState,
            [&](const std::vector<char>& payload) {
              std::vector<PendingSensorUpdate> ignoredSensors;
              return applyScenePayload(payload, true, now, ignoredSensors);
            })) {
          break;
        }
        appliedCount = i + 1u;
      }
      replayIndex = appliedCount;
      replayPaused = true;
      replayStep = false;
      replayStart = now;
      replayBaseMicros = replayIndex == 0
        ? replayFrames.front().timeMicros
        : replayFrames[replayIndex - 1u].timeMicros;
      lastStatus = "replay seek";
    }
    if (replayMode && !replayFrames.empty()) {
      if (replayIndex < replayFrames.size() && (replayStep || !replayPaused)) {
        const uint64_t targetMicros = replayStep
          ? replayFrames[replayIndex].timeMicros
          : replayBaseMicros + static_cast<uint64_t>(
              std::chrono::duration<double, std::micro>(now - replayStart).count() * replaySpeed);
        while (replayIndex < replayFrames.size() && replayFrames[replayIndex].timeMicros <= targetMicros) {
          const auto& frame = replayFrames[replayIndex];
          stats.lastPayloadBytes = static_cast<int>(frame.payload.size());
          stats.bytes += frame.payload.size();
          stats.updates++;
          std::vector<PendingSensorUpdate> pending;
          raisin::tcp_viewer::applyRecordedFrame(frame, clearSceneState,
              [&](const std::vector<char>& payload) {
                return applyScenePayload(payload, true, now, pending);
              });
          ++replayIndex;
          if (replayStep) break;
        }
        replayStep = false;
        if (replayIndex >= replayFrames.size()) {
          if (options.replayLoop) {
            replayIndex = 0;
            replayStart = now;
            replayBaseMicros = replayFrames.front().timeMicros;
            clearSceneState();
          } else {
            replayPaused = true;
            replayIndex = replayFrames.size();
          }
        }
      } else {
        replayStep = false;
      }
    }
    if (!replayMode && autoConnect && !localSimulation.active() && !inspector.active && !client.isConnected() &&
        now >= nextAutoConnectAttempt) {
      ConnectionEntry endpoint;
      if (!normalizeConnectionEndpoint(host, port, endpoint)) {
        lastStatus = "invalid endpoint";
        nextAutoConnectAttempt = now + kAutoConnectInterval;
      } else {
        connectToEndpoint(
          endpoint, false, "auto-connecting",
          isLoopbackHostName(endpoint.host) ? "waiting for local server" : "auto-connect failed",
          kAutoConnectTimeoutMs);
        scheduleAutoConnectRetry(pane, endpoint, now);
      }
    }

    requestedTag = 0;
    requestedIndex = 0;
    requestedEntry = nullptr;
    scene.getVisualInfo(viewer->getTargetVisual(), requestedTag, requestedIndex, requestedEntry);
    if (requestedEntry && (requestedEntry->shape == raisim::Shape::Ground ||
                            requestedEntry->shape == raisim::Shape::HeightMap)) {
      viewer->setTargetVisual(nullptr);
      requestedTag = 0;
      requestedIndex = 0;
      requestedEntry = nullptr;
    }
    if (requestedEntry &&
        (controlSelectionTag != requestedTag || controlSelectionIndex != requestedIndex)) {
      controlSelectionTag = requestedTag;
      controlSelectionIndex = requestedIndex;
      controlBodyIdx = std::max(0, requestedEntry->localBodyIdx);
      controlBodyFollowsSelection = true;
      controlPosePosition = requestedEntry->lastPos;
      controlPoseQuat = requestedEntry->lastQuat;
      controlPoseTag = requestedTag;
      controlPoseInitialized = true;
      controlGc.clear();
      controlGcTag = 0;
      controlGcDirty = false;
    } else if (!requestedEntry) {
      controlSelectionTag = 0;
      controlSelectionIndex = -1;
      controlGcDirty = false;
    }
    scene.setSelectionTag(hasForcedTargetOffset ? 0 : requestedTag);

    // ----- Signal workbench sampling -----
    // Record the current selection plus every pinned object, so pinned plots do
    // not freeze the moment the user clicks something else. Histories for
    // objects that are neither selected nor pinned are dropped, which keeps
    // memory bounded no matter how much of the scene has been clicked through.
    selectedSignalKey = requestedEntry ? visualMotionKey(requestedTag, requestedIndex) : 0;
    if (scene.hasServerWorldTime()) {
      const double signalTime = scene.getServerWorldTime();
      const SelectedObjectInfo& sampledSelectedInfo = scene.getSelectedInfo();
      const bool hasContactTags = scene.serverSupportsContactObjectTags();

      const auto recordSignalObject = [&](uint64_t key, uint32_t tag, int index,
                                          const VisualEntry& entry) {
        SignalObjectRecord& record = signalRecords[key];
        record.tag = tag;
        record.index = index;
        record.visual = entry.visual;
        const bool isSelection = key == selectedSignalKey;
        const SelectedObjectInfo* info =
          (isSelection && sampledSelectedInfo.valid && sampledSelectedInfo.tag == tag)
            ? &sampledSelectedInfo
            : nullptr;
        // Only rebuild the channel set for the selection: doing it for a pinned
        // object with no selected-object payload would drop its joint columns
        // (and therefore its recorded history) as soon as focus moved away.
        if ((isSelection && (!entry.isArticulated || info)) || record.history.channels().empty()) {
          record.history.setChannels(
            raisin::tcp_viewer::buildSignalChannels(entry.isArticulated, hasContactTags, info));
        }
        raisin::tcp_viewer::SignalSampleInputs inputs;
        inputs.position = entry.lastPos;
        const auto motion = motionEstimates.find(key);
        if (motion != motionEstimates.end() && motion->second.valid) {
          inputs.linearVelocity = motion->second.linearVelocity;
          inputs.angularSpeed = motion->second.angularSpeed;
          inputs.hasMotionEstimate = true;
        }
        inputs.contactCount = static_cast<float>(scene.contactCountForTag(tag));
        inputs.selectedInfo = info;
        record.history.append(signalTime,
          raisin::tcp_viewer::sampleSignalChannels(record.history.channels(), inputs), info != nullptr);

        std::string label = entry.objectName.empty() ? scene.getObjectName(tag)
                                                     : entry.objectName;
        if (label.empty()) {
          label = "tag " + std::to_string(tag);
        }
        if (index != 0) {
          label += " [" + std::to_string(index) + "]";
        }
        record.label = std::move(label);
      };

      if (requestedEntry) {
        recordSignalObject(selectedSignalKey, requestedTag, requestedIndex, *requestedEntry);
      }
      for (const uint64_t pinnedKey : pinnedSignalObjects) {
        if (pinnedKey == selectedSignalKey) {
          continue;
        }
        const auto recordIt = signalRecords.find(pinnedKey);
        if (recordIt == signalRecords.end()) {
          continue;
        }
        const std::shared_ptr<raisin::Visuals> pinnedVisual = recordIt->second.visual.lock();
        if (!pinnedVisual) {
          continue;  // The object was removed; keep the frozen history until unpinned.
        }
        uint32_t pinnedTag = 0;
        int pinnedIndex = 0;
        const VisualEntry* pinnedEntry = nullptr;
        if (scene.getVisualInfo(pinnedVisual.get(), pinnedTag, pinnedIndex, pinnedEntry) &&
            pinnedEntry && visualMotionKey(pinnedTag, pinnedIndex) == pinnedKey) {
          recordSignalObject(pinnedKey, pinnedTag, pinnedIndex, *pinnedEntry);
        }
      }

      for (auto it = signalRecords.begin(); it != signalRecords.end();) {
        if (it->first == selectedSignalKey || pinnedSignalObjects.count(it->first) != 0) {
          ++it;
          continue;
        }
        it = signalRecords.erase(it);
      }
    }
    const auto setRulerEndpoint = [&](int endpoint, const glm::vec3& point, std::string label) {
      label = trimAscii(label);
      if (label.empty()) {
        label = "scene point";
      }
      if (endpoint == 0) {
        ruler.hasA = true;
        ruler.a = point;
        ruler.aLabel = std::move(label);
        ruler.nextPoint = 1;
      } else {
        ruler.hasB = true;
        ruler.b = point;
        ruler.bLabel = std::move(label);
        ruler.nextPoint = 0;
      }
    };
    const auto appendRulerPoint = [&](const glm::vec3& point, const std::string& label) {
      if (!ruler.hasA || ruler.hasB || ruler.nextPoint == 0) {
        ruler.hasA = true;
        ruler.a = point;
        ruler.aLabel = trimAscii(label).empty() ? "scene point" : trimAscii(label);
        ruler.hasB = false;
        ruler.bLabel.clear();
        ruler.nextPoint = 1;
        lastStatus = "ruler point A set";
      } else {
        setRulerEndpoint(1, point, label);
        lastStatus = "ruler point B set";
      }
    };
    const auto rulerSelectionLabel = [&](uint32_t tag, int index, const VisualEntry* entry) {
      std::string label;
      if (entry) {
        label = entry->objectName.empty() ? scene.getObjectName(tag) : entry->objectName;
        if (label.empty()) {
          label = objectTypeLabel(entry->objectTypeRaw);
        }
      }
      if (label.empty()) {
        label = "tag " + std::to_string(tag) + ":" + std::to_string(index);
      }
      return label;
    };
    const auto setRulerEndpointFromSelection = [&](int endpoint, uint32_t tag, int index,
                                                   const VisualEntry* entry) {
      if (!entry) {
        return false;
      }
      setRulerEndpoint(endpoint, entry->lastPos, rulerSelectionLabel(tag, index, entry));
      lastStatus = endpoint == 0 ? "ruler point A set" : "ruler point B set";
      return true;
    };
    // Does the most recent beacon from this endpoint say its single client seat
    // is taken? Used to explain silence after connecting.
    const auto endpointReportedBusy = [&](const char* endpointHost, int endpointPort) {
      for (const auto& server : discoveredServers) {
        if (server.endpoint.port != endpointPort || server.endpoint.host != endpointHost) {
          continue;
        }
        const auto statusIt = server.metadata.find("status");
        return statusIt != server.metadata.end() && statusIt->second == "connected";
      }
      return false;
    };

    const uint32_t updateRequestTag = hasForcedTargetOffset ? 0 : requestedTag;

    if (client.isConnected()) {
      auto& payload = tcpPayload;
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
        if (!awaitingResponse &&
            consumeTcpUpdateSlot(now, nextTcpUpdateRequestTime, settings.tcpUpdateRateHz)) {
          // RaisimServer rejects any frame that mixes a world export with other
          // requests, and a rejected frame costs the whole connection, so the
          // export is flushed alone and everything else waits a frame.
          const auto exportIt = std::find_if(
            pendingControlRequests.begin(), pendingControlRequests.end(),
            [](const raisin::tcp_viewer::ClientRequest& request) {
              return request.type == raisin::tcp_viewer::ClientRequestType::CR_SAVE_THE_WORLD;
            });
          std::vector<raisin::tcp_viewer::ClientRequest> exportFrame;
          const bool exportOnlyFrame = exportIt != pendingControlRequests.end();
          if (exportOnlyFrame) {
            exportFrame.push_back(*exportIt);
            pendingControlRequests.erase(exportIt);
          }
          const std::vector<raisin::tcp_viewer::ClientRequest>& frameRequests =
            exportOnlyFrame ? exportFrame : pendingControlRequests;
          // Scene-file requests ride along, except with a world export, which goes alone.
          std::vector<raisin::tcp_viewer::ClientRequest> withSceneFiles;
          if (!exportOnlyFrame) {
            withSceneFiles = frameRequests;
            sceneFiles.appendRequests(withSceneFiles);
          }
          if (!sendUpdateRequest(client, updateRequestTag,
                                 withSceneFiles.empty() ? frameRequests : withSceneFiles)) {
            if (!client.lastIoWouldBlock()) {
              reportClosedServerConnection(pane);
              networkFailed = true;
            } else if (exportOnlyFrame) {
              // Nothing went out; keep the export queued for the next slot.
              pendingControlRequests.push_back(exportFrame.front());
            }
          } else {
            awaitingResponse = true;
            updateRequestSentAt = now;
            if (!exportOnlyFrame) {
              pendingControlRequests.clear();
              if (mouseForce.active) {
                mouseForce.pendingRequestIndex = kNoPendingRequest;
              }
              // The wire attachment stays live on the server, but the queued drag
              // request is gone, so the next frame has to push a fresh one.
              if (wireDrag.active) {
                wireDrag.pendingRequestIndex = kNoPendingRequest;
              }
            } else {
              // The remaining requests kept their slots, but the erase above
              // shifted them, so drop the cached indices.
              mouseForce.pendingRequestIndex = kNoPendingRequest;
              wireDrag.pendingRequestIndex = kNoPendingRequest;
            }
          }
        }

        if (!networkFailed && awaitingResponse) {
          if (!client.recvMessage(payload)) {
            if (!client.lastIoWouldBlock()) {
              reportClosedServerConnection(pane);
              networkFailed = true;
            } else if (now - updateRequestSentAt > kServerResponseTimeout) {
              // Silence is its own failure mode. RaisimServer serves one client
              // and only calls accept() while it has none, so a connect to a
              // server that is already taken completes in the kernel backlog and
              // is then ignored — indistinguishable, at the socket, from a
              // server that is merely slow. The beacon is what tells them apart:
              // it reports the seat as taken by the *other* client.
              lastStatus = sceneReceived ? "no reply from server"
                         : endpointReportedBusy(host, port) ? "server already occupied"
                                                            : "no reply from server";
              networkFailed = true;
            }
          } else {
            awaitingResponse = false;
            stats.lastRoundTripMs = std::chrono::duration<double, std::milli>(
              now - updateRequestSentAt).count();
            stats.lastPayloadBytes = static_cast<int>(payload.size());
            stats.bytes += payload.size();
            stats.updates++;
            if (sessionRecorder.active()) {
              sessionRecorder.record(payload, now, sessionStatus);
            }
            std::vector<PendingSensorUpdate> pending;
            const bool parsedOk = applyScenePayload(payload, false, now, pending);
            // applyScenePayload may erase the selected VisualEntry or rehash
            // the visual map. Do not use the pointer captured earlier this frame.
            refreshRequestedEntry(scene, *viewer,
                                  requestedTag, requestedIndex, requestedEntry);
            if (!parsedOk) {
              networkFailed = lastStatus.find("disconnect") != std::string::npos;
            } else if (!pending.empty()) {
              if (!sensorRenderer.render(*viewer, pending, sensorStatus)) {
                lastStatus = "sensor render failed";
                networkFailed = true;
              } else if (!sendSensorUpdate(client, pending, nextSensorUpdateIndex)) {
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
        clearSceneState();
      }
    }

    if (requestFrameScene) {
      frameScene(*viewer, scene);
      requestFrameScene = false;
    }
    if (requestFrameSelected) {
      uint32_t actionTag = 0;
      int actionIndex = 0;
      const VisualEntry* actionEntry = nullptr;
      scene.getVisualInfo(viewer->getTargetVisual(), actionTag, actionIndex, actionEntry);
      frameSelected(*viewer, actionEntry);
      requestFrameSelected = false;
    }
    if (requestResetCamera) {
      viewer->setTargetVisual(nullptr);
      applyCameraLookAt(viewer->getCamera(), defaultCameraPos, defaultCameraTarget);
      requestResetCamera = false;
    }

    if (exportScenePending && scene.selectableObjectCount() > 0) {
      if (assetDiagnostics.empty()) {
        assetDiagnostics = collectAssetDiagnostics(scene);
      }
      exportSceneJson(options.exportScenePath, scene, assetDiagnostics, captureStatus);
      exportScenePending = false;
    }

    // Creates, refreshes or removes the world axes to match showWorldFrame.
    const auto syncWorldFrame = [&]() {
      if (showWorldFrame) {
        if (!worldFrame) {
          worldFrame = viewer->addCoordinateFrame("world_frame");
        }
        if (worldFrame) {
          worldFrame->poses.resize(1);
          setTcpViewerIdentityPose(worldFrame->poses[0]);
          worldFrame->frameSize = computeWorldFrameSize(viewer->getCamera());
        }
      } else if (worldFrame) {
        viewer->removeCoordinateFrame("world_frame");
        worldFrame.reset();
      }
    };
    syncWorldFrame();

    // A scene the server shares brings its own camera, lights and render settings.
    if (sceneFiles.update(*viewer, scene) && !sceneFiles.sync().scene()->cameras.empty())
      autoFrameApplied = true;
    if (sceneFiles.takeViewerSettingsReset()) pane.appliedSettingsSerial = 0;

    auto& cam = viewer->getCamera();
    if (!sceneFiles.applied()) {
      cam.nearPlane = settings.cameraNear;
      cam.farPlane = settings.cameraFar;
      cam.zNear = settings.cameraNear;
      cam.zFar = settings.cameraFar;
      cam.zoom = settings.cameraFovDeg;
    }
    cam.movementSpeed = cameraSpeed;
    // Render settings are shared: whichever pane's panel edited them raised
    // settingsDirty, and the serial bump makes every other pane — including one
    // created later — re-apply them to its own renderer exactly once.
    if (settingsDirty) {
      ++settingsSerial;
      settingsSavePending = true;
      lastSettingsDirtyTime = now;
      settingsDirty = false;
    }
    if (pane.appliedSettingsSerial != settingsSerial) {
      settings.recentConnections = recentConnections;
      settings.resourceDirs = resourceDirs;
      applyViewerSettings(*viewer, settings);
      pane.appliedSettingsSerial = settingsSerial;
    }
    if (pane.appliedResourceDirSerial != resourceDirSerial) {
      scene.setResourceSearchPaths(resourceDirs);
      sceneFiles.setResourceDirs(resourceDirs);
      pane.appliedResourceDirSerial = resourceDirSerial;
    }
    if (settingsSavePending && now - lastSettingsDirtyTime >= kSettingsSaveDebounce) {
      capturePaneSettings();
      if (!options.noSaveSettings) saveViewerSettings(settings);
      settingsSavePending = false;
    }
    const bool weatherControlsLight = settings.skyEnabled && settings.skyWeatherEnabled &&
      weatherDefaultEnabledForQuality(settings.renderQuality);
    if (sceneFiles.applied()) {
      // The scene's lights and weather stay as applyRscene() set them.
    } else if (weatherControlsLight) {
      viewer->updateWeather(static_cast<double>(std::max(0.0f, io.DeltaTime)));
      applyWeatherLightStrength(viewer->getLight(), viewer->getRenderQualitySettings(), lightStrength);
    } else {
      auto& lightRef = viewer->getLight();
      lightRef.type = raisin::LightType::DIRECTIONAL;
      lightRef.ambient = settings.mainLightAmbient * ambientStrength;
      lightRef.diffuse = settings.mainLightDiffuse * lightStrength;
      lightRef.specular = settings.mainLightSpecular * lightStrength;
      lightRef.direction = lightDirectionFromYawPitch(lightYawDeg, lightPitchDeg);
      lightRef.setShadowsEnabled(settings.shadowsEnabled);
      lightRef.setShadowResolution(settings.shadowResolution);
      lightRef.setShadowParams(settings.shadowBias, settings.shadowStrength, settings.shadowPcfRadius);
    }

    scene.updateContactVisuals(
      showContactPoints, contactPointSize, showContactForces, contactForceSize,
      contactForceAbsolute);

    // Recover body poses from the visual-to-body offsets, once for both overlays.
    const auto bodyFramePoses = (showBodyFrames || showComMarkers)
      ? scene.getBodyFrames() : std::vector<raisin::tcp_viewer::BodyFramePose>{};
    {
      const bool wantFrames = showBodyFrames;
      if (wantFrames && !bodyFramesNode) {
        bodyFramesNode = viewer->addCoordinateFrame("body_frames");
      }
      if (bodyFramesNode) {
        bodyFramesNode->enable(wantFrames);
      }
      if (wantFrames && bodyFramesNode) {
        bodyFramesNode->poses.clear();
        bodyFramesNode->poses.reserve(bodyFramePoses.size());
        for (const auto& frame : bodyFramePoses) {
          raisin::CoordinateFrame::Pose p;
          p.position = frame.position;
          const auto& q = frame.quaternionWxyz;
          p.quaternion = glm::quat::wxyz(q.x, q.y, q.z, q.w);
          bodyFramesNode->poses.push_back(p);
        }
        bodyFramesNode->frameSize = bodyFrameSize;
      }
    }

    // COM markers: spheres at each body's frame origin. For single rigid bodies
    // this is the COM exactly; for articulated system links it's the link frame
    // origin (not the COM offset within that link — server doesn't stream that).
    {
      const auto comMarkerName = [](size_t i) {
        return "com_marker_" + std::to_string(i);
      };
      if (!showComMarkers) {
        for (size_t i = comMarkers.size(); i-- > 0;) {
          viewer->removeVisualObject(comMarkerName(i));
        }
        comMarkers.clear();
      } else {
        std::vector<glm::vec3> targets;
        targets.reserve(bodyFramePoses.size());
        for (const auto& frame : bodyFramePoses) targets.push_back(frame.position);
        const float r = std::max(0.001f, comMarkerSize);
        while (comMarkers.size() < targets.size()) {
          const size_t idx = comMarkers.size();
          auto v = viewer->addVisualSphere(comMarkerName(idx), r, 0.2f, 0.8f, 1.0f, 1.0f);
          comMarkers.push_back(v);
        }
        while (comMarkers.size() > targets.size()) {
          const size_t idx = comMarkers.size() - 1;
          viewer->removeVisualObject(comMarkerName(idx));
          comMarkers.pop_back();
        }
        for (size_t i = 0; i < comMarkers.size(); ++i) {
          if (!comMarkers[i]) continue;
          comMarkers[i]->setSphereSize(r);
          comMarkers[i]->setPosition(targets[i]);
        }
      }
    }

    // Pose grabber: shadow the server's stream while the gizmo is enabled. The
    // applyScenePayload calls above just updated visual->setPosition/Orientation
    // from the server's view of the world; here we force the held body back to
    // the user-controlled pose BEFORE renderViewer reads the visual state.
    // - Seed the hold on first enable (or when selection changes) from the
    //   body's current server pose so there's no jump.
    // - The drag handler (later in the frame) updates heldPos/heldQuat as the
    //   user drags; nothing is sent to the server during the drag.
    // - On deactivation (gizmo turned off, Esc, or selection change), if the
    //   user actually moved the body, emit one CR_SET_POSE with the final
    //   pose so the server commits to it.
    auto releasePoseHold = [&]() {
      if (!poseGrabber.heldActive) return;
      const bool canSend = client.isConnected() && scene.serverSupportsSimControl();
      if (poseGrabber.heldDirty && canSend && poseGrabber.heldTag != 0) {
        raisin::tcp_viewer::ClientRequest r;
        r.type = raisin::tcp_viewer::ClientRequestType::CR_SET_POSE;
        r.visTag = poseGrabber.heldTag;
        r.vec3a = poseGrabber.heldPos;
        r.quat = poseGrabber.heldQuat;
        // Coalesce with any in-flight pose request for the same tag.
        bool replaced = false;
        for (auto& pending : pendingControlRequests) {
          if (pending.type == raisin::tcp_viewer::ClientRequestType::CR_SET_POSE &&
              pending.visTag == r.visTag) {
            pending = r;
            replaced = true;
            break;
          }
        }
        if (!replaced) pendingControlRequests.push_back(r);
      }
      poseGrabber.heldActive = false;
      poseGrabber.heldDirty = false;
      poseGrabber.heldTag = 0;
      poseGrabber.dragging = false;
      poseGrabber.axis = -1;
    };
    if (poseGrabber.enabled && requestedEntry) {
      // Selection changed mid-hold: commit the previous body's pose, then
      // seed a fresh hold on the new body.
      if (poseGrabber.heldActive && poseGrabber.heldTag != requestedTag) {
        releasePoseHold();
      }
      if (!poseGrabber.heldActive) {
        poseGrabber.heldActive = true;
        poseGrabber.heldDirty = false;
        poseGrabber.heldTag = requestedTag;
        poseGrabber.heldPos = requestedEntry->lastPos;
        poseGrabber.heldQuat = requestedEntry->lastQuat;
      }
      if (requestedEntry->visual) {
        requestedEntry->visual->setPosition(poseGrabber.heldPos);
        requestedEntry->visual->setOrientation(poseGrabber.heldQuat);
      }
    } else if (poseGrabber.heldActive) {
      releasePoseHold();
    }

    const bool canQueueSimControl = client.isConnected() && scene.serverSupportsSimControl();
    const bool shiftForceModifierHeld = isShiftModifierHeld(io);
    const bool shiftForceCaptureRequested =
      shouldRequestMouseForceCapture(
        mouseForceEnabled, shiftForceModifierHeld, ruler.enabled, angle.enabled);
    const bool mouseForceSuppressViewportInput = shouldSuppressViewportForMouseForce(
      mouseForce.active, shiftForceCaptureRequested, io.MouseDown[ImGuiMouseButton_Left]);
    const bool wireDragModifierHeld = isWireDragModifierHeld(io);
    const bool wireDragCaptureRequested = shouldRequestWireDragCapture(
      wireDragEnabled, wireDragModifierHeld, shiftForceModifierHeld, ruler.enabled, angle.enabled);
    const bool wireDragSuppressViewportInput = shouldSuppressViewportForWireDrag(
      wireDrag.active, wireDragCaptureRequested, io.MouseDown[ImGuiMouseButton_Left]);
    // While the gizmo is being dragged, or while the user is left-pressing with
    // a selected body and the gizmo enabled, swallow viewport input so the
    // camera doesn't orbit/pan during gizmo manipulation.
    const bool poseGrabberSuppressViewportInput =
      poseGrabber.dragging ||
      (poseGrabber.enabled && requestedEntry && io.MouseDown[ImGuiMouseButton_Left]);
    viewportState = ViewerViewportState{};

    const auto tFrameStart = std::chrono::steady_clock::now();
    const bool measureToolActive = ruler.enabled || angle.enabled;
    const bool rulerCapturesViewportInput = measureToolActive && !mouseForce.active;
    const bool allowViewportInput = !mouseForceSuppressViewportInput &&
                                    !wireDragSuppressViewportInput &&
                                    !rulerCapturesViewportInput &&
                                    !poseGrabberSuppressViewportInput;
    const bool allowClickSelection = !mouseForce.active && !shiftForceCaptureRequested &&
                                     !wireDrag.active && !wireDragCaptureRequested &&
                                     !rulerCapturesViewportInput && !poseGrabber.dragging;
    updateCameraFrustums(*viewer, scene, cameraFrustums);
    // WASD/Space reach the camera through the process-wide SDL keyboard state,
    // so only the focused pane may act on them; otherwise one key press walks
    // every pane's camera at once. Mouse input stays scoped by hover.
    viewer->setKeyboardInputEnabled(focused);
    renderViewer(*viewer, window, paneRect, pane.id, allowViewportInput, allowClickSelection,
                 &viewportState);

    if (ruler.enabled && !mouseForce.active && viewportState.hovered &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      glm::vec3 rulerPoint(0.0f);
      if (readWorldPointAtCursor(viewer->getCamera(), viewportState, rulerPoint)) {
        appendRulerPoint(rulerPoint, "scene point");
      } else {
        lastStatus = "ruler pick missed";
      }
    }

    if (angle.enabled && !mouseForce.active && viewportState.hovered &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      glm::vec3 picked(0.0f);
      if (readWorldPointAtCursor(viewer->getCamera(), viewportState, picked)) {
        if (angle.picked >= 3) {
          angle.picked = 0;
        }
        if (angle.picked == 0)      { angle.a = picked; angle.aLabel = formatRulerPoint(picked); }
        else if (angle.picked == 1) { angle.b = picked; angle.bLabel = formatRulerPoint(picked); }
        else                         { angle.c = picked; angle.cLabel = formatRulerPoint(picked); }
        angle.picked++;
        lastStatus = angle.picked >= 3
          ? std::string("angle = ") + formatAngleDegrees(computeAngleRadians(angle.a, angle.b, angle.c))
          : std::string("angle: pick ") + nextAnglePointLabel(angle);
      } else {
        lastStatus = "angle pick missed";
      }
    }

    // Pose grabber gizmo: translate handles (3 colored arrows) plus rotate
    // handles (3 colored rings). Drag a handle to set position or orientation
    // via CR_SET_POSE. While dragging, the body's visual pose is overridden so
    // the incoming server stream doesn't snap the body back under the cursor.
    GizmoScreenLayout gizmoLayout;
    PoseGrabberHit gizmoHover;
    const bool poseGrabberPickable = poseGrabber.enabled && requestedEntry && !mouseForce.active &&
                                     !shiftForceCaptureRequested && !wireDrag.active &&
                                     !wireDragCaptureRequested && !ruler.enabled &&
                                     !angle.enabled && canQueueSimControl;
    // The gizmo lives at the user-controlled (held) pose, not the server's
    // pose. This keeps the handles attached to what the user actually sees.
    const glm::vec3 gizmoOriginWorld = (poseGrabber.heldActive && requestedEntry)
        ? poseGrabber.heldPos
        : (requestedEntry ? requestedEntry->lastPos : glm::vec3(0.0f));
    const glm::vec4 gizmoOriginQuat = (poseGrabber.heldActive && requestedEntry)
        ? poseGrabber.heldQuat
        : (requestedEntry ? requestedEntry->lastQuat : glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    // Gizmo handles are world-axis-aligned. Translate/rotate produce deltas
    // expressed in the world frame; those deltas are composed with the body's
    // current held pose by the drag handler below.
    if (poseGrabberPickable) {
      gizmoLayout = computeGizmoLayout(viewer->getCamera(), viewportState, gizmoOriginWorld);
      if (!poseGrabber.dragging && viewportState.hovered) {
        gizmoHover = pickPoseGrabberHandle(gizmoLayout, io.MousePos);
      }
    }
    if (poseGrabberPickable && !poseGrabber.dragging && viewportState.hovered &&
        gizmoHover.axis >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      poseGrabber.dragging = true;
      poseGrabber.mode = gizmoHover.mode;
      poseGrabber.axis = gizmoHover.axis;
      poseGrabber.tag = requestedTag;
      poseGrabber.index = requestedIndex;
      poseGrabber.anchorWorld = gizmoOriginWorld;
      poseGrabber.anchorQuat = gizmoOriginQuat;
      poseGrabber.anchorMouse = io.MousePos;
      poseGrabber.currentTarget = poseGrabber.anchorWorld;
      poseGrabber.currentQuat = poseGrabber.anchorQuat;
      // For rotation, capture mouse angle around the body's screen center so
      // angular deltas are relative to that starting position.
      ImVec2 bodyScreen;
      if (projectWorldToViewport(viewer->getCamera(), viewportState,
                                 poseGrabber.anchorWorld, bodyScreen)) {
        poseGrabber.anchorScreenAngle =
          std::atan2(io.MousePos.y - bodyScreen.y, io.MousePos.x - bodyScreen.x);
      }
    }
    if (poseGrabber.dragging) {
      // The picked axis is a WORLD axis (gizmo handles are world-aligned).
      // Translate moves along that world axis; rotate applies a world-frame
      // rotation around it as a pre-multiplication onto the anchor pose
      // (which is the held pose captured at drag start).
      glm::vec3 worldAxis(0.0f);
      worldAxis[poseGrabber.axis] = 1.0f;
      // anchorQuat is wxyz: x holds w.
      const glm::quat aq = glm::quat::wxyz(poseGrabber.anchorQuat.x, poseGrabber.anchorQuat.y,
                                           poseGrabber.anchorQuat.z, poseGrabber.anchorQuat.w);

      if (poseGrabber.mode == PoseGrabberGesture::Mode::Translate) {
        ImVec2 originScreen, axisScreen;
        const bool okA = projectWorldToViewport(viewer->getCamera(), viewportState,
                                                poseGrabber.anchorWorld, originScreen);
        const bool okB = projectWorldToViewport(viewer->getCamera(), viewportState,
                                                poseGrabber.anchorWorld + worldAxis, axisScreen);
        if (okA && okB) {
          const float axDx = axisScreen.x - originScreen.x;
          const float axDy = axisScreen.y - originScreen.y;
          const float axLen2 = axDx * axDx + axDy * axDy;
          if (axLen2 > 1e-3f) {
            const float mDx = io.MousePos.x - poseGrabber.anchorMouse.x;
            const float mDy = io.MousePos.y - poseGrabber.anchorMouse.y;
            // Scalar projection of mouse drag onto the screen-axis direction;
            // pixel distance / pixels-per-metre = world distance along worldAxis.
            const float worldDelta = (mDx * axDx + mDy * axDy) / axLen2;
            poseGrabber.currentTarget = poseGrabber.anchorWorld + worldAxis * worldDelta;
          }
        }
      } else {
        // Rotation: world-frame delta applied to the held (anchor) pose via
        // pre-multiplication.  q_new = dq_world * q_anchor.  Pre-multiplication
        // means dq rotates the body in the world frame — exactly what a
        // world-aligned gizmo should do.
        ImVec2 bodyScreen;
        if (projectWorldToViewport(viewer->getCamera(), viewportState,
                                   poseGrabber.anchorWorld, bodyScreen)) {
          const float curAngle =
            std::atan2(io.MousePos.y - bodyScreen.y, io.MousePos.x - bodyScreen.x);
          float delta = curAngle - poseGrabber.anchorScreenAngle;
          while (delta > 3.14159265f) delta -= 6.2831853f;
          while (delta < -3.14159265f) delta += 6.2831853f;
          // Screen +Y points down → "visually CCW" mouse motion is math CW.
          const float screenSpaceDelta = -delta;
          // If the picked world axis points toward the camera, screen-CCW =
          // positive right-hand-rule rotation. Otherwise flip the sign.
          const float axisDotView = glm::dot(worldAxis, viewer->getCamera().front);
          const float sign = (axisDotView < 0.0f) ? +1.0f : -1.0f;
          const float rotAngle = sign * screenSpaceDelta;
          const glm::quat dq = glm::angleAxis(rotAngle, worldAxis);
          // Pre-multiply: world-frame rotation applied to the held pose.
          const glm::quat nq = glm::normalize(dq * aq);
          // Pack glm::quat back into wxyz vec4 storage.
          poseGrabber.currentQuat = glm::vec4(nq.w, nq.x, nq.y, nq.z);
          poseGrabber.currentTarget = poseGrabber.anchorWorld;
        }
      }

      // Update the held pose so next frame's pre-render override paints the
      // body at the dragged target. No CR_SET_POSE is queued during the drag —
      // the server only sees the final pose when the gizmo is deactivated
      // (handled in releasePoseHold above).
      poseGrabber.heldPos = poseGrabber.currentTarget;
      poseGrabber.heldQuat = poseGrabber.currentQuat;
      poseGrabber.heldDirty = true;
      if (requestedEntry && requestedEntry->visual) {
        requestedEntry->visual->setPosition(poseGrabber.currentTarget);
        requestedEntry->visual->setOrientation(poseGrabber.currentQuat);
      }
      // Keep the side panel's pose authoring widgets in sync so the user can
      // see numeric values without round-tripping through the server.
      controlPosePosition = poseGrabber.currentTarget;
      controlPoseQuat = poseGrabber.currentQuat;
      controlPoseTag = poseGrabber.tag;
      controlPoseInitialized = true;

      if (!io.MouseDown[ImGuiMouseButton_Left]) {
        poseGrabber.dragging = false;
        poseGrabber.axis = -1;
      }
    }

    struct MouseForceStartTarget {
      uint32_t tag = 0;
      int index = 0;
      int localBodyIdx = 0;
      const VisualEntry* entry = nullptr;
      raisin::Visuals* visual = nullptr;
      glm::vec3 clickedWorldPoint{0.0f};
      bool fromPick = false;
      bool hasClickedWorldPoint = false;
    };
    const auto resolveMouseForceStartTarget = [&]() {
      MouseForceStartTarget target;
      if (requestedEntry && supportsTcpViewerForceControl(requestedEntry)) {
        target.tag = requestedTag;
        target.index = requestedIndex;
        target.localBodyIdx = std::max(0, controlBodyIdx);
        target.entry = requestedEntry;
        target.visual = viewer->getTargetVisual();
      }

      if (viewer && viewportState.hovered) {
        if (auto* pickedVisual = viewer->pickTargetVisualAt(viewportState.cursorX, viewportState.cursorY)) {
          uint32_t pickedTag = 0;
          int pickedIndex = 0;
          const VisualEntry* pickedEntry = nullptr;
          if (scene.getVisualInfo(pickedVisual, pickedTag, pickedIndex, pickedEntry) &&
              pickedEntry && supportsTcpViewerForceControl(pickedEntry) &&
              !isContactEntry(pickedEntry)) {
            target.tag = pickedTag;
            target.index = pickedIndex;
            target.localBodyIdx = pickedEntry->isArticulated
              ? std::max(0, pickedEntry->localBodyIdx)
              : 0;
            target.entry = pickedEntry;
            target.visual = pickedVisual;
            target.fromPick = true;
            target.hasClickedWorldPoint =
              readWorldPointAtCursor(viewer->getCamera(), viewportState, target.clickedWorldPoint);
          }
        }
      }
      return target;
    };

    if (!mouseForce.active && shiftForceCaptureRequested && viewportState.hovered &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      const MouseForceStartTarget forceTarget = resolveMouseForceStartTarget();
      if (!canQueueSimControl) {
        if (!client.isConnected()) {
          lastStatus = "mouse force: disconnected";
        } else if (!scene.serverSupportsSimControl()) {
          lastStatus = "mouse force: server lacks sim control";
        }
      } else if (!forceTarget.entry) {
        lastStatus = requestedEntry ? "mouse force: unsupported object"
                                    : "mouse force: select a body";
      } else {
        if (forceTarget.fromPick && forceTarget.visual) {
          viewer->setTargetVisual(forceTarget.visual);
          requestedTag = forceTarget.tag;
          requestedIndex = forceTarget.index;
          requestedEntry = forceTarget.entry;
        }
        const glm::vec3 applicationPoint = mouseForceStartApplicationPoint(
          *forceTarget.entry, controlPointOffset, forceTarget.hasClickedWorldPoint,
          forceTarget.clickedWorldPoint);
        ImVec2 anchorScreen = io.MousePos;
        ImVec2 applicationScreen;
        if (forceTarget.hasClickedWorldPoint &&
            projectWorldToViewport(viewer->getCamera(), viewportState, applicationPoint,
              applicationScreen)) {
          anchorScreen = applicationScreen;
        } else if (isMouseWithinForceStartRadius(viewer->getCamera(), viewportState,
            *forceTarget.entry, applicationPoint, io.MousePos, applicationScreen)) {
          anchorScreen = applicationScreen;
        }
        mouseForce.active = true;
        mouseForce.tag = forceTarget.tag;
        mouseForce.index = forceTarget.index;
        mouseForce.localBodyIdx = std::max(0, forceTarget.localBodyIdx);
        mouseForce.applicationPoint = applicationPoint;
        mouseForce.localApplicationPoint =
          visualWorldPointToLocal(*forceTarget.entry, applicationPoint);
        mouseForce.force = glm::vec3(0.0f);
        mouseForce.pressMouse = anchorScreen;
        mouseForce.currentMouse = io.MousePos;
      }
    }
    const auto erasePendingRequest = [&](size_t idx) {
      if (idx != kNoPendingRequest && idx < pendingControlRequests.size()) {
        pendingControlRequests.erase(pendingControlRequests.begin() + static_cast<long>(idx));
      }
    };
    const auto cancelPendingMouseForce = [&]() {
      erasePendingRequest(mouseForce.pendingRequestIndex);
      mouseForce.pendingRequestIndex = kNoPendingRequest;
    };
    const auto queueOrUpdateMouseForce = [&]() {
      raisin::tcp_viewer::ClientRequest r;
      r.type = raisin::tcp_viewer::ClientRequestType::CR_APPLY_FORCE;
      r.visTag = mouseForce.tag;
      r.localBodyIdx = std::max(0, mouseForce.localBodyIdx);
      r.vec3a = mouseForce.applicationPoint;
      r.vec3b = mouseForce.force;

      const size_t idx = mouseForce.pendingRequestIndex;
      if (idx != kNoPendingRequest && idx < pendingControlRequests.size()) {
        auto& pending = pendingControlRequests[idx];
        if (pending.type == raisin::tcp_viewer::ClientRequestType::CR_APPLY_FORCE &&
            pending.visTag == mouseForce.tag) {
          pending = r;
          return;
        }
      }
      pendingControlRequests.push_back(r);
      mouseForce.pendingRequestIndex = pendingControlRequests.size() - 1;
    };

    if (mouseForce.active) {
      const bool mouseButtonDown = io.MouseDown[ImGuiMouseButton_Left];
      const VisualEntry* activeForceEntry = nullptr;
      if (requestedEntry && requestedTag == mouseForce.tag && requestedIndex == mouseForce.index) {
        activeForceEntry = requestedEntry;
      }

      if (!mouseButtonDown || !activeForceEntry) {
        cancelPendingMouseForce();
        mouseForce = MouseForceGesture{};
        if (mouseButtonDown) lastStatus = "mouse force target lost";
      } else {
        mouseForce.applicationPoint = visualLocalPointToWorld(
          *activeForceEntry, mouseForce.localApplicationPoint);
        mouseForce.currentMouse = io.MousePos;
        const ImVec2 dragPixels(mouseForce.currentMouse.x - mouseForce.pressMouse.x,
                                mouseForce.currentMouse.y - mouseForce.pressMouse.y);
        mouseForce.force = mouseForceFromDragPixels(viewer->getCamera(), dragPixels, mouseForceScale);
        controlForce = mouseForce.force;
        const float dragLen = std::sqrt(dragPixels.x * dragPixels.x + dragPixels.y * dragPixels.y);
        const bool shouldApplyMouseForce = dragLen >= 4.0f &&
          glm::length(mouseForce.force) > 1.0e-4f && client.isConnected();
        if (shouldApplyMouseForce) {
          queueOrUpdateMouseForce();
          lastStatus = "mouse force applying";
        } else {
          cancelPendingMouseForce();
        }
      }
    }

    // ----- Interaction wire (CR_ATTACH_WIRE + CR_DRAG_OBJECT) -----
    // Ctrl-drag (or Cmd-drag on macOS) pulls a body with a mass-scaled spring
    // instead of teleporting it, so the solver keeps contacts and joints
    // consistent while it moves.
    const auto releaseWireDrag = [&]() {
      erasePendingRequest(wireDrag.pendingRequestIndex);
      wireDrag = WireDragGesture{};
    };
    const auto queueOrUpdateWireDrag = [&]() {
      using raisin::tcp_viewer::ClientRequestType;
      // The attach only has to be sent once; the server keeps the grabbed body
      // and local attachment point until the wire goes slack.
      if (!wireDrag.attachQueued) {
        raisin::tcp_viewer::ClientRequest attach;
        attach.type = ClientRequestType::CR_ATTACH_WIRE;
        attach.visTag = wireDrag.tag;
        attach.localBodyIdx = std::max(0, wireDrag.localBodyIdx);
        attach.point = glm::dvec3(wireDrag.attachPoint);
        pendingControlRequests.push_back(attach);
        wireDrag.attachQueued = true;
        // The drag that follows lands after the attach, so its index shifts.
        wireDrag.pendingRequestIndex = kNoPendingRequest;
      }
      raisin::tcp_viewer::ClientRequest drag;
      drag.type = ClientRequestType::CR_DRAG_OBJECT;
      drag.stiffness = wireDragStiffness;
      drag.point = glm::dvec3(wireDrag.target);

      const size_t idx = wireDrag.pendingRequestIndex;
      if (idx != kNoPendingRequest && idx < pendingControlRequests.size() &&
          pendingControlRequests[idx].type == ClientRequestType::CR_DRAG_OBJECT) {
        pendingControlRequests[idx] = drag;
        return;
      }
      pendingControlRequests.push_back(drag);
      wireDrag.pendingRequestIndex = pendingControlRequests.size() - 1;
    };

    if (!wireDrag.active && !mouseForce.active && wireDragCaptureRequested &&
        viewportState.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      const MouseForceStartTarget wireTarget = resolveMouseForceStartTarget();
      if (!canQueueSimControl) {
        lastStatus = client.isConnected() ? "wire drag: server lacks sim control"
                                          : "wire drag: disconnected";
      } else if (!wireTarget.entry) {
        lastStatus = requestedEntry ? "wire drag: unsupported object" : "wire drag: select a body";
      } else {
        if (wireTarget.fromPick && wireTarget.visual) {
          viewer->setTargetVisual(wireTarget.visual);
          requestedTag = wireTarget.tag;
          requestedIndex = wireTarget.index;
          requestedEntry = wireTarget.entry;
        }
        // Grab where the user actually clicked when the depth read succeeded;
        // otherwise fall back to the body origin.
        const glm::vec3 grabPoint = wireTarget.hasClickedWorldPoint
          ? wireTarget.clickedWorldPoint
          : wireTarget.entry->lastPos;
        wireDrag.active = true;
        wireDrag.attachQueued = false;
        wireDrag.tag = wireTarget.tag;
        wireDrag.index = wireTarget.index;
        wireDrag.localBodyIdx = std::max(0, wireTarget.localBodyIdx);
        wireDrag.attachPoint = grabPoint;
        wireDrag.localAttachPoint = visualWorldPointToLocal(*wireTarget.entry, grabPoint);
        wireDrag.target = grabPoint;
        wireDrag.pendingRequestIndex = kNoPendingRequest;
        lastStatus = "wire attached";
      }
    }

    if (wireDrag.active) {
      const VisualEntry* wireEntry = nullptr;
      if (requestedEntry && requestedTag == wireDrag.tag && requestedIndex == wireDrag.index) {
        wireEntry = requestedEntry;
      }
      if (!io.MouseDown[ImGuiMouseButton_Left]) {
        // Dropping the button simply stops resending CR_DRAG_OBJECT; the server
        // zeroes the wire stiffness on the next request frame.
        releaseWireDrag();
        lastStatus = "wire released";
      } else if (!wireEntry) {
        releaseWireDrag();
        lastStatus = "wire drag target lost";
      } else if (!canQueueSimControl) {
        releaseWireDrag();
        lastStatus = "wire drag: connection lost";
      } else {
        wireDrag.attachPoint = visualLocalPointToWorld(*wireEntry, wireDrag.localAttachPoint);
        glm::vec3 target = wireDrag.target;
        if (wireDragTargetFromCursor(viewer->getCamera(), viewportState, wireDrag.attachPoint,
                                     io.MousePos, target)) {
          wireDrag.target = target;
        }
        queueOrUpdateWireDrag();
      }
    }

    drawMouseForcePreview(mouseForce, viewportState, viewer->getCamera());
    drawWireDragPreview(wireDrag, viewportState, viewer->getCamera());
    if (ruler.enabled) {
      drawRulerOverlay(ruler, viewportState, viewer->getCamera());
      drawRulerCursorIcon(ruler, viewportState);
    }
    if (angle.enabled) {
      drawAngleOverlay(angle, viewportState, viewer->getCamera());
      drawAngleCursorIcon(angle, viewportState);
    }
    if (poseGrabberPickable || poseGrabber.dragging) {
      // Gizmo display stays world-axis-aligned and follows the held position
      // (or the in-progress drag target).
      const glm::vec3 drawOriginWorld =
        poseGrabber.dragging ? poseGrabber.currentTarget : gizmoOriginWorld;
      GizmoScreenLayout draw =
        computeGizmoLayout(viewer->getCamera(), viewportState, drawOriginWorld);
      const int activeAxis = poseGrabber.dragging ? poseGrabber.axis : -1;
      drawPoseGrabberOverlay(draw,
                             poseGrabber.mode, activeAxis,
                             gizmoHover.mode, gizmoHover.axis);
    }

    // Headless test harness uses --inspect-close-after-frames; surface per-frame
    // render times so close/reload regressions are easy to spot in CI output.
    if (options.inspectCloseAfterFrames >= 0) {
      const auto frameMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - tFrameStart).count();
      // Filter out the chatter: only log fast warmup-period frames if they're slow.
      // The post-load frames are always logged so close/reload regressions show up.
      const bool inspectorFocus = inspector.active ||
                                  frameSerial < 5 ||
                                  frameSerial == options.inspectAfterFrames ||
                                  frameSerial == options.inspectAfterFrames + 1 ||
                                  frameMs > 50;
      if (inspectorFocus) {
        std::cerr << "[inspect-close-test] frame=" << frameSerial << " renderViewer="
                  << frameMs << "ms\n";
      }
    }
    // Diagnostic: at frame 0 (empty viewer) and frame just before the inspect load,
    // print how many shaders the renderer has actually compiled so we can tell what
    // the lazy path needs vs what background warmup adds.
    if (options.inspectCloseAfterFrames >= 0 &&
        (frameSerial == 1 ||
         frameSerial == options.inspectAfterFrames + 1)) {
      auto d = viewer->shaderWarmupDiagnostics();
      std::cerr << "[shader-diag] frame=" << frameSerial
                << " registered=" << d.shaderProgramCount
                << " linked=" << d.linkedProgramCount;
      auto names = viewer->linkedShaderNames();
      std::cerr << " {";
      for (size_t i = 0; i < names.size(); ++i) {
        std::cerr << (i ? ", " : "") << names[i];
      }
      std::cerr << "}\n";
    }
    // Background shader warmup: compile one targeted shader per frame so the viewer
    // stays interactive while heavy programs (pbrMeshHigh, etc.) build up.
    if (shaderWarmupActive && pane.id == leadPaneId) {
      if (shaderWarmupIdx >= kViewerWarmupShaders.size()) {
        shaderWarmupActive = false;
        const auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - shaderWarmupStart).count();
        std::cerr << "[rayrai] shader warmup complete: " << totalMs << " ms wall time\n";
      } else {
        const std::string& name = kViewerWarmupShaders[shaderWarmupIdx++];
        const long long ms = viewer->compileShaderByName(name);
        std::cerr << "[rayrai] compiled '" << name << "' in " << ms << " ms\n";
      }
    }
    if (pane.id == leadPaneId) {
      frameSerial++;
      fpsMeasureFrames++;
    }
    stats.frames++;
    // Headless deferred --inspect: load only after warmup has had time to tick.
    if (options.inspectAfterFrames >= 0 && !options.inspectorPath.empty() &&
        !inspector.active && frameSerial == options.inspectAfterFrames) {
      std::cerr << "[inspect-after-frames] triggering inspect load at frame "
                << frameSerial << "\n";
      const auto loadStart = std::chrono::steady_clock::now();
      if (!loadAsInspector(options.inspectorPath.string())) {
        std::cerr << "ERROR: deferred --inspect failed: " << inspector.lastError << "\n";
      } else {
        const auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - loadStart).count();
        std::cerr << "[inspect-after-frames] load returned in " << loadMs << " ms\n";
      }
    }
    // Headless close-inspector reproducer. After the chosen number of frames, click
    // the close path; after another batch, quit. Lets CI catch close-inspector segfaults.
    if (options.inspectCloseAfterFrames >= 0 && inspector.active &&
        frameSerial == options.inspectCloseAfterFrames) {
      std::cerr << "[inspect-close-test] closing inspector at frame " << frameSerial << "\n";
      closeInspector();
    }
    // After close, if --inspect-reload was passed, trigger a second load so we can
    // observe the post-close first-frame cost (should be ~zero because shaders cached).
    if (options.inspectCloseAfterFrames >= 0 && !inspector.active &&
        !options.inspectReloadPath.empty() &&
        frameSerial == options.inspectCloseAfterFrames + 2) {
      std::cerr << "[inspect-close-test] reloading "
                << options.inspectReloadPath.string() << "\n";
      loadAsInspector(options.inspectReloadPath.string());
      options.inspectReloadPath.clear();  // one-shot
    }
    if (options.inspectCloseAfterFrames >= 0 && !inspector.active &&
        options.inspectReloadPath.empty() &&
        frameSerial >= options.inspectCloseAfterFrames * 2 + 2) {
      std::cerr << "[inspect-close-test] exiting after post-close frames\n";
      quit = true;
    }
    if (screenshotRequested) {
      const std::filesystem::path captureDir(screenshotDirBuf);
      const std::filesystem::path path = pendingScreenshotPath.empty()
        ? timestampedCapturePath(captureDir, "rayrai_tcp_viewer")
        : pendingScreenshotPath;
      saveViewerTexturePng(*viewer, path, captureStatus);
      pendingScreenshotPath.clear();
      screenshotRequested = false;
      if (options.exitAfterScreenshot) {
        quit = true;
      }
    }
    // One readback feeds both recorders, so enabling the PNG sequence and the
    // video encoder together costs a single glGetTexImage per frame.
    const bool captureThisFrame = frameSerial % std::max(1, recordEveryNFrames) == 0;
    const bool wantPngFrame = recordPngSequence && captureThisFrame;
    const bool wantVideoFrame = videoEncoder.framesDue() != 0;
    if (wantPngFrame || wantVideoFrame) {
      int captureWidth = 0;
      int captureHeight = 0;
      if (captureViewerRgba(*viewer, captureRgba, captureWidth, captureHeight, captureStatus)) {
        if (wantPngFrame) {
          const std::string frameName =
            raisin::tcp_viewer::pngSequenceFrameName(recordFramePrefix, recordFrameIndex++);
          const std::filesystem::path frameDirectory = serverRequestedRecording
            ? serverRecordFrameDirectory
            : std::filesystem::path(screenshotDirBuf);
          saveRgbaPng(captureRgba, captureWidth, captureHeight, frameDirectory / frameName,
            captureStatus);
        }
        if (wantVideoFrame &&
            !videoEncoder.writeTimedFrameRgba(captureRgba.data(), captureRgba.size(), videoStatus)) {
          // writeFrameRgba() already closed the encoder (resized window, or
          // ffmpeg died); surface why so the recording does not fail silently.
          captureStatus = videoStatus;
          serverRequestedRecording = false;
        }
      }
    }
    updateStatsWindow(stats, now);

    const auto drawViewOptions = [&]() {
      ImGui::SeparatorText("Interface");
      ImGui::SetNextItemWidth(ImGui::GetFontSize() * 10.0f);
      if (drawInlineLabelSliderFloat("ui_scale_panel", "UI Scale", &uiScale, 0.8f, 2.6f, "%.2f")) {
        uiScaleUserSet = true;
        settings.uiScale = uiScale;
        settings.uiScaleUserSet = true;
        settingsDirty = true;
      }
      ImGui::SameLine();
      if (drawCompactCheckbox("Show collapsed logo", &settings.showCollapsedLogo)) {
        settingsDirty = true;
      }

      ImGui::SeparatorText("Camera");
      // Scene/selection framing is keyboard-only (F and C, listed in the H
      // shortcuts overlay); it does not need panel space in two places.
      if (drawIconTextButton(uiIcons, TcpViewerIconKind::Refresh, "Reset Camera", "view_reset_camera")) {
        requestResetCamera = true;
      }
      {
        const auto applyOrtho = [&](OrthoView v) {
          glm::vec3 mn, mx;
          if (!scene.computeSceneBounds(mn, mx)) {
            mn = glm::vec3(-1.0f);
            mx = glm::vec3(1.0f);
          }
          applyOrthoView(*viewer, v, mn, mx);
        };
        auto& cam = viewer->getCamera();
        const bool isOrtho =
          cam.getProjectionMode() == raisin::Camera::ProjectionMode::ORTHOGRAPHIC;
        ImGui::TextDisabled("Orthographic views (%s)", isOrtho ? "ortho active" : "perspective");
        // Icon plus label: the six faces read as a set, and the border glyph
        // marks which face of the box you end up looking at.
        // Three faces per row.
        struct OrthoButton {
          TcpViewerIconKind icon;
          const char* label;
          const char* id;
          OrthoView view;
        };
        constexpr OrthoButton orthoButtons[] = {
          {TcpViewerIconKind::ViewTop, "Top", "view_top", OrthoView::Top},
          {TcpViewerIconKind::ViewBottom, "Bottom", "view_bottom", OrthoView::Bottom},
          {TcpViewerIconKind::ViewFront, "Front", "view_front", OrthoView::Front},
          {TcpViewerIconKind::ViewBack, "Back", "view_back", OrthoView::Back},
          {TcpViewerIconKind::ViewLeft, "Left", "view_left", OrthoView::Left},
          {TcpViewerIconKind::ViewRight, "Right", "view_right", OrthoView::Right}};
        for (size_t i = 0; i < std::size(orthoButtons); ++i) {
          if (i % 3 != 0) {
            ImGui::SameLine();
          }
          const OrthoButton& button = orthoButtons[i];
          if (drawIconTextButton(uiIcons, button.icon, button.label, button.id)) {
            applyOrtho(button.view);
          }
        }
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::ViewPerspective,
              isOrtho ? "Perspective" : "Perspective (active)", "view_perspective")) {
          cam.setProjectionMode(raisin::Camera::ProjectionMode::PERSPECTIVE);
        }
      }

      ImGui::SeparatorText("Bookmarks");
      // Two bookmarks per row: each is a Set/Restore pair, and stacking four
      // pairs vertically cost four lines for what fits in two.
      // The theme's CellPadding.x is 1px, which left the two columns almost
      // touching. Padding applies to both sides of a cell, so this is half the
      // gap that ends up between them.
      ImVec2 bookmarkCellPadding = compactControlCellPadding(ImGui::GetStyle().CellPadding);
      bookmarkCellPadding.x = std::round(ImGui::GetFontSize() * 0.5f);
      ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, bookmarkCellPadding);
      if (ImGui::BeginTable("##bookmarks", 2, ImGuiTableFlags_SizingFixedFit)) {
        for (int i = 0; i < static_cast<int>(cameraBookmarks.size()); ++i) {
          ImGui::TableNextColumn();
          ImGui::PushID(i);
          CameraBookmark& bookmark = cameraBookmarks[static_cast<size_t>(i)];
          const std::string setLabel = "Set " + std::to_string(i + 1);
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Save, setLabel.c_str(), "set_bookmark")) {
            bookmark.valid = true;
            bookmark.position = viewer->getCamera().getPosition();
            bookmark.target = viewer->getCamera().target;
          }
          ImGui::SameLine();
          ImGui::BeginDisabled(!bookmark.valid);
          const std::string restoreLabel = "Restore " + std::to_string(i + 1);
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Focus, restoreLabel.c_str(), "restore_bookmark")) {
            applyCameraLookAt(viewer->getCamera(), bookmark.position, bookmark.target);
          }
          ImGui::EndDisabled();
          ImGui::PopID();
        }
        ImGui::EndTable();
      }
      ImGui::PopStyleVar();

      ImGui::SeparatorText("Window");
      if (drawIconTextButton(uiIcons, TcpViewerIconKind::Options, "Toggle Fullscreen", "toggle_fullscreen")) {
        toggleFullscreenDesktop();
      }
    };

    // Everything that writes pixels to disk lives in one section, sharing one
    // output directory: a still, an encoded movie, and a raw frame sequence are
    // three answers to the same question, so splitting them across the panel
    // (and duplicating the still capture in the Connection tab) only made the
    // user hunt. The protocol log is deliberately a separate section below,
    // because it is not pixels.

    // One browse affordance for every path field, so they all open the same
    // picker and behave identically.
    const auto drawBrowseButton = [&](const char* id, FileBrowserMode mode, const char* title,
                                      const std::filesystem::path& startPath,
                                      std::vector<std::string> extensions,
                                      std::function<void(const std::filesystem::path&)> onAccept) {
      ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
      const bool pressed = drawIconOnlyButton(uiIcons, TcpViewerIconKind::Folder, "Browse...", id);
      if (pressed) {
        openFileBrowser(fileBrowser, mode, title, startPath, std::move(extensions),
                        std::move(onAccept));
      }
      return pressed;
    };

    const auto drawCaptureOptions = [&]() {
      const float controlWidth = fontScaledTextControlWidth(28.0f);

      ImGui::SeparatorText("Screenshot");
      // Shared by the still capture and the PNG frame sequence below.
      ImGui::TextDisabled("Output folder");
      drawCompactPathInput("##ScreenshotDirectory", screenshotDirBuf, sizeof(screenshotDirBuf),
        controlWidth, pathFieldEditing, "path/to/output/folder");
      drawBrowseButton("browse_screenshot_dir", FileBrowserMode::Folder, "Select output folder",
        std::filesystem::path(screenshotDirBuf), {},
        [&](const std::filesystem::path& chosen) {
          std::snprintf(screenshotDirBuf, sizeof(screenshotDirBuf), "%s",
                        chosen.string().c_str());
        });

      // --- Still image ---
      if (drawIconTextButton(uiIcons, TcpViewerIconKind::Camera, "Screenshot",
                             "capture_screenshot")) {
        screenshotRequested = true;
      }

      ImGui::SeparatorText("Video");
      {
        if (!ffmpegAvailable) {
          ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + controlWidth);
          ImGui::TextDisabled(
            "Install ffmpeg (or point $RAYRAI_FFMPEG at a build) to record video directly. "
            "Until then, recordings fall back to the PNG frame sequence below.");
          ImGui::PopTextWrapPos();
        }
        ImGui::BeginDisabled(!ffmpegAvailable);
        ImGui::BeginDisabled(videoEncoder.isOpen());
        drawCompactPathInput("##VideoFile", videoPathBuf, sizeof(videoPathBuf), controlWidth,
          pathFieldEditing, "path/to/clip.mp4");
        drawBrowseButton("browse_video_path", FileBrowserMode::SaveFile, "Select video file",
          std::filesystem::path(videoPathBuf), {"mp4", "mov", "mkv"},
          [&](const std::filesystem::path& chosen) {
            std::snprintf(videoPathBuf, sizeof(videoPathBuf), "%s", chosen.string().c_str());
          });
        float videoFpsUi = static_cast<float>(videoFramesPerSecond);
        if (drawStepperFloat("video_fps", "Frame rate (fps)", &videoFpsUi, 1.0f,
              static_cast<float>(raisin::tcp_viewer::kMinVideoFramesPerSecond),
              static_cast<float>(raisin::tcp_viewer::kMaxVideoFramesPerSecond), "%.0f")) {
          videoFramesPerSecond = static_cast<double>(videoFpsUi);
        }
        // libx264 CRF, so a lower number means a bigger, better-looking file.
        drawStepperInt("video_quality", "CRF (lower = better)", &videoQuality, 1,
          raisin::tcp_viewer::kMinVideoQuality, raisin::tcp_viewer::kMaxVideoQuality);
        ImGui::EndDisabled();
        if (!videoEncoder.isOpen()) {
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Video, "Start Recording",
                                 "start_video_recording")) {
            raisin::tcp_viewer::VideoEncoderSettings encoderSettings;
            encoderSettings.width = viewer->getCamera().rtWidth();
            encoderSettings.height = viewer->getCamera().rtHeight();
            encoderSettings.framesPerSecond = videoFramesPerSecond;
            encoderSettings.quality = videoQuality;
            const auto output = raisin::tcp_viewer::nextAvailableOutputPath(
                std::filesystem::path(videoPathBuf));
            std::snprintf(videoPathBuf, sizeof(videoPathBuf), "%s",
                          output.string().c_str());
            videoEncoder.open(output, encoderSettings, videoStatus);
            captureStatus = videoStatus;
          }
        } else {
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Stop, "Stop Recording",
                                 "stop_video_recording")) {
            videoEncoder.close(videoStatus);
            captureStatus = videoStatus;
          }
          ImGui::TextDisabled("%zu frames | %.1f MiB | %dx%d", videoEncoder.frameCount(),
            static_cast<double>(videoEncoder.bytesWritten()) / (1024.0 * 1024.0),
            videoEncoder.settings().width, videoEncoder.settings().height);
        }
        ImGui::EndDisabled();
      }

      ImGui::SeparatorText("PNG sequence");
      {
        drawCompactCheckbox("Save every rendered frame", &recordPngSequence);
        ImGui::BeginDisabled(!recordPngSequence);
        drawInlineLabelSliderInt("png_sequence_every_n_frames", "Every N frames",
          &recordEveryNFrames, 1, 120);
        ImGui::EndDisabled();
        // Rescues a sequence recorded before ffmpeg was available, or one whose
        // recording was interrupted.
        ImGui::BeginDisabled(!ffmpegAvailable || videoEncoder.isOpen() || recordPngSequence ||
                             recordFrameIndex <= 0);
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Export, "Encode To Video",
                               "encode_png_sequence")) {
          raisin::tcp_viewer::VideoEncoderSettings encoderSettings;
          encoderSettings.framesPerSecond = videoFramesPerSecond;
          encoderSettings.quality = videoQuality;
          const std::filesystem::path frameDirectory = serverRecordFrameDirectory.empty()
            ? std::filesystem::path(screenshotDirBuf)
            : serverRecordFrameDirectory;
          raisin::tcp_viewer::encodePngSequenceToVideo(frameDirectory, recordFramePrefix,
            std::filesystem::path(videoPathBuf), encoderSettings, videoStatus);
          captureStatus = videoStatus;
        }
        ImGui::EndDisabled();
      }

      // One status line for all three sections, so a screenshot, a recording,
      // and a sequence encode all report in the same place.
      if (!captureStatus.empty()) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + controlWidth);
        ImGui::TextDisabled("%s", shortenPathLabel(captureStatus, 90).c_str());
        ImGui::PopTextWrapPos();
      }

      // Not pixels: this records the protocol stream. The old "Start TCP
      // Recording" label sat next to the video controls and read as if it
      // produced a video file.
      ImGui::SeparatorText("Session Replay Log");
      ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + controlWidth);
      ImGui::TextDisabled(
        "Records the scene updates themselves to a .rrtcs file, not video frames. Replay it later "
        "with --replay-session to scrub the timeline and re-render from any camera. For a playable "
        "movie, use Video above.");
      ImGui::PopTextWrapPos();
      drawCompactPathInput("##SessionFile", sessionPathBuf, sizeof(sessionPathBuf),
        controlWidth, pathFieldEditing, "path/to/session.rrtcs");
      drawBrowseButton("browse_session_path", FileBrowserMode::SaveFile, "Select session log",
        std::filesystem::path(sessionPathBuf), {"rrtcs"},
        [&](const std::filesystem::path& chosen) {
          std::snprintf(sessionPathBuf, sizeof(sessionPathBuf), "%s", chosen.string().c_str());
        });
      if (!sessionRecorder.active()) {
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Save, "Start Session Log",
                               "start_session_log")) {
          const auto output = raisin::tcp_viewer::nextAvailableOutputPath(
              std::filesystem::path(sessionPathBuf));
          std::snprintf(sessionPathBuf, sizeof(sessionPathBuf), "%s",
                        output.string().c_str());
          sessionRecorder.open(output, sessionStatus);
        }
      } else {
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Stop, "Stop Session Log",
                               "stop_session_log")) {
          sessionRecorder.close();
          sessionStatus = "recorded " + std::to_string(sessionRecorder.frameCount()) +
                          " frames to " + sessionRecorder.pathString();
        }
        ImGui::TextDisabled("%zu frames | %.1f MiB", sessionRecorder.frameCount(),
          static_cast<double>(sessionRecorder.byteCount()) / (1024.0 * 1024.0));
      }

      if (replayMode) {
        ImGui::SeparatorText("Replay");
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Refresh,
              replayPaused ? "Resume Replay" : "Pause Replay", "toggle_replay")) {
          replayPaused = !replayPaused;
          replayStart = std::chrono::steady_clock::now();
          replayBaseMicros = replayIndex < replayFrames.size() ? replayFrames[replayIndex].timeMicros
                                                                : replayFrames.back().timeMicros;
        }
        ImGui::SameLine();
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Step, "Step", "step_replay")) {
          replayPaused = true;
          replayStep = true;
        }
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Home, "Restart Replay", "restart_replay")) {
          clearSceneState();
          replayIndex = 0;
          replayStart = std::chrono::steady_clock::now();
          replayBaseMicros = replayFrames.empty() ? 0 : replayFrames.front().timeMicros;
          replayPaused = false;
        }
        drawInlineLabelSliderFloat("replay_speed", "Replay speed", &replaySpeed, 0.05f, 8.0f, "%.2f");
        const uint64_t firstMicros = replayFrames.empty() ? 0 : replayFrames.front().timeMicros;
        const uint64_t lastMicros = replayFrames.empty() ? 0 : replayFrames.back().timeMicros;
        const float durationSeconds = static_cast<float>(lastMicros - firstMicros) / 1.0e6f;
        const uint64_t currentMicros = replayIndex == 0
          ? firstMicros
          : replayFrames[std::min(replayIndex - 1u, replayFrames.size() - 1u)].timeMicros;
        float timelineSeconds = static_cast<float>(currentMicros - firstMicros) / 1.0e6f;
        if (durationSeconds > 0.0f && drawInlineLabelSliderFloat(
              "replay_timeline", "Timeline", &timelineSeconds, 0.0f,
              durationSeconds, "%.3f s")) {
          replayPaused = true;
          const uint64_t targetMicros = firstMicros + static_cast<uint64_t>(
            std::max(0.0f, timelineSeconds) * 1.0e6f);
          replaySeekIndex = findSessionFrameAtOrAfter(replayFrames, targetMicros);
        }
        ImGui::TextDisabled("frame %zu / %zu", std::min(replayIndex, replayFrames.size()), replayFrames.size());
      }
      if (!captureStatus.empty()) {
        ImGui::TextDisabled("%s", shortenPathLabel(captureStatus, 80).c_str());
      }
      if (!sessionStatus.empty()) {
        ImGui::TextDisabled("%s", shortenPathLabel(sessionStatus, 90).c_str());
      }
    };

    const auto drawRenderingOptions = [&]() {
      bool changed = false;
      bool detailChanged = false;
      const ImGuiStyle& renderStyle = ImGui::GetStyle();
      const float colorSwatchWidth = std::round(ImGui::GetFontSize() * 4.2f);
      const auto comboWidthFor = [&](const char* const* items, int itemCount) {
        float maxTextWidth = 0.0f;
        for (int i = 0; i < itemCount; ++i) {
          maxTextWidth = std::max(maxTextWidth, ImGui::CalcTextSize(items[i]).x);
        }
        return std::ceil(maxTextWidth + renderStyle.FramePadding.x * 2.0f +
                         renderStyle.ItemInnerSpacing.x + ImGui::GetFrameHeight());
      };
      const auto drawBackgroundColorPicker = [&]() {
        glm::vec4 color = settings.backgroundColorRgb255 / 255.0f;
        color.r = std::clamp(color.r, 0.0f, 1.0f);
        color.g = std::clamp(color.g, 0.0f, 1.0f);
        color.b = std::clamp(color.b, 0.0f, 1.0f);
        color.a = std::clamp(color.a, 0.0f, 1.0f);
        bool colorChanged = false;
        ImGui::PushID("BackgroundColor");
        const ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_DisplayRGB |
          ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueBar |
          ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_AlphaPreviewHalf;
        if (ImGui::ColorButton("##swatch", ImVec4(color.r, color.g, color.b, color.a),
              colorFlags, ImVec2(colorSwatchWidth, ImGui::GetFrameHeight()))) {
          ImGui::OpenPopup("picker");
        }
        if (ImGui::BeginPopup("picker")) {
          colorChanged |= ImGui::ColorPicker4("##picker", &color.x, colorFlags);
          ImGui::EndPopup();
        }
        ImGui::PopID();
        if (colorChanged) {
          settings.backgroundColorRgb255 = glm::vec4(
            std::clamp(color.r, 0.0f, 1.0f) * 255.0f,
            std::clamp(color.g, 0.0f, 1.0f) * 255.0f,
            std::clamp(color.b, 0.0f, 1.0f) * 255.0f,
            std::clamp(color.a, 0.0f, 1.0f) * 255.0f);
        }
        return colorChanged;
      };
      // Swatch first, then its label, so all three light colours share one row.
      // The chip is square -- a wide bar read as a slider track, and squaring it
      // is also what makes three of them fit on a single line.
      const auto drawLightColorPicker = [&](const char* label, const char* id, glm::vec3& color) {
        bool colorChanged = false;
        ImGui::PushID(id);
        const ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_DisplayRGB |
          ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueBar |
          ImGuiColorEditFlags_HDR;
        const float swatchExtent = ImGui::GetFrameHeight();
        if (ImGui::ColorButton("##swatch", ImVec4(color.r, color.g, color.b, 1.0f),
              colorFlags, ImVec2(swatchExtent, swatchExtent))) {
          ImGui::OpenPopup("picker");
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("%s light colour", label);
        }
        if (ImGui::BeginPopup("picker")) {
          colorChanged |= ImGui::ColorPicker3("##picker", &color.x, colorFlags);
          ImGui::EndPopup();
        }
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::PopID();
        return colorChanged;
      };
      ImGui::SeparatorText("Quality");
      int quality = std::clamp(settings.renderQuality, 0, 4);
      constexpr const char* qualityItems[] = {"Fast", "Balanced", "High", "Ultra", "Custom"};
      // A discrete slider rather than a dropdown: the presets are ordered from
      // cheapest to most expensive, so the position itself carries meaning and
      // stepping between neighbours takes one drag instead of two clicks. The
      // format string is the preset name, so the slider shows "High", not "2".
      ImGui::SetNextItemWidth(fontScaledTextControlWidth(16.0f));
      if (compactSliderInt("##RenderQuality", &quality, 0,
            IM_ARRAYSIZE(qualityItems) - 1, qualityItems[quality])) {
        quality = std::clamp(quality, 0, IM_ARRAYSIZE(qualityItems) - 1);
        settings.renderQualityUserSet = true;
        if (quality == 4) {
          settings.renderQuality = quality;
        } else {
          copyRenderDefaultsToSettings(settings, quality);
          settings.renderQualityUserSet = true;
        }
        changed = true;
      }

      ImGui::SeparatorText("Background");
      detailChanged |= drawBackgroundColorPicker();

      ImGui::SeparatorText("Sky");
      detailChanged |= drawCompactCheckbox("Enabled", &settings.skyEnabled);
      const bool weatherAllowed = weatherDefaultEnabledForQuality(settings.renderQuality);
      if (!weatherAllowed) {
        settings.skyWeatherEnabled = false;
      }
      ImGui::BeginDisabled(!settings.skyEnabled);
      ImGui::BeginDisabled(!weatherAllowed);
      detailChanged |= drawCompactCheckbox("Weather model", &settings.skyWeatherEnabled);
      ImGui::EndDisabled();
      if (weatherAllowed && settings.skyWeatherEnabled) {
        constexpr const char* weatherPresetItems[] = {
          "Clear", "Hazy", "Overcast", "Fog", "Rain", "Heavy Rain",
          "Snow", "Storm", "Night Clear", "Night Rain", "Custom"};
        int weatherPreset = std::clamp(settings.skyWeatherPreset, 0, 10);
        ImGui::TextUnformatted("Weather");
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 9.0f);
        if (ImGui::Combo("##SkyWeatherPreset", &weatherPreset, weatherPresetItems,
              IM_ARRAYSIZE(weatherPresetItems))) {
          copyWeatherPresetToSettings(settings, weatherPreset);
          detailChanged = true;
        }
        constexpr const char* weatherQualityItems[] = {"Low", "Medium", "High", "Ultra"};
        int weatherQuality = std::clamp(settings.skyWeatherQuality, 0, 3);
        ImGui::TextUnformatted("Quality");
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 7.0f);
        if (ImGui::Combo("##SkyWeatherQuality", &weatherQuality, weatherQualityItems,
              IM_ARRAYSIZE(weatherQualityItems))) {
          settings.skyWeatherQuality = weatherQuality;
          detailChanged = true;
        }
        if (ImGui::TreeNodeEx("Time & Sun", ImGuiTreeNodeFlags_DefaultOpen)) {
          detailChanged |= drawInlineLabelSliderFloat("render_sky_time_of_day",
            "Time", &settings.skyTimeOfDayHours, 0.0f, 24.0f, "%.2f h");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_latitude",
            "Latitude", &settings.skyLatitude, -89.9f, 89.9f, "%.1f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_longitude",
            "Longitude (deg)", &settings.skyLongitude, -180.0f, 180.0f, "%.2f deg");
          detailChanged |= drawCompactCheckbox(
            "Automatic solar offset", &settings.skyAutomaticUtcOffset);
          ImGui::BeginDisabled(settings.skyAutomaticUtcOffset);
          detailChanged |= drawInlineLabelSliderFloat("render_sky_utc_offset",
            "Civil UTC offset", &settings.skyUtcOffsetHours, -12.0f, 14.0f, "%+.2f h");
          ImGui::EndDisabled();
          if (settings.skyAutomaticUtcOffset) {
            ImGui::Text("Solar offset: %+.2f h", settings.skyLongitude / 15.0f);
          }
          detailChanged |= drawInlineLabelSliderInt("render_sky_year",
            "Year", &settings.skyYear, 1900, 2500);
          detailChanged |= drawInlineLabelSliderInt("render_sky_month",
            "Month", &settings.skyMonth, 1, 12);
          detailChanged |= drawInlineLabelSliderInt("render_sky_day",
            "Day", &settings.skyDay, 1, 31);
          detailChanged |= drawCompactCheckbox("Explicit sun", &settings.skyUseExplicitSunAngles);
          ImGui::BeginDisabled(!settings.skyUseExplicitSunAngles);
          detailChanged |= drawInlineLabelSliderFloat("render_sky_sun_azimuth",
            "Azimuth", &settings.skySunAzimuthDeg, 0.0f, 360.0f, "%.1f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_sun_elevation",
            "Elevation", &settings.skySunElevationDeg, -8.0f, 89.0f, "%.1f");
          ImGui::EndDisabled();
          detailChanged |= drawInlineLabelSliderFloat("render_sky_sun_size",
            "Sun size", &settings.skySunSize, 0.001f, 0.08f, "%.3f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_moon_size",
            "Moon size", &settings.skyMoonSize, 0.001f, 0.08f, "%.3f");
          ImGui::TreePop();
        }
        if (ImGui::TreeNodeEx("Clouds", ImGuiTreeNodeFlags_DefaultOpen)) {
          // Cloud quality: "Auto" picks per render preset (Fast/Balanced → Texture,
          // High/Ultra → Volumetric) and follows the active weather preset; Off
          // disables clouds entirely; Texture/Volumetric force a specific path.
          constexpr const char* cloudQualityItems[] = {"Auto", "Off", "Texture", "Volumetric"};
          int cloudQ = std::clamp(settings.skyCloudQuality, 0, 3);
          ImGui::TextUnformatted("Quality");
          ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8.0f);
          if (ImGui::Combo("##SkyCloudQuality", &cloudQ, cloudQualityItems,
                IM_ARRAYSIZE(cloudQualityItems))) {
            settings.skyCloudQuality = cloudQ;
            detailChanged = true;
          }
          detailChanged |= drawInlineLabelSliderFloat("render_sky_cloud_coverage",
            "Coverage", &settings.skyCloudCoverage, 0.0f, 1.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_cloud_density",
            "Density", &settings.skyCloudDensity, 0.0f, 1.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_cloud_altitude",
            "Altitude", &settings.skyCloudAltitudeMeters, 20.0f, 12000.0f, "%.0f m");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_cloud_thickness",
            "Thickness", &settings.skyCloudThicknessMeters, 1.0f, 4000.0f, "%.0f m");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_cloud_shadow",
            "Shadow", &settings.skyCloudShadowStrength, 0.0f, 1.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_cloud_scale",
            "Scale", &settings.skyCloudScale, 0.01f, 2.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_cloud_animation",
            "Animation", &settings.skyCloudAnimationSpeed, 0.0f, 200.0f, "%.1f");
          ImGui::TreePop();
        }
        if (ImGui::TreeNodeEx("Atmosphere", ImGuiTreeNodeFlags_DefaultOpen)) {
          detailChanged |= drawInlineLabelSliderFloat("render_sky_air_turbidity",
            "Turbidity", &settings.skyAirTurbidity, 1.0f, 12.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_visibility",
            "Visibility", &settings.skyVisibilityMeters, 1.0f, 100000.0f, "%.0f m");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_fog_density",
            "Fog", &settings.skyFogDensity, 0.0f, 1.0f, "%.4f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_fog_anisotropy",
            "Fog phase", &settings.skyFogAnisotropy, -0.85f, 0.85f, "%.2f");
          ImGui::TextUnformatted("Fog color");
          detailChanged |= ImGui::ColorEdit3("##SkyFogColor", &settings.skyFogColor.x);
          detailChanged |= drawInlineLabelSliderFloat("render_sky_ground_albedo",
            "Ground albedo", &settings.skyGroundAlbedo, 0.0f, 1.0f, "%.2f");
          ImGui::TreePop();
        }
        if (ImGui::TreeNodeEx("Weather Effects", ImGuiTreeNodeFlags_DefaultOpen)) {
          detailChanged |= drawInlineLabelSliderFloat("render_sky_precipitation",
            "Precip", &settings.skyPrecipitationRate, 0.0f, 1.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_rain_occlusion",
            "Rain occlusion", &settings.skyRainOcclusionStrength, 0.0f, 1.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_snow",
            "Snow", &settings.skySnowCoverage, 0.0f, 1.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_humidity",
            "Humidity", &settings.skyHumidity, 0.0f, 1.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_wetness",
            "Wetness", &settings.skyWetness, 0.0f, 1.0f, "%.2f");
          detailChanged |= drawCompactCheckbox("Accumulate wetness", &settings.skyWetnessAccumulationEnabled);
          ImGui::BeginDisabled(!settings.skyWetnessAccumulationEnabled);
          detailChanged |= drawInlineLabelSliderFloat("render_sky_wetness_accum_rate",
            "Wet gain", &settings.skyWetnessAccumulationRate, 0.0f, 4.0f, "%.2f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_wetness_dry_rate",
            "Drying", &settings.skyWetnessDryingRate, 0.0f, 4.0f, "%.2f");
          ImGui::EndDisabled();
          detailChanged |= drawInlineLabelSliderFloat("render_sky_lightning",
            "Lightning", &settings.skyLightningRate, 0.0f, 16.0f, "%.2f");
          detailChanged |= drawCompactCheckbox("Lens droplets", &settings.skyLensDropletsEnabled);
          ImGui::BeginDisabled(!settings.skyLensDropletsEnabled);
          detailChanged |= drawInlineLabelSliderFloat("render_sky_lens_droplet_strength",
            "Droplets", &settings.skyLensDropletStrength, 0.0f, 1.0f, "%.2f");
          ImGui::EndDisabled();
          ImGui::TreePop();
        }
        if (ImGui::TreeNodeEx("Wind", ImGuiTreeNodeFlags_DefaultOpen)) {
          detailChanged |= drawInlineLabelSliderFloat("render_sky_wind_direction",
            "Direction", &settings.skyWindDirectionDeg, 0.0f, 360.0f, "%.1f");
          detailChanged |= drawInlineLabelSliderFloat("render_sky_wind_speed",
            "Speed", &settings.skyWindSpeed, 0.0f, 80.0f, "%.1f m/s");
          detailChanged |= drawInlineLabelSliderInt("render_sky_seed",
            "Seed", &settings.skyWeatherSeed, 1, 1000000);
          ImGui::TreePop();
        }
      } else {
        detailChanged |= drawInlineLabelSliderFloat("render_sky_sun_strength",
          "Sun", &settings.skySunStrength, 0.0f, 8.0f, "%.2f");
        detailChanged |= drawInlineLabelSliderFloat("render_sky_sun_size",
          "Sun size", &settings.skySunSize, 0.001f, 0.08f, "%.3f");
      }
      ImGui::EndDisabled();

      ImGui::SeparatorText("Camera");
      detailChanged |= drawInlineLabelSliderFloat("render_move_speed", "Move speed", &settings.cameraSpeed, 0.1f, 30.0f, "%.1f");
      detailChanged |= drawInlineLabelSliderFloat("render_fov_deg", "FOV (deg)", &settings.cameraFovDeg, 20.0f, 100.0f, "%.1f");
      detailChanged |= drawInlineLabelSliderFloat("render_near_clip", "Near clip", &settings.cameraNear, 0.001f, 1.0f, "%.3f");
      detailChanged |= drawInlineLabelSliderFloat("render_far_clip", "Far clip", &settings.cameraFar, 10.0f, 5000.0f, "%.0f");

      ImGui::SeparatorText("Light");
      detailChanged |= drawInlineLabelSliderFloat("render_key_strength", "Key strength", &settings.lightStrength, 0.0f, 2.0f, "%.2f");
      ImGui::BeginDisabled(settings.skyEnabled && settings.skyWeatherEnabled);
      detailChanged |= drawInlineLabelSliderFloat("render_light_yaw", "Yaw (deg)", &settings.lightYawDeg, -180.0f, 180.0f, "%.1f");
      detailChanged |= drawInlineLabelSliderFloat("render_light_pitch", "Pitch (deg)", &settings.lightPitchDeg, -89.0f, 89.0f, "%.1f");
      detailChanged |= drawInlineLabelSliderFloat("render_ambient", "Ambient", &settings.ambientStrength, 0.0f, 2.0f, "%.2f");
      // One property of one light, so all three share a row. Square chips make
      // that fit without shrinking anything.
      detailChanged |= drawLightColorPicker("Ambient", "LightAmbientColor",
        settings.mainLightAmbient);
      ImGui::SameLine();
      detailChanged |= drawLightColorPicker("Diffuse", "LightDiffuseColor",
        settings.mainLightDiffuse);
      ImGui::SameLine();
      detailChanged |= drawLightColorPicker("Specular", "LightSpecularColor",
        settings.mainLightSpecular);
      ImGui::EndDisabled();
      detailChanged |= drawCompactCheckbox("Fill/rim lights", &settings.addViewerFillLights);

      ImGui::SeparatorText("Shadows");
      detailChanged |= drawCompactCheckbox("Enabled", &settings.shadowsEnabled);
      int shadowResolutionIndex =
        settings.shadowResolution <= 1024 ? 0 : settings.shadowResolution <= 2048 ? 1 :
        settings.shadowResolution <= 4096 ? 2 : 3;
      constexpr const char* shadowResolutionItems[] = {"1024", "2048", "4096", "8192"};
      ImGui::TextUnformatted("Shadow map");
      ImGui::SetNextItemWidth(comboWidthFor(shadowResolutionItems, IM_ARRAYSIZE(shadowResolutionItems)));
      if (ImGui::Combo("##ShadowMap", &shadowResolutionIndex, shadowResolutionItems,
            IM_ARRAYSIZE(shadowResolutionItems))) {
        constexpr int values[] = {1024, 2048, 4096, 8192};
        settings.shadowResolution = values[shadowResolutionIndex];
        detailChanged = true;
      }
      detailChanged |= drawInlineLabelSliderFloat("render_shadow_bias", "Bias", &settings.shadowBias, 0.0f, 0.01f, "%.5f");
      detailChanged |= drawInlineLabelSliderFloat("render_shadow_strength", "Strength", &settings.shadowStrength, 0.0f, 1.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_shadow_pcf_radius", "PCF", &settings.shadowPcfRadius, 0.0f, 4.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_shadow_ortho_half_size", "Ortho half", &settings.shadowOrthoHalfSize, 1.0f, 100.0f, "%.1f");
      detailChanged |= drawInlineLabelSliderFloat("render_shadow_near", "Near", &settings.shadowNear, 0.01f, 10.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_shadow_far", "Far", &settings.shadowFar, 1.0f, 250.0f, "%.1f");
      detailChanged |= drawInlineLabelSliderFloat("render_shadow_center_offset", "Center offset", &settings.shadowCenterOffset, 0.0f, 80.0f, "%.1f");
      detailChanged |= drawCompactCheckbox("Update every frame", &settings.updateShadowsEveryFrame);
      detailChanged |= drawInlineLabelSliderInt("render_shadowed_light_budget", "Light budget", &settings.shadowedLightBudget, 0, 8);
      detailChanged |= drawInlineLabelSliderInt("render_point_shadow_lights", "Point lights", &settings.maxPointShadowLights, 0, 8);
      detailChanged |= drawInlineLabelSliderFloat("render_additional_shadow_resolution_scale",
        "Extra map scale", &settings.additionalShadowResolutionScale, 0.05f, 2.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_point_shadow_resolution_scale",
        "Point map scale", &settings.pointShadowResolutionScale, 0.05f, 2.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderInt("render_min_additional_resolution",
        "Min map size", &settings.minAdditionalShadowResolution, 64, 2048);
      detailChanged |= drawCompactCheckbox("Auto imported light", &settings.autoSelectImportedShadowLight);

      ImGui::SeparatorText("Post");
      detailChanged |= drawInlineLabelSliderFloat("render_fog_density", "Fog", &settings.fogDensity, 0.0f, 0.08f, "%.4f");
      detailChanged |= drawInlineLabelSliderFloat("render_gamma", "Gamma", &settings.gamma, 0.5f, 2.5f, "%.2f");
      int colorMode = std::clamp(settings.colorMode, 0, 4);
      const char* colorModeItems[] = {
        colorModeName(0), colorModeName(1), colorModeName(2), colorModeName(3), colorModeName(4)};
      ImGui::TextUnformatted("Color mode");
      ImGui::SetNextItemWidth(comboWidthFor(colorModeItems, IM_ARRAYSIZE(colorModeItems)));
      if (ImGui::Combo("##ColorMode", &colorMode, colorModeItems, IM_ARRAYSIZE(colorModeItems))) {
        settings.colorMode = colorMode;
        detailChanged = true;
      }
      detailChanged |= drawCompactCheckbox("FXAA", &settings.fxaaEnabled);
      detailChanged |= drawCompactCheckbox("Bloom", &settings.bloomEnabled);
      ImGui::BeginDisabled(!settings.bloomEnabled);
      detailChanged |= drawInlineLabelSliderFloat("render_bloom_threshold", "Threshold", &settings.bloomThreshold, 0.0f, 4.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_bloom_strength", "Strength", &settings.bloomStrength, 0.0f, 2.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_bloom_radius", "Radius", &settings.bloomRadius, 0.0f, 12.0f, "%.1f");
      detailChanged |= drawInlineLabelSliderFloat("render_bloom_knee", "Knee", &settings.bloomKnee, 0.0f, 1.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderInt("render_bloom_quality", "Quality", &settings.bloomQuality, 0, 3);
      ImGui::EndDisabled();
      detailChanged |= drawCompactCheckbox("Screen-space AO", &settings.screenSpaceAoEnabled);
      ImGui::BeginDisabled(!settings.screenSpaceAoEnabled);
      detailChanged |= drawInlineLabelSliderFloat("render_ao_radius", "Radius", &settings.screenSpaceAoRadius, 0.05f, 10.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_ao_strength", "Strength", &settings.screenSpaceAoStrength, 0.0f, 4.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_ao_bias", "Bias", &settings.screenSpaceAoBias, 0.0f, 0.25f, "%.3f");
      ImGui::EndDisabled();
      detailChanged |= drawCompactCheckbox("Opaque depth prepass", &settings.opaqueDepthPrepass);
      detailChanged |= drawCompactCheckbox("Depth of field", &settings.depthOfFieldEnabled);
      ImGui::BeginDisabled(!settings.depthOfFieldEnabled);
      detailChanged |= drawInlineLabelSliderFloat("render_dof_focus_distance",
        "Focus distance", &settings.depthOfFieldFocusDistance, 0.05f, 30.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_dof_focus_range",
        "Focus range", &settings.depthOfFieldFocusRange, 1.0f, 100.0f, "%.1f");
      detailChanged |= drawInlineLabelSliderFloat("render_dof_max_blur_radius",
        "Max blur radius", &settings.depthOfFieldMaxRadius, 0.0f, 8.0f, "%.2f");
      ImGui::EndDisabled();

      ImGui::SeparatorText("PBR");
      detailChanged |= drawCompactCheckbox("High fidelity", &settings.highFidelityPbr);
      detailChanged |= drawCompactCheckbox("Tone mapping", &settings.pbrToneMapping);
      detailChanged |= drawInlineLabelSliderFloat("render_pbr_exposure", "Exposure", &settings.pbrExposure, 0.1f, 4.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_pbr_environment_max_lod", "Environment LOD", &settings.pbrEnvironmentMaxLod, 0.0f, 12.0f, "%.1f");
      detailChanged |= drawInlineLabelSliderFloat("render_pbr_environment_intensity", "Environment", &settings.pbrEnvironmentIntensity, 0.0f, 4.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_pbr_key_intensity", "Key", &settings.pbrKeyLightIntensity, 0.0f, 4.0f, "%.2f");

      ImGui::SeparatorText("Ground");
      detailChanged |= drawCompactCheckbox("Reflective checkerboard", &settings.reflectiveGround);
      ImGui::BeginDisabled(!settings.reflectiveGround);
      detailChanged |= drawInlineLabelSliderFloat("render_ground_roughness",
        "Roughness", &settings.reflectiveGroundRoughness, 0.02f, 1.0f, "%.2f");
      detailChanged |= drawInlineLabelSliderFloat("render_ground_metallic",
        "Metallic", &settings.reflectiveGroundMetallic, 0.0f, 1.0f, "%.2f");
      ImGui::EndDisabled();

      ImGui::SeparatorText("Advanced");
      detailChanged |= drawCompactCheckbox("Sort transparent", &settings.sortTransparentInstances);
      detailChanged |= drawInlineLabelSliderInt("render_additional_lights_per_frame",
        "Lights per frame", &settings.maxAdditionalLightsPerFrame, 0, 16);
      detailChanged |= drawInlineLabelSliderFloat("render_min_light_influence",
        "Min influence", &settings.minAdditionalLightInfluence, 0.0f, 1.0f, "%.3f");

      if (drawIconTextButton(uiIcons, TcpViewerIconKind::Refresh, "Reset Rendering Settings", "reset_rendering_settings")) {
        ViewerSettings defaults;
        applyAutomaticRenderQualityIfUnset(defaults, gpuQuality.quality);
        defaults.uiScale = settings.uiScale;
        defaults.uiScaleUserSet = settings.uiScaleUserSet;
        defaults.showCollapsedLogo = settings.showCollapsedLogo;
        settings = defaults;
        changed = true;
      }

      changed |= detailChanged;
      if (changed) {
        if (detailChanged) {
          settings.renderQuality = 4;
          settings.renderQualityUserSet = true;
        }
        cameraSpeed = settings.cameraSpeed;
        lightYawDeg = settings.lightYawDeg;
        lightPitchDeg = settings.lightPitchDeg;
        lightStrength = settings.lightStrength;
        ambientStrength = settings.ambientStrength;
        settingsDirty = true;
      }
    };

    // Scene-file prompt and download progress, below the menu bar.
    sceneFiles.drawOverlay(ImVec2(paneOrigin.x, paneOrigin.y + menuBarHeight),
                           ImVec2(uiSize.x, uiSize.y - menuBarHeight), int(pane.id));

    const ImVec2 overlayBase(paneOrigin.x + 12.0f, paneOrigin.y + 12.0f + menuBarHeight);
    // Right edge the left overlay actually occupied this frame, so the object
    // inspector can avoid it. Starts at the pane's left edge: with no overlay
    // drawn the inspector may use the whole pane.
    float overlayRightEdge = paneOrigin.x;
    const ImVec2 overlayPos(overlayBase.x + overlayOffset.x, overlayBase.y + overlayOffset.y);
    const float collapsedPanelPadding = std::max(2.0f, std::round(ImGui::GetFontSize() * 0.18f));
    const bool collapsedLogoVisible = overlayMinimized && settings.showCollapsedLogo && raisimLogo.valid();
    // Inspector mode is a modal alternative to the TCP client UI — hide the entire
    // left overlay (connection / render / objects / diagnostics tabs) while it's
    // active so the user isn't tempted to interact with mutually-exclusive state.
    const bool overlayVisible = !inspector.active;
    bool overlayHovered = false;
    if (overlayVisible) {
    // The collapsed logo sits on a light chip because the wordmark is dark ink;
    // that chip is fully opaque so no scene colour bleeds through the logo.
    ImGui::SetNextWindowBgAlpha(collapsedLogoVisible ? kCollapsedLogoBackdropAlpha : 0.5f);
    ImGui::SetNextWindowPos(overlayPos, ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
      overlayMinimized ? ImVec2(collapsedPanelPadding, collapsedPanelPadding) : ImVec2(12.0f, 10.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, ImGui::GetStyle().WindowBorderSize);
    const int overlayColorPushCount = collapsedLogoVisible ? 2 : 0;
    if (collapsedLogoVisible) {
      ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.96f, 0.97f, 0.99f, 1.0f));
      ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.86f, 0.88f, 0.92f, 0.95f));
    }
    // Auto-resize alone would let a tall panel run past the bottom of a small
    // pane and over its neighbour, so cap it at the pane the panel belongs to.
    ImGui::SetNextWindowSizeConstraints(
      ImVec2(0.0f, 0.0f),
      ImVec2(FLT_MAX, std::max(ImGui::GetFontSize() * 6.0f,
                               uiSize.y - (overlayPos.y - paneOrigin.y) - 12.0f)));
    if (ImGui::Begin(overlayWindowName.c_str(), nullptr, kOverlayPanelWindowFlags)) {
      overlayHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
      overlayRightEdge = ImGui::GetWindowPos().x + ImGui::GetWindowSize().x;
      const bool overlayHoverShouldOpen = overlayMinimized && overlayHovered &&
                                          !overlayCollapsedHoveredLastFrame;
      const ImGuiStyle& style = ImGui::GetStyle();
      if (overlayHoverShouldOpen) {
        overlayMinimized = false;
        overlayLastInteractionTime = now;
      }

      // With no session, this pane is a connect prompt: every other tab, and
      // most of the Connection tab, describes a scene that is not there yet.
      const bool sessionActive =
        (client.isConnected() && sceneReceived) || replayMode || localSimulation.active();

      // The Connection tab's contents, which are also drawn on their own when
      // there is no session and the tab bar would hold a single tab.
      const auto drawConnectionTab = [&]() {
        ImGui::PushStyleColor(ImGuiCol_TextDisabled,
          raionrobotics_imgui_secondary_text_color());
        ConnectionEntry current;
        current.host = host;
        current.port = port;
        std::string preview = formatConnectionLabel(current);
        if (preview.empty()) {
          preview = "set host:port";
        }
        const float comboLabelWidth = ImGui::CalcTextSize(preview.c_str()).x;
        const float minHostTextWidth = ImGui::CalcTextSize("255.255.255.255").x;
        const float minPortTextWidth = ImGui::CalcTextSize("0").x;
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
            int parsed = port;
            if (parsePortStrict(portBuf, parsed)) {
              port = parsed;
            }
          }
          ImGui::SeparatorText("Remote server");
          ImGui::TextDisabled("Enter any DNS name or IP address and port above.");
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Save, "Save Endpoint", "save_endpoint")) {
            ConnectionEntry endpoint;
            if (normalizeConnectionEndpoint(host, port, endpoint)) {
              setPaneEndpoint(pane, endpoint.host, endpoint.port);
              recordConnection(recentConnections, endpoint.host, endpoint.port);
              settingsDirty = true;
            } else {
              lastStatus = "invalid endpoint";
            }
          }
          // While disconnected the beacons are already listed below, so showing
          // them here too would just be a second copy of the same list. Saved
          // endpoints are not in that list and stay here either way.
          if (sessionActive) {
            ImGui::SeparatorText("Detected RaisimServer beacons");
            if (!discoveryStatus.empty()) {
              ImGui::TextDisabled("%s; showing protocol %d only", discoveryStatus.c_str(),
                raisin::tcp_viewer::kProtocolVersion);
            }
            bool shownDetectedServer = false;
            for (const auto& server : discoveredServers) {
              shownDetectedServer = true;
              const std::string label = formatServerLabel(server);
              if (ImGui::Selectable(label.c_str())) {
                setPaneEndpoint(pane, server.endpoint.host, server.endpoint.port);
                if (!client.isConnected()) {
                  connectToEndpoint(server.endpoint, true, "connecting", "connect failed");
                }
                ImGui::CloseCurrentPopup();
              }
            }
            if (!shownDetectedServer) {
              ImGui::TextDisabled("No compatible RaisimServer beacons detected");
            }
          }
          ImGui::SeparatorText("Saved / recent endpoints");
          if (recentConnections.empty()) {
            ImGui::TextDisabled("No saved or recent endpoints");
          } else {
            for (const auto& entry : recentConnections) {
              const std::string label = formatConnectionLabel(entry);
              if (ImGui::Selectable(label.c_str())) {
                setPaneEndpoint(pane, entry.host, entry.port);
              }
            }
          }
          ImGui::EndCombo();
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();

        ImGui::SameLine();
        if (!client.isConnected()) {
          ImGui::BeginDisabled(inspector.active);
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Connect, "Connect", "connect")) {
            ConnectionEntry endpoint;
            if (!normalizeConnectionEndpoint(host, port, endpoint)) {
              lastStatus = "invalid endpoint";
            } else {
              connectToEndpoint(endpoint, true, "connecting", "connect failed");
            }
          }
          ImGui::EndDisabled();
        } else {
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Disconnect, "Disconnect", "disconnect")) {
            client.disconnect();
            localSimulation.stop();
            connectingLocalSimulation = false;
            autoConnect = false;
            awaitingResponse = false;
            awaitingSensorAck = false;
            lastStatus = "disconnected";
            clearSceneState();
          }
        }

        // Auto-connect belongs with the button it modifies, so it shares that
        // row instead of costing the panel another line.
        ImGui::SameLine();
        if (drawCompactCheckbox("Auto-connect", &autoConnect)) {
          if (autoConnect) {
            nextAutoConnectAttempt = now;
          }
        }
        if (inspector.active && !client.isConnected()) {
          ImGui::SameLine();
          ImGui::TextDisabled("(inspector active — close to connect)");
        }

        // Disconnected: the endpoint row above, then the live server list. It is
        // a child rather than a combo so the servers are on screen without a
        // click to expand, and a click on a row connects straight away.
        //
        // Only beacons go in here. A saved endpoint is just an address someone
        // typed once — nothing says anything is listening on it now — so those
        // stay in the endpoint editor above rather than masquerading as
        // discovered servers.
        if (!sessionActive) {
          const ImGuiStyle& promptStyle = ImGui::GetStyle();
          ImGui::SeparatorText("RaiSim servers on this network");

          // One row per server, one column per fact the beacon carries: what is
          // running, where to reach it, which machine it is on, and whether its
          // single client seat is taken. Nothing is behind a hover.
          struct ServerRow {
            std::string name;
            std::string address;
            std::string port;
            std::string computer;
            std::string availability;
            bool busy = false;
          };
          std::vector<ServerRow> serverRows;
          serverRows.reserve(discoveredServers.size());
          for (const auto& server : discoveredServers) {
            const auto exeIt = server.metadata.find("exe");
            const auto hostnameIt = server.metadata.find("hostname");
            const auto statusIt = server.metadata.find("status");
            ServerRow serverRow;
            serverRow.name = exeIt != server.metadata.end() && !exeIt->second.empty()
              ? exeIt->second
              : std::string("RaisimServer");
            serverRow.address = formatEndpointHost(server.endpoint.host);
            serverRow.port = std::to_string(server.endpoint.port);
            serverRow.computer = hostnameIt != server.metadata.end() ? hostnameIt->second
                                                                    : std::string();
            serverRow.busy =
              statusIt != server.metadata.end() && statusIt->second == "connected";
            serverRow.availability = serverRow.busy ? "in use" : "free";
            serverRows.push_back(std::move(serverRow));
          }

          // Fixed-fit columns need an explicit outer width, because the panel
          // auto-resizes and a fill width would chase its own content.
          constexpr int kServerColumns = 5;
          const char* const columnTitles[kServerColumns] = {
            "Name", "Address", "Port", "Computer", "Availability"};
          float columnWidths[kServerColumns];
          for (int column = 0; column < kServerColumns; ++column) {
            columnWidths[column] = ImGui::CalcTextSize(columnTitles[column]).x;
          }
          const auto widen = [&columnWidths](int column, const std::string& text) {
            columnWidths[column] = std::max(columnWidths[column],
                                            ImGui::CalcTextSize(text.c_str()).x);
          };
          for (const ServerRow& serverRow : serverRows) {
            widen(0, serverRow.name);
            widen(1, serverRow.address);
            widen(2, serverRow.port);
            widen(3, serverRow.computer);
            widen(4, serverRow.availability);
          }
          // 10 px of breathing room on each side of every cell, so the columns
          // do not run into each other or into the borders.
          constexpr float kServerCellPaddingX = 10.0f;
          float listWidth = promptStyle.ScrollbarSize + 2.0f;
          for (int column = 0; column < kServerColumns; ++column) {
            listWidth += columnWidths[column] + kServerCellPaddingX * 2.0f;
          }

          // Five servers fit without scrolling; a longer list grows a little
          // further before it starts to scroll. The header takes one more row.
          const float rowHeight = ImGui::GetTextLineHeightWithSpacing();
          const size_t visibleRows = std::clamp<size_t>(serverRows.size(), 5, 10);
          const float listHeight =
            rowHeight * static_cast<float>(visibleRows + 1) + promptStyle.CellPadding.y * 2.0f;

          constexpr ImGuiTableFlags kServerTableFlags =
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter |
            ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit |
            ImGuiTableFlags_ScrollY;
          ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,
                              ImVec2(kServerCellPaddingX, promptStyle.CellPadding.y));
          if (ImGui::BeginTable("##available_servers", kServerColumns, kServerTableFlags,
                                ImVec2(listWidth, listHeight))) {
            for (int column = 0; column < kServerColumns; ++column) {
              ImGui::TableSetupColumn(columnTitles[column], ImGuiTableColumnFlags_WidthFixed,
                                      columnWidths[column]);
            }
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();

            for (size_t row = 0; row < serverRows.size(); ++row) {
              const ServerEntry& server = discoveredServers[row];
              const ServerRow& serverRow = serverRows[row];
              ImGui::TableNextRow();
              ImGui::PushID(static_cast<int>(row));
              // Amber marks a server whose client seat is taken. The row stays
              // clickable: the beacon is a second or two old, so the seat may
              // have been given up since it was sent.
              if (serverRow.busy) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.78f, 0.25f, 1.0f));
              }
              ImGui::TableSetColumnIndex(0);
              // The name cell carries the click target for the whole row, so
              // there is no dead space between the columns.
              if (ImGui::Selectable(serverRow.name.c_str(), false,
                                    ImGuiSelectableFlags_SpanAllColumns)) {
                setPaneEndpoint(pane, server.endpoint.host, server.endpoint.port);
                connectToEndpoint(server.endpoint, true, "connecting", "connect failed");
              }
              ImGui::TableSetColumnIndex(1);
              ImGui::TextUnformatted(serverRow.address.c_str());
              ImGui::TableSetColumnIndex(2);
              ImGui::TextUnformatted(serverRow.port.c_str());
              ImGui::TableSetColumnIndex(3);
              ImGui::TextUnformatted(serverRow.computer.c_str());
              ImGui::TableSetColumnIndex(4);
              ImGui::TextUnformatted(serverRow.availability.c_str());
              if (serverRow.busy) {
                ImGui::PopStyleColor();
              }
              ImGui::PopID();
            }
            ImGui::EndTable();
          }
          ImGui::PopStyleVar();
          if (serverRows.empty()) {
            ImGui::TextDisabled("Listening for beacons, or type a host and port above");
          }
          if (!discoveryStatus.empty()) {
            ImGui::TextDisabled("%s", discoveryStatus.c_str());
          }
          // Amber while a connect is in flight, red once it is idle or failed:
          // the prompt now stays up through the handshake, so a plain red line
          // would call a connection that is still being made a failure.
          const bool connectPending = client.isConnected() || connectingLocalSimulation;
          ImGui::TextColored(connectPending ? ImVec4(0.95f, 0.75f, 0.25f, 1.0f)
                                            : ImVec4(0.9f, 0.2f, 0.2f, 1.0f),
                             "Status: %s", lastStatus.c_str());
        } else {
          if (localSimulation.active()) {
            ImGui::Text("Local world: %s", shortenPathLabel(localSimulation.worldPath().filename().string(), 32).c_str());
            ImGui::SameLine();
            if (ImGui::Button("Stop simulation")) {
              client.disconnect(); localSimulation.stop(); clearSceneState();
              awaitingResponse = awaitingSensorAck = false;
              connectingLocalSimulation = autoConnect = false;
              lastStatus = "Local simulation stopped";
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
          ImGui::TextColored(statusColor, "Status: %s", lastStatus.c_str());
          ImGui::SameLine();
          ImGui::TextUnformatted(worldText);
          ImGui::TextDisabled("Heightmap colors: server color map");
          ImGui::TextDisabled("FPS %.1f | updates %.1f Hz", stats.fps, stats.updateHz);
          ImGui::TextDisabled("Objects %zu | visuals %zu | instanced %zu | point clouds %zu",
            scene.selectableObjectCount(), scene.visualCount(), scene.instancedCount(), scene.pointCloudCount());
          ImGui::TextDisabled("Assets unresolved %zu | sensor requests %d | session %s",
            stats.unresolvedAssets, stats.pendingSensorRequests,
            sessionRecorder.active() ? "recording" : (replayMode ? "replay" : "live"));
          // A table ignores ItemSpacing.y, so match the loose checkboxes' margin
          // through cell padding instead.
          ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,
            compactControlCellPadding(ImGui::GetStyle().CellPadding));
          if (ImGui::BeginTable("##viewer_checkboxes", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableNextColumn();
            drawCompactCheckbox("Verbose parsing", &verboseParsing);

            ImGui::TableNextColumn();
            if (drawCompactCheckbox("Show Collision Bodies", &showCollisionBodies)) {
              viewer->setShowCollisionBodies(showCollisionBodies);
              scene.setShowCollisionBodies(showCollisionBodies);
            }

            ImGui::TableNextColumn();
            if (drawCompactCheckbox("X-ray (transparent)", &forceTransparent)) {
              scene.setForceTransparent(forceTransparent);
            }

            ImGui::TableNextColumn();
            if (drawCompactCheckbox("Show World Frame", &showWorldFrame)) {
              syncWorldFrame();
            }

            ImGui::TableNextColumn();
            drawCompactCheckbox("Show Body Frames", &showBodyFrames);

            ImGui::TableNextColumn();
            drawCompactCheckbox("Show COM Markers", &showComMarkers);

            ImGui::TableNextColumn();
            drawCompactCheckbox("Pose Grabber (drag axes)", &poseGrabber.enabled);

            ImGui::TableNextColumn();
            drawCompactCheckbox("Show Contact Points", &showContactPoints);

            ImGui::TableNextColumn();
            drawCompactCheckbox("Show Contact Forces", &showContactForces);

            ImGui::TableNextColumn();
            ImGui::BeginDisabled(!showContactForces);
            drawCompactCheckbox("Force Scale: Absolute", &contactForceAbsolute);
            ImGui::EndDisabled();

            ImGui::EndTable();
          }
          ImGui::PopStyleVar();

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
          ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,
            ImVec2(8.0f,
                   compactControlCellPadding(ImGui::GetStyle().CellPadding).y));
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
            if (drawOverlaySlider("##camera_speed", "Camera Speed", &cameraSpeed, 0.1f, 30.0f, "%.1f",
                  leftValueWidth, leftItemWidth, false)) {
              settings.cameraSpeed = cameraSpeed;
              settingsDirty = true;
            }

            ImGui::TableNextColumn();
            if (drawOverlaySlider("##light_yaw", "Light Yaw (deg)", &lightYawDeg, -180.0f, 180.0f,
                  "%.1f", rightValueWidth, rightItemWidth, false)) {
              settings.lightYawDeg = lightYawDeg;
              settingsDirty = true;
            }

            ImGui::TableNextColumn();
            if (drawOverlaySlider("##light_pitch", "Light Pitch (deg)", &lightPitchDeg, -89.0f, 89.0f,
                  "%.1f", leftValueWidth, leftItemWidth, false)) {
              settings.lightPitchDeg = lightPitchDeg;
              settingsDirty = true;
            }

            ImGui::TableNextColumn();
            if (drawOverlaySlider("##light_strength", "Light Strength", &lightStrength, 0.0f, 2.0f,
                  "%.2f", rightValueWidth, rightItemWidth, false)) {
              settings.lightStrength = lightStrength;
              settingsDirty = true;
            }

            ImGui::TableNextColumn();
            if (drawOverlaySlider("##ambient_strength", "Ambient Strength", &ambientStrength, 0.0f,
                  2.0f, "%.2f", leftValueWidth, leftItemWidth, false)) {
              settings.ambientStrength = ambientStrength;
              settingsDirty = true;
            }

            ImGui::TableNextColumn();
            ImGui::Dummy(ImVec2(rightItemWidth, 0.0f));
            ImGui::EndTable();
          }
          ImGui::PopStyleVar();

          // The directory list is collapsed into a dropdown: it is usually
          // empty or one entry, and when it is not, a stack of paths was eating
          // the panel. Each expanded row carries its own remove button.
          ImGui::SeparatorText("Resource directories");
          {
            const float removeSize = ImGui::GetFrameHeight();
            const float innerSpacing = ImGui::GetStyle().ItemInnerSpacing.x;
            const float addWidth = iconTextButtonSize("Add").x;
            const float comboWidth = std::max(ImGui::GetFontSize() * 8.0f,
              (fontScaledTextControlWidth(22.4f) - addWidth - innerSpacing) * 1.1f);

            std::string preview;
            if (resourceDirs.empty()) {
              preview = "none";
            } else if (resourceDirs.size() == 1) {
              preview = shortenPathLabel(resourceDirs.front(), 28);
            } else {
              preview = std::to_string(resourceDirs.size()) + " directories";
            }

            ImGui::SetNextItemWidth(comboWidth);
            if (ImGui::BeginCombo("##resource_dirs", preview.c_str())) {
              if (resourceDirs.empty()) {
                ImGui::TextDisabled("No resource directories");
              }
              // Removal is deferred so the vector is not mutated mid-iteration.
              size_t removeIndex = resourceDirs.size();
              for (size_t i = 0; i < resourceDirs.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::Button("x", ImVec2(removeSize, removeSize))) {
                  removeIndex = i;
                }
                if (ImGui::IsItemHovered()) {
                  ImGui::SetTooltip("Remove this directory");
                }
                ImGui::SameLine(0.0f, innerSpacing);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(shortenPathLabel(resourceDirs[i], 44).c_str());
                if (ImGui::IsItemHovered()) {
                  ImGui::SetTooltip("%s", resourceDirs[i].c_str());
                }
                ImGui::PopID();
              }
              if (removeIndex < resourceDirs.size()) {
                resourceDirs.erase(resourceDirs.begin() + static_cast<long>(removeIndex));
                ++resourceDirSerial;
                settingsDirty = true;
              }
              ImGui::EndCombo();
            }
            ImGui::SameLine(0.0f, innerSpacing);
            if (drawIconTextButton(uiIcons, TcpViewerIconKind::Folder, "Add", "resource_add",
                                   ImVec2(addWidth, 0.0f))) {
              // Start from the last directory added, so adding siblings is quick.
              openFileBrowser(fileBrowser, FileBrowserMode::Folder, "Select resource folder",
                resourceDirs.empty() ? std::filesystem::path()
                                     : std::filesystem::path(resourceDirs.back()),
                {},
                [&](const std::filesystem::path& chosen) {
                  const std::string path = chosen.string();
                  recordResourceDir(resourceDirs, path);
                  ++resourceDirSerial;
                  settingsDirty = true;
                  lastStatus = "added resource directory " + shortenPathLabel(path, 40);
                });
            }
          }
        }
        // Balances the TextDisabled push at the top of this tab.
        ImGui::PopStyleColor();
      };

      if (overlayMinimized) {
        drawCollapsedLeftPanelLogo(collapsedLogoVisible ? raisimLogo : TcpViewerImageTexture{});
      } else if (!sessionActive) {
        // A tab bar holding a single tab is just a title with extra steps, so the
        // connect prompt is drawn on its own.
        drawConnectionTab();
      } else if (beginIconTabBar("##LeftTabs")) {
        if (beginIconTabItem(uiIcons, TcpViewerIconKind::Connect, "Connection", "tab_connection")) {
          drawConnectionTab();
          ImGui::EndTabItem();
        }

        if (beginIconTabItem(uiIcons, TcpViewerIconKind::Options, "Options", "tab_options")) {
          drawViewOptions();
          ImGui::EndTabItem();
        }

        // Everything that writes a file lives here rather than sharing the
        // Options tab: screenshots, video, the PNG frame sequence, the session
        // replay log, and the replay transport.
        if (beginIconTabItem(uiIcons, TcpViewerIconKind::Video, "Record", "tab_record")) {
          ImGui::PushStyleColor(ImGuiCol_TextDisabled,
            raionrobotics_imgui_secondary_text_color());
          drawCaptureOptions();
          ImGui::PopStyleColor();
          ImGui::EndTabItem();
        }

        if (beginIconTabItem(uiIcons, TcpViewerIconKind::Render, "Render", "tab_render")) {
          drawRenderingOptions();
          ImGui::EndTabItem();
        }

        if (beginIconTabItem(uiIcons, TcpViewerIconKind::Objects, "Objects", "tab_objects")) {
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
          auto items = scene.getSelectableObjects();
          const std::string filterLower = toLowerAscii(trimAscii(objectFilterBuf));
          items.erase(std::remove_if(items.begin(), items.end(), [&](const ObjectListItem& item) {
            return isContactItem(item) || (hideCollisionObjects && item.isCollision) ||
                   !objectMatchesFilter(item, filterLower);
          }), items.end());
          std::sort(items.begin(), items.end(), [&](const ObjectListItem& lhs, const ObjectListItem& rhs) {
            if (groupObjectsByType && lhs.objectTypeRaw != rhs.objectTypeRaw) {
              return objectTypeLabelLess(lhs.objectTypeRaw, rhs.objectTypeRaw);
            }
            return objectLessByMode(lhs, rhs, objectSortMode);
          });

          const float objectIconSize = std::round(ImGui::GetFontSize() * 1.05f);
          const float objectIconWidth = objectIconSize + style.ItemInnerSpacing.x;
          float objectContentWidth = ImGui::CalcTextSize("No matching objects").x;
          for (const auto& item : items) {
            const std::string nameText = item.name.empty() ?
              ("tag " + std::to_string(item.tag)) : item.name;
            objectContentWidth = std::max(objectContentWidth,
              objectIconWidth + ImGui::CalcTextSize(objectTypeLabel(item.objectTypeRaw)).x +
              ImGui::CalcTextSize(": ").x + ImGui::CalcTextSize(nameText.c_str()).x);
            if (groupObjectsByType) {
              objectContentWidth = std::max(objectContentWidth,
                ImGui::CalcTextSize(objectTypeLabel(item.objectTypeRaw)).x);
            }
          }
          const float objectListPadding = style.FramePadding.x * 2.0f +
                                          style.WindowPadding.x * 2.0f +
                                          style.ScrollbarSize;
          const float minObjectListWidth = std::round(ImGui::GetFontSize() * 16.0f);
          const float maxObjectListWidth = std::max(minObjectListWidth,
            std::min(uiSize.x * 0.32f, ImGui::GetFontSize() * 36.0f));
          const float objectListWidth = std::clamp(
            objectContentWidth + objectListPadding, minObjectListWidth, maxObjectListWidth);

          constexpr const char* sortItems[] = {"Name", "Type", "Tag", "Index"};
          const float objectSortWidth = comboWidthForTextItems(sortItems, IM_ARRAYSIZE(sortItems));
          const float sortLabelWidth = style.ItemInnerSpacing.x + ImGui::CalcTextSize("Sort").x;
          const float objectFilterWidth = std::max(ImGui::GetFontSize() * 8.0f,
            objectListWidth - style.ItemSpacing.x - objectSortWidth - sortLabelWidth);
          ImGui::SetNextItemWidth(objectFilterWidth);
          ImGui::InputTextWithHint("##ObjectFilter", "filter name, type, tag", objectFilterBuf,
            sizeof(objectFilterBuf));
          ImGui::SameLine();
          ImGui::SetNextItemWidth(objectSortWidth);
          ImGui::Combo("Sort", &objectSortMode, sortItems, IM_ARRAYSIZE(sortItems));
          drawCompactCheckbox("Group by type", &groupObjectsByType);
          ImGui::SameLine();
          drawCompactCheckbox("Hide collisions", &hideCollisionObjects);
          ImGui::TextDisabled("%zu shown / %zu selectable | visuals %zu",
            items.size(), scene.selectableObjectCount(), scene.visualCount());
          ImGui::TextDisabled("instanced %zu | point clouds %zu",
            scene.instancedCount(), scene.pointCloudCount());

          if (items.empty()) {
            ImGui::TextDisabled("No matching objects");
          } else {
            const ImU32 typeColor = ImGui::GetColorU32(ImVec4(0.35f, 0.8f, 1.0f, 1.0f));
            const float listHeight = ImGui::GetFontSize() * 14.0f;
            const auto drawObjectRow = [&](const ObjectListItem& item) {
              const char* typeName = objectTypeLabel(item.objectTypeRaw);
              const bool selected = hasSelected && item.tag == selectedTag && item.index == selectedIndex;
              const std::string id =
                "##obj_" + std::to_string(item.tag) + "_" + std::to_string(item.index);
              if (ImGui::Selectable(id.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
                if (viewer) {
                  viewer->setTargetVisual(item.visual.get());
                }
              }
              if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("tag %u, index %d%s", item.tag, item.index,
                  item.isCollision ? ", collision" : "");
              }
              const ImVec2 itemMin = ImGui::GetItemRectMin();
              const ImVec2 textPos(itemMin.x + ImGui::GetStyle().FramePadding.x,
                itemMin.y + ImGui::GetStyle().FramePadding.y);
              float textX = textPos.x;
              const TcpViewerIconKind iconKind = objectTypeIconKind(item.objectTypeRaw);
              if (const TcpViewerIcon* icon = uiIcons.get(iconKind)) {
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                drawList->AddImage(reinterpret_cast<ImTextureID>(uint64_t(icon->texture)),
                  ImVec2(textX, textPos.y), ImVec2(textX + objectIconSize, textPos.y + objectIconSize),
                  ImVec2(0, 0), ImVec2(1, 1),
                  ImGui::GetColorU32(tcpViewerIconTint(iconKind, ImGui::IsItemHovered(), selected)));
                textX += objectIconWidth;
              }
              const std::string nameText =
                item.name.empty() ? ("tag " + std::to_string(item.tag)) : item.name;
              const char* sep = ": ";
              const ImVec2 typeSize = ImGui::CalcTextSize(typeName);
              const ImVec2 sepSize = ImGui::CalcTextSize(sep);
              ImDrawList* drawList = ImGui::GetWindowDrawList();
              drawList->AddText(
                ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(textX, textPos.y), typeColor, typeName);
              drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
                ImVec2(textX + typeSize.x, textPos.y), ImGui::GetColorU32(ImGuiCol_Text),
                sep);
              drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
                ImVec2(textX + typeSize.x + sepSize.x, textPos.y),
                ImGui::GetColorU32(item.isCollision ? ImGuiCol_TextDisabled : ImGuiCol_Text),
                nameText.c_str());
            };

            if (ImGui::BeginChild("##ObjectList", ImVec2(objectListWidth, listHeight), true)) {
              if (groupObjectsByType) {
                int currentType = std::numeric_limits<int>::min();
                for (const auto& item : items) {
                  if (item.objectTypeRaw != currentType) {
                    currentType = item.objectTypeRaw;
                    ImGui::SeparatorText(objectTypeLabel(currentType));
                  }
                  drawObjectRow(item);
                }
              } else {
                for (const auto& item : items) {
                  drawObjectRow(item);
                }
              }
              ImGui::EndChild();
            }
          }

          ImGui::SeparatorText("Ruler");
          drawCompactCheckbox("Measure (M)", &ruler.enabled);
          ImGui::SameLine();
          ImGui::TextDisabled("next %s", nextRulerPointLabel(ruler));
          ImGui::BeginDisabled(!hasSelected);
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Focus, "Set A", "ruler_set_a")) {
            setRulerEndpointFromSelection(0, selectedTag, selectedIndex, selectedEntry);
          }
          ImGui::SameLine();
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Focus, "Set B", "ruler_set_b")) {
            setRulerEndpointFromSelection(1, selectedTag, selectedIndex, selectedEntry);
          }
          ImGui::EndDisabled();
          ImGui::SameLine();
          ImGui::BeginDisabled(!ruler.hasA && !ruler.hasB);
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Reset, "Clear", "ruler_clear")) {
            ruler = RulerToolState{};
            lastStatus = "ruler cleared";
          }
          ImGui::EndDisabled();
          const std::string pointAText = ruler.hasA ? formatRulerPoint(ruler.a) : "--";
          const std::string pointBText = ruler.hasB ? formatRulerPoint(ruler.b) : "--";
          ImGui::TextDisabled("A %s", pointAText.c_str());
          if (ruler.hasA && !ruler.aLabel.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%s", shortenPathLabel(ruler.aLabel, 34).c_str());
          }
          ImGui::TextDisabled("B %s", pointBText.c_str());
          if (ruler.hasB && !ruler.bLabel.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%s", shortenPathLabel(ruler.bLabel, 34).c_str());
          }
          if (ruler.hasA && ruler.hasB) {
            const std::string distanceText = formatRulerDistance(glm::distance(ruler.a, ruler.b));
            ImGui::Text("Distance %s", distanceText.c_str());
          }

          // ----- Simulation control (requires server SIM_CONTROL feature + CAP_SIM_CONTROL) -----
          ImGui::SeparatorText("Simulation");
          const bool serverSimControl = scene.serverSupportsSimControl();
          const bool canControlSim = client.isConnected() && serverSimControl;
          ImGui::BeginDisabled(!canControlSim);
          using raisin::tcp_viewer::ClientRequestType;
          // Pause / Resume toggle — icon-only with hover tooltip.
          if (drawIconOnlyButton(uiIcons,
                simPaused ? TcpViewerIconKind::Play : TcpViewerIconKind::Pause,
                simPaused ? "Resume simulation" : "Pause simulation",
                simPaused ? "sim_resume" : "sim_pause")) {
            raisin::tcp_viewer::ClientRequest r;
            r.type = simPaused ? ClientRequestType::CR_RESUME : ClientRequestType::CR_PAUSE;
            pendingControlRequests.push_back(r);
            simPaused = !simPaused;
          }
          // Step buttons — auto-pause first if running, then queue the step(s).
          auto queueStep = [&](int n) {
            raisin::tcp_viewer::ClientRequest r;
            r.type = ClientRequestType::CR_STEP_N;
            r.stepCount = n;
            pendingControlRequests.push_back(r);
            if (!simPaused) {
              raisin::tcp_viewer::ClientRequest p;
              p.type = ClientRequestType::CR_PAUSE;
              pendingControlRequests.push_back(p);
              simPaused = true;
            }
          };
          ImGui::SameLine();
          if (drawIconOnlyButton(uiIcons, TcpViewerIconKind::Step,
                                 "Step 1 frame", "sim_step")) {
            queueStep(1);
          }
          ImGui::SameLine();
          if (drawIconOnlyButton(uiIcons, TcpViewerIconKind::StepFast,
                                 "Step 10 frames", "sim_step10")) {
            queueStep(10);
          }
          ImGui::EndDisabled();
          if (!client.isConnected()) {
            ImGui::TextDisabled("Sim control: disconnected");
          } else if (!serverSimControl) {
            ImGui::TextDisabled("Sim control: server does not advertise SIM_CONTROL");
          }

          // ----- Scene editing (CR_SPAWN_* / CR_REMOVE_OBJECT / CR_SAVE_THE_WORLD) -----
          // These predate the SIM_CONTROL feature bit and are not gated by it, so
          // they work against any protocol-matched server.
          ImGui::SeparatorText("Scene editing");
          const bool canEditScene = client.isConnected();
          const bool serverIsLocal = localSimulation.active() || isLoopbackHostName(host);
          if (!canEditScene) {
            ImGui::TextDisabled("Scene editing: disconnected");
          }
          ImGui::BeginDisabled(!canEditScene);
          if (ImGui::TreeNode("Add object")) {
            raisin::tcp_viewer::ClientRequest spawnRequest;
            const glm::vec3 dropPoint = viewer->getCamera().target;
            if (drawSpawnForm(uiIcons, spawnForm, canEditScene, dropPoint, serverIsLocal,
                              spawnRequest, spawnStatus, &fileBrowser)) {
              pendingControlRequests.push_back(std::move(spawnRequest));
              lastStatus = spawnStatus;
            }
            ImGui::TreePop();
          }

          const bool canRemoveSelection = canEditScene && requestedEntry != nullptr &&
                                          requestedTag != 0 && !isContactEntry(requestedEntry);
          ImGui::BeginDisabled(!canRemoveSelection);
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Delete, "Delete Selected",
                                 "remove_selected_object")) {
            raisin::tcp_viewer::ClientRequest remove;
            remove.type = raisin::tcp_viewer::ClientRequestType::CR_REMOVE_OBJECT;
            remove.visTag = requestedTag;
            pendingControlRequests.push_back(remove);
            // The visual disappears with the next scene update; drop the
            // selection now so the panels do not point at a dead tag.
            viewer->setTargetVisual(nullptr);
            lastStatus = "remove queued for tag " + std::to_string(requestedTag);
          }
          ImGui::EndDisabled();
          if (canEditScene && !canRemoveSelection) {
            ImGui::TextDisabled("Select an object to delete it");
          }

          if (ImGui::TreeNode("Export world")) {
            ImGui::SetNextItemWidth(fontScaledTextControlWidth(26.0f));
            ImGui::InputText("##WorldExportPath", worldExportPathBuf, sizeof(worldExportPathBuf));
            if (serverIsLocal) {
              ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
              if (drawIconOnlyButton(uiIcons, TcpViewerIconKind::Folder, "Browse...",
                                     "browse_world_export")) {
                openFileBrowser(fileBrowser, FileBrowserMode::SaveFile, "Select world XML",
                  std::filesystem::path(worldExportPathBuf), {"xml"},
                  [&](const std::filesystem::path& chosen) {
                    std::snprintf(worldExportPathBuf, sizeof(worldExportPathBuf), "%s",
                                  chosen.string().c_str());
                  });
              }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("XML path (server-side)");
            const std::string exportPath = trimAscii(std::string(worldExportPathBuf));
            ImGui::BeginDisabled(!canEditScene || exportPath.empty());
            if (drawIconTextButton(uiIcons, TcpViewerIconKind::Export, "Export World XML",
                                   "export_world_xml")) {
              raisin::tcp_viewer::ClientRequest save;
              save.type = raisin::tcp_viewer::ClientRequestType::CR_SAVE_THE_WORLD;
              save.file = exportPath;
              // Flushed on its own frame; see the send path above.
              pendingControlRequests.push_back(std::move(save));
              lastStatus = "world export queued";
            }
            ImGui::EndDisabled();
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + fontScaledTextControlWidth(26.0f));
            ImGui::TextDisabled(
              "The server writes the file, so the path is relative to the simulation host.");
            ImGui::PopTextWrapPos();
            ImGui::TreePop();
          }
          ImGui::EndDisabled();

          ImGui::SeparatorText("Selected control");
          const SelectedObjectInfo& controlInfo = scene.getSelectedInfo();
          const bool hasControlSelection = requestedEntry != nullptr && requestedTag != 0;
          if (!hasControlSelection) {
            ImGui::TextDisabled("Select an object to apply controls");
          } else {
            const bool forceSupported = supportsTcpViewerForceControl(requestedEntry);
            const bool poseSupported = supportsTcpViewerPoseControl(requestedEntry);
            const bool gcSupported = requestedEntry->isArticulated;
            const int selectedBodyIdx = requestedEntry->isArticulated ?
              std::max(0, requestedEntry->localBodyIdx) : 0;
            if (controlBodyFollowsSelection) {
              controlBodyIdx = selectedBodyIdx;
            }

            std::string selectedName = requestedEntry->objectName.empty() ?
              scene.getObjectName(requestedTag) : requestedEntry->objectName;
            if (selectedName.empty()) selectedName = objectTypeLabel(requestedEntry->objectTypeRaw);
            ImGui::TextDisabled("%s | tag %u | body %d", selectedName.c_str(),
              requestedTag, selectedBodyIdx);

            const float controlVecWidth = std::round(ImGui::GetFontSize() * 12.5f);
            const auto drawVec3Control = [&](const char* label, const char* id, glm::vec3& value,
                                             float speed) {
              ImGui::TextUnformatted(label);
              ImGui::SetNextItemWidth(controlVecWidth);
              return compactDragFloat3(id, &value.x, speed, -100000.0f, 100000.0f, "%.3g");
            };
            const auto syncPoseFromSelection = [&]() {
              controlPosePosition = requestedEntry->lastPos;
              controlPoseQuat = requestedEntry->lastQuat;
              controlPoseTag = requestedTag;
              controlPoseInitialized = true;
            };

            if (requestedEntry->isArticulated) {
              if (drawCompactCheckbox("Body follows selection", &controlBodyFollowsSelection)) {
                if (controlBodyFollowsSelection) controlBodyIdx = selectedBodyIdx;
              }
              ImGui::BeginDisabled(controlBodyFollowsSelection);
              ImGui::SetNextItemWidth(ImGui::GetFontSize() * 5.0f);
              ImGui::InputInt("Body", &controlBodyIdx);
              if (controlBodyIdx < 0) controlBodyIdx = 0;
              ImGui::EndDisabled();
            }

            ImGui::BeginDisabled(!canControlSim || !forceSupported);
            drawCompactCheckbox("Shift-drag force", &mouseForceEnabled);
            ImGui::BeginDisabled(!mouseForceEnabled);
            drawInlineLabelSliderFloat("mouse_force_scale", "Mouse accel", &mouseForceScale,
              kMinMouseForceAccelPerPixel, kMaxMouseForceAccelPerPixel, "%.2f m/s^2/px");
            ImGui::EndDisabled();
            drawVec3Control("Force", "##selected_force", controlForce, 0.25f);
            drawVec3Control("Point offset", "##selected_force_offset", controlPointOffset, 0.01f);
            if (drawIconTextButton(uiIcons, TcpViewerIconKind::Force, "Apply Force", "apply_selected_force")) {
              raisin::tcp_viewer::ClientRequest r;
              r.type = ClientRequestType::CR_APPLY_FORCE;
              r.visTag = requestedTag;
              r.localBodyIdx = std::max(0, controlBodyIdx);
              r.vec3a = requestedEntry->lastPos + controlPointOffset;
              r.vec3b = controlForce;
              pendingControlRequests.push_back(r);
              lastStatus = "force queued";
            }
            drawVec3Control("Torque", "##selected_torque", controlTorque, 0.05f);
            if (drawIconTextButton(uiIcons, TcpViewerIconKind::Torque, "Apply Torque", "apply_selected_torque")) {
              raisin::tcp_viewer::ClientRequest r;
              r.type = ClientRequestType::CR_APPLY_TORQUE;
              r.visTag = requestedTag;
              r.localBodyIdx = std::max(0, controlBodyIdx);
              r.vec3a = controlTorque;
              pendingControlRequests.push_back(r);
              lastStatus = "torque queued";
            }
            ImGui::EndDisabled();
            if (!forceSupported) {
              ImGui::TextDisabled("Force/torque: unsupported object type");
            }

            // The interaction wire pulls with a mass-scaled spring rather than
            // teleporting, so it is the gesture to use on an articulated system
            // that must stay physically consistent while it is moved.
            ImGui::BeginDisabled(!canControlSim || !forceSupported);
            drawCompactCheckbox(kWireDragGestureLabel, &wireDragEnabled);
            ImGui::BeginDisabled(!wireDragEnabled);
            drawInlineLabelSliderFloat("wire_drag_stiffness", "Wire stiffness", &wireDragStiffness,
              kMinWireDragStiffness, kMaxWireDragStiffness, "%.0f N/m/kg");
            ImGui::EndDisabled();
            ImGui::EndDisabled();

            if (poseSupported) {
              if (!controlPoseInitialized || controlPoseTag != requestedTag) {
                syncPoseFromSelection();
              }
              if (ImGui::TreeNodeEx("Pose", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::BeginDisabled(!canControlSim);
                if (drawIconTextButton(uiIcons, TcpViewerIconKind::Refresh, "Sync Pose", "sync_selected_pose")) {
                  syncPoseFromSelection();
                }
                drawVec3Control("Position", "##selected_pose_position", controlPosePosition, 0.01f);
                ImGui::TextUnformatted("Quaternion WXYZ");
                ImGui::SetNextItemWidth(controlVecWidth);
                compactDragFloat4("##selected_pose_quat", &controlPoseQuat.x, 0.005f,
                  -1.0f, 1.0f, "%.3f");
                if (drawIconTextButton(uiIcons, TcpViewerIconKind::Save, "Set Pose", "set_selected_pose")) {
                  raisin::tcp_viewer::ClientRequest r;
                  r.type = ClientRequestType::CR_SET_POSE;
                  r.visTag = requestedTag;
                  r.vec3a = controlPosePosition;
                  r.quat = normalizedWxyz(controlPoseQuat);
                  controlPoseQuat = r.quat;
                  pendingControlRequests.push_back(r);
                  lastStatus = "pose queued";
                }
                ImGui::EndDisabled();
                ImGui::TreePop();
              }
            }

            if (gcSupported) {
              const bool hasGc = controlInfo.valid && controlInfo.isArticulated &&
                controlInfo.tag == requestedTag && !controlInfo.generalizedCoordinates.empty();
              if (!hasGc) {
                ImGui::TextDisabled("Joint controls: waiting for selected object data");
              } else {
                if (controlGcTag != requestedTag || (!controlGcDirty &&
                    controlGc.size() != controlInfo.generalizedCoordinates.size())) {
                  controlGc = controlInfo.generalizedCoordinates;
                  controlGcTag = requestedTag;
                  controlGcDirty = false;
                }
                if (ImGui::TreeNodeEx("Generalized coordinates", ImGuiTreeNodeFlags_DefaultOpen)) {
                  ImGui::BeginDisabled(!canControlSim);
                  if (drawIconTextButton(uiIcons, TcpViewerIconKind::Refresh, "Sync GC", "sync_selected_gc")) {
                    controlGc = controlInfo.generalizedCoordinates;
                    controlGcTag = requestedTag;
                    controlGcDirty = false;
                  }
                  for (size_t i = 0; i < controlInfo.jointNames.size(); ++i) {
                    const int32_t offset = i < controlInfo.jointGcOffsets.size() ?
                      controlInfo.jointGcOffsets[i] : -1;
                    const int32_t dim = i < controlInfo.jointGcDims.size() ?
                      controlInfo.jointGcDims[i] : 0;
                    const int32_t type = i < controlInfo.jointTypes.size() ?
                      controlInfo.jointTypes[i] : int32_t(raisim::Joint::Type::FIXED);
                    ImGui::PushID(static_cast<int>(i));
                    drawJointTypeIcon(uiIcons, type);
                    ImGui::Text("%s [%s]", controlInfo.jointNames[i].c_str(),
                      tcpViewerJointTypeLabel(type));
                    const bool validSlice = offset >= 0 &&
                      static_cast<size_t>(offset + std::max<int32_t>(1, dim) - 1) < controlGc.size();
                    if (!validSlice || dim == 0) {
                      ImGui::TextDisabled("not editable");
                    } else if (dim == 1) {
                      ImGui::SetNextItemWidth(controlVecWidth);
                      if (compactDragFloat("##joint_q", &controlGc[static_cast<size_t>(offset)],
                            0.005f, -1000.0f, 1000.0f, "%.5g")) {
                        controlGcDirty = true;
                      }
                    } else if (dim == 4) {
                      ImGui::SetNextItemWidth(controlVecWidth);
                      if (compactDragFloat4("##joint_quat", &controlGc[static_cast<size_t>(offset)],
                            0.005f, -1.0f, 1.0f, "%.3f")) {
                        controlGcDirty = true;
                      }
                    } else if (dim == 7) {
                      ImGui::TextUnformatted("Position");
                      ImGui::SetNextItemWidth(controlVecWidth);
                      if (compactDragFloat3("##floating_pos", &controlGc[static_cast<size_t>(offset)],
                            0.01f, -1000.0f, 1000.0f, "%.4g")) {
                        controlGcDirty = true;
                      }
                      ImGui::TextUnformatted("Quaternion WXYZ");
                      ImGui::SetNextItemWidth(controlVecWidth);
                      if (compactDragFloat4("##floating_quat", &controlGc[static_cast<size_t>(offset + 3)],
                            0.005f, -1.0f, 1.0f, "%.3f")) {
                        controlGcDirty = true;
                      }
                    }
                    ImGui::PopID();
                  }
                  if (drawIconTextButton(uiIcons, TcpViewerIconKind::Robot, "Set GC", "set_selected_gc")) {
                    for (size_t i = 0; i < controlInfo.jointTypes.size(); ++i) {
                      const auto type = static_cast<raisim::Joint::Type>(controlInfo.jointTypes[i]);
                      const int32_t offset = i < controlInfo.jointGcOffsets.size() ?
                        controlInfo.jointGcOffsets[i] : -1;
                      if (offset < 0) continue;
                      if (type == raisim::Joint::Type::SPHERICAL) {
                        normalizeWxyzSlice(controlGc, static_cast<size_t>(offset));
                      } else if (type == raisim::Joint::Type::FLOATING) {
                        normalizeWxyzSlice(controlGc, static_cast<size_t>(offset + 3));
                      }
                    }
                    raisin::tcp_viewer::ClientRequest r;
                    r.type = ClientRequestType::CR_SET_GC;
                    r.visTag = requestedTag;
                    r.gc = controlGc;
                    pendingControlRequests.push_back(std::move(r));
                    controlGcDirty = false;
                    lastStatus = "gc queued";
                  }
                  ImGui::EndDisabled();
                  ImGui::TreePop();
                }
              }
            }
          }

          ImGui::EndTabItem();
        }
        if (beginIconTabItem(uiIcons, TcpViewerIconKind::Diagnostics, "Diagnostics", "tab_diagnostics")) {
          constexpr const char* packetColumnNames[] = {"t", "bytes", "src", "ok", "obj", "vis", "sens", "miss"};
          constexpr float packetColumnWidths[] = {62.0f, 62.0f, 46.0f, 34.0f, 44.0f, 44.0f, 44.0f, 44.0f};
          float packetTableWidth = ImGui::GetStyle().ScrollbarSize + ImGui::GetStyle().CellPadding.x * 16.0f;
          for (const float width : packetColumnWidths) {
            packetTableWidth += width;
          }
          const float diagnosticsContentWidth = std::max(packetTableWidth,
            iconTextButtonSize("Refresh Assets").x + style.ItemSpacing.x +
              iconTextButtonSize("Export Scene JSON").x);

          ImGui::SeparatorText("Security");
          ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + diagnosticsContentWidth);
          ImGui::TextDisabled("TCP viewer traffic is plain and unauthenticated; use SSH/VPN/TLS proxy for untrusted networks.");
          ImGui::PopTextWrapPos();
          ImGui::SeparatorText("Data Transfer");
          const double diagnosticsNowSeconds = std::chrono::duration<double>(now - steadyStart).count();
          refreshDiagnosticsPresentation(
            diagnosticsPresentation, packetSamples, stats, diagnosticsNowSeconds);
          const float peakTransferRate = maxTransferRate(diagnosticsPresentation.transferRates);
          const float graphMax = std::max(1.0f, peakTransferRate * 1.15f);
          char transferOverlay[96];
          std::snprintf(transferOverlay, sizeof(transferOverlay), "current %.1f KiB/s | peak %.1f KiB/s",
            diagnosticsPresentation.rxKbps, peakTransferRate);
          ImGui::PlotLines("##DataTransferRate", diagnosticsPresentation.transferRates.data(),
            static_cast<int>(diagnosticsPresentation.transferRates.size()), 0,
            transferOverlay, 0.0f, graphMax,
            ImVec2(diagnosticsContentWidth, ImGui::GetFontSize() * 6.0f));
          if (!diagnosticsPresentation.roundTripTimes.empty()) {
            char timingOverlay[128];
            std::snprintf(timingOverlay, sizeof(timingOverlay),
              "RTT %.2f ms | avg %.2f | jitter %.2f | max %.2f",
              diagnosticsPresentation.timing.currentMs,
              diagnosticsPresentation.timing.averageMs,
              diagnosticsPresentation.timing.jitterMs,
              diagnosticsPresentation.timing.maximumMs);
            ImGui::PlotLines("##RoundTripTime", diagnosticsPresentation.roundTripTimes.data(),
              static_cast<int>(diagnosticsPresentation.roundTripTimes.size()), 0, timingOverlay,
              0.0f, FLT_MAX,
              ImVec2(diagnosticsContentWidth, ImGui::GetFontSize() * 5.0f));
          }
          float updateRateHz = settings.tcpUpdateRateHz;
          if (drawInlineLabelSliderFloat("diagnostics_update_rate", "Target",
                &updateRateHz, kTcpUpdateRateMinHz, kTcpUpdateRateMaxHz, "%.0f Hz")) {
            settings.tcpUpdateRateHz = sanitizeTcpUpdateRateHz(updateRateHz);
            nextTcpUpdateRequestTime = now;
            settingsDirty = true;
          }

          ImGui::SeparatorText("Packets");
          ImGui::TextDisabled("Recent %zu packets | parse errors %d | RX %.1f KiB/s",
            diagnosticsPresentation.packetSamples.size(), diagnosticsPresentation.parseErrors,
            diagnosticsPresentation.rxKbps);
          const float packetHeight = ImGui::GetFontSize() * 8.0f;
          if (ImGui::BeginChild("##PacketHistory", ImVec2(diagnosticsContentWidth, packetHeight), true)) {
            if (ImGui::BeginTable("##packet_table", 8,
                  ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit)) {
              for (size_t column = 0; column < std::size(packetColumnNames); ++column) {
                ImGui::TableSetupColumn(packetColumnNames[column], ImGuiTableColumnFlags_WidthFixed,
                                        packetColumnWidths[column]);
              }
              ImGui::TableHeadersRow();
              for (auto it = diagnosticsPresentation.packetSamples.rbegin();
                   it != diagnosticsPresentation.packetSamples.rend(); ++it) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%.2f", it->timeSeconds);
                ImGui::TableSetColumnIndex(1); ImGui::Text("%d", it->bytes);
                ImGui::TableSetColumnIndex(2); ImGui::TextUnformatted(it->replay ? "replay" : "live");
                ImGui::TableSetColumnIndex(3); ImGui::TextUnformatted(it->parsed ? "yes" : "no");
                ImGui::TableSetColumnIndex(4); ImGui::Text("%zu", it->objects);
                ImGui::TableSetColumnIndex(5); ImGui::Text("%zu", it->visuals);
                ImGui::TableSetColumnIndex(6); ImGui::Text("%d", it->pendingSensors);
                ImGui::TableSetColumnIndex(7); ImGui::Text("%zu", it->unresolvedAssets);
              }
              ImGui::EndTable();
            }
            ImGui::EndChild();
          }

          ImGui::SeparatorText("Assets");
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Refresh, "Refresh Assets", "refresh_assets")) {
            assetDiagnostics = collectAssetDiagnostics(scene);
            assetDiagnosticsDirty = false;
            stats.unresolvedAssets = unresolvedAssetCount(scene);
            diagnosticsPresentation.unresolvedAssets = stats.unresolvedAssets;
          }
          ImGui::SameLine();
          if (drawIconTextButton(uiIcons, TcpViewerIconKind::Export, "Export Scene JSON", "export_scene_json")) {
            const std::filesystem::path path = options.exportScenePath.empty()
              ? timestampedDataPath(std::filesystem::path(screenshotDirBuf), "rayrai_tcp_viewer_scene", ".json")
              : options.exportScenePath;
            if (assetDiagnosticsDirty || assetDiagnostics.empty()) {
              assetDiagnostics = collectAssetDiagnostics(scene);
              assetDiagnosticsDirty = false;
            }
            exportSceneJson(path, scene, assetDiagnostics, captureStatus);
          }
          ImGui::TextDisabled("%zu assets | %zu unresolved", assetDiagnostics.size(),
            diagnosticsPresentation.unresolvedAssets);
          const float assetHeight = ImGui::GetFontSize() * 9.0f;
          if (ImGui::BeginChild("##AssetDiagnostics", ImVec2(diagnosticsContentWidth, assetHeight), true)) {
            if (assetDiagnostics.empty()) {
              ImGui::TextDisabled("No mesh assets observed yet");
            } else if (ImGui::BeginTable("##asset_table", 5,
                         ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
              ImGui::TableSetupColumn("state", ImGuiTableColumnFlags_WidthFixed, 70.0f);
              ImGui::TableSetupColumn("tag", ImGuiTableColumnFlags_WidthFixed, 52.0f);
              ImGui::TableSetupColumn("object");
              ImGui::TableSetupColumn("mesh");
              ImGui::TableSetupColumn("resolved/resource");
              ImGui::TableHeadersRow();
              for (const auto& asset : assetDiagnostics) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextColored(asset.resolved ? ImVec4(0.35f, 0.85f, 0.45f, 1.0f)
                                                   : ImVec4(1.0f, 0.45f, 0.25f, 1.0f),
                  "%s", asset.resolved ? "resolved" : "missing");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%u", asset.tag);
                ImGui::TableSetColumnIndex(2); ImGui::TextUnformatted(shortenPathLabel(asset.name, 32).c_str());
                ImGui::TableSetColumnIndex(3); ImGui::TextUnformatted(shortenPathLabel(asset.meshFile, 44).c_str());
                ImGui::TableSetColumnIndex(4);
                const std::string pathLabel = asset.resolved ? asset.meshPath : asset.resourceDir;
                ImGui::TextUnformatted(shortenPathLabel(pathLabel, 56).c_str());
              }
              ImGui::EndTable();
            }
            ImGui::EndChild();
          }

          ImGui::SeparatorText("Server Metadata");
          bool shownMetadata = false;
          for (const auto& server : discoveredServers) {
            if (!server.remoteBeacon || server.metadata.empty()) continue;
            shownMetadata = true;
            if (ImGui::TreeNode(formatConnectionLabel(server.endpoint).c_str())) {
              for (const auto& item : server.metadata) {
                ImGui::TextDisabled("%s: %s", item.first.c_str(), item.second.c_str());
              }
              ImGui::TreePop();
            }
          }
          if (!shownMetadata) {
            ImGui::TextDisabled("No beacon metadata received yet");
          }
          ImGui::EndTabItem();
        }
        if (beginIconTabItem(uiIcons, TcpViewerIconKind::Help, "Help", "tab_help")) {
          ImGui::PushStyleColor(ImGuiCol_TextDisabled,
            raionrobotics_imgui_secondary_text_color());
          ImGui::SeparatorText("Keyboard and mouse");
          drawShortcutTable("##shortcuts_tab");
          ImGui::PopStyleColor();
          ImGui::EndTabItem();
        }
        endIconTabBar();
      }
    }
    ImGui::End();
    if (overlayColorPushCount > 0) {
      ImGui::PopStyleColor(overlayColorPushCount);
    }
    ImGui::PopStyleVar(2);
    const bool overlayInteractionActive = ImGui::IsAnyItemActive() ||
                                          ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopup);
    if (!overlayMinimized) {
      if (overlayHovered || overlayInteractionActive || options.keepOverlayOpen) {
        overlayLastInteractionTime = now;
      } else if (now - overlayLastInteractionTime >= kOverlayAutoCollapseDelay) {
        overlayMinimized = true;
      }
    }
    overlayCollapsedHoveredLastFrame = overlayMinimized && overlayHovered;
    } // end if (overlayVisible)

    const raisin::Visuals* selectedVisual = viewer->getTargetVisual();
    const VisualEntry* selectedEntry = nullptr;
    uint32_t selectedTag = 0;
    int selectedIndex = 0;
    scene.getVisualInfo(selectedVisual, selectedTag, selectedIndex, selectedEntry);

    if (selectedEntry && isContactEntry(selectedEntry)) {
      selectedEntry = nullptr;
    }

    // A split pane can be too narrow for both panels. The object inspector is
    // right-anchored, so in a narrow pane it used to land on top of the left
    // overlay and both became unreadable; give the overlay the room it took and
    // let the inspector shrink into what is left, down to its minimised width.
    const float detailGap = 12.0f;
    const float detailRoom =
      paneOrigin.x + uiSize.x - detailGap - std::max(paneOrigin.x, overlayRightEdge + detailGap);
    const float detailMinimisedWidth = std::round(ImGui::GetFontSize() * 9.5f);
    if (selectedEntry && !detailMinimized && detailRoom < detailMinimisedWidth) {
      // Not even the minimised inspector fits beside the overlay: collapsing the
      // overlay is the only way to see the selection in a pane this narrow.
      selectedEntry = nullptr;
    }

    if (selectedEntry) {
      const float detailPanelWidth = std::max(
        std::min(detailRoom, std::round(ImGui::GetFontSize() * (detailMinimized ? 9.5f : 18.0f))),
        detailMinimisedWidth);
      const ImVec2 detailsBasePos(paneOrigin.x + uiSize.x - detailPanelWidth - detailGap,
                                  paneOrigin.y + 12.0f + menuBarHeight);
      ImVec2 detailsPos = detailsBasePos;
      detailsPos.x += detailOffset.x;
      detailsPos.y += detailOffset.y;
      ImGui::SetNextWindowBgAlpha(0.5f);
      ImGui::SetNextWindowPos(detailsPos, ImGuiCond_Always);
      const ImVec2 detailMinSize(detailPanelWidth, 0.0f);
      const ImVec2 detailMaxSize(detailPanelWidth,
        std::max(ImGui::GetFontSize() * 8.0f, uiSize.y - menuBarHeight - 24.0f));
      ImGui::SetNextWindowSizeConstraints(detailMinSize, detailMaxSize);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
      if (ImGui::Begin(detailWindowName.c_str(), nullptr, kOverlayPanelWindowFlags)) {
        const ImGuiStyle& detailStyle = ImGui::GetStyle();
        const float detailToggleWidth =
          (ImGui::CalcTextSize("-").x + detailStyle.FramePadding.x * 2.0f) * 1.6f;
        const float detailDragWidth = std::max(0.0f,
          ImGui::GetContentRegionAvail().x - detailToggleWidth - detailStyle.ItemSpacing.x);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.35f, 0.8f, 1.0f, 1.0f));
        ImGui::Selectable("Selected Object", false, ImGuiSelectableFlags_None,
          ImVec2(detailDragWidth, 0.0f));
        ImGui::PopStyleColor();
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
          ImVec2 delta = ImGui::GetIO().MouseDelta;
          detailOffset.x = std::min(0.0f, detailOffset.x + delta.x);
          detailOffset.y = std::max(0.0f, detailOffset.y + delta.y);
        }
        ImGui::SameLine();
        if (ImGui::Button(detailMinimized ? "+##detail_toggle" : "-##detail_toggle",
              ImVec2(detailToggleWidth, ImGui::GetFrameHeight()))) {
          detailMinimized = !detailMinimized;
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("%s", detailMinimized ? "Expand" : "Minimize");
        }

        if (!detailMinimized) {
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

          const auto selectedSensors = scene.getSensorsForTag(selectedTag);
          // Record operating points every update, so the plots have a trail when opened.
          if (selectedInfo.isArticulated && !selectedInfo.actuators.empty()) {
            actuatorTraces.update(selectedInfo, scene.getServerWorldTime());
          }
          if (ImGui::BeginTabBar("##selected_object_tabs")) {
            if (ImGui::BeginTabItem("Object")) {
          // Opens a label/value row and leaves the cursor in the value cell.
          const auto beginPropRow = [](const char* label) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(label);
            ImGui::TableSetColumnIndex(1);
          };
          if (ImGui::BeginTable("##selected_props", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);

            beginPropRow("Name");
            ImGui::TextColored(nameColor, "%s", objectName.c_str());

            beginPropRow("Tag");
            ImGui::TextColored(tagColor, "%u", selectedTag);

            beginPropRow("Index");
            ImGui::TextColored(indexColor, "%d", selectedIndex);

            beginPropRow("Body");
            ImGui::TextColored(indexColor, "%d", selectedEntry->localBodyIdx);

            beginPropRow("Type");
            ImGui::TextColored(shapeColor, "%s", objectTypeLabel(selectedEntry->objectTypeRaw));

            if (!selectedEntry->meshFile.empty()) {
              std::string meshLabel = selectedEntry->meshFile;
              const size_t slashPos = meshLabel.find_last_of("/\\");
              if (slashPos != std::string::npos && slashPos + 1 < meshLabel.size()) {
                meshLabel = meshLabel.substr(slashPos + 1);
              }
              beginPropRow("Mesh");
              ImGui::TextColored(metaColor, "%s", meshLabel.c_str());
            }

            beginPropRow("Articulated");
            ImGui::TextColored(metaColor, "%s", selectedEntry->isArticulated ? "yes" : "no");

            beginPropRow("Collision");
            ImGui::TextColored(metaColor, "%s", selectedEntry->isCollision ? "yes" : "no");

            beginPropRow("Pos");
            ImGui::TextColored(metaColor, "%.3f %.3f %.3f", selectedEntry->lastPos.x,
              selectedEntry->lastPos.y, selectedEntry->lastPos.z);

            beginPropRow("Quat");
            ImGui::TextColored(metaColor, "%.3f %.3f %.3f %.3f", selectedEntry->lastQuat.x,
              selectedEntry->lastQuat.y, selectedEntry->lastQuat.z, selectedEntry->lastQuat.w);

            beginPropRow("Size");
            ImGui::TextColored(metaColor, "%.3f %.3f %.3f", selectedEntry->lastSize.x,
              selectedEntry->lastSize.y, selectedEntry->lastSize.z);

            beginPropRow("Color");
            ImGui::TextColored(metaColor, "%.2f %.2f %.2f %.2f", selectedEntry->lastColor.r,
              selectedEntry->lastColor.g, selectedEntry->lastColor.b, selectedEntry->lastColor.a);

            const auto motionIt = motionEstimates.find(visualMotionKey(selectedTag, selectedIndex));
            if (motionIt != motionEstimates.end() && motionIt->second.valid) {
              const auto& motion = motionIt->second;
              beginPropRow("Lin vel");
              ImGui::TextColored(metaColor, "%.3f %.3f %.3f", motion.linearVelocity.x,
                motion.linearVelocity.y, motion.linearVelocity.z);

              beginPropRow("Speed");
              ImGui::TextColored(metaColor, "%.3f m/s", glm::length(motion.linearVelocity));

              beginPropRow("Angular");
              ImGui::TextColored(metaColor, "%.3f rad/s", motion.angularSpeed);
            }

            if (!selectedEntry->resourceDir.empty()) {
              beginPropRow("Resource");
              ImGui::TextColored(metaColor, "%s", shortenPathLabel(selectedEntry->resourceDir, 56).c_str());
            }

            ImGui::EndTable();
          }

          // ----- Live signal workbench -----
          {
            const uint64_t selectedKey = visualMotionKey(selectedTag, selectedIndex);
            const auto selectedRecordIt = signalRecords.find(selectedKey);
            const raisin::tcp_viewer::SignalHistory* selectedHistory =
              selectedRecordIt == signalRecords.end() ? nullptr : &selectedRecordIt->second.history;
            const std::vector<raisin::tcp_viewer::SignalChannelDesc>& availableChannels =
              selectedHistory ? selectedHistory->channels()
                              : kEmptySignalChannels;

            ImGui::SeparatorText("Live Signals");
            if (!signalPlotChannelsInitialized && !availableChannels.empty()) {
              signalPlotChannels = raisin::tcp_viewer::defaultSignalChannelKeys(availableChannels);
              signalPlotChannelsInitialized = true;
            }

            const bool selectionPinned = pinnedSignalObjects.count(selectedKey) != 0;
            bool pinToggle = selectionPinned;
            if (drawCompactCheckbox("Keep recording when deselected", &pinToggle)) {
              if (pinToggle) {
                pinnedSignalObjects.insert(selectedKey);
              } else {
                pinnedSignalObjects.erase(selectedKey);
              }
            }
            if (!pinnedSignalObjects.empty()) {
              ImGui::SameLine();
              ImGui::TextDisabled("(%zu pinned)", pinnedSignalObjects.size());
            }

            if (ImGui::TreeNode("Channels")) {
              if (availableChannels.empty()) {
                ImGui::TextDisabled("No channels yet; waiting for scene updates");
              }
              for (const auto& channel : availableChannels) {
                const auto existing = std::find(signalPlotChannels.begin(),
                  signalPlotChannels.end(), channel.key);
                bool shown = existing != signalPlotChannels.end();
                ImGui::PushID(channel.key.c_str());
                if (drawCompactCheckbox(channel.label.c_str(), &shown)) {
                  if (shown) {
                    signalPlotChannels.push_back(channel.key);
                  } else {
                    signalPlotChannels.erase(existing);
                  }
                  signalPlotChannelsInitialized = true;
                }
                if (channel.scope == raisin::tcp_viewer::SignalChannelScope::SelectionOnly) {
                  ImGui::SameLine();
                  ImGui::TextDisabled("(selection only)");
                }
                ImGui::PopID();
              }
              ImGui::TreePop();
            }

            const ImVec2 plotSize(std::max(240.0f, ImGui::GetContentRegionAvail().x),
                                  ImGui::GetFontSize() * 4.5f);
            // Plot every recorded object so a pinned trace stays visible next to
            // the current selection.
            std::vector<uint64_t> plottedKeys;
            plottedKeys.reserve(signalRecords.size());
            if (selectedHistory) {
              plottedKeys.push_back(selectedKey);
            }
            for (const auto& [key, record] : signalRecords) {
              if (key != selectedKey) {
                plottedKeys.push_back(key);
              }
            }

            std::vector<float> series;
            for (const uint64_t key : plottedKeys) {
              const auto recordIt = signalRecords.find(key);
              if (recordIt == signalRecords.end()) {
                continue;
              }
              const SignalObjectRecord& record = recordIt->second;
              if (record.history.size() < 2) {
                continue;
              }
              if (plottedKeys.size() > 1) {
                ImGui::TextDisabled("%s%s", record.label.c_str(),
                  key == selectedKey ? " (selected)" : " (pinned)");
              }
              ImGui::PushID(static_cast<int>(key & 0x7fffffffu));
              for (const std::string& channelKey : signalPlotChannels) {
                if (!record.history.series(channelKey, series) || series.empty()) {
                  continue;
                }
                const int channelIndex = record.history.channelIndex(channelKey);
                const auto& channel =
                  record.history.channels()[static_cast<size_t>(channelIndex)];
                ImGui::TextUnformatted(channel.label.c_str());
                char currentValue[80];
                const bool current = record.history.sampleAvailable(record.history.size() - 1,
                  static_cast<size_t>(channelIndex));
                std::snprintf(currentValue, sizeof(currentValue),
                  ((current ? "Current " : "Unavailable; last ") + channel.valueFormat).c_str(), series.back());
                ImGui::PlotLines(("##signal_" + channelKey).c_str(), series.data(),
                  static_cast<int>(series.size()), 0, currentValue, FLT_MAX, FLT_MAX, plotSize);
              }
              ImGui::PopID();
            }
            if (signalPlotChannels.empty()) {
              ImGui::TextDisabled("Pick channels above to plot them");
            }

            ImGui::BeginDisabled(!selectedHistory || selectedHistory->empty() ||
                                 signalPlotChannels.empty());
            if (drawIconTextButton(uiIcons, TcpViewerIconKind::Export, "Export CSV",
                                   "export_signal_csv")) {
              const std::filesystem::path csvPath =
                raisin::tcp_viewer::timestampedSignalCsvPath(
                  std::filesystem::path(screenshotDirBuf), objectName,
                  std::time(nullptr));
              raisin::tcp_viewer::writeSignalCsv(csvPath, *selectedHistory, signalPlotChannels,
                signalExportStatus);
              lastStatus = signalExportStatus;
            }
            ImGui::EndDisabled();
            if (!signalExportStatus.empty()) {
              ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + plotSize.x);
              ImGui::TextDisabled("%s", signalExportStatus.c_str());
              ImGui::PopTextWrapPos();
            }
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
                  const int32_t type = i < selectedInfo.jointTypes.size() ?
                    selectedInfo.jointTypes[i] : int32_t(raisim::Joint::Type::FIXED);
                  drawJointTypeIcon(uiIcons, type);
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
              ImGui::EndTabItem();
            }
            if (!selectedSensors.empty()) {
              const std::string sensorTabLabel =
                "Sensors (" + std::to_string(selectedSensors.size()) + ")";
              if (ImGui::BeginTabItem(sensorTabLabel.c_str())) {
                drawObjectSensors(
                  uiIcons, selectedSensors, sensorRenderer.previewsForTag(selectedTag),
                  cameraFrustums);
                ImGui::EndTabItem();
              }
            }
            if (selectedInfo.isArticulated && !selectedInfo.actuators.empty()) {
              const std::string actuatorTabLabel =
                "Actuators (" + std::to_string(selectedInfo.actuators.size()) + ")";
              if (ImGui::BeginTabItem(actuatorTabLabel.c_str())) {
                raisin::tcp_viewer::drawObjectActuators(selectedInfo, actuatorTraces);
                ImGui::EndTabItem();
              }
            }
            ImGui::EndTabBar();
          }
        }
      }
      ImGui::End();
      ImGui::PopStyleVar();
    }

    // ----- Local AS inspector window — styled to match the left "Raisim TCP" panel -----
    if (inspector.active && inspector.as) {
      // Mirror the left overlay's positioning/style so the inspector visually replaces it
      // (the left overlay is hidden while inspector.active).
      const ImVec2 inspectorBase(paneOrigin.x + 12.0f, paneOrigin.y + 12.0f + menuBarHeight);
      // Both width and height auto-fit the joint list. Width is locked to whatever
      // the longest joint label needs plus the slider budget. Height grows with the
      // joint count, but is capped at displaySize so the window never pushes
      // off-screen — only then does a vertical scrollbar appear.
      const ImGuiStyle& imStyle = ImGui::GetStyle();
      const float namePixelBudget = [&]() {
        float w = ImGui::CalcTextSize("Articulated System Inspector").x;
        for (const auto& j : inspector.joints) {
          // The label row prints "<name>  [type]"; account for the type tag width too.
          const std::string row = j.name + "  [floating]";
          w = std::max(w, ImGui::CalcTextSize(row.c_str()).x);
        }
        return w;
      }();
      const float sliderBudget = 220.0f;   // room for the DragFloat/SliderFloat widgets
      const float wantedWidth = std::min(
          uiSize.x - (inspectorBase.x - paneOrigin.x) - 20.0f,
          namePixelBudget + sliderBudget + imStyle.WindowPadding.x * 2.0f + 24.0f);
      const float maxHeight =
        std::max(240.0f, uiSize.y - (inspectorBase.y - paneOrigin.y) - 16.0f);

      ImGui::SetNextWindowBgAlpha(0.5f);
      ImGui::SetNextWindowPos(inspectorBase, ImGuiCond_Always);
      // Lock width via constraints (min==max) and let height auto-fit up to maxHeight.
      // AlwaysAutoResize keeps the window snug around its content every frame; the
      // ScrollY flag turns scroll on only when content exceeds the height cap.
      ImGui::SetNextWindowSizeConstraints(ImVec2(wantedWidth, 0.0f),
                                          ImVec2(wantedWidth, maxHeight));
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
      ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, ImGui::GetStyle().WindowBorderSize);
      if (ImGui::Begin(inspectorWindowName.c_str(), nullptr, kOverlayPanelWindowFlags)) {
        ImGui::PushStyleColor(ImGuiCol_TextDisabled,
          raionrobotics_imgui_secondary_text_color());
        // Title row — mirrors the "Status" line in the connection panel.
        const std::filesystem::path src(inspector.sourceFile);
        if (const TcpViewerIcon* robotIcon = uiIcons.get(TcpViewerIconKind::Robot)) {
          const float lineH = ImGui::GetFontSize() + 4.0f;
          ImGui::Image(reinterpret_cast<ImTextureID>(uint64_t(robotIcon->texture)),
                       ImVec2(lineH, lineH),
                       ImVec2(0, 0), ImVec2(1, 1),
                       tcpViewerIconTint(TcpViewerIconKind::Robot, false, false));
          ImGui::SameLine();
        }
        ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.0f, 1.0f), "Articulated System Inspector");
        ImGui::TextUnformatted(src.filename().string().c_str());
        ImGui::TextDisabled("%s", shortenPathLabel(inspector.sourceFile, 60).c_str());
        ImGui::TextDisabled("DoF %d  GC %zu  Joints %zu",
                            int(inspector.as->getDOF()),
                            inspector.gc.size(),
                            inspector.joints.size());
        if (!lastStatus.empty()) {
          ImGui::TextDisabled("Status: %s", lastStatus.c_str());
        }
        ImGui::Separator();
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Reset, "Reset pose", "reset_pose")) {
          std::fill(inspector.gc.begin(), inspector.gc.end(), 0.0);
          // For multi-DoF joints with a quaternion component, set qw=1 (identity rotation).
          for (const auto& j : inspector.joints) {
            if (j.type == raisim::Joint::Type::FLOATING && j.gcOffset + 3 < int(inspector.gc.size())) {
              inspector.gc[j.gcOffset + 3] = 1.0; // qw for floating: pos(3) + quat(wxyz)
            } else if (j.type == raisim::Joint::Type::SPHERICAL &&
                       j.gcOffset < int(inspector.gc.size())) {
              inspector.gc[j.gcOffset] = 1.0; // qw for spherical
            }
          }
          Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(
              inspector.gc.data(), Eigen::Index(inspector.gc.size()));
          inspector.as->setGeneralizedCoordinate(q);
        }
        ImGui::SameLine();
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Exit, "Close inspector",
                               "close_inspector")) {
          closeInspector();
        } else {
          ImGui::Separator();
          bool gcChanged = false;
          inspector.dragging = false;
          // Edits the wxyz quaternion stored at gc[offset..offset+3], renormalized.
          const auto dragNormalizedQuat = [&](const char* id, int offset) {
            float quat[4] = {float(inspector.gc[offset]),
                             float(inspector.gc[offset + 1]),
                             float(inspector.gc[offset + 2]),
                             float(inspector.gc[offset + 3])};
            if (compactDragFloat4(id, quat, 0.01f, -1.0f, 1.0f)) {
              const double n = std::sqrt(double(quat[0]) * quat[0] + double(quat[1]) * quat[1] +
                                         double(quat[2]) * quat[2] + double(quat[3]) * quat[3]);
              const double inv = n > 1e-6 ? 1.0 / n : 1.0;
              inspector.gc[offset] = quat[0] * inv;
              inspector.gc[offset + 1] = quat[1] * inv;
              inspector.gc[offset + 2] = quat[2] * inv;
              inspector.gc[offset + 3] = quat[3] * inv;
              gcChanged = true;
            }
            if (ImGui::IsItemActive()) inspector.dragging = true;
          };
          for (size_t ji = 0; ji < inspector.joints.size(); ++ji) {
            const auto& j = inspector.joints[ji];
            ImGui::PushID(int(ji));
            const char* typeLabel = "?";
            switch (j.type) {
              case raisim::Joint::Type::FIXED: typeLabel = "fixed"; break;
              case raisim::Joint::Type::REVOLUTE: typeLabel = "rev"; break;
              case raisim::Joint::Type::PRISMATIC: typeLabel = "pris"; break;
              case raisim::Joint::Type::SPHERICAL: typeLabel = "sph  quat"; break;
              case raisim::Joint::Type::FLOATING: typeLabel = "float  pos  quat"; break;
            }
            drawJointTypeIcon(uiIcons, static_cast<int32_t>(j.type));
            ImGui::Text("%s  [%s]", j.name.c_str(), typeLabel);
            if (j.type == raisim::Joint::Type::REVOLUTE ||
                j.type == raisim::Joint::Type::PRISMATIC) {
              float v = float(inspector.gc[j.gcOffset]);
              const bool useSlider = j.hasLimits;
              bool changed = false;
              if (useSlider) {
                changed = compactSliderFloat("##v", &v, float(j.minLimit), float(j.maxLimit),
                                             "%.4f");
              } else {
                changed = compactDragFloat("##v", &v, 0.01f, 0.0f, 0.0f, "%.4f");
              }
              if (changed) {
                inspector.gc[j.gcOffset] = v;
                gcChanged = true;
              }
              if (ImGui::IsItemActive()) inspector.dragging = true;
            } else if (j.type == raisim::Joint::Type::FLOATING) {
              float pos[3] = {float(inspector.gc[j.gcOffset]),
                              float(inspector.gc[j.gcOffset + 1]),
                              float(inspector.gc[j.gcOffset + 2])};
              if (compactDragFloat3("##float_pos", pos, 0.05f)) {
                inspector.gc[j.gcOffset] = pos[0];
                inspector.gc[j.gcOffset + 1] = pos[1];
                inspector.gc[j.gcOffset + 2] = pos[2];
                gcChanged = true;
              }
              if (ImGui::IsItemActive()) inspector.dragging = true;
              dragNormalizedQuat("##float_quat", j.gcOffset + 3);
            } else if (j.type == raisim::Joint::Type::SPHERICAL) {
              dragNormalizedQuat("##sph_quat", j.gcOffset);
            }
            ImGui::PopID();
          }
          if (gcChanged) {
            Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(
                inspector.gc.data(), Eigen::Index(inspector.gc.size()));
            inspector.as->setGeneralizedCoordinate(q);
          }
        }
        ImGui::PopStyleColor();
      }
      ImGui::End();
      ImGui::PopStyleVar(2);
    }

    // Shared file/folder picker. Drawn at top level rather than as a popup
    // inside a panel so it is not clipped by the panels' child windows and can
    // be centred on the viewport. Every control that needs a path opens this
    // one window, so they all navigate identically.
    if (fileBrowser.open) {
      const ImGuiViewport* browserViewport = ImGui::GetMainViewport();
      const float browserWidth = ImGui::GetFontSize() * 28.0f;
      const float browserHeight = ImGui::GetFontSize() * 24.0f;
      ImGui::SetNextWindowPos(
        ImVec2(browserViewport->WorkPos.x + browserViewport->WorkSize.x * 0.5f,
               browserViewport->WorkPos.y + browserViewport->WorkSize.y * 0.5f),
        ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
      ImGui::SetNextWindowSize(ImVec2(browserWidth, browserHeight), ImGuiCond_Appearing);
      // Same palette and padding as the side panels: the browser is part of the
      // same UI, and ImGui's stock window colours read as a different product.
      ImGui::SetNextWindowBgAlpha(0.5f);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
      bool browserOpen = true;
      const std::string browserTitle = fileBrowser.title + paneWindowSuffix + "##file_browser";
      const bool browserVisible =
        ImGui::Begin(browserTitle.c_str(), &browserOpen, ImGuiWindowFlags_NoSavedSettings);
      // Pushed inside the window so the secondary text matches the panels' own.
      ImGui::PushStyleColor(ImGuiCol_TextDisabled, raionrobotics_imgui_secondary_text_color());
      if (browserVisible) {
        const bool savingFile = fileBrowser.mode == FileBrowserMode::SaveFile;
        const bool pickingFile = fileBrowser.mode != FileBrowserMode::Folder;

        // Breadcrumb: one button per ancestor, so any level above the current
        // folder is a single click rather than repeated presses of Up. The
        // keyboard button swaps in the editable field, which is still the only
        // way to paste a path.
        if (fileBrowser.editingPath) {
          // Full width while editing: the whole path has to be visible to fix.
          ImGui::SetNextItemWidth(-FLT_MIN);
          if (ImGui::InputText("##file_browser_path", fileBrowser.pathBuf,
                sizeof(fileBrowser.pathBuf), ImGuiInputTextFlags_EnterReturnsTrue)) {
            navigateFileBrowser(fileBrowser, std::filesystem::path(fileBrowser.pathBuf));
            fileBrowser.editingPath = false;
          }
          if (ImGui::IsItemDeactivated()) {
            fileBrowser.editingPath = false;
          }
        } else {
          std::filesystem::path crumbPath;
          bool firstCrumb = true;
          bool needSlash = false;
          int crumbIndex = 0;
          std::filesystem::path crumbTarget;
          const float slashWidth = ImGui::CalcTextSize("/").x;
          const float innerSpacing = ImGui::GetStyle().ItemInnerSpacing.x;
          // Right edge available to the row, captured before anything is drawn.
          // There is no horizontal scrollbar, so a path too long for one line has
          // to wrap rather than run off the window.
          const float contentRight =
            ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
          // Zero horizontal item spacing so the separators sit flush against the
          // folder buttons and the row reads as one path.
          ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
            ImVec2(0.0f, ImGui::GetStyle().ItemSpacing.y));
          for (const auto& part : fileBrowser.current) {
            crumbPath /= part;
            const std::string label = part.string();
            if (label.empty()) {
              continue;
            }
            // The root component already carries its own separator, so it is
            // drawn as the leading "/" and does not get one prefixed.
            const bool isRoot = firstCrumb && crumbPath == fileBrowser.current.root_path();
            const float slashSpan = needSlash ? slashWidth : 0.0f;
            const float buttonSpan =
              ImGui::CalcTextSize(label.c_str()).x + ImGui::GetStyle().FramePadding.x * 2.0f;
            if (!firstCrumb) {
              // GetItemRectMax() is the real right edge of the previous item;
              // the cursor has already advanced to the next line by now, so it
              // cannot be used to measure the current line. A separator and its
              // folder are measured together so they never split across lines.
              const float previousRight = ImGui::GetItemRectMax().x;
              if (previousRight + slashSpan + buttonSpan <= contentRight) {
                ImGui::SameLine(0.0f, 0.0f);
              }
            }
            if (needSlash) {
              ImGui::AlignTextToFramePadding();
              ImGui::TextUnformatted("/");
              ImGui::SameLine(0.0f, 0.0f);
            }
            // Index-keyed so repeated folder names along one path stay distinct.
            ImGui::PushID(crumbIndex++);
            if (ImGui::Button(label.c_str())) {
              crumbTarget = crumbPath;
            }
            if (ImGui::IsItemHovered()) {
              ImGui::SetTooltip("%s", crumbPath.string().c_str());
            }
            ImGui::PopID();
            firstCrumb = false;
            needSlash = !isRoot;
          }
          // The edit affordance joins the same wrapping flow, so it never pushes
          // the last folder off the line.
          const float editSpan = ImGui::GetFrameHeight() + innerSpacing;
          if (!firstCrumb && ImGui::GetItemRectMax().x + editSpan <= contentRight) {
            ImGui::SameLine(0.0f, innerSpacing);
          }
          if (drawIconOnlyButton(uiIcons, TcpViewerIconKind::Help, "Type or paste a path",
                                 "file_browser_edit_path")) {
            fileBrowser.editingPath = true;
          }
          ImGui::PopStyleVar();
          if (!crumbTarget.empty()) {
            navigateFileBrowser(fileBrowser, crumbTarget);
          }
        }

        ImGui::BeginDisabled(!fileBrowser.current.has_parent_path() ||
                             fileBrowser.current.parent_path() == fileBrowser.current);
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::FolderUp, "Up", "file_browser_up")) {
          navigateFileBrowser(fileBrowser, fileBrowser.current.parent_path());
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Refresh, "Refresh",
                               "file_browser_refresh")) {
          refreshFileBrowser(fileBrowser);
        }
        if (!fileBrowser.extensions.empty()) {
          ImGui::SameLine();
          std::string filterText;
          for (const std::string& extension : fileBrowser.extensions) {
            filterText += (filterText.empty() ? "." : "  .") + extension;
          }
          ImGui::AlignTextToFramePadding();
          ImGui::TextDisabled("%s", filterText.c_str());
        }

        // Reserve the footer: the name field (file modes) plus the button row.
        const float footerRows = savingFile || pickingFile ? 2.0f : 1.0f;
        const float footerHeight =
          ImGui::GetFrameHeightWithSpacing() * footerRows + ImGui::GetStyle().ItemSpacing.y;
        if (ImGui::BeginChild("##file_browser_list", ImVec2(0.0f, -footerHeight),
                              ImGuiChildFlags_Border)) {
          if (!fileBrowser.error.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextDisabled("%s", fileBrowser.error.c_str());
            ImGui::PopTextWrapPos();
          } else if (fileBrowser.entries.empty()) {
            ImGui::TextDisabled(pickingFile ? "Nothing matching here" : "No sub-folders here");
          } else {
            const float iconSize = std::round(ImGui::GetFontSize() * 0.95f);
            for (size_t i = 0; i < fileBrowser.entries.size(); ++i) {
              const FileBrowserState::Entry& entry = fileBrowser.entries[i];
              ImGui::PushID(static_cast<int>(i));
              const std::string name = entry.path.filename().string();
              const std::string label = name.empty() ? entry.path.string() : name;
              const TcpViewerIconKind kind = fileBrowserIconKind(entry.path, entry.isDirectory);

              // Draw the row first, then paint its icon into the leading gap,
              // so the whole row stays one clickable Selectable.
              const bool clicked = ImGui::Selectable(
                (std::string("      ") + label).c_str(),
                !entry.isDirectory && name == fileBrowser.nameBuf);
              if (const TcpViewerIcon* icon = uiIcons.get(kind)) {
                const ImVec2 rowMin = ImGui::GetItemRectMin();
                const ImVec2 rowMax = ImGui::GetItemRectMax();
                const float centreY = (rowMin.y + rowMax.y) * 0.5f;
                const ImVec2 iconMin(rowMin.x + ImGui::GetStyle().ItemInnerSpacing.x,
                                     std::round(centreY - iconSize * 0.5f));
                ImGui::GetWindowDrawList()->AddImage(
                  (ImTextureID)(intptr_t)icon->texture, iconMin,
                  ImVec2(iconMin.x + iconSize, iconMin.y + iconSize),
                  ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f),
                  ImGui::GetColorU32(tcpViewerIconTint(kind, false, false)));
              }
              if (clicked) {
                // Directories navigate; files only select, so a stray click
                // never commits a choice. Accepting is always an explicit press.
                if (entry.isDirectory) {
                  navigateFileBrowser(fileBrowser, entry.path);
                  ImGui::PopID();
                  break;
                }
                std::snprintf(fileBrowser.nameBuf, sizeof(fileBrowser.nameBuf), "%s",
                              name.c_str());
              }
              ImGui::PopID();
            }
          }
        }
        ImGui::EndChild();

        if (pickingFile) {
          ImGui::TextDisabled("Name");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(-FLT_MIN);
          ImGui::InputText("##file_browser_name", fileBrowser.nameBuf,
                           sizeof(fileBrowser.nameBuf));
        }

        std::filesystem::path chosen = fileBrowser.current;
        if (pickingFile) {
          chosen = fileBrowser.nameBuf[0] == '\0'
            ? std::filesystem::path()
            : fileBrowser.current / std::filesystem::path(fileBrowser.nameBuf);
        }
        std::error_code chosenEc;
        const bool chosenUsable = !fileBrowser.error.empty() || chosen.empty()
          ? false
          : (fileBrowser.mode == FileBrowserMode::OpenFile
               ? std::filesystem::is_regular_file(chosen, chosenEc)
               : true);

        const char* acceptLabel = fileBrowser.mode == FileBrowserMode::Folder
          ? "Use This Folder"
          : (savingFile ? "Save Here" : "Open");
        ImGui::BeginDisabled(!chosenUsable);
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Add, acceptLabel,
                               "file_browser_accept")) {
          if (fileBrowser.onAccept) {
            fileBrowser.onAccept(chosen);
          }
          fileBrowser.open = false;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (drawIconTextButton(uiIcons, TcpViewerIconKind::Exit, "Cancel",
                               "file_browser_cancel")) {
          fileBrowser.open = false;
        }
        if (fileBrowser.mode == FileBrowserMode::OpenFile && !chosenUsable && !chosen.empty()) {
          ImGui::SameLine();
          ImGui::TextDisabled("no such file");
        }
      }
      ImGui::PopStyleColor();
      ImGui::End();
      ImGui::PopStyleVar();
      if (!browserOpen) {
        fileBrowser.open = false;
      }
    }


  };

  // ----- split-pane chrome -------------------------------------------------
  // Splitting or closing rewrites both the layout tree and the pane map that the
  // per-frame loop walks, so the menu queues a request and the loop applies it
  // once every pane has been drawn.
  struct PaneCommand {
    enum class Kind { Split, Close };
    Kind kind = Kind::Split;
    uint32_t pane = 0;
    raisin::tcp_viewer::SplitOrientation orientation =
      raisin::tcp_viewer::SplitOrientation::Vertical;
  };
  std::vector<PaneCommand> paneCommands;
  std::vector<raisin::tcp_viewer::PaneRect> paneRects;
  std::vector<raisin::tcp_viewer::SplitterHandle> paneSplitters;
  uint32_t contextMenuPane = 0;
  uint32_t draggingSplitter = 0;
  // A closed pane's texture is still referenced by the draw data for the frame
  // the user closed it in, so the renderer is destroyed one frame later.
  std::vector<std::unique_ptr<ViewerPane>> retiredPanes;

  auto applyPaneCommands = [&]() {
    for (const PaneCommand& command : paneCommands) {
      if (command.kind == PaneCommand::Kind::Split) {
        const uint32_t created = paneLayout.split(command.pane, command.orientation);
        if (created == 0) continue;
        panes.emplace(created, createPane(created));
        // The new pane opens on the view it was split off, so the split reads as
        // dividing what is on screen rather than jumping somewhere else.
        if (const auto sourceIt = panes.find(command.pane); sourceIt != panes.end()) {
          auto& sourceCamera = sourceIt->second->viewer->getCamera();
          auto& createdPane = *panes[created];
          applyCameraLookAt(createdPane.viewer->getCamera(), sourceCamera.getPosition(),
                            sourceCamera.target);
          createdPane.defaultCameraPos = createdPane.viewer->getCamera().getPosition();
          createdPane.defaultCameraTarget = createdPane.viewer->getCamera().target;
        }
        paneLayout.focus(created);
      } else {
        if (!paneLayout.close(command.pane)) continue;
        const auto it = panes.find(command.pane);
        if (it != panes.end()) {
          it->second->client.disconnect();
          it->second->localSimulation.stop();
          retiredPanes.push_back(std::move(it->second));
          panes.erase(it);
        }
      }
      paneLayoutDirty = true;
    }
    paneCommands.clear();
  };


  while (!quit && !gSignalQuit.load(std::memory_order_relaxed)) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL2_ProcessEvent(&event);
      if (event.type == SDL_QUIT)
        quit = true;
      if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
          event.window.windowID == SDL_GetWindowID(window))
        quit = true;
      if (event.type == SDL_DROPFILE && event.drop.file) {
        // The pane that has focus owns the scene a drop would replace, and its
        // load handler lives inside paneFrame, so hand the path over instead of
        // loading from the event pump.
        pendingDropPath = event.drop.file;
        SDL_free(event.drop.file);
      }
    }

    int fbW = 0;
    int fbH = 0;
    SDL_GL_GetDrawableSize(window, &fbW, &fbH);
    const ImVec2 displaySize(static_cast<float>(fbW), static_cast<float>(fbH));
    // SDL_GL_GetDrawableSize reports physical pixels, which is what the UI-scale
    // heuristic below wants. ImGui, though, lays windows out in logical points,
    // so anything that positions or clamps a window has to use the window size
    // instead: on a HiDPI display the two differ by the backing scale, and a
    // right-anchored panel placed in pixels ends up off-screen.
    int winW = 0;
    int winH = 0;
    SDL_GetWindowSize(window, &winW, &winH);
    const ImVec2 uiSize(static_cast<float>(winW > 0 ? winW : fbW),
                        static_cast<float>(winH > 0 ? winH : fbH));
    const float scaleX = displaySize.x / 1920.0f;
    const float scaleY = displaySize.y / 1080.0f;
    defaultUiScale = std::clamp(std::min(scaleX, scaleY) * 1.25f, 1.1f, 2.6f);
    const bool displaySizeChanged = displaySize.x != lastDisplaySize.x ||
                                    displaySize.y != lastDisplaySize.y;
    uiScale = resolveUiScaleForDisplay(uiScale, defaultUiScale, uiScaleUserSet,
                                       uiScaleInitialized, displaySizeChanged);
    uiScaleInitialized = true;
    lastDisplaySize = displaySize;
    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.0f;
    if (!baseStyleCaptured) {
      baseStyle = ImGui::GetStyle();
      baseStyleCaptured = true;
    }
    if (std::abs(appliedUiScale - uiScale) > kUiScaleEpsilon) {
      const float fontSize = std::max(1.0f, std::round(kBaseFontSize * uiScale * kFontScale));
      ImFontConfig fontConfig;
      fontConfig.SizePixels = fontSize;
      fontConfig.OversampleH = 3;
      fontConfig.OversampleV = 2;
      fontConfig.PixelSnapH = false;
      fontConfig.RasterizerDensity = std::max(fontRasterizerDensity,
        std::clamp(std::max(io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y), 1.0f, 2.0f));
      fontConfig.RasterizerMultiply = 1.04f;
      io.Fonts->Clear();
      io.Fonts->TexDesiredWidth = 2048;
      io.Fonts->TexGlyphPadding = 2;
      ImFont* uiFont = nullptr;
      if (!robotoFontPath.empty()) {
        uiFont = io.Fonts->AddFontFromFileTTF(robotoFontPath.c_str(), fontSize, &fontConfig);
      }
      static bool fontSelectionLogged = false;
      if (!fontSelectionLogged) {
        if (uiFont) {
          std::cerr << "INFO: TCP viewer font " << robotoFontPath
                    << " size_px=" << fontSize
                    << " rasterizer_density=" << fontConfig.RasterizerDensity << "\n";
        } else {
          std::cerr << "WARN: TCP viewer Roboto font not found; using ImGui default font\n";
        }
        fontSelectionLogged = true;
      }
      io.FontDefault = uiFont ? uiFont : io.Fonts->AddFontDefault(&fontConfig);
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

    // Panes closed last frame: their textures have been through the renderer by
    // now, so the GL objects can go.
    for (auto& retired : retiredPanes) {
      retired->shutdown();
    }
    retiredPanes.clear();

    // ----- split-pane layout -----------------------------------------------
    // A 2 px line. The grab band below is deliberately wider than the line, so a
    // divider this thin is still easy to catch with the pointer.
    const float splitterThickness = 2.0f;
    paneLayout.layout(0.0f, 0.0f, uiSize.x, uiSize.y, splitterThickness, paneRects, paneSplitters);
    leadPaneId = paneRects.empty() ? 0 : paneRects.front().pane;

    // Give every pane's backing window its slot in ImGui's window order before
    // any panel is drawn. Without this pass a later pane's image would paint over
    // an earlier pane's panels wherever one overflows its rectangle.
    for (const auto& rect : paneRects) {
      char paneWindowName[64];
      std::snprintf(paneWindowName, sizeof(paneWindowName), "Viewer##pane%u", rect.pane);
      ImGui::SetNextWindowPos(ImVec2(rect.x, rect.y));
      ImGui::SetNextWindowSize(ImVec2(std::max(1.0f, rect.width), std::max(1.0f, rect.height)));
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
      ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
      ImGui::Begin(paneWindowName, nullptr, kPaneViewportWindowFlags);
      ImGui::End();
      ImGui::PopStyleVar(2);
    }

    // Focus follows the pointer on any click or wheel, the way clicking into a
    // terminator pane does. It is resolved before the panes run so the pane the
    // user just clicked already owns this frame's keyboard shortcuts.
    const ImVec2 mousePos = io.MousePos;
    const auto paneUnderCursor = [&]() -> uint32_t {
      for (const auto& rect : paneRects) {
        if (mousePos.x >= rect.x && mousePos.x < rect.x + rect.width &&
            mousePos.y >= rect.y && mousePos.y < rect.y + rect.height) {
          return rect.pane;
        }
      }
      return 0;
    };
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
        ImGui::IsMouseClicked(ImGuiMouseButton_Right) || io.MouseWheel != 0.0f) {
      const uint32_t hoveredPane = paneUnderCursor();
      if (hoveredPane != 0) paneLayout.focus(hoveredPane);
    }

    if (!io.WantTextInput && io.KeyCtrl && io.KeyShift) {
      if (ImGui::IsKeyPressed(ImGuiKey_O, false)) {
        paneCommands.push_back({PaneCommand::Kind::Split, paneLayout.focused(),
                                raisin::tcp_viewer::SplitOrientation::Horizontal});
      }
      if (ImGui::IsKeyPressed(ImGuiKey_E, false)) {
        paneCommands.push_back({PaneCommand::Kind::Split, paneLayout.focused(),
                                raisin::tcp_viewer::SplitOrientation::Vertical});
      }
      if (ImGui::IsKeyPressed(ImGuiKey_W, false) && paneLayout.paneCount() > 1) {
        paneCommands.push_back({PaneCommand::Kind::Close, paneLayout.focused(),
                                raisin::tcp_viewer::SplitOrientation::Vertical});
      }
    }

    for (const auto& rect : paneRects) {
      const auto paneIt = panes.find(rect.pane);
      if (paneIt == panes.end()) continue;
      paneFrame(*paneIt->second, rect, rect.pane == paneLayout.focused());
    }

    // ----- dividers ---------------------------------------------------------
    // Drawn in a regular window layer below popups, and grabbed
    // over a slightly wider band than the drawn line so a thin divider is still
    // easy to hit.
    ImDrawList* chromeDrawList = raisin::tcp_viewer::paneChromeDrawList(uiSize);
    const bool popupOpen = ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopup);
    if (!paneSplitters.empty()) {
      const float grabPadding = std::max(2.0f, std::round(2.0f * uiScale));
      for (const auto& handle : paneSplitters) {
        const bool vertical = handle.orientation == raisin::tcp_viewer::SplitOrientation::Vertical;
        const ImVec2 dividerMin(handle.x, handle.y);
        const ImVec2 dividerMax(handle.x + handle.width, handle.y + handle.height);
        const bool over = !popupOpen && mousePos.x >= dividerMin.x - grabPadding &&
                          mousePos.x <= dividerMax.x + grabPadding &&
                          mousePos.y >= dividerMin.y - grabPadding &&
                          mousePos.y <= dividerMax.y + grabPadding;
        if (over && draggingSplitter == 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
          draggingSplitter = handle.node;
        }
        const bool active = draggingSplitter == handle.node;
        if (over || active) {
          ImGui::SetMouseCursor(vertical ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
        }
        chromeDrawList->AddRectFilled(dividerMin, dividerMax, ImGui::GetColorU32(
          (over || active) ? ImVec4(0.36f, 0.55f, 0.85f, 0.95f)
                           : ImVec4(0.09f, 0.10f, 0.12f, 1.0f)));
        if (active) {
          const float usable = vertical
            ? std::max(1.0f, handle.regionWidth - handle.width)
            : std::max(1.0f, handle.regionHeight - handle.height);
          const float offset = vertical
            ? mousePos.x - handle.regionX - handle.width * 0.5f
            : mousePos.y - handle.regionY - handle.height * 0.5f;
          paneLayout.setSplitterRatio(handle.node, offset / usable);
          paneLayoutDirty = true;
        }
      }
      if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) draggingSplitter = 0;
    }

    // The focused pane gets a border so it is obvious which one the keyboard
    // shortcuts and the panels' Connect buttons are about to act on.
    if (paneRects.size() > 1) {
      for (const auto& rect : paneRects) {
        if (rect.pane != paneLayout.focused()) continue;
        chromeDrawList->AddRect(
          ImVec2(rect.x, rect.y), ImVec2(rect.x + rect.width, rect.y + rect.height),
          ImGui::GetColorU32(ImVec4(0.36f, 0.55f, 0.85f, 0.85f)), 0.0f, 0,
          std::max(1.0f, std::round(2.0f * uiScale)));
      }
    }

    // ----- context menu -----------------------------------------------------
    // Only a right-click on a pane's rendered image opens it: over a panel the
    // image is not hovered, so the panel keeps its own right-click behaviour.
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && draggingSplitter == 0) {
      for (const auto& rect : paneRects) {
        const auto paneIt = panes.find(rect.pane);
        if (paneIt == panes.end() || !paneIt->second->viewportState.hovered) continue;
        contextMenuPane = rect.pane;
        ImGui::OpenPopup("##pane_context_menu");
        break;
      }
    }
    if (ImGui::BeginPopup("##pane_context_menu")) {
      // Name the pane by what it is connected to, which is what tells two panes
      // apart on screen; the pane id is an implementation detail.
      if (const auto menuPaneIt = panes.find(contextMenuPane); menuPaneIt != panes.end()) {
        const ViewerPane& menuPane = *menuPaneIt->second;
        ImGui::TextDisabled("%s:%d — %s", menuPane.host, menuPane.port,
                            menuPane.lastStatus.c_str());
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Split Horizontally", "Ctrl+Shift+O")) {
        paneCommands.push_back({PaneCommand::Kind::Split, contextMenuPane,
                                raisin::tcp_viewer::SplitOrientation::Horizontal});
      }
      if (ImGui::MenuItem("Split Vertically", "Ctrl+Shift+E")) {
        paneCommands.push_back({PaneCommand::Kind::Split, contextMenuPane,
                                raisin::tcp_viewer::SplitOrientation::Vertical});
      }
      ImGui::Separator();
      ImGui::BeginDisabled(paneLayout.paneCount() <= 1);
      if (ImGui::MenuItem("Close Pane", "Ctrl+Shift+W")) {
        paneCommands.push_back({PaneCommand::Kind::Close, contextMenuPane,
                                raisin::tcp_viewer::SplitOrientation::Vertical});
      }
      ImGui::EndDisabled();
      ImGui::EndPopup();
    }

    applyPaneCommands();

    // A split, a close or a divider drag is a settings change like any other, so
    // it rides the same debounce as the render settings.
    if (paneLayoutDirty) {
      paneLayoutDirty = false;
      settingsDirty = true;
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window);
  }

  if (logExitFps) {
    // The pane that owned the command line may have been closed; report on the
    // focused pane, falling back to any surviving one.
    const ViewerPane* summaryPane = nullptr;
    if (const auto focusedIt = panes.find(paneLayout.focused()); focusedIt != panes.end()) {
      summaryPane = focusedIt->second.get();
    } else if (!panes.empty()) {
      summaryPane = panes.begin()->second.get();
    }
    const double elapsedSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - fpsMeasureStart).count();
    const double fps = elapsedSeconds > 1e-9
      ? static_cast<double>(fpsMeasureFrames) / elapsedSeconds
      : 0.0;
    std::cerr << "[rayrai tcp viewer fps] frames=" << fpsMeasureFrames
              << " seconds=" << std::fixed << std::setprecision(3) << elapsedSeconds
              << " fps=" << std::setprecision(1) << fps
              << " connected=" << (summaryPane && summaryPane->client.isConnected() ? "yes" : "no")
              << " auto_connect=" << (summaryPane && summaryPane->autoConnect ? "yes" : "no") << "\n";
  }

  if (settingsDirty || settingsSavePending) {
    settings.recentConnections = recentConnections;
    settings.resourceDirs = resourceDirs;
    capturePaneSettings();
    if (!options.noSaveSettings) saveViewerSettings(settings);
    settingsDirty = false;
    settingsSavePending = false;
  }

  raisimLogo.release();
  uiIcons.release();
  for (auto& retired : retiredPanes) {
    retired->shutdown();
  }
  retiredPanes.clear();
  for (auto& [paneId, pane] : panes) {
    (void)paneId;
    pane->shutdown();
  }
  panes.clear();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(context);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return viewerExitCode;
}
#endif  // RAYRAI_TCP_VIEWER_NO_MAIN
