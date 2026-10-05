// Viewer helpers, included inside the implementation namespace.
std::string toLowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string trimAscii(const std::string& value) {
  const auto first = std::find_if_not(value.begin(), value.end(),
    [](unsigned char c) { return std::isspace(c); });
  const auto last = std::find_if_not(value.rbegin(), value.rend(),
    [](unsigned char c) { return std::isspace(c); }).base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

void sortServerEntries(std::vector<ServerEntry>& servers) {
  std::sort(servers.begin(), servers.end(), [](const ServerEntry& lhs, const ServerEntry& rhs) {
    if (lhs.remoteBeacon != rhs.remoteBeacon) {
      return !lhs.remoteBeacon;
    }
    if (lhs.bindHost != rhs.bindHost) {
      return lhs.bindHost < rhs.bindHost;
    }
    if (lhs.endpoint.port != rhs.endpoint.port) {
      return lhs.endpoint.port < rhs.endpoint.port;
    }
    return lhs.protocol < rhs.protocol;
  });
}

ServerEntry serverEntryFromDiscovered(const DiscoveredServer& discovered) {
  ServerEntry server;
  server.endpoint.host = discovered.endpointHost;
  server.endpoint.port = discovered.endpointPort;
  server.bindHost = discovered.bindHost;
  server.process = discovered.process;
  server.protocol = discovered.protocol;
  server.metadata = discovered.metadata;
  server.remoteBeacon = discovered.remoteBeacon;
  server.lastSeen = discovered.lastSeen;
  return server;
}

std::vector<ServerEntry> serverEntriesFromDiscovered(
  const std::vector<DiscoveredServer>& discoveredServers) {
  std::vector<ServerEntry> entries;
  entries.reserve(discoveredServers.size());
  for (const auto& discovered : discoveredServers) {
    entries.push_back(serverEntryFromDiscovered(discovered));
  }
  sortServerEntries(entries);
  return entries;
}

std::string formatServerLabel(const ServerEntry& server) {
  std::string label = formatConnectionLabel(ConnectionEntry{server.bindHost, server.endpoint.port});
  if (server.bindHost != server.endpoint.host) {
    label += " (connect " + formatConnectionLabel(server.endpoint) + ")";
  }
  if (!server.protocol.empty()) {
    label += " [" + server.protocol + "]";
  }
  if (!server.process.empty()) {
    label += " - " + shortenPathLabel(server.process, 72);
  }
  return label;
}

bool readEnvBool(const char* name, bool defaultValue) {
  const char* rawValue = std::getenv(name);
  if (!rawValue || !*rawValue) {
    return defaultValue;
  }
  const std::string value = toLowerAscii(rawValue);
  return !(value == "0" || value == "false" || value == "off" || value == "no");
}

bool shouldQuitForInitialServerWait(double waitForServerSeconds, bool replayMode,
                                    bool connected, bool everConnected,
                                    double wallElapsedSeconds) {
  return waitForServerSeconds > 0.0 && !replayMode && !connected && !everConnected &&
         wallElapsedSeconds >= waitForServerSeconds;
}

float readEnvFloatClamped(const char* name, float defaultValue, float minValue, float maxValue) {
  const char* rawValue = std::getenv(name);
  if (!rawValue || !*rawValue) {
    return defaultValue;
  }
  float parsed = defaultValue;
  if (!parseFloatStrict(rawValue, parsed)) {
    return defaultValue;
  }
  return std::clamp(parsed, minValue, maxValue);
}

struct GpuQualityInfo {
  std::string vendor;
  std::string renderer;
  std::string version;
  int maxTextureSize = 0;
  int maxSamples = 0;
  int maxTextureImageUnits = 0;
};

struct GpuQualityRecommendation {
  int quality = 1;
  GpuQualityInfo gpu;
};

bool containsToken(const std::string& haystack, const std::string& token) {
  return haystack.find(token) != std::string::npos;
}

bool containsAnyToken(const std::string& haystack, std::initializer_list<const char*> tokens) {
  for (const char* token : tokens) {
    if (token != nullptr && containsToken(haystack, token)) {
      return true;
    }
  }
  return false;
}

int automaticRenderQualityFromGpuInfo(const GpuQualityInfo& info) {
  const std::string gpuText = toLowerAscii(info.vendor + " " + info.renderer + " " + info.version);
  if (containsAnyToken(gpuText, {
        "llvmpipe", "softpipe", "swiftshader", "software rasterizer",
        "microsoft basic render", "mesa offscreen", "swrast"})) {
    return 0;
  }

  int score = 0;
  const bool highEndDiscrete = containsAnyToken(gpuText, {
    "rtx 4090", "rtx 4080", "rtx 4070", "rtx 3090", "rtx 3080", "rtx 3070",
    "rx 7900", "rx 7800", "rx 7700", "rx 6950", "rx 6900", "rx 6800",
    "radeon pro w7", "radeon pro w6"});
  const bool lowEndDiscrete = containsAnyToken(gpuText, {
    "gtx 1050", "gtx 1060", "gtx 1650", "gtx 1660", "mx150", "mx250", "mx350",
    "rx 550", "rx 560", "rx 570", "rx 580"});
  const bool nvidia = containsAnyToken(gpuText, {"nvidia", "geforce", "quadro", "rtx", "gtx"});
  const bool amd = containsAnyToken(gpuText, {"amd", "radeon", "advanced micro devices"});
  const bool intel = containsAnyToken(gpuText, {"intel", "uhd graphics", "iris", "arc"});
  const bool apple = containsAnyToken(gpuText, {"apple", "metal"});

  if (highEndDiscrete) {
    score += 6;
  } else if (lowEndDiscrete) {
    score += 2;
  } else if (nvidia || amd || apple) {
    score += 4;
  } else if (intel) {
    score += containsToken(gpuText, "arc") ? 3 : 2;
  } else {
    score += 1;
  }

  if (info.maxTextureSize >= 32768) {
    score += 2;
  } else if (info.maxTextureSize >= 16384) {
    score += 1;
  } else if (info.maxTextureSize > 0 && info.maxTextureSize < 8192) {
    score -= 1;
  }

  if (info.maxSamples >= 8) {
    score += 1;
  } else if (info.maxSamples > 0 && info.maxSamples < 4) {
    score -= 1;
  }

  if (info.maxTextureImageUnits >= 32) {
    score += 1;
  } else if (info.maxTextureImageUnits > 0 && info.maxTextureImageUnits < 16) {
    score -= 1;
  }

  if (score <= 1) {
    return 0;
  }
  if (score <= 4) {
    return 1;
  }
  if (score <= 8) {
    return 2;
  }
  return 3;
}

std::string glStringValue(gl::GLenum name) {
  const auto* raw = reinterpret_cast<const char*>(gl::glGetString(name));
  return raw != nullptr ? std::string(raw) : std::string();
}

GpuQualityRecommendation recommendRenderQualityForCurrentGpu() {
  GpuQualityRecommendation recommendation;
  recommendation.gpu.vendor = glStringValue(gl::GL_VENDOR);
  recommendation.gpu.renderer = glStringValue(gl::GL_RENDERER);
  recommendation.gpu.version = glStringValue(gl::GL_VERSION);

  const auto glInt = [](gl::GLenum name) {
    gl::GLint value = 0;
    gl::glGetIntegerv(name, &value);
    return static_cast<int>(value);
  };
  recommendation.gpu.maxTextureSize = glInt(gl::GL_MAX_TEXTURE_SIZE);
  recommendation.gpu.maxSamples = glInt(gl::GL_MAX_SAMPLES);
  recommendation.gpu.maxTextureImageUnits = glInt(gl::GL_MAX_TEXTURE_IMAGE_UNITS);

  recommendation.quality = automaticRenderQualityFromGpuInfo(recommendation.gpu);
  return recommendation;
}

raisin::RayraiWindow::RenderQualityPreset qualityPresetFromIndex(int quality) {
  switch (std::clamp(quality, 0, 4)) {
    case 0: return raisin::RayraiWindow::RenderQualityPreset::Fast;
    case 2: return raisin::RayraiWindow::RenderQualityPreset::High;
    case 3: return raisin::RayraiWindow::RenderQualityPreset::Ultra;
    case 4: return raisin::RayraiWindow::RenderQualityPreset::Custom;
    case 1:
    default: return raisin::RayraiWindow::RenderQualityPreset::Balanced;
  }
}

raisin::RayraiWindow::WeatherPreset weatherPresetFromIndex(int preset) {
  switch (std::clamp(preset, 0, 10)) {
    case 0: return raisin::RayraiWindow::WeatherPreset::Clear;
    case 1: return raisin::RayraiWindow::WeatherPreset::Hazy;
    case 2: return raisin::RayraiWindow::WeatherPreset::Overcast;
    case 3: return raisin::RayraiWindow::WeatherPreset::Fog;
    case 4: return raisin::RayraiWindow::WeatherPreset::Rain;
    case 5: return raisin::RayraiWindow::WeatherPreset::HeavyRain;
    case 6: return raisin::RayraiWindow::WeatherPreset::Snow;
    case 7: return raisin::RayraiWindow::WeatherPreset::Storm;
    case 8: return raisin::RayraiWindow::WeatherPreset::NightClear;
    case 9: return raisin::RayraiWindow::WeatherPreset::NightRain;
    case 10:
    default: return raisin::RayraiWindow::WeatherPreset::Custom;
  }
}

raisin::RayraiWindow::WeatherQuality weatherQualityFromIndex(int quality) {
  switch (std::clamp(quality, 0, 3)) {
    case 0: return raisin::RayraiWindow::WeatherQuality::Low;
    case 1: return raisin::RayraiWindow::WeatherQuality::Medium;
    case 3: return raisin::RayraiWindow::WeatherQuality::Ultra;
    case 2:
    default: return raisin::RayraiWindow::WeatherQuality::High;
  }
}

float normalizedDegrees(float degrees) {
  if (!std::isfinite(degrees)) {
    return 0.0f;
  }
  float wrapped = std::fmod(degrees, 360.0f);
  if (wrapped < 0.0f) {
    wrapped += 360.0f;
  }
  return wrapped;
}

glm::vec3 windDirectionFromDegrees(float degrees) {
  const float radians = normalizedDegrees(degrees) * kDegToRad;
  return glm::normalize(glm::vec3(std::cos(radians), std::sin(radians), 0.0f));
}

float windDirectionDegreesFromVector(const glm::vec3& direction) {
  if (glm::dot(direction, direction) <= 1.0e-8f) {
    return 0.0f;
  }
  return normalizedDegrees(std::atan2(direction.y, direction.x) * kRadToDeg);
}

std::string findRobotoFontPath(const std::filesystem::path& binaryDir) {
  if (const char* envPath = std::getenv("RAYRAI_TCP_VIEWER_FONT")) {
    if (*envPath && std::filesystem::exists(envPath)) {
      return envPath;
    }
  }

  std::vector<std::filesystem::path> candidates;
  candidates.push_back(binaryDir / kRobotoFontRelativePath);
  candidates.push_back(binaryDir / ".." / kRobotoFontRelativePath);
  candidates.push_back(binaryDir / ".." / ".." / kRobotoFontRelativePath);
  candidates.push_back(binaryDir / ".." / ".." / ".." / kRobotoFontRelativePath);
  candidates.push_back(binaryDir / ".." / "share/rayrai" / kRobotoFontRelativePath);
  candidates.push_back(std::filesystem::current_path() / kRobotoFontRelativePath);
  candidates.push_back(std::filesystem::current_path() / ".." / kRobotoFontRelativePath);

  const std::filesystem::path sourceDir = std::filesystem::path(__FILE__).parent_path();
  if (!sourceDir.empty()) {
    candidates.push_back(sourceDir / ".." / ".." / ".." / kRobotoFontRelativePath);
    candidates.push_back(sourceDir / ".." / ".." / ".." / ".." / kRobotoFontRelativePath);
    candidates.push_back(sourceDir / ".." / ".." / ".." / ".." / "raisim2Lib" / kRobotoFontRelativePath);
  }

  std::error_code ec;
  for (const auto& path : candidates) {
    const std::filesystem::path absolute = std::filesystem::absolute(path, ec).lexically_normal();
    if (!ec && std::filesystem::exists(absolute, ec) && !ec) {
      return absolute.string();
    }
    ec.clear();
  }

  return {};
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

bool parseVec3Env(const char* value, glm::vec3& vec) {
  float values[3]{};
  if (!parseFloatListStrict(value, values, 3)) {
    return false;
  }
  vec = glm::vec3(values[0], values[1], values[2]);
  return glm::length(vec) > 1e-4f;
}

bool parseCameraLookAtEnv(const char* value, glm::vec3& pos, glm::vec3& target) {
  float values[6]{};
  if (!parseFloatListStrict(value, values, 6)) {
    return false;
  }
  pos = glm::vec3(values[0], values[1], values[2]);
  target = glm::vec3(values[3], values[4], values[5]);
  return glm::length(target - pos) > 1e-4f;
}

void applyCameraLookAt(raisin::Camera& camera, const glm::vec3& pos, const glm::vec3& target) {
  camera.setCameraFixedTarget(false);
  camera.setCameraFixedDistance(false);
  camera.targetObject = nullptr;
  camera.position = pos;
  camera.target = target;
  camera.worldUp = glm::vec3(0.0f, 0.0f, 1.0f);
  const glm::vec3 dir = glm::normalize(target - pos);
  yawPitchFromDirection(dir, camera.yaw, camera.pitch);
  camera.front = dir;
  camera.update(false);
}

void frameBounds(raisin::RayraiWindow& viewer, const glm::vec3& minBound, const glm::vec3& maxBound) {
  const glm::vec3 center = 0.5f * (minBound + maxBound);
  const glm::vec3 extent = glm::max(maxBound - minBound, glm::vec3(0.25f));
  const float radius = std::max(0.25f, 0.5f * glm::length(extent));
  const float fovy = glm::radians(std::max(20.0f, viewer.getCamera().zoom));
  const float distance = std::max(1.0f, radius / std::tan(0.5f * fovy) * 1.55f);
  const glm::vec3 viewDir = glm::normalize(glm::vec3(1.55f, -1.85f, 1.05f));
  applyCameraLookAt(viewer.getCamera(), center + viewDir * distance, center);
}

enum class OrthoView { Top, Bottom, Front, Back, Left, Right };

// Snap the camera to a canonical orthographic axis-aligned view of the given
// world-space bounds. Up-axis is +Z (raisim convention).
void applyOrthoView(raisin::RayraiWindow& viewer, OrthoView v,
                    const glm::vec3& minBound, const glm::vec3& maxBound) {
  auto& camera = viewer.getCamera();
  const glm::vec3 center = 0.5f * (minBound + maxBound);
  const glm::vec3 extent = glm::max(maxBound - minBound, glm::vec3(0.25f));
  const float radius = std::max(0.25f, 0.5f * glm::length(extent));
  const float distance = std::max(radius * 2.5f, 1.0f);

  glm::vec3 dir{0.0f};   // camera-from-target direction
  switch (v) {
    case OrthoView::Top:    dir = glm::vec3(0.0f, 0.0f, 1.0f); break;
    case OrthoView::Bottom: dir = glm::vec3(0.0f, 0.0f, -1.0f); break;
    case OrthoView::Front:  dir = glm::vec3(1.0f, 0.0f, 0.0f); break;
    case OrthoView::Back:   dir = glm::vec3(-1.0f, 0.0f, 0.0f); break;
    case OrthoView::Right:  dir = glm::vec3(0.0f, 1.0f, 0.0f); break;
    case OrthoView::Left:   dir = glm::vec3(0.0f, -1.0f, 0.0f); break;
  }
  applyCameraLookAt(camera, center + dir * distance, center);
  camera.setProjectionMode(raisin::Camera::ProjectionMode::ORTHOGRAPHIC);
  camera.orthoScale = std::max(0.5f, std::max(extent.x, std::max(extent.y, extent.z)) * 1.25f);
}

bool frameScene(raisin::RayraiWindow& viewer, const RemoteScene& scene) {
  glm::vec3 minBound, maxBound;
  if (!scene.computeSceneBounds(minBound, maxBound)) {
    return false;
  }
  frameBounds(viewer, minBound, maxBound);
  return true;
}

bool frameSelected(raisin::RayraiWindow& viewer, const VisualEntry* entry) {
  if (!entry) {
    return false;
  }
  const glm::vec3 halfExtent = glm::max(glm::vec3(entry->lastSize) * 0.5f, glm::vec3(0.25f));
  frameBounds(viewer, entry->lastPos - halfExtent, entry->lastPos + halfExtent);
  return true;
}

glm::vec3 normalizedOr(const glm::vec3& value, const glm::vec3& fallback) {
  const float len = glm::length(value);
  if (std::isfinite(len) && len > 1.0e-6f) {
    return value / len;
  }
  const float fallbackLen = glm::length(fallback);
  if (std::isfinite(fallbackLen) && fallbackLen > 1.0e-6f) {
    return fallback / fallbackLen;
  }
  return glm::vec3(1.0f, 0.0f, 0.0f);
}

glm::vec3 mouseForceFromDragPixels(
  const glm::vec3& cameraRight, const glm::vec3& cameraUp, const ImVec2& dragPixels,
  float accelerationPerPixel) {
  const float scale = std::isfinite(accelerationPerPixel)
      ? std::max(0.0f, accelerationPerPixel)
      : 0.0f;
  const glm::vec3 right = normalizedOr(cameraRight, glm::vec3(1.0f, 0.0f, 0.0f));
  const glm::vec3 up = normalizedOr(cameraUp, glm::vec3(0.0f, 0.0f, 1.0f));
  return (right * dragPixels.x - up * dragPixels.y) * scale;
}

glm::vec3 mouseForceFromDragPixels(
  const raisin::Camera& camera, const ImVec2& dragPixels, float accelerationPerPixel) {
  return mouseForceFromDragPixels(camera.right, camera.up, dragPixels, accelerationPerPixel);
}

// quat holds w, x, y, z in its x, y, z, w components (VisualEntry::lastQuat).
glm::quat normalizedQuatFromWxyz(const glm::vec4& quat) {
  const float norm2 = glm::dot(quat, quat);
  if (!std::isfinite(norm2) || norm2 <= 1.0e-12f) {
    return glm::quat::wxyz(1.0f, 0.0f, 0.0f, 0.0f);
  }
  const glm::vec4 q = quat / std::sqrt(norm2);
  return glm::quat::wxyz(q.x, q.y, q.z, q.w);
}


glm::vec3 visualWorldPointToLocal(const VisualEntry& entry, const glm::vec3& worldPoint) {
  const glm::quat q = normalizedQuatFromWxyz(entry.lastQuat);
  return glm::inverse(q) * (worldPoint - entry.lastPos);
}

glm::vec3 visualLocalPointToWorld(const VisualEntry& entry, const glm::vec3& localPoint) {
  const glm::quat q = normalizedQuatFromWxyz(entry.lastQuat);
  return entry.lastPos + q * localPoint;
}

glm::vec3 mouseForceStartApplicationPoint(
    const VisualEntry& entry, const glm::vec3& fallbackOffset, bool hasClickedWorldPoint,
    const glm::vec3& clickedWorldPoint) {
  if (hasClickedWorldPoint && std::isfinite(clickedWorldPoint.x) &&
      std::isfinite(clickedWorldPoint.y) && std::isfinite(clickedWorldPoint.z)) {
    return clickedWorldPoint;
  }
  return entry.lastPos + fallbackOffset;
}

/**
 * @brief Project a world point with an explicit view-projection matrix.
 *
 * Split out from the camera overload so overlay geometry can be unit-tested
 * without a GL context (raisin::Camera owns framebuffer handles).
 */
bool projectWorldToViewportWithMatrix(
  const glm::mat4& viewProjection, const ViewerViewportState& viewport, const glm::vec3& world,
  ImVec2& screen) {
  if (viewport.size.x <= 1.0f || viewport.size.y <= 1.0f) {
    return false;
  }
  const glm::vec4 clip = viewProjection * glm::vec4(world, 1.0f);
  if (!std::isfinite(clip.x) || !std::isfinite(clip.y) || !std::isfinite(clip.z) ||
      !std::isfinite(clip.w) || clip.w <= 1.0e-6f) {
    return false;
  }
  const glm::vec3 ndc = glm::vec3(clip) / clip.w;
  if (!std::isfinite(ndc.x) || !std::isfinite(ndc.y) || !std::isfinite(ndc.z)) {
    return false;
  }
  screen.x = viewport.origin.x + (ndc.x * 0.5f + 0.5f) * viewport.size.x;
  screen.y = viewport.origin.y + (0.5f - ndc.y * 0.5f) * viewport.size.y;
  return std::isfinite(screen.x) && std::isfinite(screen.y);
}

bool projectWorldToViewport(
  const raisin::Camera& camera, const ViewerViewportState& viewport, const glm::vec3& world,
  ImVec2& screen) {
  return projectWorldToViewportWithMatrix(
    camera.getProjectionMatrix() * camera.getViewMatrix(), viewport, world, screen);
}

float screenDistancePixels(const ImVec2& a, const ImVec2& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

std::string formatRulerDistance(float meters) {
  if (!std::isfinite(meters) || meters < 0.0f) {
    return "--";
  }
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.3f m", meters);
  return buffer;
}

std::string formatRulerPoint(const glm::vec3& point) {
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "(%.3f, %.3f, %.3f)",
    point.x, point.y, point.z);
  return buffer;
}

const char* nextRulerPointLabel(const RulerToolState& ruler) {
  if (!ruler.hasA || (ruler.hasA && ruler.hasB) || ruler.nextPoint == 0) {
    return "A";
  }
  return "B";
}

std::string formatAngleDegrees(float radians) {
  if (!std::isfinite(radians)) return "--";
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.2f°", radians * 57.2957795f);
  return buffer;
}

const char* nextAnglePointLabel(const AngleToolState& angle) {
  if (angle.picked >= 3) return "A";
  static const char* names[] = {"A", "B", "C"};
  return names[angle.picked];
}

float computeAngleRadians(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
  // Angle at vertex B between rays BA and BC.
  const glm::vec3 ba = a - b;
  const glm::vec3 bc = c - b;
  const float lba = glm::length(ba);
  const float lbc = glm::length(bc);
  if (lba <= 1e-6f || lbc <= 1e-6f) return std::numeric_limits<float>::quiet_NaN();
  float cosA = glm::dot(ba, bc) / (lba * lbc);
  cosA = std::clamp(cosA, -1.0f, 1.0f);
  return std::acos(cosA);
}

bool readWorldPointAtCursor(
  const raisin::Camera& camera, const ViewerViewportState& viewport, glm::vec3& world) {
  const int width = camera.rtWidth();
  const int height = camera.rtHeight();
  const int x = viewport.cursorX;
  const int y = viewport.cursorY;
  if (!viewport.hovered || width <= 0 || height <= 0 || x < 0 || y < 0 || x >= width || y >= height ||
      camera.getSceneFbo() == 0) {
    return false;
  }

  gl::GLint previousReadFramebuffer = 0;
  gl::glGetIntegerv(gl::GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
  gl::glBindFramebuffer(gl::GL_READ_FRAMEBUFFER, camera.getSceneFbo());
  float depth = 1.0f;
  const int readY = (height - 1) - y;
  gl::glReadPixels(x, readY, 1, 1, gl::GL_DEPTH_COMPONENT, gl::GL_FLOAT, &depth);
  gl::glBindFramebuffer(gl::GL_READ_FRAMEBUFFER,
    static_cast<gl::GLuint>(previousReadFramebuffer));

  if (!std::isfinite(depth) || depth < 0.0f || depth >= 1.0f) {
    return false;
  }

  // The depth buffer gives the first rendered surface hit by the camera ray.
  const float ndcX = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width)) * 2.0f - 1.0f;
  const float ndcY = 1.0f - ((static_cast<float>(y) + 0.5f) / static_cast<float>(height)) * 2.0f;
  const float ndcZ = depth * 2.0f - 1.0f;
  const glm::vec4 clip(ndcX, ndcY, ndcZ, 1.0f);
  const glm::mat4 invViewProjection = glm::inverse(camera.getProjectionMatrix() * camera.getViewMatrix());
  const glm::vec4 worldH = invViewProjection * clip;
  if (!std::isfinite(worldH.x) || !std::isfinite(worldH.y) || !std::isfinite(worldH.z) ||
      !std::isfinite(worldH.w) || std::abs(worldH.w) <= 1.0e-6f) {
    return false;
  }

  world = glm::vec3(worldH) / worldH.w;
  return std::isfinite(world.x) && std::isfinite(world.y) && std::isfinite(world.z);
}

float mouseForceStartRadiusPixels(
  const raisin::Camera& camera, const ViewerViewportState& viewport,
  const VisualEntry& entry, const glm::vec3& applicationPoint) {
  constexpr float kMinRadius = 24.0f;
  constexpr float kMaxRadius = 96.0f;
  ImVec2 center;
  if (!projectWorldToViewport(camera, viewport, applicationPoint, center)) {
    return kMinRadius;
  }

  const glm::vec3 halfSize = glm::max(glm::vec3(entry.lastSize) * 0.5f, glm::vec3(0.05f));
  const float worldRadius = std::max(0.05f, glm::length(halfSize));
  const glm::vec3 right = normalizedOr(camera.right, glm::vec3(1.0f, 0.0f, 0.0f));
  const glm::vec3 up = normalizedOr(camera.up, glm::vec3(0.0f, 0.0f, 1.0f));

  float radius = kMinRadius;
  ImVec2 edge;
  if (projectWorldToViewport(camera, viewport, applicationPoint + right * worldRadius, edge)) {
    radius = std::max(radius, screenDistancePixels(center, edge));
  }
  if (projectWorldToViewport(camera, viewport, applicationPoint + up * worldRadius, edge)) {
    radius = std::max(radius, screenDistancePixels(center, edge));
  }
  return std::clamp(radius, kMinRadius, kMaxRadius);
}

bool isMouseWithinForceStartRadius(
  const raisin::Camera& camera, const ViewerViewportState& viewport, const VisualEntry& entry,
  const glm::vec3& applicationPoint, const ImVec2& mousePos, ImVec2& anchorScreen) {
  if (!projectWorldToViewport(camera, viewport, applicationPoint, anchorScreen)) {
    return false;
  }
  const float radius = mouseForceStartRadiusPixels(camera, viewport, entry, applicationPoint);
  return screenDistancePixels(mousePos, anchorScreen) <= radius;
}

bool isShiftModifierHeld(const ImGuiIO& io) {
  if (io.KeyShift || (io.KeyMods & ImGuiMod_Shift) != 0) {
    return true;
  }
  return (SDL_GetModState() & KMOD_SHIFT) != KMOD_NONE;
}

bool shouldRequestMouseForceCapture(
    bool mouseForceEnabled, bool shiftHeld, bool rulerEnabled, bool angleEnabled) {
  return mouseForceEnabled && shiftHeld && !rulerEnabled && !angleEnabled;
}

bool shouldSuppressViewportForMouseForce(
    bool mouseForceActive, bool shiftForceCaptureRequested, bool leftMouseDown) {
  return mouseForceActive || (shiftForceCaptureRequested && leftMouseDown);
}

bool isWireDragModifierHeld(const ImGuiIO& io) {
  // Ctrl everywhere, plus Cmd on macOS where Ctrl-click is a right-click gesture.
  if (io.KeyCtrl || (io.KeyMods & ImGuiMod_Ctrl) != 0) {
    return true;
  }
  const SDL_Keymod state = SDL_GetModState();
  if ((state & KMOD_CTRL) != KMOD_NONE) {
    return true;
  }
#if defined(__APPLE__)
  if (io.KeySuper || (io.KeyMods & ImGuiMod_Super) != 0 || (state & KMOD_GUI) != KMOD_NONE) {
    return true;
  }
#endif
  return false;
}

// The wire drag and the force drag both capture a left-drag on a body, so only
// one may arm at a time. Shift (force) wins because it is the older gesture.
bool shouldRequestWireDragCapture(
    bool wireDragEnabled, bool wireModifierHeld, bool shiftHeld, bool rulerEnabled,
    bool angleEnabled) {
  return wireDragEnabled && wireModifierHeld && !shiftHeld && !rulerEnabled && !angleEnabled;
}

bool shouldSuppressViewportForWireDrag(
    bool wireDragActive, bool wireDragCaptureRequested, bool leftMouseDown) {
  return wireDragActive || (wireDragCaptureRequested && leftMouseDown);
}

/**
 * @brief Unproject a viewport pixel into a world-space ray.
 *
 * Inverting the view-projection keeps the ray correct for both perspective and
 * orthographic projections. Takes the matrix rather than the camera so it can be
 * unit-tested without a GL context.
 */
bool viewportCursorRayWithMatrix(
  const glm::mat4& viewProjection, const ViewerViewportState& viewport, const ImVec2& cursor,
  glm::vec3& origin, glm::vec3& direction) {
  if (viewport.size.x <= 1.0f || viewport.size.y <= 1.0f) {
    return false;
  }
  const glm::mat4 inverseViewProjection = glm::inverse(viewProjection);
  const float ndcX = ((cursor.x - viewport.origin.x) / viewport.size.x) * 2.0f - 1.0f;
  const float ndcY = 1.0f - ((cursor.y - viewport.origin.y) / viewport.size.y) * 2.0f;
  const glm::vec4 nearClip = inverseViewProjection * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
  const glm::vec4 farClip = inverseViewProjection * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
  if (!std::isfinite(nearClip.w) || !std::isfinite(farClip.w) ||
      std::abs(nearClip.w) <= 1.0e-9f || std::abs(farClip.w) <= 1.0e-9f) {
    return false;
  }
  const glm::vec3 nearPoint = glm::vec3(nearClip) / nearClip.w;
  const glm::vec3 farPoint = glm::vec3(farClip) / farClip.w;
  const glm::vec3 delta = farPoint - nearPoint;
  const float length = glm::length(delta);
  if (!std::isfinite(length) || length <= 1.0e-9f) {
    return false;
  }
  origin = nearPoint;
  direction = delta / length;
  return true;
}

/**
 * @brief Where the interaction wire should pull, given the cursor.
 *
 * The target slides in the plane through @p anchorWorld whose normal is
 * @p planeNormal (the camera's forward axis), so the body follows the cursor
 * without changing its distance from the camera.
 */
bool wireDragTargetWithMatrix(
  const glm::mat4& viewProjection, const glm::vec3& planeNormal,
  const ViewerViewportState& viewport, const glm::vec3& anchorWorld, const ImVec2& cursor,
  glm::vec3& target) {
  glm::vec3 origin(0.0f);
  glm::vec3 direction(0.0f);
  if (!viewportCursorRayWithMatrix(viewProjection, viewport, cursor, origin, direction)) {
    return false;
  }
  const glm::vec3 normal = normalizedOr(planeNormal, glm::vec3(0.0f, 1.0f, 0.0f));
  const float denominator = glm::dot(direction, normal);
  if (!std::isfinite(denominator) || std::abs(denominator) <= 1.0e-6f) {
    return false;
  }
  const float distance = glm::dot(anchorWorld - origin, normal) / denominator;
  if (!std::isfinite(distance)) {
    return false;
  }
  const glm::vec3 hit = origin + direction * distance;
  if (!std::isfinite(hit.x) || !std::isfinite(hit.y) || !std::isfinite(hit.z)) {
    return false;
  }
  target = hit;
  return true;
}

bool wireDragTargetFromCursor(
  const raisin::Camera& camera, const ViewerViewportState& viewport, const glm::vec3& anchorWorld,
  const ImVec2& cursor, glm::vec3& target) {
  return wireDragTargetWithMatrix(camera.getProjectionMatrix() * camera.getViewMatrix(),
    camera.front, viewport, anchorWorld, cursor, target);
}

// ---------------------------------------------------------------------------
// Scene editing (CR_SPAWN_* / CR_REMOVE_OBJECT / CR_SAVE_THE_WORLD)
// ---------------------------------------------------------------------------

struct SpawnShapeInfo {
  raisin::tcp_viewer::ClientRequestType type;
  const char* label;
  /** Server-side geometry file; RaisimServer rejects the frame when it is missing. */
  bool needsFile;
  /** Whether the server requires a positive mass for this shape. */
  bool needsMass;
  /** Accepted extensions for `needsFile` shapes; empty means "any extension". */
  const char* fileExtensions;
};

inline constexpr std::array<SpawnShapeInfo, 8> kSpawnShapes{{
  {raisin::tcp_viewer::ClientRequestType::CR_SPAWN_BOX, "Box", false, true, ""},
  {raisin::tcp_viewer::ClientRequestType::CR_SPAWN_SPHERE, "Sphere", false, true, ""},
  {raisin::tcp_viewer::ClientRequestType::CR_SPAWN_CYLINDER, "Cylinder", false, true, ""},
  {raisin::tcp_viewer::ClientRequestType::CR_SPAWN_CAPSULE, "Capsule", false, true, ""},
  {raisin::tcp_viewer::ClientRequestType::CR_SPAWN_MESH, "Mesh", true, true, ""},
  {raisin::tcp_viewer::ClientRequestType::CR_SPAWN_AS, "Articulated system", true, false,
   "urdf xml"},
  {raisin::tcp_viewer::ClientRequestType::CR_SPAWN_PLANE, "Ground plane", false, false, ""},
  {raisin::tcp_viewer::ClientRequestType::CR_SPAWN_HEIGHT_MAP, "Height map (PNG)", true, false,
   "png"},
}};

const SpawnShapeInfo& spawnShapeAt(int index) {
  const size_t clamped = static_cast<size_t>(
    std::clamp(index, 0, static_cast<int>(kSpawnShapes.size()) - 1));
  return kSpawnShapes[clamped];
}

/** Editable state of the spawn palette. Kept flat so ImGui can bind to it directly. */
struct SpawnFormState {
  int shapeIndex = 0;
  char name[96] = "";
  char appearance[96] = "";
  char file[512] = "";
  float mass = 1.0f;
  int bodyType = static_cast<int>(raisin::tcp_viewer::SpawnBodyType::Dynamic);
  float boxExtent[3] = {0.4f, 0.4f, 0.4f};
  float radius = 0.2f;
  float height = 0.4f;
  float groundHeight = 0.0f;
  float heightMapCenter[2] = {0.0f, 0.0f};
  float heightMapSize[2] = {10.0f, 10.0f};
  float heightMapHeightScale = 1.0f;
  float heightMapHeightOffset = 0.0f;
  float position[3] = {0.0f, 0.0f, 1.0f};
  float linearVelocity[3] = {0.0f, 0.0f, 0.0f};
  float angularVelocity[3] = {0.0f, 0.0f, 0.0f};
  float quatWxyz[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  /** Drop the object in front of the camera instead of at the typed position. */
  bool useCameraPlacement = true;
};

/**
 * @brief Compact a path for display as `../parent/leaf`.
 *
 * A full absolute path overflows the panel's fields, and the part a user
 * actually recognises is the parent folder plus the leaf. Returns the path
 * unchanged when it already fits @p maxChars, and falls back to trimming the
 * leaf when even `../parent/leaf` is too long.
 */
std::string compactPathDisplay(const std::string& value, size_t maxChars) {
  if (maxChars == 0 || value.size() <= maxChars) {
    return value;
  }
  // Split on either separator so Windows paths compact the same way.
  std::vector<std::string> parts;
  std::string part;
  for (const char c : value) {
    if (c == '/' || c == '\\') {
      if (!part.empty()) {
        parts.push_back(part);
        part.clear();
      }
    } else {
      part.push_back(c);
    }
  }
  if (!part.empty()) {
    parts.push_back(part);
  }
  if (parts.size() < 2) {
    // Nothing to drop; keep the tail, which holds the file name.
    return value.size() <= maxChars ? value : "..." + value.substr(value.size() - (maxChars - 3));
  }

  // Keep whichever separator the path itself used, so a Windows path is not
  // displayed as though it had been rewritten.
  const bool backslashPath =
    value.find('\\') != std::string::npos && value.find('/') == std::string::npos;
  const std::string sep = backslashPath ? "\\" : "/";
  const std::string leaf = parts.back();
  const std::string parent = parts[parts.size() - 2];
  const std::string compact = ".." + sep + parent + sep + leaf;
  if (compact.size() <= maxChars) {
    return compact;
  }
  // Even the two-component form is too wide: keep the leaf, which is the part
  // that distinguishes one output file from another.
  const std::string leafOnly = "..." + sep + leaf;
  if (leafOnly.size() <= maxChars || maxChars <= 4) {
    return leafOnly;
  }
  return "..." + sep + leaf.substr(leaf.size() - (maxChars - 4));
}

/**
 * @brief A path field that shows `../parent/leaf` until it is clicked.
 *
 * The buffer always holds the full path; only the idle display is compacted, so
 * typing and pasting still operate on the real value. @p editingField tracks
 * which field is in edit mode -- one id is enough because ImGui can only have
 * one active item. The read-only and editable states share an ImGui id, so the
 * click that starts editing also lands the caret in the field.
 *
 * @return True when the buffer was edited this frame.
 */
bool drawCompactPathInput(const char* id, char* buf, size_t bufSize, float maxWidth,
                          ImGuiID& editingField, const char* hint = nullptr) {
  const ImGuiID fieldId = ImGui::GetID(id);
  const float framePadding = ImGui::GetStyle().FramePadding.x * 2.0f;
  const float charWidth = std::max(1.0f, ImGui::CalcTextSize("0").x);
  // Editing needs room to type, so the field opens to its full allowance; the
  // idle display shrinks to whatever the compacted path actually needs.
  if (editingField == fieldId) {
    ImGui::SetNextItemWidth(maxWidth);
    const bool changed = hint != nullptr
      ? ImGui::InputTextWithHint(id, hint, buf, bufSize)
      : ImGui::InputText(id, buf, bufSize);
    if (ImGui::IsItemDeactivated()) {
      editingField = 0;
    }
    return changed;
  }

  const size_t maxChars =
    static_cast<size_t>(std::max(4.0f, (maxWidth - framePadding) / charWidth));
  const std::string display = compactPathDisplay(buf, maxChars);
  // Tight to content: the frame is only as wide as the text it shows, floored
  // so an empty path still leaves a clickable target.
  const float contentWidth = std::min(maxWidth,
    std::max(ImGui::GetFontSize() * 6.0f, ImGui::CalcTextSize(display.c_str()).x + framePadding));
  ImGui::SetNextItemWidth(contentWidth);

  // A read-only InputText keeps the framed look of the editable state, and
  // activating it hands over to the editable branch on the next frame.
  std::vector<char> displayBuf(std::max<size_t>(display.size() + 1, 2), '\0');
  std::snprintf(displayBuf.data(), displayBuf.size(), "%s", display.c_str());
  ImGui::InputText(id, displayBuf.data(), displayBuf.size(), ImGuiInputTextFlags_ReadOnly);
  if (ImGui::IsItemActivated()) {
    editingField = fieldId;
  }
  if (ImGui::IsItemHovered() && display != buf) {
    ImGui::SetTooltip("%s", buf);
  }
  return false;
}

/**
 * @brief The keyboard/mouse shortcut list.
 *
 * Shared by the Help tab and the `H` overlay so the two can never drift apart;
 * adding a binding here shows up in both.
 */
void drawShortcutTable(const char* tableId) {
  const auto row = [](const char* keys, const char* desc) {
    ImGui::TableNextColumn();
    ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.35f, 1.0f), "%s", keys);
    ImGui::TableNextColumn();
    // Never wrapped: a binding split over two lines is harder to scan than a
    // table that is simply wider, so both columns size to their content and the
    // panel widens to fit.
    ImGui::TextUnformatted(desc);
  };
  // The theme's CellPadding.x is 1px, which leaves the key and action columns
  // almost touching. Padding applies to both sides of a cell, so this is half
  // the gap that ends up between them.
  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,
    ImVec2(std::round(ImGui::GetFontSize() * 0.5f), ImGui::GetStyle().CellPadding.y));
  if (!ImGui::BeginTable(tableId, 2,
        ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
    ImGui::PopStyleVar();
    return;
  }
  ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed);
  ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed);
  ImGui::TableHeadersRow();
  row("F", "Frame entire scene");
  row("C", "Frame selected object");
  row("R", "Reset camera");
  row("M", "Cycle measure tool: off -> ruler (2pt) -> angle (3pt) -> off");
  row("G", "Toggle pose grabber (drag XYZ axes on selected body)");
  row("Esc", "Cancel measure tool / exit fullscreen");
  row("F11", "Toggle fullscreen");
  row("F12", "Screenshot");
  // These rows describe Camera::update() and RayraiWindow::processMouseEvents();
  // keep them in step with those functions.
  row("WASD", "Move free camera (viewport focused; not while following a body)");
  row("Space", "Move free camera up");
  row("Left-drag", "Rotate camera (orbit when following a body, pan in orthographic view)");
  row("Scroll", "Dolly / zoom");
  row("Shift+Left-drag", "Apply mouse force to selected body");
#if defined(__APPLE__)
  row("Cmd+Left-drag", "Pull a body with the interaction wire (spring, not teleport)");
#else
  row("Ctrl+Left-drag", "Pull a body with the interaction wire (spring, not teleport)");
#endif
  ImGui::EndTable();
  ImGui::PopStyleVar();
}

/** What a file browser session is being used to choose. */
enum class FileBrowserMode {
  Folder,    /**< Pick an existing directory. */
  OpenFile,  /**< Pick an existing file. */
  SaveFile   /**< Name a file, which need not exist yet. */
};

/**
 * @brief State for the shared file/folder picker.
 *
 * Deliberately a plain std::filesystem browser rather than a native dialog: the
 * viewer ships on Linux, macOS, and Windows, and a native picker would mean
 * three backends plus a dependency. One instance is reused by every control
 * that needs a path, so they all behave and look the same.
 */
struct FileBrowserState {
  bool open = false;
  FileBrowserMode mode = FileBrowserMode::Folder;
  /** Window title, naming what is being chosen. */
  std::string title = "Select folder";
  /** Directory currently being listed. */
  std::filesystem::path current;

  struct Entry {
    std::filesystem::path path;
    bool isDirectory = false;
  };
  /** Contents of `current`: directories first, then files, each sorted. */
  std::vector<Entry> entries;

  /** Set when the listing could not be read, shown instead of an empty list. */
  std::string error;
  /** Editable directory field, so a path can be pasted instead of navigated to. */
  char pathBuf[512] = "";
  /** True while the breadcrumb is swapped for the editable path field. */
  bool editingPath = false;
  /** File name for SaveFile mode, or the highlighted file in OpenFile mode. */
  char nameBuf[256] = "";
  /** Lower-case extensions without the dot; empty accepts every file. */
  std::vector<std::string> extensions;
  /** Applied to the chosen path when the user accepts. */
  std::function<void(const std::filesystem::path&)> onAccept;
};

/** Lower-case extension of `path` without the leading dot. */
std::string lowerCaseFileExtension(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  if (!extension.empty() && extension.front() == '.') {
    extension.erase(extension.begin());
  }
  return toLowerAscii(extension);
}

/** True when `path`'s extension passes the browser's filter. */
bool fileBrowserAcceptsExtension(const FileBrowserState& browser,
                                 const std::filesystem::path& path) {
  if (browser.extensions.empty()) {
    return true;
  }
  const std::string extension = lowerCaseFileExtension(path);
  return std::find(browser.extensions.begin(), browser.extensions.end(), extension) !=
         browser.extensions.end();
}

/** Re-list `current`, sorting case-insensitively and reporting unreadable dirs. */
void refreshFileBrowser(FileBrowserState& browser) {
  browser.entries.clear();
  browser.error.clear();
  std::error_code ec;
  if (browser.current.empty() || !std::filesystem::is_directory(browser.current, ec)) {
    browser.error = "not a directory: " + browser.current.string();
    return;
  }
  std::filesystem::directory_iterator it(browser.current, ec);
  if (ec) {
    browser.error = "cannot read: " + ec.message();
    return;
  }
  // Iterate with the error_code increment rather than a range-for: the range
  // version uses the throwing increment, and a directory that turns unreadable
  // part-way through is an ordinary thing for a file browser to meet.
  const std::filesystem::directory_iterator end;
  while (it != end) {
    std::error_code entryEc;
    // is_directory follows symlinks, which is what a picker should do; an
    // unreadable entry is skipped rather than aborting the whole listing.
    const bool isDirectory = std::filesystem::is_directory(it->path(), entryEc) && !entryEc;
    if (isDirectory) {
      browser.entries.push_back({it->path(), true});
    } else if (browser.mode != FileBrowserMode::Folder &&
               fileBrowserAcceptsExtension(browser, it->path())) {
      browser.entries.push_back({it->path(), false});
    }
    it.increment(ec);
    if (ec) {
      browser.error = "listing stopped: " + ec.message();
      break;
    }
  }
  // Directories first so navigation targets are grouped at the top.
  std::sort(browser.entries.begin(), browser.entries.end(),
    [](const FileBrowserState::Entry& a, const FileBrowserState::Entry& b) {
      if (a.isDirectory != b.isDirectory) {
        return a.isDirectory;
      }
      return toLowerAscii(a.path.filename().string()) < toLowerAscii(b.path.filename().string());
    });
  std::snprintf(browser.pathBuf, sizeof(browser.pathBuf), "%s", browser.current.string().c_str());
}

/** Point the browser at `path`, falling back to the process's directory. */
void navigateFileBrowser(FileBrowserState& browser, const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::path target = path;
  if (target.empty() || !std::filesystem::is_directory(target, ec)) {
    target = std::filesystem::current_path(ec);
    if (ec) {
      target = std::filesystem::path("/");
    }
  }
  browser.current = std::filesystem::weakly_canonical(target, ec);
  if (ec || browser.current.empty()) {
    browser.current = target;
  }
  refreshFileBrowser(browser);
}

/**
 * @brief Start a browser session.
 *
 * @param startPath A directory to open, or a file whose directory is opened and
 *   whose name pre-fills the name field.
 * @param extensions Lower-case extensions without the dot; empty accepts all.
 */
void openFileBrowser(FileBrowserState& browser, FileBrowserMode mode, std::string title,
                     const std::filesystem::path& startPath,
                     std::vector<std::string> extensions,
                     std::function<void(const std::filesystem::path&)> onAccept) {
  browser.mode = mode;
  browser.title = std::move(title);
  browser.extensions = std::move(extensions);
  browser.onAccept = std::move(onAccept);
  browser.nameBuf[0] = '\0';

  std::error_code ec;
  std::filesystem::path startDirectory = startPath;
  if (!startPath.empty() && !std::filesystem::is_directory(startPath, ec)) {
    // A file path: browse its folder and carry the name over, which makes
    // "change this output file" a one-click operation.
    startDirectory = startPath.parent_path();
    const std::string name = startPath.filename().string();
    std::snprintf(browser.nameBuf, sizeof(browser.nameBuf), "%s", name.c_str());
  }
  navigateFileBrowser(browser, startDirectory);
  browser.open = true;
}


/**
 * @brief Whether a host string names this machine.
 *
 * Used to decide when the viewer may check a spawn geometry path itself: only a
 * loopback server is guaranteed to share this filesystem.
 */
bool isLoopbackHostName(const std::string& host) {
  const std::string trimmed = toLowerAscii(trimAscii(host));
  if (trimmed.empty()) {
    return false;
  }
  return trimmed == "localhost" || trimmed == "::1" || trimmed == "[::1]" ||
         trimmed.rfind("127.", 0) == 0;
}

bool allComponentsFinite(const float* values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

bool spawnExtensionAccepted(const SpawnShapeInfo& shape, const std::string& path) {
  const std::string accepted(shape.fileExtensions ? shape.fileExtensions : "");
  const std::string extension = lowerCaseFileExtension(path);
  if (accepted.empty()) {
    // RaisimServer only requires that a mesh path have *some* extension.
    return !extension.empty();
  }
  std::istringstream stream(accepted);
  std::string candidate;
  while (stream >> candidate) {
    if (candidate == extension) return true;
  }
  return false;
}

/**
 * @brief Translate the spawn palette into a wire request.
 *
 * Mirrors every check in RaisimServer::validateDecodedClientRequests(), because
 * a request the server rejects is a DeserializationError, and that drops the
 * whole connection rather than just the request.
 *
 * @return Empty string on success, otherwise why the request cannot be sent.
 */
std::string buildSpawnRequest(
    const SpawnFormState& form, raisin::tcp_viewer::ClientRequest& request) {
  const SpawnShapeInfo& shape = spawnShapeAt(form.shapeIndex);
  request = raisin::tcp_viewer::ClientRequest{};
  request.type = shape.type;
  request.name = trimAscii(std::string(form.name));
  request.appearance = trimAscii(std::string(form.appearance));
  request.file = trimAscii(std::string(form.file));
  request.mass = form.mass;
  request.bodyType = form.bodyType;

  if (form.bodyType < 0 || form.bodyType > 2) {
    return "invalid body type";
  }
  if (!allComponentsFinite(form.position, 3) || !allComponentsFinite(form.linearVelocity, 3) ||
      !allComponentsFinite(form.angularVelocity, 3) || !std::isfinite(form.mass)) {
    return "spawn fields must be finite numbers";
  }
  request.vec3a = glm::vec3(form.position[0], form.position[1], form.position[2]);
  request.linVel =
    glm::vec3(form.linearVelocity[0], form.linearVelocity[1], form.linearVelocity[2]);
  request.angVel =
    glm::vec3(form.angularVelocity[0], form.angularVelocity[1], form.angularVelocity[2]);

  const glm::vec4 quatWxyz(form.quatWxyz[0], form.quatWxyz[1], form.quatWxyz[2], form.quatWxyz[3]);
  const float quatNorm2 = glm::dot(quatWxyz, quatWxyz);
  if (!std::isfinite(quatNorm2) || quatNorm2 <= 1.0e-12f) {
    return "orientation quaternion is degenerate";
  }
  const float invQuatNorm = 1.0f / std::sqrt(quatNorm2);
  request.quat = quatWxyz * invQuatNorm;

  if (shape.needsMass && !(form.mass > 0.0f)) {
    return "mass must be positive";
  }
  if (shape.needsFile) {
    if (request.file.empty()) {
      return "geometry file is required";
    }
    if (!spawnExtensionAccepted(shape, request.file)) {
      const std::string accepted(shape.fileExtensions ? shape.fileExtensions : "");
      return accepted.empty() ? "mesh path needs a file extension"
                              : "file extension must be one of: " + accepted;
    }
  }

  switch (shape.type) {
    case raisin::tcp_viewer::ClientRequestType::CR_SPAWN_BOX:
      if (!(form.boxExtent[0] > 0.0f) || !(form.boxExtent[1] > 0.0f) ||
          !(form.boxExtent[2] > 0.0f)) {
        return "box dimensions must be positive";
      }
      request.size[0] = form.boxExtent[0];
      request.size[1] = form.boxExtent[1];
      request.size[2] = form.boxExtent[2];
      break;
    case raisin::tcp_viewer::ClientRequestType::CR_SPAWN_SPHERE:
      if (!(form.radius > 0.0f)) {
        return "radius must be positive";
      }
      request.size[0] = form.radius;
      break;
    case raisin::tcp_viewer::ClientRequestType::CR_SPAWN_CYLINDER:
      if (!(form.radius > 0.0f) || !(form.height > 0.0f)) {
        return "radius and height must be positive";
      }
      request.size[0] = form.radius;
      request.size[1] = form.height;
      break;
    case raisin::tcp_viewer::ClientRequestType::CR_SPAWN_CAPSULE:
      if (!(form.radius > 0.0f) || form.height < 0.0f) {
        return "radius must be positive and height nonnegative";
      }
      request.size[0] = form.radius;
      request.size[1] = form.height;
      break;
    case raisin::tcp_viewer::ClientRequestType::CR_SPAWN_PLANE:
      if (!std::isfinite(form.groundHeight)) {
        return "ground height must be finite";
      }
      request.size[0] = form.groundHeight;
      break;
    case raisin::tcp_viewer::ClientRequestType::CR_SPAWN_HEIGHT_MAP:
      if (!allComponentsFinite(form.heightMapCenter, 2) ||
          !allComponentsFinite(form.heightMapSize, 2) ||
          !std::isfinite(form.heightMapHeightScale) ||
          !std::isfinite(form.heightMapHeightOffset)) {
        return "height-map fields must be finite numbers";
      }
      if (!(form.heightMapSize[0] > 0.0f) || !(form.heightMapSize[1] > 0.0f)) {
        return "height-map x/y size must be positive";
      }
      request.size[0] = form.heightMapCenter[0];
      request.size[1] = form.heightMapCenter[1];
      request.size[2] = form.heightMapSize[0];
      request.size[3] = form.heightMapSize[1];
      request.size[4] = form.heightMapHeightScale;
      request.size[5] = form.heightMapHeightOffset;
      break;
    case raisin::tcp_viewer::ClientRequestType::CR_SPAWN_MESH:
    case raisin::tcp_viewer::ClientRequestType::CR_SPAWN_AS:
      // Mesh scale is not part of the spawn payload; the loader uses the file's units.
      break;
    default:
      return "unsupported spawn shape";
  }

  if (!allComponentsFinite(request.size.data(), request.size.size())) {
    return "spawn dimensions must be finite numbers";
  }
  return {};
}

/** `p` nudged one pixel down-right, where overlay drop shadows are drawn. */
ImVec2 shadowOffset(const ImVec2& p) {
  return ImVec2(p.x + 1.0f, p.y + 1.0f);
}

/** A line over its one-pixel drop shadow. */
void addShadowedLine(ImDrawList* drawList, const ImVec2& a, const ImVec2& b, ImU32 shadow,
                     float shadowThickness, ImU32 color, float thickness) {
  drawList->AddLine(shadowOffset(a), shadowOffset(b), shadow, shadowThickness);
  drawList->AddLine(a, b, color, thickness);
}

/** Text over its one-pixel drop shadow. */
void addShadowedText(ImDrawList* drawList, const ImVec2& pos, ImU32 shadow, ImU32 color,
                     const char* text) {
  drawList->AddText(shadowOffset(pos), shadow, text);
  drawList->AddText(pos, color, text);
}

void drawMouseForcePreview(
  const MouseForceGesture& gesture, const ViewerViewportState& viewport, const raisin::Camera& camera) {
  if (!gesture.active) {
    return;
  }
  ImVec2 origin = gesture.pressMouse;
  projectWorldToViewport(camera, viewport, gesture.applicationPoint, origin);
  const ImVec2 drag(gesture.currentMouse.x - gesture.pressMouse.x,
                    gesture.currentMouse.y - gesture.pressMouse.y);
  const ImVec2 tip(origin.x + drag.x, origin.y + drag.y);
  const float len = std::sqrt(drag.x * drag.x + drag.y * drag.y);

  ImDrawList* drawList = ImGui::GetForegroundDrawList();
  const ImU32 shadow = IM_COL32(5, 8, 12, 190);
  const ImU32 lineColor = IM_COL32(255, 210, 74, 255);
  const ImU32 fillColor = IM_COL32(255, 168, 48, 245);
  drawList->AddCircleFilled(origin, 5.0f, shadow, 24);
  drawList->AddCircleFilled(origin, 3.5f, lineColor, 24);
  if (len >= 2.0f) {
    addShadowedLine(drawList, origin, tip, shadow, 5.0f, lineColor, 3.0f);
    const ImVec2 dir(drag.x / len, drag.y / len);
    const ImVec2 normal(-dir.y, dir.x);
    const float headLen = std::min(22.0f, std::max(10.0f, len * 0.24f));
    const float headHalfWidth = headLen * 0.42f;
    const ImVec2 base(tip.x - dir.x * headLen, tip.y - dir.y * headLen);
    const ImVec2 p1(tip.x, tip.y);
    const ImVec2 p2(base.x + normal.x * headHalfWidth, base.y + normal.y * headHalfWidth);
    const ImVec2 p3(base.x - normal.x * headHalfWidth, base.y - normal.y * headHalfWidth);
    drawList->AddTriangleFilled(shadowOffset(p1), shadowOffset(p2), shadowOffset(p3), shadow);
    drawList->AddTriangleFilled(p1, p2, p3, fillColor);
  }

  char label[96];
  std::snprintf(label, sizeof(label), "%.1f N", glm::length(gesture.force));
  const ImVec2 labelPos(tip.x + 10.0f, tip.y - ImGui::GetFontSize() * 0.5f);
  addShadowedText(drawList, labelPos, shadow, lineColor, label);
}

void drawWireDragPreview(
  const WireDragGesture& gesture, const ViewerViewportState& viewport,
  const raisin::Camera& camera) {
  if (!gesture.active) {
    return;
  }
  ImVec2 anchor;
  ImVec2 target;
  if (!projectWorldToViewport(camera, viewport, gesture.attachPoint, anchor) ||
      !projectWorldToViewport(camera, viewport, gesture.target, target)) {
    return;
  }

  ImDrawList* drawList = ImGui::GetForegroundDrawList();
  const ImU32 shadow = IM_COL32(5, 8, 12, 190);
  const ImU32 wireColor = IM_COL32(120, 226, 255, 255);
  addShadowedLine(drawList, anchor, target, shadow, 4.0f, wireColor, 2.0f);
  drawList->AddCircleFilled(anchor, 5.0f, shadow, 24);
  drawList->AddCircleFilled(anchor, 3.5f, wireColor, 24);
  drawList->AddCircle(target, 7.0f, shadow, 24, 4.0f);
  drawList->AddCircle(target, 6.0f, wireColor, 24, 2.0f);

  char label[96];
  std::snprintf(label, sizeof(label), "wire %.3f m",
    glm::distance(gesture.attachPoint, gesture.target));
  const ImVec2 labelPos(target.x + 10.0f, target.y - ImGui::GetFontSize() * 0.5f);
  addShadowedText(drawList, labelPos, shadow, wireColor, label);
}

void drawRulerOverlay(
  const RulerToolState& ruler, const ViewerViewportState& viewport, const raisin::Camera& camera) {
  if (!ruler.hasA && !ruler.hasB) {
    return;
  }

  ImVec2 screenA;
  ImVec2 screenB;
  const bool visibleA = ruler.hasA && projectWorldToViewport(camera, viewport, ruler.a, screenA);
  const bool visibleB = ruler.hasB && projectWorldToViewport(camera, viewport, ruler.b, screenB);
  if (!visibleA && !visibleB) {
    return;
  }

  ImDrawList* drawList = ImGui::GetForegroundDrawList();
  const ImU32 shadow = IM_COL32(4, 7, 11, 205);
  const ImU32 lineColor = IM_COL32(74, 214, 255, 255);
  const ImU32 pointAColor = IM_COL32(255, 214, 84, 255);
  const ImU32 pointBColor = IM_COL32(120, 255, 166, 255);
  const auto drawEndpoint = [&](const ImVec2& pos, const char* label, ImU32 color) {
    drawList->AddCircleFilled(shadowOffset(pos), 6.0f, shadow, 28);
    drawList->AddCircleFilled(pos, 4.0f, color, 28);
    addShadowedText(drawList, ImVec2(pos.x + 8.0f, pos.y - ImGui::GetFontSize() * 0.5f), shadow,
      color, label);
  };

  if (visibleA && visibleB) {
    const ImVec2 delta(screenB.x - screenA.x, screenB.y - screenA.y);
    const float len = std::max(1.0f, std::sqrt(delta.x * delta.x + delta.y * delta.y));
    const ImVec2 dir(delta.x / len, delta.y / len);
    const ImVec2 normal(-dir.y, dir.x);
    const float tickHalf = 8.0f;
    addShadowedLine(drawList, screenA, screenB, shadow, 5.0f, lineColor, 2.5f);
    drawList->AddLine(ImVec2(screenA.x - normal.x * tickHalf, screenA.y - normal.y * tickHalf),
      ImVec2(screenA.x + normal.x * tickHalf, screenA.y + normal.y * tickHalf), lineColor, 2.5f);
    drawList->AddLine(ImVec2(screenB.x - normal.x * tickHalf, screenB.y - normal.y * tickHalf),
      ImVec2(screenB.x + normal.x * tickHalf, screenB.y + normal.y * tickHalf), lineColor, 2.5f);

    const std::string label = formatRulerDistance(glm::distance(ruler.a, ruler.b));
    const ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
    const ImVec2 mid((screenA.x + screenB.x) * 0.5f, (screenA.y + screenB.y) * 0.5f);
    const ImVec2 labelPos(mid.x - textSize.x * 0.5f, mid.y - textSize.y - 12.0f);
    const ImVec2 pad(7.0f, 4.0f);
    drawList->AddRectFilled(ImVec2(labelPos.x - pad.x, labelPos.y - pad.y),
      ImVec2(labelPos.x + textSize.x + pad.x, labelPos.y + textSize.y + pad.y),
      IM_COL32(8, 13, 20, 220), 5.0f);
    drawList->AddRect(ImVec2(labelPos.x - pad.x, labelPos.y - pad.y),
      ImVec2(labelPos.x + textSize.x + pad.x, labelPos.y + textSize.y + pad.y),
      IM_COL32(74, 214, 255, 180), 5.0f);
    drawList->AddText(labelPos, lineColor, label.c_str());
  }

  if (visibleA) drawEndpoint(screenA, "A", pointAColor);
  if (visibleB) drawEndpoint(screenB, "B", pointBColor);
}

// World-space axis arrows + rotation rings projected to screen at the given
// world position. `origin` is the screen-space body center.
struct GizmoScreenLayout {
  static constexpr int kRingSamples = 32;
  ImVec2 origin{0.0f, 0.0f};
  ImVec2 axisTip[3]{};                          // translate-arrow tips, X/Y/Z
  ImVec2 ringSamples[3][kRingSamples]{};        // rotation-ring screen samples
  bool ringPointVisible[3][kRingSamples]{};
  bool visible = false;
  float worldScale = 1.0f;                       // metres of one axis arrow
};

// Compute the screen-space layout of a world-axis-aligned gizmo at the given
// world position. The handles always point along world X/Y/Z regardless of
// the body's orientation — translate/rotate then act in the world frame, and
// the resulting delta is composed with the body's held pose by the drag
// handler.
GizmoScreenLayout computeGizmoLayout(
  const raisin::Camera& camera, const ViewerViewportState& viewport, const glm::vec3& world) {
  GizmoScreenLayout L;
  if (!projectWorldToViewport(camera, viewport, world, L.origin)) return L;

  // Pick an axis length that occupies a consistent pixel span at the body's
  // depth — about 90 px.
  const glm::vec3 probeAxis(1.0f, 0.0f, 0.0f);
  ImVec2 probeScreen;
  float pixelsPerMetre = 0.0f;
  if (projectWorldToViewport(camera, viewport, world + probeAxis, probeScreen)) {
    const float dx = probeScreen.x - L.origin.x;
    const float dy = probeScreen.y - L.origin.y;
    pixelsPerMetre = std::sqrt(dx * dx + dy * dy);
  }
  const float targetPixels = 90.0f;
  const float worldLen = pixelsPerMetre > 1e-4f
    ? std::clamp(targetPixels / pixelsPerMetre, 0.05f, 5.0f)
    : 0.5f;
  L.worldScale = worldLen;

  const glm::vec3 axes[3] = {
    glm::vec3(worldLen, 0.0f, 0.0f),
    glm::vec3(0.0f, worldLen, 0.0f),
    glm::vec3(0.0f, 0.0f, worldLen),
  };
  for (int i = 0; i < 3; ++i) {
    if (!projectWorldToViewport(camera, viewport, world + axes[i], L.axisTip[i])) {
      L.axisTip[i] = L.origin;
    }
  }

  // Rotation rings: a circle in the world plane perpendicular to each world
  // axis, radius slightly larger than the translate arrows so handles don't
  // overlap visually.
  const float ringRadius = worldLen * 1.05f;
  for (int a = 0; a < 3; ++a) {
    glm::vec3 u(0.0f), v(0.0f);
    if (a == 0)      { u = glm::vec3(0.0f, 1.0f, 0.0f); v = glm::vec3(0.0f, 0.0f, 1.0f); }
    else if (a == 1) { u = glm::vec3(1.0f, 0.0f, 0.0f); v = glm::vec3(0.0f, 0.0f, 1.0f); }
    else             { u = glm::vec3(1.0f, 0.0f, 0.0f); v = glm::vec3(0.0f, 1.0f, 0.0f); }
    for (int s = 0; s < GizmoScreenLayout::kRingSamples; ++s) {
      const float t = static_cast<float>(s) /
                      static_cast<float>(GizmoScreenLayout::kRingSamples) * 6.2831853f;
      const glm::vec3 worldPt = world + (u * std::cos(t) + v * std::sin(t)) * ringRadius;
      ImVec2 sc;
      L.ringPointVisible[a][s] = projectWorldToViewport(camera, viewport, worldPt, sc);
      L.ringSamples[a][s] = sc;
    }
  }
  L.visible = true;
  return L;
}

// Distance from point p to segment (a, b) in screen space.
float pointSegmentDistance(const ImVec2& p, const ImVec2& a, const ImVec2& b) {
  const float dx = b.x - a.x;
  const float dy = b.y - a.y;
  const float l2 = dx * dx + dy * dy;
  if (l2 < 1e-3f) {
    const float ex = p.x - a.x;
    const float ey = p.y - a.y;
    return std::sqrt(ex * ex + ey * ey);
  }
  const float t = std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / l2, 0.0f, 1.0f);
  const float cx = a.x + dx * t;
  const float cy = a.y + dy * t;
  const float ex = p.x - cx;
  const float ey = p.y - cy;
  return std::sqrt(ex * ex + ey * ey);
}

// Returns the picked handle: mode (Translate/Rotate) and axis (0,1,2), or
// (Translate, -1) for "nothing hit". Picks translate over rotate when both are
// equidistant — translate arrows render in front of rotation rings.
struct PoseGrabberHit {
  PoseGrabberGesture::Mode mode = PoseGrabberGesture::Mode::Translate;
  int axis = -1;
};

PoseGrabberHit pickPoseGrabberHandle(const GizmoScreenLayout& L, const ImVec2& mouse) {
  PoseGrabberHit hit;
  if (!L.visible) return hit;
  constexpr float kHitRadius = 9.0f;
  float bestDist = kHitRadius;

  // Translate arrows.
  for (int i = 0; i < 3; ++i) {
    const float d = pointSegmentDistance(mouse, L.origin, L.axisTip[i]);
    if (d < bestDist) {
      bestDist = d;
      hit.mode = PoseGrabberGesture::Mode::Translate;
      hit.axis = i;
    }
  }
  // Rotation rings: min distance to any consecutive sample segment.
  for (int a = 0; a < 3; ++a) {
    for (int s = 0; s < GizmoScreenLayout::kRingSamples; ++s) {
      const int sNext = (s + 1) % GizmoScreenLayout::kRingSamples;
      if (!L.ringPointVisible[a][s] || !L.ringPointVisible[a][sNext]) continue;
      const float d = pointSegmentDistance(mouse, L.ringSamples[a][s], L.ringSamples[a][sNext]);
      if (d < bestDist) {
        bestDist = d;
        hit.mode = PoseGrabberGesture::Mode::Rotate;
        hit.axis = a;
      }
    }
  }
  return hit;
}

void drawPoseGrabberOverlay(const GizmoScreenLayout& L,
                            PoseGrabberGesture::Mode activeMode, int activeAxis,
                            PoseGrabberGesture::Mode hoverMode, int hoverAxis) {
  if (!L.visible) return;
  ImDrawList* drawList = ImGui::GetForegroundDrawList();
  const ImU32 shadow = IM_COL32(0, 0, 0, 180);
  const ImU32 axisColors[3] = {
    IM_COL32(255, 90, 90, 255),    // X = red
    IM_COL32(110, 220, 110, 255),  // Y = green
    IM_COL32(110, 160, 255, 255),  // Z = blue
  };
  const ImU32 hot = IM_COL32(255, 230, 60, 255);
  const char* labels[3] = {"X", "Y", "Z"};

  // Rotation rings first (drawn behind translate arrows). Use the axis color
  // at lower alpha, full alpha when hot.
  for (int a = 0; a < 3; ++a) {
    const bool isHot =
      (activeAxis >= 0 && activeMode == PoseGrabberGesture::Mode::Rotate && activeAxis == a) ||
      (activeAxis < 0  && hoverMode == PoseGrabberGesture::Mode::Rotate && hoverAxis == a);
    const ImU32 col = isHot ? hot : ((axisColors[a] & 0x00FFFFFF) | (160u << 24));
    const float thickness = isHot ? 3.0f : 1.6f;
    for (int s = 0; s < GizmoScreenLayout::kRingSamples; ++s) {
      const int sNext = (s + 1) % GizmoScreenLayout::kRingSamples;
      if (!L.ringPointVisible[a][s] || !L.ringPointVisible[a][sNext]) continue;
      drawList->AddLine(L.ringSamples[a][s], L.ringSamples[a][sNext], col, thickness);
    }
  }

  // Translate arrows on top.
  for (int i = 0; i < 3; ++i) {
    const bool isHot =
      (activeAxis >= 0 && activeMode == PoseGrabberGesture::Mode::Translate && activeAxis == i) ||
      (activeAxis < 0  && hoverMode == PoseGrabberGesture::Mode::Translate && hoverAxis == i);
    const ImU32 col = isHot ? hot : axisColors[i];
    addShadowedLine(drawList, L.origin, L.axisTip[i], shadow, 4.0f, col, isHot ? 3.5f : 2.5f);
    drawList->AddCircleFilled(L.axisTip[i], isHot ? 6.0f : 4.5f, col, 24);
    drawList->AddText(ImVec2(L.axisTip[i].x + 6.0f, L.axisTip[i].y - 6.0f), col, labels[i]);
  }
  drawList->AddCircleFilled(L.origin, 4.0f, shadow, 24);
  drawList->AddCircleFilled(L.origin, 2.5f, IM_COL32(240, 240, 240, 255), 24);
}

void drawAngleOverlay(
  const AngleToolState& angle, const ViewerViewportState& viewport, const raisin::Camera& camera) {
  if (angle.picked == 0) return;

  ImVec2 sA, sB, sC;
  const bool visA = angle.picked >= 1 && projectWorldToViewport(camera, viewport, angle.a, sA);
  const bool visB = angle.picked >= 2 && projectWorldToViewport(camera, viewport, angle.b, sB);
  const bool visC = angle.picked >= 3 && projectWorldToViewport(camera, viewport, angle.c, sC);
  if (!visA && !visB && !visC) return;

  ImDrawList* drawList = ImGui::GetForegroundDrawList();
  const ImU32 shadow = IM_COL32(4, 7, 11, 205);
  const ImU32 armColor = IM_COL32(255, 168, 84, 255);
  const ImU32 arcColor = IM_COL32(120, 230, 120, 255);
  const ImU32 ptColor[3] = {
    IM_COL32(255, 214, 84, 255),
    IM_COL32(120, 230, 120, 255),
    IM_COL32(74, 214, 255, 255)};
  const char* lbls[3] = {"A", "B (vertex)", "C"};

  if (visA && visB) addShadowedLine(drawList, sA, sB, shadow, 4.0f, armColor, 2.0f);
  if (visB && visC) addShadowedLine(drawList, sB, sC, shadow, 4.0f, armColor, 2.0f);

  if (angle.picked >= 3 && visA && visB && visC) {
    const ImVec2 ba(sA.x - sB.x, sA.y - sB.y);
    const ImVec2 bc(sC.x - sB.x, sC.y - sB.y);
    const float lba = std::sqrt(ba.x * ba.x + ba.y * ba.y);
    const float lbc = std::sqrt(bc.x * bc.x + bc.y * bc.y);
    if (lba > 1.0f && lbc > 1.0f) {
      const float aA = std::atan2(ba.y, ba.x);
      const float aC = std::atan2(bc.y, bc.x);
      float d = aC - aA;
      while (d > 3.14159265f) d -= 6.2831853f;
      while (d < -3.14159265f) d += 6.2831853f;
      const float arcR = std::min(lba, lbc) * 0.30f;
      const int steps = 28;
      ImVec2 prev(sB.x + std::cos(aA) * arcR, sB.y + std::sin(aA) * arcR);
      for (int i = 1; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        const float ang = aA + d * t;
        const ImVec2 cur(sB.x + std::cos(ang) * arcR, sB.y + std::sin(ang) * arcR);
        drawList->AddLine(prev, cur, arcColor, 2.0f);
        prev = cur;
      }
      const float angRad = computeAngleRadians(angle.a, angle.b, angle.c);
      const std::string label = formatAngleDegrees(angRad);
      const ImVec2 tsz = ImGui::CalcTextSize(label.c_str());
      const float midA = aA + d * 0.5f;
      const ImVec2 labelPos(sB.x + std::cos(midA) * (arcR + 14.0f) - tsz.x * 0.5f,
                            sB.y + std::sin(midA) * (arcR + 14.0f) - tsz.y * 0.5f);
      const ImVec2 pad(6.0f, 3.0f);
      drawList->AddRectFilled(ImVec2(labelPos.x - pad.x, labelPos.y - pad.y),
        ImVec2(labelPos.x + tsz.x + pad.x, labelPos.y + tsz.y + pad.y),
        IM_COL32(8, 13, 20, 220), 4.0f);
      drawList->AddText(labelPos, arcColor, label.c_str());
    }
  }

  const auto drawPoint = [&](const ImVec2& pos, const char* label, ImU32 color) {
    drawList->AddCircleFilled(shadowOffset(pos), 6.0f, shadow, 24);
    drawList->AddCircleFilled(pos, 4.0f, color, 24);
    addShadowedText(drawList, ImVec2(pos.x + 8.0f, pos.y - ImGui::GetFontSize() * 0.5f), shadow,
      color, label);
  };
  if (visA) drawPoint(sA, lbls[0], ptColor[0]);
  if (visB) drawPoint(sB, lbls[1], ptColor[1]);
  if (visC) drawPoint(sC, lbls[2], ptColor[2]);
}

void drawRulerCursorIcon(const RulerToolState& ruler, const ViewerViewportState& viewport) {
  if (!ruler.enabled) {
    return;
  }

  const ImGuiIO& io = ImGui::GetIO();
  if (!std::isfinite(io.MousePos.x) || !std::isfinite(io.MousePos.y)) {
    return;
  }

  ImDrawList* drawList = ImGui::GetForegroundDrawList();
  const bool active = viewport.hovered;
  const ImU32 shadow = IM_COL32(4, 7, 11, active ? 190 : 120);
  const ImU32 lineColor = active ? IM_COL32(74, 214, 255, 245) : IM_COL32(166, 180, 193, 170);
  const ImU32 tickColor = active ? IM_COL32(255, 216, 92, 255) : IM_COL32(205, 214, 224, 170);
  const ImVec2 base(io.MousePos.x + 15.0f, io.MousePos.y + 18.0f);
  const ImVec2 end(base.x + 28.0f, base.y - 10.0f);
  const ImVec2 delta(end.x - base.x, end.y - base.y);
  const float len = std::max(1.0f, std::sqrt(delta.x * delta.x + delta.y * delta.y));
  const ImVec2 dir(delta.x / len, delta.y / len);
  const ImVec2 normal(-dir.y, dir.x);

  addShadowedLine(drawList, base, end, shadow, 4.5f, lineColor, 2.5f);
  for (int i = 0; i <= 4; ++i) {
    const float t = static_cast<float>(i) / 4.0f;
    const float tickHalf = (i == 0 || i == 4) ? 6.0f : 4.0f;
    const ImVec2 center(base.x + delta.x * t, base.y + delta.y * t);
    const ImVec2 a(center.x - normal.x * tickHalf, center.y - normal.y * tickHalf);
    const ImVec2 b(center.x + normal.x * tickHalf, center.y + normal.y * tickHalf);
    addShadowedLine(drawList, a, b, shadow, 3.0f, tickColor, 1.7f);
  }

  const char* label = nextRulerPointLabel(ruler);
  const ImVec2 labelPos(end.x + 5.0f, end.y - ImGui::GetFontSize() * 0.5f);
  addShadowedText(drawList, labelPos, shadow, tickColor, label);
}

void drawAngleCursorIcon(const AngleToolState& angle, const ViewerViewportState& viewport) {
  if (!angle.enabled) {
    return;
  }

  const ImGuiIO& io = ImGui::GetIO();
  if (!std::isfinite(io.MousePos.x) || !std::isfinite(io.MousePos.y)) {
    return;
  }

  ImDrawList* drawList = ImGui::GetForegroundDrawList();
  const bool active = viewport.hovered;
  const ImU32 shadow = IM_COL32(4, 7, 11, active ? 190 : 120);
  const ImU32 arcColor = active ? IM_COL32(120, 230, 120, 245) : IM_COL32(166, 180, 193, 170);
  const ImU32 armColor = active ? IM_COL32(255, 168, 84, 255) : IM_COL32(205, 214, 224, 170);

  // Small protractor: flat base + half-circle arc opening upward, drawn to the
  // lower-right of the cursor so it doesn't occlude the pick point.
  const ImVec2 c(io.MousePos.x + 26.0f, io.MousePos.y + 22.0f);   // arc center
  const float r = 12.0f;

  // Flat baseline (the protractor's straight edge), pi to 0 in screen coords.
  addShadowedLine(drawList, ImVec2(c.x - r, c.y), ImVec2(c.x + r, c.y), shadow, 3.0f, arcColor, 1.7f);

  // Upper half-arc (semicircle).
  constexpr int kSteps = 22;
  ImVec2 prev(c.x - r, c.y);
  for (int i = 1; i <= kSteps; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(kSteps);
    const float a = 3.14159265f - 3.14159265f * t;   // pi → 0
    const ImVec2 cur(c.x + std::cos(a) * r, c.y - std::sin(a) * r);
    addShadowedLine(drawList, prev, cur, shadow, 3.0f, arcColor, 1.7f);
    prev = cur;
  }

  // Tick marks at 0°, 45°, 90°, 135°, 180°.
  for (int k = 0; k <= 4; ++k) {
    const float a = 3.14159265f - 3.14159265f * (static_cast<float>(k) / 4.0f);
    const float tickInner = (k == 0 || k == 4 || k == 2) ? r - 4.5f : r - 3.0f;
    const ImVec2 inner(c.x + std::cos(a) * tickInner, c.y - std::sin(a) * tickInner);
    const ImVec2 outer(c.x + std::cos(a) * r,        c.y - std::sin(a) * r);
    drawList->AddLine(inner, outer, armColor, 1.5f);
  }

  // Central pivot dot.
  drawList->AddCircleFilled(shadowOffset(c), 2.5f, shadow, 12);
  drawList->AddCircleFilled(c, 1.8f, armColor, 12);

  // Label which point comes next.
  const char* label = nextAnglePointLabel(angle);
  const ImVec2 labelPos(c.x + r + 4.0f, c.y - ImGui::GetFontSize() * 0.5f);
  addShadowedText(drawList, labelPos, shadow, armColor, label);
}

float quaternionAngularSpeed(const glm::vec4& previous, const glm::vec4& current, double dt) {
  if (dt <= 1e-9) {
    return 0.0f;
  }
  const float dotValue = std::abs(glm::dot(normalizedQuatFromWxyz(previous),
                                           normalizedQuatFromWxyz(current)));
  const float angle = 2.0f * std::acos(std::clamp(dotValue, 0.0f, 1.0f));
  return angle / static_cast<float>(dt);
}

uint64_t visualMotionKey(uint32_t tag, int index) {
  return (static_cast<uint64_t>(tag) << 32u) | static_cast<uint32_t>(index);
}

void updateMotionEstimate(MotionEstimate& estimate, const VisualEntry& entry, double timeSeconds) {
  if (estimate.hasPrevious) {
    const double dt = timeSeconds - estimate.previousTime;
    if (dt > 1e-9 && dt < 10.0) {
      estimate.linearVelocity = (entry.lastPos - estimate.previousPosition) / static_cast<float>(dt);
      estimate.angularSpeed = quaternionAngularSpeed(estimate.previousQuat, entry.lastQuat, dt);
      estimate.valid = true;
    }
  }
  estimate.previousTime = timeSeconds;
  estimate.previousPosition = entry.lastPos;
  estimate.previousQuat = entry.lastQuat;
  estimate.hasPrevious = true;
}

void updateStatsWindow(ViewerStats& stats, std::chrono::steady_clock::time_point now) {
  const double elapsed = std::chrono::duration<double>(now - stats.windowStart).count();
  if (elapsed < 1.0) {
    return;
  }
  stats.fps = static_cast<double>(stats.frames) / elapsed;
  stats.updateHz = static_cast<double>(stats.updates) / elapsed;
  stats.rxKbps = static_cast<double>(stats.bytes) / 1024.0 / elapsed;
  stats.frames = 0;
  stats.updates = 0;
  stats.bytes = 0;
  stats.windowStart = now;
}

void configureViewerFillLights(raisin::RayraiWindow& viewer, bool enabled) {
  viewer.clearAdditionalLights();
  if (!enabled) {
    return;
  }

  raisin::RayraiWindow::AdditionalLight fillLight;
  fillLight.type = raisin::LightType::DIRECTIONAL;
  fillLight.direction = glm::normalize(glm::vec3(0.45f, -0.25f, -0.85f));
  fillLight.diffuse = glm::vec3(0.14f, 0.16f, 0.20f);
  fillLight.specular = glm::vec3(0.03f);
  viewer.addAdditionalLight(fillLight);

  raisin::RayraiWindow::AdditionalLight rimLight;
  rimLight.type = raisin::LightType::POINT;
  rimLight.position = glm::vec3(-2.0f, -3.0f, 3.0f);
  rimLight.diffuse = glm::vec3(0.16f, 0.10f, 0.05f);
  rimLight.specular = glm::vec3(0.04f);
  rimLight.linear = 0.08f;
  rimLight.quadratic = 0.025f;
  viewer.addAdditionalLight(rimLight);
}

void copyWeatherSettingsToViewerSettings(
  ViewerSettings& settings, const raisin::RayraiWindow::WeatherSettings& weather) {
  settings.skyWeatherPreset = static_cast<int>(weather.preset);
  settings.skyWeatherQuality = static_cast<int>(weather.quality);
  settings.skyWeatherSeed = static_cast<int>(std::max<uint32_t>(weather.seed, 1u));
  settings.skyTimeOfDayHours = weather.timeOfDayHours;
  settings.skyLatitude = weather.latitude;
  settings.skyLongitude = weather.longitude;
  settings.skyAutomaticUtcOffset = weather.automaticUtcOffset;
  settings.skyUtcOffsetHours = weather.utcOffsetHours;
  settings.skyYear = weather.year;
  settings.skyMonth = weather.month;
  settings.skyDay = weather.day;
  settings.skyWindDirectionDeg = windDirectionDegreesFromVector(weather.windDirection);
  settings.skyWindSpeed = weather.windSpeed;
  settings.skyCloudCoverage = weather.cloudCoverage;
  settings.skyCloudDensity = weather.cloudDensity;
  settings.skyCloudAltitudeMeters = weather.cloudAltitudeMeters;
  settings.skyCloudThicknessMeters = weather.cloudThicknessMeters;
  settings.skyCloudShadowStrength = weather.cloudShadowStrength;
  settings.skyCloudScale = weather.cloudScale;
  settings.skyCloudAnimationSpeed = weather.cloudAnimationSpeed;
  settings.skyPrecipitationRate = weather.precipitationRate;
  settings.skyRainOcclusionStrength = weather.rainOcclusionStrength;
  settings.skySnowCoverage = weather.snowCoverage;
  settings.skyHumidity = weather.humidity;
  settings.skyWetness = weather.wetness;
  settings.skyWetnessAccumulationEnabled = weather.wetnessAccumulationEnabled &&
    (weather.precipitationRate > 0.0f || weather.wetness > 0.0f || weather.snowCoverage > 0.0f);
  settings.skyWetnessAccumulationRate = weather.wetnessAccumulationRate;
  settings.skyWetnessDryingRate = weather.wetnessDryingRate;
  settings.skyLightningRate = weather.lightningRate;
  settings.skyFogDensity = weather.fogDensity;
  settings.skyVisibilityMeters = weather.visibilityMeters;
  settings.skyFogColor = weather.fogColor;
  settings.skyFogAnisotropy = weather.fogAnisotropy;
  settings.skyAirTurbidity = weather.airTurbidity;
  settings.skyGroundAlbedo = weather.groundAlbedo;
  settings.skyUseExplicitSunAngles = weather.useExplicitSunAngles;
  settings.skySunAzimuthDeg = weather.sunAzimuthDegrees;
  settings.skySunElevationDeg = weather.sunElevationDegrees;
  settings.skySunSize = weather.sunDiskSize;
  settings.skyMoonSize = weather.moonDiskSize;
  settings.skyLensDropletsEnabled = weather.lensDropletsEnabled;
  settings.skyLensDropletStrength = weather.lensDropletStrength;
}

void copyWeatherPresetToSettings(ViewerSettings& settings, int preset) {
  const int clampedPreset = std::clamp(preset, 0, 10);
  settings.skyWeatherPreset = clampedPreset;
  if (clampedPreset == 10) {
    return;
  }
  const bool skyEnabled = settings.skyEnabled;
  const bool weatherEnabled = settings.skyWeatherEnabled;
  const bool automaticUtcOffset = settings.skyAutomaticUtcOffset;
  const float utcOffsetHours = settings.skyUtcOffsetHours;
  copyWeatherSettingsToViewerSettings(
    settings, raisin::RayraiWindow::defaultWeatherSettings(weatherPresetFromIndex(clampedPreset)));
  settings.skyEnabled = skyEnabled;
  settings.skyWeatherEnabled = weatherEnabled;
  settings.skyAutomaticUtcOffset = automaticUtcOffset;
  settings.skyUtcOffsetHours = utcOffsetHours;
  settings.skyWeatherPreset = clampedPreset;
}

raisin::RayraiWindow::WeatherSettings weatherSettingsFromViewerSettings(
  const ViewerSettings& settings) {
  auto weather = raisin::RayraiWindow::defaultWeatherSettings(weatherPresetFromIndex(settings.skyWeatherPreset));
  weather.enabled = settings.skyEnabled && settings.skyWeatherEnabled &&
    weatherDefaultEnabledForQuality(settings.renderQuality);
  weather.preset = weatherPresetFromIndex(settings.skyWeatherPreset);
  weather.quality = weatherQualityFromIndex(settings.skyWeatherQuality);
  weather.seed = static_cast<uint32_t>(std::max(settings.skyWeatherSeed, 1));
  weather.timeOfDayHours = settings.skyTimeOfDayHours;
  weather.latitude = settings.skyLatitude;
  weather.longitude = settings.skyLongitude;
  weather.automaticUtcOffset = settings.skyAutomaticUtcOffset;
  weather.utcOffsetHours = settings.skyUtcOffsetHours;
  weather.year = settings.skyYear;
  weather.month = settings.skyMonth;
  weather.day = settings.skyDay;
  weather.windDirection = windDirectionFromDegrees(settings.skyWindDirectionDeg);
  weather.windSpeed = settings.skyWindSpeed;
  weather.transitionSeconds = 0.0f;
  weather.affectSensors = false;
  // When the user picks a non-Custom weather preset, the preset's own
  // cloud coverage / density / altitude / thickness should win — otherwise an
  // old persisted ViewerSettings (with skyCloudCoverage = 0.05 from before
  // clouds were a thing) silently overrides "Rain" with "almost no clouds".
  const bool useCustomCloudAuthoring = settings.skyWeatherPreset == 10;
  if (useCustomCloudAuthoring) {
    weather.cloudCoverage = settings.skyCloudCoverage;
    weather.cloudDensity = settings.skyCloudDensity;
  }
  // else: weather already holds the preset defaults from defaultWeatherSettings(preset)
  weather.cloudAltitudeMeters = settings.skyCloudAltitudeMeters;
  weather.cloudThicknessMeters = settings.skyCloudThicknessMeters;
  weather.cloudShadowStrength = settings.skyCloudShadowStrength;
  weather.cloudScale = settings.skyCloudScale;
  weather.cloudAnimationSpeed = settings.skyCloudAnimationSpeed;
  weather.precipitationRate = settings.skyPrecipitationRate;
  weather.rainOcclusionStrength = settings.skyRainOcclusionStrength;
  weather.snowCoverage = settings.skySnowCoverage;
  weather.humidity = settings.skyHumidity;
  weather.wetness = settings.skyWetness;
  weather.wetnessAccumulationEnabled = settings.skyWetnessAccumulationEnabled;
  weather.wetnessAccumulationRate = settings.skyWetnessAccumulationRate;
  weather.wetnessDryingRate = settings.skyWetnessDryingRate;
  weather.lightningRate = settings.skyLightningRate;
  weather.fogDensity = settings.skyFogDensity;
  weather.visibilityMeters = settings.skyVisibilityMeters;
  weather.fogColor = settings.skyFogColor;
  weather.fogAnisotropy = settings.skyFogAnisotropy;
  weather.airTurbidity = settings.skyAirTurbidity;
  weather.groundAlbedo = settings.skyGroundAlbedo;
  weather.useExplicitSunAngles = settings.skyUseExplicitSunAngles;
  weather.sunAzimuthDegrees = settings.skySunAzimuthDeg;
  weather.sunElevationDegrees = settings.skySunElevationDeg;
  weather.sunDiskSize = settings.skySunSize;
  weather.moonDiskSize = settings.skyMoonSize;
  weather.lensDropletsEnabled = settings.skyLensDropletsEnabled;
  weather.lensDropletStrength = settings.skyLensDropletStrength;
  return weather;
}

void copyRenderDefaultsToSettings(ViewerSettings& settings, int quality) {
  const auto renderSettings =
    raisin::RayraiWindow::defaultRenderQualitySettings(qualityPresetFromIndex(quality));
  settings.renderQuality = std::clamp(quality, 0, 4);
  settings.backgroundColorRgb255 = renderSettings.backgroundColorRgb255;
  settings.mainLightAmbient = renderSettings.mainLightAmbient;
  settings.mainLightDiffuse = renderSettings.mainLightDiffuse;
  settings.mainLightSpecular = renderSettings.mainLightSpecular;
  // Give the TCP viewer's Ultra preset stronger daylight.
  // Selecting another preset restores its normal sunlight strength.
  settings.lightStrength = settings.renderQuality == 3 ? 1.6f : 1.0f;
  settings.shadowsEnabled = renderSettings.shadowsEnabled;
  settings.shadowResolution = renderSettings.shadowResolution;
  settings.shadowBias = renderSettings.shadowBias;
  settings.shadowStrength = renderSettings.shadowStrength;
  settings.shadowPcfRadius = renderSettings.shadowPcfRadius;
  settings.shadowOrthoHalfSize = renderSettings.shadowOrthoHalfSize;
  settings.shadowNear = renderSettings.shadowNear;
  settings.shadowFar = renderSettings.shadowFar;
  settings.shadowCenterOffset = renderSettings.shadowCenterOffset;
  settings.fogDensity = renderSettings.fogDensity;
  settings.gamma = renderSettings.gamma;
  settings.colorMode = static_cast<int>(renderSettings.colorMode);
  // Preserve the TCP viewer's existing Ultra display curve explicitly. The
  // old three-mode clamp selected this curve accidentally instead of AgX.
  if (settings.renderQuality == 3) {
    settings.colorMode = static_cast<int>(raisin::ViewerColorMode::UnrealPreviewApprox);
  }
  settings.fxaaEnabled = renderSettings.fxaaEnabled;
  settings.bloomEnabled = renderSettings.bloomEnabled;
  settings.bloomThreshold = renderSettings.bloomThreshold;
  settings.bloomStrength = renderSettings.bloomStrength;
  settings.bloomRadius = renderSettings.bloomRadius;
  settings.bloomKnee = renderSettings.bloomKnee;
  settings.bloomQuality = renderSettings.bloomQuality;
  settings.screenSpaceAoEnabled = renderSettings.screenSpaceAoEnabled;
  settings.screenSpaceAoRadius = renderSettings.screenSpaceAoRadius;
  settings.screenSpaceAoStrength = renderSettings.screenSpaceAoStrength;
  settings.screenSpaceAoBias = renderSettings.screenSpaceAoBias;
  settings.opaqueDepthPrepass = renderSettings.opaqueDepthPrepass;
  settings.depthOfFieldEnabled = renderSettings.depthOfFieldEnabled;
  settings.depthOfFieldFocusDistance = renderSettings.depthOfFieldFocusDistance;
  settings.depthOfFieldFocusRange = renderSettings.depthOfFieldFocusRange;
  settings.depthOfFieldMaxRadius = renderSettings.depthOfFieldMaxRadius;
  settings.highFidelityPbr = renderSettings.highFidelityPbr;
  settings.pbrToneMapping = renderSettings.pbrToneMapping;
  settings.pbrExposure = settings.renderQuality == 3 ? 0.65f : renderSettings.pbrExposure;
  settings.pbrEnvironmentMaxLod = renderSettings.pbrEnvironmentMaxLod;
  settings.pbrEnvironmentIntensity = renderSettings.pbrEnvironmentIntensity;
  settings.pbrKeyLightIntensity = renderSettings.pbrKeyLightIntensity;
  settings.skyEnabled = true;
  settings.skySunStrength = 1.8f;
  settings.skySunSize = renderSettings.proceduralSkySunSize;
  settings.skyWeatherEnabled = weatherDefaultEnabledForQuality(settings.renderQuality);
  copyWeatherPresetToSettings(settings, 0);
  settings.reflectiveGround = renderSettings.reflectiveGround;
  settings.reflectiveGroundRoughness = renderSettings.reflectiveGroundRoughness;
  settings.reflectiveGroundMetallic = renderSettings.reflectiveGroundMetallic;
  settings.shadowedLightBudget = renderSettings.shadowedLightBudget;
  settings.maxPointShadowLights = renderSettings.maxPointShadowLights;
  settings.additionalShadowResolutionScale = renderSettings.additionalShadowResolutionScale;
  settings.pointShadowResolutionScale = renderSettings.pointShadowResolutionScale;
  settings.minAdditionalShadowResolution = renderSettings.minAdditionalShadowResolution;
  settings.updateShadowsEveryFrame = renderSettings.updateShadowsEveryFrame;
  settings.maxAdditionalLightsPerFrame = renderSettings.maxAdditionalLightsPerFrame;
  settings.minAdditionalLightInfluence = renderSettings.minAdditionalLightInfluence;
  settings.autoSelectImportedShadowLight = renderSettings.autoSelectImportedShadowLight;
  settings.sortTransparentInstances = renderSettings.sortTransparentInstances;
  settings.addViewerFillLights = renderSettings.addViewerFillLights;
}

bool applyAutomaticRenderQualityIfUnset(ViewerSettings& settings, int recommendedQuality) {
  if (settings.renderQualityUserSet) {
    return false;
  }
  copyRenderDefaultsToSettings(settings, std::clamp(recommendedQuality, 0, 3));
  settings.renderQualityUserSet = false;
  sanitizeViewerSettings(settings);
  return true;
}

void applyWeatherLightStrength(raisin::Light& light,
  const raisin::RenderQualitySettings& weatherLighting, float strength) {
  // Weather may leave the previous light unchanged between updates. Always
  // scale its unmodified settings so the user's strength never compounds.
  light.diffuse = weatherLighting.mainLightDiffuse * strength;
  light.specular = weatherLighting.mainLightSpecular * strength;
}

void applyViewerSettings(raisin::RayraiWindow& viewer, const ViewerSettings& settings) {
  auto quality = qualityPresetFromIndex(settings.renderQuality);
  auto renderSettings = raisin::RayraiWindow::defaultRenderQualitySettings(quality);
  renderSettings.backgroundColorRgb255 = settings.backgroundColorRgb255;
  renderSettings.mainLightAmbient = settings.mainLightAmbient;
  renderSettings.mainLightDiffuse = settings.mainLightDiffuse;
  renderSettings.mainLightSpecular = settings.mainLightSpecular;
  renderSettings.mainLightDirection =
    lightDirectionFromYawPitch(settings.lightYawDeg, settings.lightPitchDeg);
  renderSettings.shadowsEnabled = settings.shadowsEnabled;
  renderSettings.shadowResolution = settings.shadowResolution;
  renderSettings.shadowBias = settings.shadowBias;
  renderSettings.shadowStrength = settings.shadowStrength;
  renderSettings.shadowPcfRadius = settings.shadowPcfRadius;
  renderSettings.shadowOrthoHalfSize = settings.shadowOrthoHalfSize;
  renderSettings.shadowNear = settings.shadowNear;
  renderSettings.shadowFar = settings.shadowFar;
  renderSettings.shadowCenterOffset = settings.shadowCenterOffset;
  renderSettings.fogDensity = settings.fogDensity;
  renderSettings.gamma = settings.gamma;
  renderSettings.colorMode =
    static_cast<raisin::RayraiWindow::ViewerColorMode>(std::clamp(settings.colorMode, 0, 4));
  renderSettings.fxaaEnabled = settings.fxaaEnabled;
  renderSettings.bloomEnabled = settings.bloomEnabled;
  renderSettings.bloomThreshold = settings.bloomThreshold;
  renderSettings.bloomStrength = settings.bloomStrength;
  renderSettings.bloomRadius = settings.bloomRadius;
  renderSettings.bloomKnee = settings.bloomKnee;
  renderSettings.bloomQuality = settings.bloomQuality;
  renderSettings.screenSpaceAoEnabled = settings.screenSpaceAoEnabled;
  renderSettings.screenSpaceAoRadius = settings.screenSpaceAoRadius;
  renderSettings.screenSpaceAoStrength = settings.screenSpaceAoStrength;
  renderSettings.screenSpaceAoBias = settings.screenSpaceAoBias;
  renderSettings.opaqueDepthPrepass = settings.opaqueDepthPrepass;
  renderSettings.depthOfFieldEnabled = settings.depthOfFieldEnabled;
  renderSettings.depthOfFieldFocusDistance = settings.depthOfFieldFocusDistance;
  renderSettings.depthOfFieldFocusRange = settings.depthOfFieldFocusRange;
  renderSettings.depthOfFieldMaxRadius = settings.depthOfFieldMaxRadius;
  const bool highFidelityPbrAllowed = highFidelityPbrAllowedForQuality(settings.renderQuality);
  renderSettings.highFidelityPbr = highFidelityPbrAllowed && settings.highFidelityPbr;
  if (!highFidelityPbrAllowed) {
    renderSettings.forceSimpleMaterialShading = true;
  }
  renderSettings.pbrToneMapping = settings.pbrToneMapping;
  renderSettings.pbrExposure = settings.pbrExposure;
  renderSettings.pbrEnvironmentMaxLod = settings.pbrEnvironmentMaxLod;
  renderSettings.pbrEnvironmentIntensity = settings.pbrEnvironmentIntensity;
  renderSettings.pbrKeyLightIntensity = settings.pbrKeyLightIntensity;
  renderSettings.proceduralSkyBackgroundEnabled = settings.skyEnabled;
  renderSettings.proceduralSkySunStrength = settings.skySunStrength;
  renderSettings.proceduralSkySunSize = settings.skySunSize;
  renderSettings.reflectiveGround = settings.reflectiveGround;
  renderSettings.reflectiveGroundRoughness = settings.reflectiveGroundRoughness;
  renderSettings.reflectiveGroundMetallic = settings.reflectiveGroundMetallic;
  renderSettings.shadowedLightBudget = settings.shadowedLightBudget;
  renderSettings.maxPointShadowLights = settings.maxPointShadowLights;
  renderSettings.additionalShadowResolutionScale = settings.additionalShadowResolutionScale;
  renderSettings.pointShadowResolutionScale = settings.pointShadowResolutionScale;
  renderSettings.minAdditionalShadowResolution = settings.minAdditionalShadowResolution;
  renderSettings.updateShadowsEveryFrame = settings.updateShadowsEveryFrame;
  renderSettings.maxAdditionalLightsPerFrame = settings.maxAdditionalLightsPerFrame;
  renderSettings.minAdditionalLightInfluence = settings.minAdditionalLightInfluence;
  renderSettings.autoSelectImportedShadowLight = settings.autoSelectImportedShadowLight;
  renderSettings.sortTransparentInstances = settings.sortTransparentInstances;
  renderSettings.addViewerFillLights = settings.addViewerFillLights;
  viewer.setRenderQualitySettings(renderSettings);
  viewer.setBackgroundColorRgb255(settings.backgroundColorRgb255);
  viewer.setFogDensity(settings.fogDensity);
  viewer.setShadowOrtho(settings.shadowOrthoHalfSize, settings.shadowNear, settings.shadowFar);
  viewer.setShadowCenterOffset(settings.shadowCenterOffset);
  viewer.setUpdateShadowsEveryFrame(settings.updateShadowsEveryFrame);
  configureViewerFillLights(viewer, settings.addViewerFillLights);

  auto& light = viewer.getLight();
  light.setShadowResolution(settings.shadowResolution);
  light.setShadowsEnabled(settings.shadowsEnabled);
  light.setShadowParams(settings.shadowBias, settings.shadowStrength, settings.shadowPcfRadius);
  viewer.setWeatherSettings(weatherSettingsFromViewerSettings(settings));
  // Apply the explicit cloud-quality user choice last so it wins over both
  // the render preset default and the weather-driven promotion. Auto leaves
  // cloudQuality at whatever the preset + weather logic resolved to.
  if (settings.skyCloudQuality != 0) {
    auto rs = viewer.getUserRenderQualitySettings();
    switch (settings.skyCloudQuality) {
      case 1: rs.cloudQuality = raisin::CloudQuality::Off; break;
      case 2: rs.cloudQuality = raisin::CloudQuality::Texture; break;
      case 3: rs.cloudQuality = raisin::CloudQuality::Volumetric; break;
      default: break;
    }
    // If user picked Off, make sure the layer flag follows; otherwise force
    // the cloud layer on so the chosen quality has something to draw.
    rs.proceduralCloudLayerEnabled = (settings.skyCloudQuality != 1);
    viewer.setRenderQualitySettings(rs);
    if (viewer.getWeatherSettings().enabled) {
      const auto activeWeather = viewer.getWeatherSettings();
      viewer.setWeatherSettings(activeWeather);
    }
  }
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

bool isTcpViewerSingleBodyControlType(int objectTypeRaw) {
  if (objectTypeRaw < 0 || objectTypeRaw == 10 || objectTypeRaw == 11) {
    return false;
  }
  switch (static_cast<raisim::ObjectType>(objectTypeRaw)) {
    case raisim::ObjectType::SPHERE:
    case raisim::ObjectType::BOX:
    case raisim::ObjectType::CYLINDER:
    case raisim::ObjectType::CAPSULE:
    case raisim::ObjectType::MESH:
      return true;
    default:
      return false;
  }
}

bool supportsTcpViewerForceControl(const VisualEntry* entry) {
  if (!entry) {
    return false;
  }
  return entry->isArticulated || isTcpViewerSingleBodyControlType(entry->objectTypeRaw);
}

bool supportsTcpViewerPoseControl(const VisualEntry* entry) {
  if (!entry || entry->isCollision || entry->isArticulated) {
    return false;
  }
  return isTcpViewerSingleBodyControlType(entry->objectTypeRaw);
}

const char* tcpViewerJointTypeLabel(int32_t rawType) {
  const auto type = static_cast<raisim::Joint::Type>(rawType);
  switch (type) {
    case raisim::Joint::Type::REVOLUTE: return "rev";
    case raisim::Joint::Type::PRISMATIC: return "pris";
    case raisim::Joint::Type::SPHERICAL: return "sph";
    case raisim::Joint::Type::FLOATING: return "float";
    case raisim::Joint::Type::FIXED: return "fixed";
    default: return "joint";
  }
}

const char* tcpViewerJointTypeName(int32_t rawType) {
  const auto type = static_cast<raisim::Joint::Type>(rawType);
  switch (type) {
    case raisim::Joint::Type::REVOLUTE: return "Revolute";
    case raisim::Joint::Type::PRISMATIC: return "Prismatic";
    case raisim::Joint::Type::SPHERICAL: return "Spherical";
    case raisim::Joint::Type::FLOATING: return "Floating";
    case raisim::Joint::Type::FIXED: return "Fixed";
    default: return "Unknown";
  }
}

// Unit wxyz quaternion vector; a degenerate one becomes the identity (1, 0, 0, 0).
glm::vec4 normalizedWxyz(const glm::vec4& quat) {
  const float norm2 = glm::dot(quat, quat);
  if (!std::isfinite(norm2) || norm2 <= 1e-12f) {
    return glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
  }
  return quat / std::sqrt(norm2);
}

// Normalizes the w, x, y, z values at values[offset..offset + 3] in place.
void normalizeWxyzSlice(std::vector<float>& values, size_t offset) {
  if (offset + 3 >= values.size()) {
    return;
  }
  const glm::vec4 quat = normalizedWxyz(
    glm::vec4(values[offset], values[offset + 1], values[offset + 2], values[offset + 3]));
  for (int i = 0; i < 4; ++i) {
    values[offset + static_cast<size_t>(i)] = quat[i];
  }
}

const char* objectTypeLabel(int objectTypeRaw) {
  if (objectTypeRaw == -1) {
    return "visual";
  }
  if (objectTypeRaw == 10) {
    return "deformable";
  }
  if (objectTypeRaw == 11) {
    return "granular";
  }
  if (objectTypeRaw < 0) {
    return "unknown";
  }
  switch (static_cast<raisim::ObjectType>(objectTypeRaw)) {
    case raisim::ObjectType::SPHERE: return "sphere";
    case raisim::ObjectType::BOX: return "box";
    case raisim::ObjectType::CYLINDER: return "cylinder";
    case raisim::ObjectType::CAPSULE: return "capsule";
    case raisim::ObjectType::MESH: return "mesh";
    case raisim::ObjectType::HALFSPACE: return "halfspace";
    case raisim::ObjectType::HEIGHTMAP: return "heightmap";
    case raisim::ObjectType::ARTICULATED_SYSTEM: return "articulated system";
    case raisim::ObjectType::COMPOUND: return "compound";
    default: return "unknown";
  }
}

const char* sensorTypeLabel(raisim::Sensor::Type type) {
  switch (type) {
    case raisim::Sensor::Type::RGB: return "RGB camera";
    case raisim::Sensor::Type::DEPTH: return "Depth camera";
    case raisim::Sensor::Type::IMU: return "IMU";
    case raisim::Sensor::Type::SPINNING_LIDAR: return "Spinning lidar";
    default: return "Unknown sensor";
  }
}

const char* sensorSourceLabel(raisim::Sensor::MeasurementSource source) {
  return source == raisim::Sensor::MeasurementSource::MANUAL ? "manual" : "RaiSim";
}

struct CameraFrustumUiState {
  uint32_t parentTag = 0;
  raisim::Sensor::Type type = raisim::Sensor::Type::UNKNOWN;
  std::string sensorName;
  std::string visualName;
  std::string frameName;
  bool visible = false;
  bool frameVisible = false;
  std::shared_ptr<raisin::CameraFrustum> frustum;
  std::shared_ptr<raisin::CoordinateFrame> frame;
};

using CameraFrustumUiStates = std::unordered_map<std::string, CameraFrustumUiState>;

bool isCameraSensorType(raisim::Sensor::Type type) {
  return type == raisim::Sensor::Type::RGB || type == raisim::Sensor::Type::DEPTH;
}

std::string cameraFrustumKey(const SensorInfo& sensor) {
  return std::to_string(sensor.parentTag) + ":" +
         std::to_string(static_cast<int>(sensor.type)) + ":" + sensor.name;
}

float cameraFrustumFarDistance(const SensorInfo& sensor) {
  if (sensor.type == raisim::Sensor::Type::RGB) return 10.0f;
  if (sensor.type == raisim::Sensor::Type::DEPTH && std::isfinite(sensor.clipFar)) {
    return static_cast<float>(std::max(sensor.clipFar, 1.0e-3));
  }
  return 10.0f;
}

std::string cameraSensorRangeLabel(const SensorInfo& sensor) {
  if (sensor.type != raisim::Sensor::Type::DEPTH) return {};
  char label[64];
  std::snprintf(label, sizeof(label), "%.3f~%.3f", sensor.clipNear, sensor.clipFar);
  return label;
}

float cameraFrustumHorizontalFov(const SensorInfo& sensor) {
  if (!std::isfinite(sensor.hFov) || sensor.hFov <= 0.0) {
    return glm::radians(60.0f);
  }
  return static_cast<float>(std::min(sensor.hFov, 3.124139361));
}

void removeCameraFrustum(raisin::RayraiWindow& viewer, CameraFrustumUiState& state) {
  if (!state.frustum) return;
  viewer.removeVisualObject(state.visualName);
  state.frustum.reset();
}

void removeCameraSensorFrame(raisin::RayraiWindow& viewer, CameraFrustumUiState& state) {
  if (!state.frame) return;
  viewer.removeCoordinateFrame(state.frameName);
  state.frame.reset();
}

void clearCameraFrustums(raisin::RayraiWindow& viewer, CameraFrustumUiStates& states) {
  for (auto& [key, state] : states) {
    (void)key;
    removeCameraFrustum(viewer, state);
    removeCameraSensorFrame(viewer, state);
  }
  states.clear();
}

void clearRemoteSceneAndFrustums(raisin::RayraiWindow& viewer,
                                 RemoteScene& scene,
                                 CameraFrustumUiStates& states) {
  // Frustums are viewer-owned visuals, so remove them while their names still
  // exist in the renderer's visualization map.
  clearCameraFrustums(viewer, states);
  scene.clear();
}

void refreshRequestedEntry(RemoteScene& scene,
                           raisin::RayraiWindow& viewer,
                           uint32_t& tag,
                           int& index,
                           const VisualEntry*& entry) {
  tag = 0;
  index = 0;
  entry = nullptr;
  scene.getVisualInfo(viewer.getTargetVisual(), tag, index, entry);
  if (!entry) viewer.setTargetVisual(nullptr);
}

raisin::CoordinateFrame::Pose cameraSensorFramePose(const SensorInfo& sensor) {
  raisin::CoordinateFrame::Pose pose;
  pose.position = sensor.position;
  pose.quaternion = glm::quat(sensor.orientation.w, sensor.orientation.x,
    sensor.orientation.y, sensor.orientation.z);
  const float norm = glm::length(pose.quaternion);
  pose.quaternion = std::isfinite(norm) && norm > 1.0e-6f
    ? pose.quaternion / norm
    : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  return pose;
}

void updateCameraFrustums(raisin::RayraiWindow& viewer, const RemoteScene& scene,
                          CameraFrustumUiStates& states) {
  for (auto& [key, state] : states) {
    (void)key;
    if (!state.visible && !state.frameVisible) {
      removeCameraFrustum(viewer, state);
      removeCameraSensorFrame(viewer, state);
      continue;
    }
    const auto sensors = scene.getSensorsForTag(state.parentTag);
    const auto sensor = std::find_if(sensors.begin(), sensors.end(), [&](const SensorInfo& item) {
      return item.type == state.type && item.name == state.sensorName;
    });
    if (sensor == sensors.end() || !isCameraSensorType(sensor->type)) {
      removeCameraFrustum(viewer, state);
      removeCameraSensorFrame(viewer, state);
      continue;
    }
    if (state.visible) {
      if (!state.frustum) {
        const glm::vec4 color = sensor->type == raisim::Sensor::Type::RGB
          ? glm::vec4(0.25f, 0.78f, 1.0f, 1.0f)
          : glm::vec4(1.0f, 0.66f, 0.20f, 1.0f);
        state.frustum = viewer.addCameraFrustum(state.visualName, color);
        state.frustum->setDetectable(false);
      }
      const float aspect = sensor->width > 0 && sensor->height > 0
        ? static_cast<float>(sensor->width) / static_cast<float>(sensor->height)
        : 1.0f;
      const glm::quat orientation(sensor->orientation.w, sensor->orientation.x,
                                  sensor->orientation.y, sensor->orientation.z);
      state.frustum->updatePerspective(sensor->position, orientation,
        cameraFrustumHorizontalFov(*sensor), aspect,
        static_cast<float>(sensor->clipNear), cameraFrustumFarDistance(*sensor));
    } else {
      removeCameraFrustum(viewer, state);
    }

    if (state.frameVisible) {
      if (!state.frame) {
        state.frame = viewer.addCoordinateFrame(state.frameName);
      }
      state.frame->poses.assign(1, cameraSensorFramePose(*sensor));
      state.frame->frameSize = 0.3;
    } else {
      removeCameraSensorFrame(viewer, state);
    }
  }
}

struct TcpViewerIcons;
bool drawSensorTreeNode(const TcpViewerIcons& icons, raisim::Sensor::Type type,
                        const char* label);

void drawObjectSensors(const TcpViewerIcons& icons, const std::vector<SensorInfo>& sensors,
                       const std::vector<SensorPreviewInfo>& previews,
                       CameraFrustumUiStates& cameraFrustums) {
  if (sensors.empty()) return;
  for (size_t i = 0; i < sensors.size(); ++i) {
    const auto& sensor = sensors[i];
    ImGui::PushID(static_cast<int>(i));
    const std::string label = std::string(sensorTypeLabel(sensor.type)) + "  " + sensor.name;
    if (drawSensorTreeNode(icons, sensor.type, label.c_str())) {
      ImGui::TextDisabled("source %s", sensorSourceLabel(sensor.source));
      if (isCameraSensorType(sensor.type)) {
        const std::string frustumKey = cameraFrustumKey(sensor);
        auto& state = cameraFrustums[frustumKey];
        state.parentTag = sensor.parentTag;
        state.type = sensor.type;
        state.sensorName = sensor.name;
        if (state.visualName.empty()) {
          state.visualName = "__tcp_sensor_frustum:" + frustumKey;
        }
        if (state.frameName.empty()) {
          state.frameName = "__tcp_sensor_frame:" + frustumKey;
        }
        ImGui::Checkbox("Frustum", &state.visible);
        ImGui::SameLine();
        ImGui::Checkbox("Frame", &state.frameVisible);
        const std::string rangeLabel = cameraSensorRangeLabel(sensor);
        if (!rangeLabel.empty()) {
          ImGui::SameLine();
          ImGui::TextDisabled("%s", rangeLabel.c_str());
        }
        ImGui::TextDisabled("%dx%d | clip %.3f .. %.3f m",
          sensor.width, sensor.height, sensor.clipNear, sensor.clipFar);
      } else if (sensor.type == raisim::Sensor::Type::SPINNING_LIDAR) {
        ImGui::TextDisabled("%d yaw x %d pitch samples",
          sensor.yawSamples, sensor.pitchSamples);
      }

      const auto preview = std::find_if(previews.begin(), previews.end(), [&](const auto& item) {
        return item.type == sensor.type && item.name == sensor.name;
      });
      if (preview != previews.end()) {
        ImGui::TextDisabled("render %.2f ms", preview->renderMilliseconds);
        if (preview->type == raisim::Sensor::Type::DEPTH) {
          ImGui::TextDisabled("valid depth %.3f .. %.3f m",
            preview->minimumDepth, preview->maximumDepth);
        }
        if (preview->texture != 0 && preview->width > 0 && preview->height > 0) {
          const float imageWidth = ImGui::GetContentRegionAvail().x;
          const float imageHeight = imageWidth * static_cast<float>(preview->height) /
                                    static_cast<float>(preview->width);
          ImGui::Image(reinterpret_cast<ImTextureID>(uint64_t(preview->texture)),
            ImVec2(imageWidth, imageHeight), ImVec2(0, 1), ImVec2(1, 0));
        }
      } else if (sensor.source == raisim::Sensor::MeasurementSource::MANUAL &&
                 (sensor.type == raisim::Sensor::Type::RGB ||
                  sensor.type == raisim::Sensor::Type::DEPTH)) {
        if (sensor.viewerUpdateRequested) {
          ImGui::TextDisabled("Rendering the first frame...");
        } else {
          ImGui::TextWrapped(
            "Waiting for a server request. If this persists, reinstall RaiSim and rebuild the server example.");
        }
      } else {
        ImGui::TextDisabled("No image preview for this sensor type");
      }
      ImGui::TreePop();
    }
    ImGui::PopID();
  }
}

bool objectMatchesFilter(const ObjectListItem& item, const std::string& filterLower) {
  if (filterLower.empty()) {
    return true;
  }
  std::ostringstream haystack;
  haystack << item.name << " " << objectTypeLabel(item.objectTypeRaw) << " tag " << item.tag
           << " index " << item.index << (item.isCollision ? " collision" : " visual");
  return toLowerAscii(haystack.str()).find(filterLower) != std::string::npos;
}

bool objectTypeLabelLess(int lhsObjectTypeRaw, int rhsObjectTypeRaw) {
  const int labelOrder = std::strcmp(
    objectTypeLabel(lhsObjectTypeRaw), objectTypeLabel(rhsObjectTypeRaw));
  if (labelOrder != 0) return labelOrder < 0;
  return lhsObjectTypeRaw < rhsObjectTypeRaw;
}

bool objectLessByMode(const ObjectListItem& lhs, const ObjectListItem& rhs, int mode) {
  switch (mode) {
    case 1:
      if (lhs.objectTypeRaw != rhs.objectTypeRaw) {
        return objectTypeLabelLess(lhs.objectTypeRaw, rhs.objectTypeRaw);
      }
      return lhs.name < rhs.name;
    case 2:
      return lhs.tag < rhs.tag;
    case 3:
      return lhs.index < rhs.index;
    case 0:
    default:
      return lhs.name < rhs.name;
  }
}

constexpr int kDefaultObjectSortMode = 1;

struct SensorTreeRowLayout {
  float iconX = 0.0f;
  float labelX = 0.0f;
};

SensorTreeRowLayout sensorTreeRowLayout(float rowX, float treeNodeToLabelSpacing,
                                        float iconSize, float itemInnerSpacing) {
  const float iconX = rowX + treeNodeToLabelSpacing;
  return {iconX, iconX + iconSize + itemInnerSpacing};
}

raisin::Visuals* chooseDefaultFollowTarget(const RemoteScene& scene) {
  const auto items = scene.getSelectableObjects();
  raisin::Visuals* fallback = nullptr;
  for (const auto& item : items) {
    if (isContactItem(item) || !item.visual) {
      continue;
    }
    if (!fallback) {
      fallback = item.visual.get();
    }
    if (item.objectTypeRaw >= 0 &&
        static_cast<raisim::ObjectType>(item.objectTypeRaw) == raisim::ObjectType::ARTICULATED_SYSTEM) {
      return item.visual.get();
    }
  }
  return fallback;
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
              << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

std::string csvEscape(const std::string& value) {
  const bool quote = value.find_first_of(",\n\r\"") != std::string::npos;
  if (!quote) {
    return value;
  }
  std::string out = "\"";
  for (char c : value) {
    if (c == '"') out += "\"\"";
    else out += c;
  }
  out += "\"";
  return out;
}

std::filesystem::path timestampedDataPath(const std::filesystem::path& dir, const char* prefix,
                                          const char* extension) {
  const auto now = std::chrono::system_clock::now();
  const std::time_t raw = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &raw);
#else
  localtime_r(&raw, &tm);
#endif
  std::ostringstream name;
  name << (prefix ? prefix : "rayrai") << "_" << std::put_time(&tm, "%Y%m%d_%H%M%S")
       << (extension ? extension : "");
  return raisin::tcp_viewer::nextAvailableOutputPath(dir / name.str());
}

std::vector<AssetDiagnostic> collectAssetDiagnostics(const RemoteScene& scene) {
  std::vector<AssetDiagnostic> assets;
  std::unordered_set<uint64_t> seen;
  for (const auto& snapshot : scene.getVisualEntries()) {
    const auto& entry = snapshot.entry;
    if (entry.meshFile.empty()) {
      continue;
    }
    if (!seen.insert(visualMotionKey(snapshot.tag, snapshot.index)).second) {
      continue;
    }
    AssetDiagnostic asset;
    asset.tag = snapshot.tag;
    asset.index = snapshot.index;
    asset.name = entry.objectName.empty() ? scene.getObjectName(snapshot.tag) : entry.objectName;
    asset.meshFile = entry.meshFile;
    asset.meshPath = entry.meshPath;
    asset.resourceDir = entry.resourceDir;
    asset.resolved = !entry.meshPath.empty();
    assets.push_back(std::move(asset));
  }
  std::sort(assets.begin(), assets.end(), [](const AssetDiagnostic& lhs, const AssetDiagnostic& rhs) {
    if (lhs.resolved != rhs.resolved) return !lhs.resolved;
    if (lhs.meshFile != rhs.meshFile) return lhs.meshFile < rhs.meshFile;
    return lhs.tag < rhs.tag;
  });
  return assets;
}

size_t unresolvedAssetCount(const RemoteScene& scene) {
  return scene.unresolvedAssetCount();
}

void pushPacketSample(std::deque<PacketSample>& samples, const PacketSample& sample) {
  samples.push_back(sample);
  while (samples.size() > 180) {
    samples.pop_front();
  }
}

std::array<float, kTransferRateGraphBuckets> buildTransferRateGraph(
  const std::deque<PacketSample>& samples, double endTimeSeconds,
  double windowSeconds = kTransferRateGraphWindowSeconds) {
  std::array<float, kTransferRateGraphBuckets> rates{};
  if (samples.empty() || windowSeconds <= 0.0 || !std::isfinite(windowSeconds) ||
      !std::isfinite(endTimeSeconds)) {
    return rates;
  }

  const double bucketSeconds = windowSeconds / static_cast<double>(rates.size());
  if (bucketSeconds <= 0.0) {
    return rates;
  }
  const double startTimeSeconds = endTimeSeconds - windowSeconds;
  for (const auto& sample : samples) {
    if (sample.bytes <= 0 || sample.timeSeconds < startTimeSeconds ||
        sample.timeSeconds > endTimeSeconds) {
      continue;
    }
    size_t bucket = sample.timeSeconds >= endTimeSeconds
      ? rates.size() - 1
      : static_cast<size_t>((sample.timeSeconds - startTimeSeconds) / bucketSeconds);
    bucket = std::min(bucket, rates.size() - 1);
    rates[bucket] += static_cast<float>(static_cast<double>(sample.bytes) / 1024.0 / bucketSeconds);
  }
  return rates;
}

float maxTransferRate(const std::array<float, kTransferRateGraphBuckets>& rates) {
  return *std::max_element(rates.begin(), rates.end());
}

bool refreshDiagnosticsPresentation(DiagnosticsPresentationState& presentation,
  const std::deque<PacketSample>& packetSamples, const ViewerStats& stats,
  double nowSeconds) {
  if (!std::isfinite(nowSeconds)) {
    return false;
  }
  if (presentation.initialized && nowSeconds >= presentation.lastRefreshSeconds &&
      nowSeconds - presentation.lastRefreshSeconds < kDiagnosticsPresentationIntervalSeconds) {
    return false;
  }

  presentation.initialized = true;
  presentation.lastRefreshSeconds = nowSeconds;
  presentation.packetSamples = packetSamples;
  presentation.transferRates = buildTransferRateGraph(packetSamples, nowSeconds);
  presentation.timing = summarizePacketTimings(packetSamples);
  presentation.roundTripTimes.clear();
  presentation.roundTripTimes.reserve(packetSamples.size());
  for (const auto& sample : packetSamples) {
    if (!sample.replay && sample.parsed && sample.roundTripMs > 0.0) {
      presentation.roundTripTimes.push_back(static_cast<float>(sample.roundTripMs));
    }
  }
  presentation.parseErrors = stats.parseErrors;
  presentation.rxKbps = stats.rxKbps;
  presentation.unresolvedAssets = stats.unresolvedAssets;
  return true;
}

bool exportSceneJson(const std::filesystem::path& path, const RemoteScene& scene,
                     const std::vector<AssetDiagnostic>& assets, std::string& status) {
  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      status = "scene export failed: cannot create directory";
      return false;
    }
  }
  std::ofstream output(path);
  if (!output) {
    status = "scene export failed: cannot open " + path.string();
    return false;
  }
  output << std::setprecision(9);
  output << "{\n";
  output << "  \"world_time\": ";
  if (scene.hasServerWorldTime()) output << scene.getServerWorldTime(); else output << "null";
  output << ",\n  \"counts\": {\n";
  output << "    \"selectable\": " << scene.selectableObjectCount() << ",\n";
  output << "    \"visuals\": " << scene.visualCount() << ",\n";
  output << "    \"instanced\": " << scene.instancedCount() << ",\n";
  output << "    \"point_clouds\": " << scene.pointCloudCount() << "\n  },\n";
  output << "  \"objects\": [\n";
  const auto items = scene.getSelectableObjects();
  bool first = true;
  for (const auto& item : items) {
    if (!item.visual || isContactItem(item)) continue;
    uint32_t tag = 0;
    int index = 0;
    const VisualEntry* entry = nullptr;
    if (!scene.getVisualInfo(item.visual.get(), tag, index, entry) || !entry) continue;
    if (!first) output << ",\n";
    first = false;
    output << "    {\"tag\": " << tag << ", \"index\": " << index
           << ", \"name\": \"" << jsonEscape(item.name.empty() ? entry->objectName : item.name) << "\""
           << ", \"type\": \"" << objectTypeLabel(entry->objectTypeRaw) << "\""
           << ", \"collision\": " << (entry->isCollision ? "true" : "false")
           << ", \"mesh\": \"" << jsonEscape(entry->meshFile) << "\""
           << ", \"mesh_path\": \"" << jsonEscape(entry->meshPath) << "\""
           << ", \"position\": [" << entry->lastPos.x << ", " << entry->lastPos.y << ", " << entry->lastPos.z << "]"
           << ", \"quaternion_wxyz\": [" << entry->lastQuat.x << ", " << entry->lastQuat.y << ", "
           << entry->lastQuat.z << ", " << entry->lastQuat.w << "]}";
  }
  output << "\n  ],\n  \"assets\": [\n";
  for (size_t i = 0; i < assets.size(); ++i) {
    const auto& asset = assets[i];
    output << "    {\"tag\": " << asset.tag << ", \"index\": " << asset.index
           << ", \"name\": \"" << jsonEscape(asset.name)
           << "\", \"mesh\": \"" << jsonEscape(asset.meshFile)
           << "\", \"resource_dir\": \"" << jsonEscape(asset.resourceDir)
           << "\", \"resolved_path\": \"" << jsonEscape(asset.meshPath)
           << "\", \"resolved\": " << (asset.resolved ? "true" : "false") << "}";
    if (i + 1 < assets.size()) output << ",";
    output << "\n";
  }
  output << "  ]\n}\n";
  status = "exported scene " + path.string();
  return true;
}

void writeTrajectoryRows(std::ofstream& output, const RemoteScene& scene, double timeSeconds) {
  if (!output) {
    return;
  }
  output << std::setprecision(9);
  for (const auto& item : scene.getSelectableObjects()) {
    if (!item.visual || isContactItem(item)) continue;
    uint32_t tag = 0;
    int index = 0;
    const VisualEntry* entry = nullptr;
    if (!scene.getVisualInfo(item.visual.get(), tag, index, entry) || !entry) continue;
    const std::string name = item.name.empty() ? entry->objectName : item.name;
    output << timeSeconds << ',' << tag << ',' << index << ',' << csvEscape(name) << ','
           << csvEscape(objectTypeLabel(entry->objectTypeRaw)) << ','
           << entry->lastPos.x << ',' << entry->lastPos.y << ',' << entry->lastPos.z << ','
           << entry->lastQuat.x << ',' << entry->lastQuat.y << ',' << entry->lastQuat.z << ','
           << entry->lastQuat.w << '\n';
  }
}

void loadEndpointList(const std::filesystem::path& path, std::vector<ConnectionEntry>& connections) {
  std::ifstream input(path);
  if (!input) {
    std::cerr << "WARN: failed to open server list " << path << "\n";
    return;
  }
  std::string line;
  while (std::getline(input, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) line.resize(comment);
    line = trimAscii(line);
    if (line.empty()) continue;
    ConnectionEntry entry;
    if (parseConnectionLabel(line, entry)) {
      recordConnection(connections, entry.host, entry.port);
    }
  }
}

void drawSliderInsideLabel(const char* label, const char* valueText, bool disabled) {
  const ImVec2 itemMin = ImGui::GetItemRectMin();
  const ImVec2 itemMax = ImGui::GetItemRectMax();
  const ImGuiStyle& style = ImGui::GetStyle();
  const float padding = style.FramePadding.x + 2.0f;
  const float textHeight = ImGui::GetFontSize();
  const float textY = itemMin.y + (itemMax.y - itemMin.y - textHeight) * 0.5f;
  const float itemWidth = itemMax.x - itemMin.x;
  const ImVec2 labelSize = ImGui::CalcTextSize(label);
  const ImVec2 valueSize = ImGui::CalcTextSize(valueText);
  const bool visuallyDisabled = disabled || style.Alpha < 0.999f;
  const ImU32 textColor = ImGui::GetColorU32(visuallyDisabled ? ImGuiCol_TextDisabled : ImGuiCol_Text);
  const ImU32 shadowColor = ImGui::GetColorU32(ImVec4(0.0f, 0.0f, 0.0f, visuallyDisabled ? 0.16f : 0.34f));
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  const auto drawTextWithShadow = [&](const ImVec2& pos, const char* text) {
    drawList->AddText(ImVec2(pos.x + 1.0f, pos.y + 1.0f), shadowColor, text);
    drawList->AddText(pos, textColor, text);
  };

  drawList->PushClipRect(itemMin, itemMax, true);
  if (labelSize.x + valueSize.x + padding * 3.0f <= itemWidth) {
    drawTextWithShadow(ImVec2(itemMin.x + padding, textY), label);
    drawTextWithShadow(ImVec2(itemMax.x - padding - valueSize.x, textY), valueText);
  } else {
    char combined[128];
    std::snprintf(combined, sizeof(combined), "%s: %s", label, valueText);
    const ImVec2 combinedSize = ImGui::CalcTextSize(combined);
    const float textX = itemMin.x + std::max(padding, (itemWidth - combinedSize.x) * 0.5f);
    drawTextWithShadow(ImVec2(textX, textY), combined);
  }
  drawList->PopClipRect();
}

// Checkboxes, sliders, and drag fields are the densest controls in the panel and
// the ones that push it past a screenful. ImGui derives all of their heights
// from GetFrameHeight() = font size + 2 * FramePadding.y, so trimming that
// padding shrinks exactly those widgets while leaving buttons, combos, and text
// inputs at their normal size.
// ImGui's floor for these widgets is the font height itself, so this padding is
// what remains to give back. Two pixels at the default font keeps the inline
// slider's value text (drawn inside the frame by drawSliderInsideLabel) clear of
// the border, and scales with the font so it tracks the UI Scale slider. The
// min() guarantees this never *increases* a caller's padding.
inline constexpr float kCompactControlPaddingRatio = 0.11f;
// Vertical gap *around* a compact control, in pixels. Absolute rather than a
// fraction of the font size, so it does not grow with the UI Scale slider.
// Tighter than ImGui's stock 4px: the panel is a long settings list and the
// rows are meant to pack. Only the vertical gap is touched -- ItemSpacing.x
// drives SameLine(), and changing it would move rows like
// [Connect] [Auto-connect].
inline constexpr float kCompactControlSpacingY = 3.0f;

float compactControlPaddingY(float fontSize, float stylePaddingY) {
  if (!std::isfinite(fontSize) || fontSize <= 0.0f) {
    return stylePaddingY;
  }
  return std::min(stylePaddingY, std::max(1.0f, fontSize * kCompactControlPaddingRatio));
}

/** The compact vertical gap. Deliberately independent of the panel's style. */
float compactControlSpacingY() {
  return kCompactControlSpacingY;
}

/**
 * @brief Cell padding that gives table-hosted controls the same gap.
 *
 * A table derives its row height from CellPadding, ignoring ItemSpacing.y
 * entirely, so a checkbox grid would not match the loose checkboxes around it.
 * Half the gap per cell edge adds up to the same spacing between rows. Must be
 * pushed before BeginTable(), not per cell.
 */
ImVec2 compactControlCellPadding(const ImVec2& styleCellPadding) {
  return ImVec2(styleCellPadding.x, compactControlSpacingY() * 0.5f);
}

/**
 * @brief RAII scope that makes a control shorter but gives it more margin.
 *
 * Trims FramePadding.y (which sets the widget's own height) and raises
 * ItemSpacing.y (the gap that follows it).
 */
struct CompactControlScope {
  CompactControlScope() {
    const ImGuiStyle& style = ImGui::GetStyle();
    const float fontSize = ImGui::GetFontSize();
    ImVec2 padding = style.FramePadding;
    padding.y = compactControlPaddingY(fontSize, padding.y);
    ImVec2 spacing = style.ItemSpacing;
    spacing.y = compactControlSpacingY();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, padding);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, spacing);
  }
  CompactControlScope(const CompactControlScope&) = delete;
  CompactControlScope& operator=(const CompactControlScope&) = delete;
  ~CompactControlScope() { ImGui::PopStyleVar(2); }
};

bool drawCompactCheckbox(const char* label, bool* value) {
  CompactControlScope scope;
  return ImGui::Checkbox(label, value);
}

bool compactSliderFloat(const char* label, float* value, float min, float max,
                        const char* format = "%.3f") {
  CompactControlScope scope;
  return ImGui::SliderFloat(label, value, min, max, format);
}

bool compactSliderInt(const char* label, int* value, int min, int max,
                      const char* format = "%d") {
  CompactControlScope scope;
  return ImGui::SliderInt(label, value, min, max, format);
}

bool compactDragFloat(const char* label, float* value, float speed = 1.0f, float min = 0.0f,
                      float max = 0.0f,
                      const char* format = "%.3f") {
  CompactControlScope scope;
  return ImGui::DragFloat(label, value, speed, min, max, format);
}

bool compactDragFloat2(const char* label, float value[2], float speed = 1.0f, float min = 0.0f,
                      float max = 0.0f,
                       const char* format = "%.3f") {
  CompactControlScope scope;
  return ImGui::DragFloat2(label, value, speed, min, max, format);
}

bool compactDragFloat3(const char* label, float value[3], float speed = 1.0f, float min = 0.0f,
                      float max = 0.0f,
                       const char* format = "%.3f") {
  CompactControlScope scope;
  return ImGui::DragFloat3(label, value, speed, min, max, format);
}

bool compactDragFloat4(const char* label, float value[4], float speed = 1.0f, float min = 0.0f,
                      float max = 0.0f,
                       const char* format = "%.3f") {
  CompactControlScope scope;
  return ImGui::DragFloat4(label, value, speed, min, max, format);
}

/**
 * @brief Stacked up/down arrow buttons, sized to sit beside a numeric field.
 *
 * ImGui's ArrowButton is locked to a full frame height, and InputInt's built-in
 * "-"/"+" pair is as wide as two buttons; neither fits a dense settings panel.
 * Holding an arrow repeats, which is what makes a stepper usable for a value
 * like a frame rate.
 *
 * @return +1 when the up arrow fired, -1 for down, 0 otherwise.
 */
int drawCompactStepper(const char* id, float width) {
  // The two halves must total the field's height exactly, so the lower half
  // takes the remainder when the height is odd.
  const float frameHeight = ImGui::GetFrameHeight();
  const float upperHeight = std::max(2.0f, std::floor(frameHeight * 0.5f));
  const float lowerHeight = std::max(2.0f, frameHeight - upperHeight);
  int delta = 0;
  ImGui::PushID(id);
  // Zero spacing and padding so the two halves read as one control.
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
  ImGui::PushButtonRepeat(true);
  ImGui::BeginGroup();
  for (int half = 0; half < 2; ++half) {
    const bool isUp = half == 0;
    ImGui::PushID(half);
    if (ImGui::Button("##step", ImVec2(width, isUp ? upperHeight : lowerHeight))) {
      delta = isUp ? 1 : -1;
    }
    // The font is an ASCII subset with no arrow glyphs, so draw the triangle.
    const ImVec2 itemMin = ImGui::GetItemRectMin();
    const ImVec2 itemMax = ImGui::GetItemRectMax();
    const float insetX = std::max(2.0f, (itemMax.x - itemMin.x) * 0.28f);
    const float insetY = std::max(1.5f, (itemMax.y - itemMin.y) * 0.30f);
    const float midX = (itemMin.x + itemMax.x) * 0.5f;
    const ImU32 arrowColor = ImGui::GetColorU32(ImGuiCol_Text);
    if (isUp) {
      ImGui::GetWindowDrawList()->AddTriangleFilled(
        ImVec2(midX, itemMin.y + insetY),
        ImVec2(itemMin.x + insetX, itemMax.y - insetY),
        ImVec2(itemMax.x - insetX, itemMax.y - insetY), arrowColor);
    } else {
      ImGui::GetWindowDrawList()->AddTriangleFilled(
        ImVec2(itemMin.x + insetX, itemMin.y + insetY),
        ImVec2(itemMax.x - insetX, itemMin.y + insetY),
        ImVec2(midX, itemMax.y - insetY), arrowColor);
    }
    ImGui::PopID();
  }
  ImGui::EndGroup();
  ImGui::PopButtonRepeat();
  ImGui::PopStyleVar(2);
  ImGui::PopID();
  return delta;
}

/**
 * @brief Numeric text box plus a compact stepper, with the label trailing.
 *
 * @p input draws the "##value" text box and returns whether it was edited.
 */
template <typename T, typename InputFn>
bool drawStepperField(const char* id, const char* label, T* value, T step, T min, T max,
                      float fieldWidth, InputFn input) {
  ImGui::PushID(id);
  bool changed = false;
  const float width = fieldWidth > 0.0f ? fieldWidth : ImGui::GetFontSize() * 3.6f;
  // No CompactControlScope here: these are text boxes and must match the height
  // of every other text box in the panel, not the shorter slider/checkbox rows.
  ImGui::SetNextItemWidth(width);
  changed = input();
  ImGui::SameLine(0.0f, 1.0f);
  const int delta = drawCompactStepper("stepper", std::round(ImGui::GetFontSize() * 0.62f));
  if (delta != 0) {
    *value += static_cast<T>(delta) * step;
    changed = true;
  }
  if (changed) {
    *value = std::clamp(*value, min, max);
  }
  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(label);
  ImGui::PopID();
  return changed;
}

bool drawStepperFloat(const char* id, const char* label, float* value, float step, float min,
                      float max, const char* format, float fieldWidth = 0.0f) {
  return drawStepperField(id, label, value, step, min, max, fieldWidth,
    [&] { return ImGui::InputFloat("##value", value, 0.0f, 0.0f, format); });
}

/** Integer flavour of drawStepperFloat(). */
bool drawStepperInt(const char* id, const char* label, int* value, int step, int min, int max,
                    float fieldWidth = 0.0f) {
  return drawStepperField(id, label, value, step, min, max, fieldWidth,
    [&] { return ImGui::InputInt("##value", value, 0, 0); });
}

/**
 * @brief A compact slider whose label and value are drawn inside its frame.
 *
 * @p slider draws the "##slider" control and returns whether it changed.
 */
template <typename T, typename SliderFn>
bool drawInlineLabelSlider(const char* id, const char* label, const T* value, const char* format,
                           float itemWidth, bool disabled, SliderFn slider) {
  ImGui::PushID(id ? id : label);
  if (itemWidth > 0.0f) {
    ImGui::PushItemWidth(itemWidth);
  }
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
  const bool changed = slider();
  ImGui::PopStyleColor();
  if (itemWidth > 0.0f) {
    ImGui::PopItemWidth();
  }

  char valueBuf[32];
  std::snprintf(valueBuf, sizeof(valueBuf), format, *value);
  drawSliderInsideLabel(label, valueBuf, disabled);
  ImGui::PopID();
  return changed;
}

bool drawInlineLabelSliderFloat(const char* id, const char* label, float* value, float min, float max,
                                const char* format, float itemWidth = 0.0f, bool disabled = false) {
  return drawInlineLabelSlider(id, label, value, format, itemWidth, disabled,
    [&] { return compactSliderFloat("##slider", value, min, max, format); });
}

bool drawInlineLabelSliderInt(const char* id, const char* label, int* value, int min, int max,
                              const char* format = "%d", float itemWidth = 0.0f,
                              bool disabled = false) {
  return drawInlineLabelSlider(id, label, value, format, itemWidth, disabled,
    [&] { return compactSliderInt("##slider", value, min, max, format); });
}

bool drawOverlaySlider(const char* id, const char* label, float* value, float min, float max,
  const char* format, float valueWidth, float itemWidth, bool disabled) {
  static_cast<void>(valueWidth);
  return drawInlineLabelSliderFloat(id, label, value, min, max, format, itemWidth, disabled);
}

struct TcpViewerPoseAssignPriorityLow {};
struct TcpViewerPoseAssignPriorityHigh : TcpViewerPoseAssignPriorityLow {};

template <typename Vec3>
auto setTcpViewerZeroPosition(Vec3& position, TcpViewerPoseAssignPriorityHigh)
  -> decltype(position.x, position.y, position.z, void()) {
  position.x = 0.0f;
  position.y = 0.0f;
  position.z = 0.0f;
}

template <typename Vec3>
auto setTcpViewerZeroPosition(Vec3& position, TcpViewerPoseAssignPriorityLow)
  -> decltype(position[0], position[1], position[2], void()) {
  position[0] = 0.0;
  position[1] = 0.0;
  position[2] = 0.0;
}

template <typename Quat>
auto setTcpViewerIdentityQuaternion(Quat& quaternion, TcpViewerPoseAssignPriorityHigh)
  -> decltype(quaternion.w, quaternion.x, quaternion.y, quaternion.z, void()) {
  quaternion.w = 1.0f;
  quaternion.x = 0.0f;
  quaternion.y = 0.0f;
  quaternion.z = 0.0f;
}

template <typename Quat>
auto setTcpViewerIdentityQuaternion(Quat& quaternion, TcpViewerPoseAssignPriorityLow)
  -> decltype(quaternion[0], quaternion[1], quaternion[2], quaternion[3], void()) {
  quaternion[0] = 1.0;
  quaternion[1] = 0.0;
  quaternion[2] = 0.0;
  quaternion[3] = 0.0;
}

template <typename Pose>
void setTcpViewerIdentityPose(Pose& pose) {
  setTcpViewerZeroPosition(pose.position, TcpViewerPoseAssignPriorityHigh{});
  setTcpViewerIdentityQuaternion(pose.quaternion, TcpViewerPoseAssignPriorityHigh{});
}

enum class TcpViewerIconKind {
  Connect = 0,
  Disconnect,
  Refresh,
  Save,
  Home,
  Focus,
  Camera,
  Folder,
  Export,
  Options,
  Robot,
  Reset,
  Exit,
  Pause,
  Play,
  Stop,
  Video,
  Force,
  Torque,
  Add,
  Delete,
  Render,
  Diagnostics,
  Objects,
  Help,
  ViewTop,
  ViewBottom,
  ViewLeft,
  ViewRight,
  ViewFront,
  ViewBack,
  ViewPerspective,
  FolderClosed,
  FolderUp,
  File,
  FileCode,
  FileImage,
  FileCsv,
  FileArchive,
  Step,
  StepFast,
  SensorDepth,
  SensorImu,
  SensorLidar,
  SensorUnknown,
  ObjectVisual,
  ObjectSphere,
  ObjectBox,
  ObjectCylinder,
  ObjectCapsule,
  ObjectMesh,
  ObjectGround,
  ObjectHeightmap,
  ObjectCompound,
  ObjectDeformable,
  ObjectGranular,
  Count
};

struct TcpViewerIcon {
  unsigned int texture = 0;
  int width = 0;
  int height = 0;

  [[nodiscard]] bool valid() const { return texture != 0 && width > 0 && height > 0; }
};

struct TcpViewerImageTexture {
  unsigned int texture = 0;
  int width = 0;
  int height = 0;
  ImVec2 uvMin{0.0f, 0.0f};
  ImVec2 uvMax{1.0f, 1.0f};

  [[nodiscard]] bool valid() const { return texture != 0 && width > 0 && height > 0; }

  void release() {
    raisin::deleteTextureAndInvalidateCache(texture);
    width = 0;
    height = 0;
    uvMin = ImVec2(0.0f, 0.0f);
    uvMax = ImVec2(1.0f, 1.0f);
  }
};

const char* tcpViewerIconFileName(TcpViewerIconKind kind) {
  switch (kind) {
    case TcpViewerIconKind::Connect: return "connect_uicons_sr_plug_connection.png";
    case TcpViewerIconKind::Disconnect: return "disconnect_uicons_sr_link_horizontal_slash.png";
    case TcpViewerIconKind::Refresh: return "refresh_uicons_sr_refresh.png";
    case TcpViewerIconKind::Save: return "save_uicons_sr_disk.png";
    case TcpViewerIconKind::Home: return "home_uicons_sr_house_signal.png";
    case TcpViewerIconKind::Focus: return "focus_uicons_sr_target.png";
    case TcpViewerIconKind::Camera: return "camera_uicons_sr_camera_viewfinder.png";
    case TcpViewerIconKind::Folder: return "folder_uicons_sr_folder_open.png";
    case TcpViewerIconKind::Export: return "export_uicons_sr_file_export.png";
    case TcpViewerIconKind::Options: return "options_uicons_sr_settings_sliders.png";
    case TcpViewerIconKind::Robot: return "robot_uicons_sr_robot.png";
    case TcpViewerIconKind::Reset: return "reset_uicons_sr_rotate_left.png";
    case TcpViewerIconKind::Exit: return "exit_uicons_sr_sign_out_alt.png";
    case TcpViewerIconKind::Pause: return "pause_uicons_sr_pause.png";
    case TcpViewerIconKind::Play: return "play_uicons_sr_play.png";
    case TcpViewerIconKind::Stop: return "stop_uicons_sr_stop.png";
    case TcpViewerIconKind::Force: return "force_uicons_sr_bolt.png";
    case TcpViewerIconKind::Torque: return "torque_uicons_sr_rotate_right.png";
    case TcpViewerIconKind::Video: return "video_uicons_sr_video_camera.png";
    case TcpViewerIconKind::Add: return "add_uicons_sr_square_plus.png";
    case TcpViewerIconKind::Delete: return "delete_uicons_sr_trash.png";
    case TcpViewerIconKind::Render: return "render_uicons_sr_palette.png";
    case TcpViewerIconKind::Diagnostics: return "diagnostics_uicons_sr_chart_histogram.png";
    case TcpViewerIconKind::Objects: return "objects_uicons_sr_layers.png";
    case TcpViewerIconKind::Help: return "help_uicons_sr_keyboard.png";
    case TcpViewerIconKind::ViewTop: return "view_top_uicons_sr_arrow_up.png";
    case TcpViewerIconKind::ViewBottom: return "view_bottom_uicons_sr_arrow_down.png";
    case TcpViewerIconKind::ViewLeft: return "view_left_uicons_sr_arrow_left.png";
    case TcpViewerIconKind::ViewRight: return "view_right_uicons_sr_arrow_right.png";
    // Drafting convention: a dot is the axis coming toward the viewer, an X is
    // the axis going away, so front/back need no arrow direction at all.
    case TcpViewerIconKind::ViewFront: return "view_front_uicons_sr_bullet.png";
    case TcpViewerIconKind::ViewBack: return "view_back_uicons_sr_cross.png";
    case TcpViewerIconKind::ViewPerspective: return "view_perspective_uicons_sr_panorama.png";
    case TcpViewerIconKind::FolderClosed: return "folder_closed_uicons_sr_folder.png";
    case TcpViewerIconKind::FolderUp: return "folder_up_uicons_sr_level_up_alt.png";
    case TcpViewerIconKind::File: return "file_uicons_sr_file.png";
    case TcpViewerIconKind::FileCode: return "file_code_uicons_sr_file_code.png";
    case TcpViewerIconKind::FileImage: return "file_image_uicons_sr_picture.png";
    case TcpViewerIconKind::FileCsv: return "file_csv_uicons_sr_file_csv.png";
    case TcpViewerIconKind::FileArchive: return "file_archive_uicons_sr_file_zipper.png";
    case TcpViewerIconKind::Step: return "step_uicons_sr_step_forward.png";
    case TcpViewerIconKind::StepFast: return "step_fast_uicons_sr_forward_fast.png";
    case TcpViewerIconKind::SensorDepth: return "depth_uicons_sr_scanner_image.png";
    case TcpViewerIconKind::SensorImu: return "imu_uicons_sr_compass_alt.png";
    case TcpViewerIconKind::SensorLidar: return "lidar_uicons_sr_radar.png";
    case TcpViewerIconKind::SensorUnknown: return "sensor_uicons_sr_sensor.png";
    case TcpViewerIconKind::ObjectVisual: return "visual_uicons_sr_transformation_shapes.png";
    case TcpViewerIconKind::ObjectSphere: return "sphere_uicons_sr_sphere.png";
    case TcpViewerIconKind::ObjectBox: return "box_uicons_sr_cube.png";
    case TcpViewerIconKind::ObjectCylinder: return "cylinder_uicons_sr_database.png";
    case TcpViewerIconKind::ObjectCapsule: return "capsule_uicons_sr_capsules.png";
    case TcpViewerIconKind::ObjectMesh: return "mesh_uicons_sr_vector_polygon.png";
    case TcpViewerIconKind::ObjectGround: return "ground_uicons_sr_land_layers.png";
    case TcpViewerIconKind::ObjectHeightmap: return "heightmap_uicons_sr_mountain.png";
    case TcpViewerIconKind::ObjectCompound: return "compound_uicons_sr_cubes.png";
    case TcpViewerIconKind::ObjectDeformable: return "deformable_uicons_sr_wave_square.png";
    case TcpViewerIconKind::ObjectGranular: return "granular_uicons_sr_braille.png";
    case TcpViewerIconKind::Count: break;
  }
  return "";
}

/** Icon for a directory row, or for a file row based on its extension. */
TcpViewerIconKind fileBrowserIconKind(const std::filesystem::path& path, bool isDirectory) {
  if (isDirectory) {
    return TcpViewerIconKind::FolderClosed;
  }
  const std::string extension = lowerCaseFileExtension(path);
  const auto isAny = [&](std::initializer_list<const char*> candidates) {
    return std::find(candidates.begin(), candidates.end(), extension) != candidates.end();
  };

  if (extension == "urdf") return TcpViewerIconKind::Robot;
  if (isAny({"obj", "stl", "dae", "ply", "gltf", "glb", "fbx", "usd", "usda", "usdc", "usdz"})) {
    return TcpViewerIconKind::ObjectMesh;
  }
  if (isAny({"png", "jpg", "jpeg", "bmp", "tga", "hdr", "exr", "pgm"})) {
    return TcpViewerIconKind::FileImage;
  }
  if (isAny({"mp4", "mov", "avi", "mkv", "webm"})) return TcpViewerIconKind::Video;
  if (isAny({"csv", "tsv"})) return TcpViewerIconKind::FileCsv;
  if (isAny({"zip", "gz", "tgz", "bz2", "xz", "7z", "tar", "zst"})) {
    return TcpViewerIconKind::FileArchive;
  }
  if (isAny({"xml", "json", "yaml", "yml", "lua", "py", "toml", "ini", "cfg", "mjcf", "sdf"})) {
    return TcpViewerIconKind::FileCode;
  }
  if (extension == "rrtcs") return TcpViewerIconKind::Save;
  return TcpViewerIconKind::File;
}
