// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerSettings.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace raisin::tcp_viewer
{
namespace
{

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

bool parseLongStrict(const char* value, int base, long& out) {
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
  out = std::strtol(value, &end, base);
  if (end == value || errno == ERANGE) {
    return false;
  }
  while (std::isspace(static_cast<unsigned char>(*end))) {
    ++end;
  }
  return *end == '\0';
}

bool parseFloatStrict(const char* value, float& out) {
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
  out = std::strtof(value, &end);
  if (end == value || errno == ERANGE || !std::isfinite(out)) {
    return false;
  }
  while (std::isspace(static_cast<unsigned char>(*end))) {
    ++end;
  }
  return *end == '\0';
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

std::string toLowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool parseBoolValue(const std::string& rawValue, bool fallback) {
  const std::string value = toLowerAscii(trimAscii(rawValue));
  if (value == "true" || value == "yes" || value == "on" || value == "1") {
    return true;
  }
  if (value == "false" || value == "no" || value == "off" || value == "0") {
    return false;
  }
  return fallback;
}

float parseFloatValue(const std::string& rawValue, float fallback) {
  float parsed = fallback;
  return parseFloatStrict(rawValue.c_str(), parsed) ? parsed : fallback;
}

int parseIntValue(const std::string& rawValue, int fallback) {
  long parsed = 0;
  if (!parseLongStrict(rawValue.c_str(), 10, parsed) ||
      parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

bool parseVec3Text(const std::string& value, glm::vec3& out) {
  float values[3]{};
  if (!parseFloatListStrict(value.c_str(), values, 3)) {
    return false;
  }
  out = glm::vec3(values[0], values[1], values[2]);
  return true;
}

bool parseVec4Text(const std::string& value, glm::vec4& out) {
  float values[4]{};
  if (!parseFloatListStrict(value.c_str(), values, 4)) {
    return false;
  }
  out = glm::vec4(values[0], values[1], values[2], values[3]);
  return true;
}

glm::vec3 parseVec3Value(const std::string& rawValue, const glm::vec3& fallback) {
  glm::vec3 parsed = fallback;
  return parseVec3Text(rawValue, parsed) ? parsed : fallback;
}

glm::vec4 parseVec4Value(const std::string& rawValue, const glm::vec4& fallback) {
  glm::vec4 parsed = fallback;
  return parseVec4Text(rawValue, parsed) ? parsed : fallback;
}

float normalizedDegreesForSettings(float degrees) {
  if (!std::isfinite(degrees)) {
    return 0.0f;
  }
  float wrapped = std::fmod(degrees, 360.0f);
  if (wrapped < 0.0f) {
    wrapped += 360.0f;
  }
  return wrapped;
}

template <typename T>
T clampValue(T value, T minValue, T maxValue) {
  return std::clamp(value, minValue, maxValue);
}

glm::vec3 clampVec3(const glm::vec3& value, float minValue, float maxValue) {
  return glm::vec3(
    clampValue(value.x, minValue, maxValue),
    clampValue(value.y, minValue, maxValue),
    clampValue(value.z, minValue, maxValue));
}

glm::vec4 clampVec4(const glm::vec4& value, float minValue, float maxValue) {
  return glm::vec4(
    clampValue(value.x, minValue, maxValue),
    clampValue(value.y, minValue, maxValue),
    clampValue(value.z, minValue, maxValue),
    clampValue(value.w, minValue, maxValue));
}

} // namespace

bool parsePortStrict(const std::string& value, int& port) {
  long parsed = 0;
  if (!parseLongStrict(value.c_str(), 10, parsed) || parsed <= 0 || parsed > 65535) {
    return false;
  }
  port = static_cast<int>(parsed);
  return true;
}

std::string normalizeConnectionHost(const std::string& value) {
  std::string host = trimAscii(value);
  if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
    host = trimAscii(host.substr(1, host.size() - 2));
  }
  if (host.empty()) {
    return {};
  }
  const bool hasInvalidChar = std::any_of(host.begin(), host.end(), [](unsigned char c) {
    return std::iscntrl(c) || std::isspace(c);
  });
  return hasInvalidChar ? std::string{} : host;
}

bool normalizeConnectionEndpoint(const std::string& host, int port, ConnectionEntry& entry) {
  if (port <= 0 || port > 65535) {
    return false;
  }
  const std::string normalizedHost = normalizeConnectionHost(host);
  if (normalizedHost.empty()) {
    return false;
  }
  entry.host = normalizedHost;
  entry.port = port;
  return true;
}

std::string formatEndpointHost(const std::string& host) {
  if (host.find(':') != std::string::npos &&
      !(host.size() >= 2 && host.front() == '[' && host.back() == ']')) {
    return "[" + host + "]";
  }
  return host;
}

std::string formatConnectionLabel(const ConnectionEntry& entry) {
  if (entry.host.empty()) {
    return {};
  }
  return formatEndpointHost(entry.host) + ":" + std::to_string(entry.port);
}

bool parseConnectionLabel(const std::string& value, ConnectionEntry& entry) {
  const std::string trimmed = trimAscii(value);
  if (trimmed.empty()) {
    return false;
  }

  std::string hostPart;
  std::string portPart;
  if (trimmed.front() == '[') {
    const auto close = trimmed.find(']');
    if (close == std::string::npos || close + 2 > trimmed.size() || trimmed[close + 1] != ':') {
      return false;
    }
    hostPart = trimmed.substr(1, close - 1);
    portPart = trimmed.substr(close + 2);
  } else {
    const auto sep = trimmed.rfind(':');
    if (sep == std::string::npos || sep == 0 || sep + 1 >= trimmed.size()) {
      return false;
    }
    hostPart = trimmed.substr(0, sep);
    if (hostPart.find(':') != std::string::npos) {
      return false;
    }
    portPart = trimmed.substr(sep + 1);
  }

  int parsedPort = 0;
  if (!parsePortStrict(trimAscii(portPart), parsedPort)) {
    return false;
  }

  return normalizeConnectionEndpoint(hostPart, parsedPort, entry);
}

void recordConnection(std::vector<ConnectionEntry>& connections, const std::string& host, int port) {
  ConnectionEntry normalized;
  if (!normalizeConnectionEndpoint(host, port, normalized)) {
    return;
  }
  connections.erase(
    std::remove_if(connections.begin(), connections.end(),
      [&](const ConnectionEntry& entry) {
        return entry.host == normalized.host && entry.port == normalized.port;
      }),
    connections.end());
  connections.insert(connections.begin(), normalized);
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

float sanitizeTcpUpdateRateHz(float value) {
  if (!std::isfinite(value)) {
    return kTcpUpdateRateDefaultHz;
  }
  return std::clamp(value, kTcpUpdateRateMinHz, kTcpUpdateRateMaxHz);
}

std::chrono::steady_clock::duration tcpUpdatePeriodForHz(float rateHz) {
  const double seconds = 1.0 / static_cast<double>(sanitizeTcpUpdateRateHz(rateHz));
  return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(seconds));
}

bool consumeTcpUpdateSlot(std::chrono::steady_clock::time_point now,
                          std::chrono::steady_clock::time_point& nextRequestTime,
                          float rateHz) {
  if (now < nextRequestTime) {
    return false;
  }
  nextRequestTime = now + tcpUpdatePeriodForHz(rateHz);
  return true;
}

const char* qualityName(int quality) {
  static constexpr const char* kNames[] = {"Fast", "Balanced", "High", "Ultra", "Custom"};
  return kNames[std::clamp(quality, 0, 4)];
}

const char* colorModeName(int colorMode) {
  static constexpr const char* kNames[] = {
    "Fast Linear", "ACES Approx", "Unreal Preview", "Filmic Approx", "AgX Approx"};
  return kNames[std::clamp(colorMode, 0, 4)];
}

int qualityIndexFromName(const std::string& rawValue, int fallback) {
  const std::string value = toLowerAscii(trimAscii(rawValue));
  if (value == "fast" || value == "0") return 0;
  if (value == "balanced" || value == "balance" || value == "1") return 1;
  if (value == "high" || value == "2") return 2;
  if (value == "ultra" || value == "3") return 3;
  if (value == "custom" || value == "4") return 4;
  return fallback;
}

int colorModeIndexFromName(const std::string& rawValue, int fallback) {
  const std::string value = toLowerAscii(trimAscii(rawValue));
  if (value == "fast_linear" || value == "fast linear" || value == "linear" || value == "0") return 0;
  if (value == "aces_approx" || value == "aces approx" || value == "aces" || value == "1") return 1;
  if (value == "unreal_preview" || value == "unreal preview" || value == "unreal" || value == "2") return 2;
  if (value == "filmic_approx" || value == "filmic approx" || value == "filmic" || value == "3") return 3;
  if (value == "agx_approx" || value == "agx approx" || value == "agx" || value == "4") return 4;
  return fallback;
}

int cloudQualityIndexFromName(const std::string& rawValue, int fallback) {
  const std::string value = toLowerAscii(trimAscii(rawValue));
  if (value == "auto" || value == "0") return 0;
  if (value == "off" || value == "none" || value == "1") return 1;
  if (value == "texture" || value == "2d" || value == "fast" || value == "2") return 2;
  if (value == "volumetric" || value == "3d" || value == "ultra" || value == "3") return 3;
  return fallback;
}

const char* cloudQualityName(int index) {
  switch (std::clamp(index, 0, 3)) {
    case 0: return "auto";
    case 1: return "off";
    case 2: return "texture";
    case 3: return "volumetric";
  }
  return "auto";
}

const char* weatherPresetName(int preset) {
  static constexpr const char* kNames[] = {
    "Clear", "Hazy", "Overcast", "Fog", "Rain", "Heavy Rain",
    "Snow", "Storm", "Night Clear", "Night Rain", "Custom"};
  return kNames[std::clamp(preset, 0, 10)];
}

const char* weatherQualityName(int quality) {
  static constexpr const char* kNames[] = {"Low", "Medium", "High", "Ultra"};
  return kNames[std::clamp(quality, 0, 3)];
}

int weatherPresetIndexFromName(const std::string& rawValue, int fallback) {
  const std::string value = toLowerAscii(trimAscii(rawValue));
  if (value == "clear" || value == "0") return 0;
  if (value == "hazy" || value == "1") return 1;
  if (value == "overcast" || value == "2") return 2;
  if (value == "fog" || value == "3") return 3;
  if (value == "rain" || value == "4") return 4;
  if (value == "heavy_rain" || value == "heavy rain" || value == "5") return 5;
  if (value == "snow" || value == "6") return 6;
  if (value == "storm" || value == "7") return 7;
  if (value == "night_clear" || value == "night clear" || value == "8") return 8;
  if (value == "night_rain" || value == "night rain" || value == "9") return 9;
  if (value == "custom" || value == "10") return 10;
  return fallback;
}

int weatherQualityIndexFromName(const std::string& rawValue, int fallback) {
  const std::string value = toLowerAscii(trimAscii(rawValue));
  if (value == "low" || value == "0") return 0;
  if (value == "medium" || value == "1") return 1;
  if (value == "high" || value == "2") return 2;
  if (value == "ultra" || value == "3") return 3;
  return fallback;
}

bool weatherDefaultEnabledForQuality(int quality) {
  const int clampedQuality = std::clamp(quality, 0, 4);
  return clampedQuality >= 2;
}

bool highFidelityPbrAllowedForQuality(int quality) {
  const int clampedQuality = std::clamp(quality, 0, 4);
  return clampedQuality >= 2;
}

void sanitizeViewerSettings(ViewerSettings& settings) {
  settings.renderQuality = clampValue(settings.renderQuality, 0, 4);
  settings.backgroundColorRgb255 = clampVec4(settings.backgroundColorRgb255, 0.0f, 255.0f);
  settings.mainLightAmbient = clampVec3(settings.mainLightAmbient, 0.0f, 4.0f);
  settings.mainLightDiffuse = clampVec3(settings.mainLightDiffuse, 0.0f, 4.0f);
  settings.mainLightSpecular = clampVec3(settings.mainLightSpecular, 0.0f, 4.0f);
  settings.cameraSpeed = clampValue(settings.cameraSpeed, 0.1f, 30.0f);
  settings.cameraFovDeg = clampValue(settings.cameraFovDeg, 20.0f, 100.0f);
  settings.cameraNear = clampValue(settings.cameraNear, 0.001f, 1.0f);
  settings.cameraFar = clampValue(settings.cameraFar, 10.0f, 5000.0f);
  if (settings.cameraFar <= settings.cameraNear) {
    settings.cameraFar = std::min(5000.0f, std::max(10.0f, settings.cameraNear * 10.0f));
  }
  settings.lightYawDeg = clampValue(settings.lightYawDeg, -180.0f, 180.0f);
  settings.lightPitchDeg = clampValue(settings.lightPitchDeg, -89.0f, 89.0f);
  settings.lightStrength = clampValue(settings.lightStrength, 0.0f, 2.0f);
  settings.ambientStrength = clampValue(settings.ambientStrength, 0.0f, 2.0f);
  settings.shadowResolution = clampValue(settings.shadowResolution, 64, 8192);
  settings.shadowBias = clampValue(settings.shadowBias, 0.0f, 0.01f);
  settings.shadowStrength = clampValue(settings.shadowStrength, 0.0f, 1.0f);
  settings.shadowPcfRadius = clampValue(settings.shadowPcfRadius, 0.0f, 4.0f);
  settings.shadowOrthoHalfSize = clampValue(settings.shadowOrthoHalfSize, 1.0f, 100.0f);
  settings.shadowNear = clampValue(settings.shadowNear, 0.01f, 10.0f);
  settings.shadowFar = clampValue(settings.shadowFar, 1.0f, 250.0f);
  if (settings.shadowFar <= settings.shadowNear) {
    settings.shadowFar = std::min(250.0f, settings.shadowNear + 1.0f);
  }
  settings.shadowCenterOffset = clampValue(settings.shadowCenterOffset, 0.0f, 80.0f);
  settings.fogDensity = clampValue(settings.fogDensity, 0.0f, 0.08f);
  settings.gamma = clampValue(settings.gamma, 0.5f, 2.5f);
  settings.colorMode = clampValue(settings.colorMode, 0, 4);
  settings.bloomThreshold = clampValue(settings.bloomThreshold, 0.0f, 4.0f);
  settings.bloomStrength = clampValue(settings.bloomStrength, 0.0f, 2.0f);
  settings.bloomRadius = clampValue(settings.bloomRadius, 0.0f, 12.0f);
  settings.bloomKnee = clampValue(settings.bloomKnee, 0.0f, 1.0f);
  settings.bloomQuality = clampValue(settings.bloomQuality, 0, 3);
  settings.screenSpaceAoRadius = clampValue(settings.screenSpaceAoRadius, 0.05f, 10.0f);
  settings.screenSpaceAoStrength = clampValue(settings.screenSpaceAoStrength, 0.0f, 4.0f);
  settings.screenSpaceAoBias = clampValue(settings.screenSpaceAoBias, 0.0f, 0.25f);
  settings.depthOfFieldFocusDistance = clampValue(settings.depthOfFieldFocusDistance, 0.05f, 30.0f);
  settings.depthOfFieldFocusRange = clampValue(settings.depthOfFieldFocusRange, 1.0f, 100.0f);
  settings.depthOfFieldMaxRadius = clampValue(settings.depthOfFieldMaxRadius, 0.0f, 8.0f);
  settings.pbrExposure = clampValue(settings.pbrExposure, 0.1f, 4.0f);
  settings.pbrEnvironmentMaxLod = clampValue(settings.pbrEnvironmentMaxLod, 0.0f, 12.0f);
  settings.pbrEnvironmentIntensity = clampValue(settings.pbrEnvironmentIntensity, 0.0f, 4.0f);
  settings.pbrKeyLightIntensity = clampValue(settings.pbrKeyLightIntensity, 0.0f, 4.0f);
  if (!highFidelityPbrAllowedForQuality(settings.renderQuality)) {
    settings.highFidelityPbr = false;
  }
  settings.skySunStrength = clampValue(settings.skySunStrength, 0.0f, 8.0f);
  settings.skySunSize = clampValue(settings.skySunSize, 0.001f, 0.08f);
  if (!weatherDefaultEnabledForQuality(settings.renderQuality)) {
    settings.skyWeatherEnabled = false;
  }
  settings.skyWeatherPreset = clampValue(settings.skyWeatherPreset, 0, 10);
  settings.skyWeatherQuality = clampValue(settings.skyWeatherQuality, 0, 3);
  settings.skyWeatherSeed = clampValue(settings.skyWeatherSeed, 1, 1000000);
  settings.skyTimeOfDayHours = clampValue(settings.skyTimeOfDayHours, 0.0f, 24.0f);
  settings.skyLatitude = clampValue(settings.skyLatitude, -89.9f, 89.9f);
  settings.skyLongitude = clampValue(settings.skyLongitude, -180.0f, 180.0f);
  settings.skyUtcOffsetHours = clampValue(settings.skyUtcOffsetHours, -12.0f, 14.0f);
  settings.skyYear = clampValue(settings.skyYear, 1900, 2500);
  settings.skyMonth = clampValue(settings.skyMonth, 1, 12);
  settings.skyDay = clampValue(settings.skyDay, 1, 31);
  settings.skyWindDirectionDeg = normalizedDegreesForSettings(settings.skyWindDirectionDeg);
  settings.skyWindSpeed = clampValue(settings.skyWindSpeed, 0.0f, 80.0f);
  settings.skyCloudCoverage = clampValue(settings.skyCloudCoverage, 0.0f, 1.0f);
  settings.skyCloudDensity = clampValue(settings.skyCloudDensity, 0.0f, 1.0f);
  settings.skyCloudAltitudeMeters = clampValue(settings.skyCloudAltitudeMeters, 20.0f, 12000.0f);
  settings.skyCloudThicknessMeters = clampValue(settings.skyCloudThicknessMeters, 1.0f, 4000.0f);
  settings.skyCloudShadowStrength = clampValue(settings.skyCloudShadowStrength, 0.0f, 1.0f);
  settings.skyCloudScale = clampValue(settings.skyCloudScale, 0.01f, 2.0f);
  settings.skyCloudAnimationSpeed = clampValue(settings.skyCloudAnimationSpeed, 0.0f, 200.0f);
  settings.skyCloudQuality = clampValue(settings.skyCloudQuality, 0, 3);
  settings.skyPrecipitationRate = clampValue(settings.skyPrecipitationRate, 0.0f, 1.0f);
  settings.skyRainOcclusionStrength = clampValue(settings.skyRainOcclusionStrength, 0.0f, 1.0f);
  settings.skySnowCoverage = clampValue(settings.skySnowCoverage, 0.0f, 1.0f);
  settings.skyHumidity = clampValue(settings.skyHumidity, 0.0f, 1.0f);
  settings.skyWetness = clampValue(settings.skyWetness, 0.0f, 1.0f);
  settings.skyWetnessAccumulationRate = clampValue(settings.skyWetnessAccumulationRate, 0.0f, 4.0f);
  settings.skyWetnessDryingRate = clampValue(settings.skyWetnessDryingRate, 0.0f, 4.0f);
  settings.skyLightningRate = clampValue(settings.skyLightningRate, 0.0f, 16.0f);
  settings.skyFogDensity = clampValue(settings.skyFogDensity, 0.0f, 1.0f);
  settings.skyVisibilityMeters = clampValue(settings.skyVisibilityMeters, 1.0f, 100000.0f);
  settings.skyFogColor = clampVec3(settings.skyFogColor, 0.0f, 4.0f);
  settings.skyFogAnisotropy = clampValue(settings.skyFogAnisotropy, -0.85f, 0.85f);
  settings.skyAirTurbidity = clampValue(settings.skyAirTurbidity, 1.0f, 12.0f);
  settings.skyGroundAlbedo = clampValue(settings.skyGroundAlbedo, 0.0f, 1.0f);
  settings.skySunAzimuthDeg = normalizedDegreesForSettings(settings.skySunAzimuthDeg);
  settings.skySunElevationDeg = clampValue(settings.skySunElevationDeg, -8.0f, 89.0f);
  settings.skyMoonSize = clampValue(settings.skyMoonSize, 0.001f, 0.08f);
  settings.skyLensDropletStrength = clampValue(settings.skyLensDropletStrength, 0.0f, 1.0f);
  settings.reflectiveGroundRoughness = clampValue(settings.reflectiveGroundRoughness, 0.02f, 1.0f);
  settings.reflectiveGroundMetallic = clampValue(settings.reflectiveGroundMetallic, 0.0f, 1.0f);
  settings.shadowedLightBudget = clampValue(settings.shadowedLightBudget, 0, 8);
  settings.maxPointShadowLights = clampValue(settings.maxPointShadowLights, 0, 8);
  settings.additionalShadowResolutionScale = clampValue(settings.additionalShadowResolutionScale, 0.05f, 2.0f);
  settings.pointShadowResolutionScale = clampValue(settings.pointShadowResolutionScale, 0.05f, 2.0f);
  settings.minAdditionalShadowResolution = clampValue(settings.minAdditionalShadowResolution, 64, 2048);
  settings.maxAdditionalLightsPerFrame = clampValue(settings.maxAdditionalLightsPerFrame, 0, 16);
  settings.minAdditionalLightInfluence = clampValue(settings.minAdditionalLightInfluence, 0.0f, 1.0f);
  settings.uiScale = clampValue(settings.uiScale, 0.8f, 2.6f);
  settings.tcpUpdateRateHz = sanitizeTcpUpdateRateHz(settings.tcpUpdateRateHz);
}

std::filesystem::path settingsFilePath() {
  const char* home = std::getenv("HOME");
  if (home && *home) {
    return std::filesystem::path(home) / ".rayrai" / "settings.yaml";
  }
#if defined(_WIN32)
  const char* appData = std::getenv("APPDATA");
  if (appData && *appData)
    return std::filesystem::path(appData) / "rayrai" / "settings.yaml";
  const char* userProfile = std::getenv("USERPROFILE");
  if (userProfile && *userProfile)
    return std::filesystem::path(userProfile) / ".rayrai" / "settings.yaml";
#endif
  std::error_code ec;
  const auto cwd = std::filesystem::current_path(ec);
  return (ec ? std::filesystem::path(".") : cwd) / ".rayrai" / "settings.yaml";
}

namespace
{

/**
 * @brief Visits every scalar setting in file order.
 *
 * Called as `visit(key, field)`, or as `visit(key, field, indexFromName, name)`
 * for enum settings that are stored by name. Works for const and mutable
 * settings, so loading and saving share one key table.
 */
template <typename Settings, typename Visitor>
void forEachScalarSetting(Settings& s, Visitor&& visit) {
  visit("render_quality", s.renderQuality, qualityIndexFromName, qualityName);
  visit("render_quality_user_set", s.renderQualityUserSet);
  visit("background_color_rgb255", s.backgroundColorRgb255);
  visit("main_light_ambient", s.mainLightAmbient);
  visit("main_light_diffuse", s.mainLightDiffuse);
  visit("main_light_specular", s.mainLightSpecular);
  visit("camera_speed", s.cameraSpeed);
  visit("camera_fov_deg", s.cameraFovDeg);
  visit("camera_near", s.cameraNear);
  visit("camera_far", s.cameraFar);
  visit("light_yaw_deg", s.lightYawDeg);
  visit("light_pitch_deg", s.lightPitchDeg);
  visit("light_strength", s.lightStrength);
  visit("ambient_strength", s.ambientStrength);
  visit("shadows_enabled", s.shadowsEnabled);
  visit("shadow_resolution", s.shadowResolution);
  visit("shadow_bias", s.shadowBias);
  visit("shadow_strength", s.shadowStrength);
  visit("shadow_pcf_radius", s.shadowPcfRadius);
  visit("shadow_ortho_half_size", s.shadowOrthoHalfSize);
  visit("shadow_near", s.shadowNear);
  visit("shadow_far", s.shadowFar);
  visit("shadow_center_offset", s.shadowCenterOffset);
  visit("fog_density", s.fogDensity);
  visit("gamma", s.gamma);
  visit("color_mode", s.colorMode, colorModeIndexFromName, colorModeName);
  visit("fxaa_enabled", s.fxaaEnabled);
  visit("bloom_enabled", s.bloomEnabled);
  visit("bloom_threshold", s.bloomThreshold);
  visit("bloom_strength", s.bloomStrength);
  visit("bloom_radius", s.bloomRadius);
  visit("bloom_knee", s.bloomKnee);
  visit("bloom_quality", s.bloomQuality);
  visit("screen_space_ao_enabled", s.screenSpaceAoEnabled);
  visit("screen_space_ao_radius", s.screenSpaceAoRadius);
  visit("screen_space_ao_strength", s.screenSpaceAoStrength);
  visit("screen_space_ao_bias", s.screenSpaceAoBias);
  visit("opaque_depth_prepass", s.opaqueDepthPrepass);
  visit("depth_of_field_enabled", s.depthOfFieldEnabled);
  visit("depth_of_field_focus_distance", s.depthOfFieldFocusDistance);
  visit("depth_of_field_focus_range", s.depthOfFieldFocusRange);
  visit("depth_of_field_max_radius", s.depthOfFieldMaxRadius);
  visit("high_fidelity_pbr", s.highFidelityPbr);
  visit("pbr_tone_mapping", s.pbrToneMapping);
  visit("pbr_exposure", s.pbrExposure);
  visit("pbr_environment_max_lod", s.pbrEnvironmentMaxLod);
  visit("pbr_environment_intensity", s.pbrEnvironmentIntensity);
  visit("pbr_key_light_intensity", s.pbrKeyLightIntensity);
  visit("sky_enabled", s.skyEnabled);
  visit("sky_sun_strength", s.skySunStrength);
  visit("sky_sun_size", s.skySunSize);
  visit("sky_weather_enabled", s.skyWeatherEnabled);
  visit("sky_weather_preset", s.skyWeatherPreset, weatherPresetIndexFromName, weatherPresetName);
  visit("sky_weather_quality", s.skyWeatherQuality, weatherQualityIndexFromName, weatherQualityName);
  visit("sky_weather_seed", s.skyWeatherSeed);
  visit("sky_time_of_day_hours", s.skyTimeOfDayHours);
  visit("sky_latitude", s.skyLatitude);
  visit("sky_longitude", s.skyLongitude);
  visit("sky_automatic_utc_offset", s.skyAutomaticUtcOffset);
  visit("sky_utc_offset_hours", s.skyUtcOffsetHours);
  visit("sky_year", s.skyYear);
  visit("sky_month", s.skyMonth);
  visit("sky_day", s.skyDay);
  visit("sky_wind_direction_deg", s.skyWindDirectionDeg);
  visit("sky_wind_speed", s.skyWindSpeed);
  visit("sky_cloud_coverage", s.skyCloudCoverage);
  visit("sky_cloud_density", s.skyCloudDensity);
  visit("sky_cloud_altitude_m", s.skyCloudAltitudeMeters);
  visit("sky_cloud_thickness_m", s.skyCloudThicknessMeters);
  visit("sky_cloud_shadow_strength", s.skyCloudShadowStrength);
  visit("sky_cloud_scale", s.skyCloudScale);
  visit("sky_cloud_animation_speed", s.skyCloudAnimationSpeed);
  visit("sky_cloud_quality", s.skyCloudQuality, cloudQualityIndexFromName, cloudQualityName);
  visit("sky_precipitation_rate", s.skyPrecipitationRate);
  visit("sky_rain_occlusion_strength", s.skyRainOcclusionStrength);
  visit("sky_snow_coverage", s.skySnowCoverage);
  visit("sky_humidity", s.skyHumidity);
  visit("sky_wetness", s.skyWetness);
  visit("sky_wetness_accumulation", s.skyWetnessAccumulationEnabled);
  visit("sky_wetness_accumulation_rate", s.skyWetnessAccumulationRate);
  visit("sky_wetness_drying_rate", s.skyWetnessDryingRate);
  visit("sky_lightning_rate", s.skyLightningRate);
  visit("sky_fog_density", s.skyFogDensity);
  visit("sky_visibility_m", s.skyVisibilityMeters);
  visit("sky_fog_color", s.skyFogColor);
  visit("sky_fog_anisotropy", s.skyFogAnisotropy);
  visit("sky_air_turbidity", s.skyAirTurbidity);
  visit("sky_ground_albedo", s.skyGroundAlbedo);
  visit("sky_use_explicit_sun_angles", s.skyUseExplicitSunAngles);
  visit("sky_sun_azimuth_deg", s.skySunAzimuthDeg);
  visit("sky_sun_elevation_deg", s.skySunElevationDeg);
  visit("sky_moon_size", s.skyMoonSize);
  visit("sky_lens_droplets_enabled", s.skyLensDropletsEnabled);
  visit("sky_lens_droplet_strength", s.skyLensDropletStrength);
  visit("reflective_ground", s.reflectiveGround);
  visit("reflective_ground_roughness", s.reflectiveGroundRoughness);
  visit("reflective_ground_metallic", s.reflectiveGroundMetallic);
  visit("shadowed_light_budget", s.shadowedLightBudget);
  visit("max_point_shadow_lights", s.maxPointShadowLights);
  visit("additional_shadow_resolution_scale", s.additionalShadowResolutionScale);
  visit("point_shadow_resolution_scale", s.pointShadowResolutionScale);
  visit("min_additional_shadow_resolution", s.minAdditionalShadowResolution);
  visit("update_shadows_every_frame", s.updateShadowsEveryFrame);
  visit("max_additional_lights_per_frame", s.maxAdditionalLightsPerFrame);
  visit("min_additional_light_influence", s.minAdditionalLightInfluence);
  visit("auto_select_imported_shadow_light", s.autoSelectImportedShadowLight);
  visit("sort_transparent_instances", s.sortTransparentInstances);
  visit("add_viewer_fill_lights", s.addViewerFillLights);
  visit("ui_scale", s.uiScale);
  visit("ui_scale_user_set", s.uiScaleUserSet);
  visit("show_collapsed_logo", s.showCollapsedLogo);
  visit("tcp_update_rate_hz", s.tcpUpdateRateHz);
}

using SettingIndexFromName = int (*)(const std::string&, int);
using SettingName = const char* (*)(int);

/** Assigns the setting whose key matches, keeping the old value when unparsable. */
struct SettingLoader {
  const std::string& key;
  const std::string& value;
  bool matched = false;

  bool hit(const char* name) {
    if (matched || key != name) {
      return false;
    }
    matched = true;
    return true;
  }
  void operator()(const char* name, bool& field) {
    if (hit(name)) field = parseBoolValue(value, field);
  }
  void operator()(const char* name, int& field) {
    if (hit(name)) field = parseIntValue(value, field);
  }
  void operator()(const char* name, float& field) {
    if (hit(name)) field = parseFloatValue(value, field);
  }
  void operator()(const char* name, glm::vec3& field) {
    if (hit(name)) field = parseVec3Value(value, field);
  }
  void operator()(const char* name, glm::vec4& field) {
    if (hit(name)) field = parseVec4Value(value, field);
  }
  void operator()(const char* name, int& field, SettingIndexFromName indexFromName, SettingName) {
    if (hit(name)) field = indexFromName(value, field);
  }
};

/** Writes one `key: value` line per setting. */
struct SettingWriter {
  std::ostream& output;

  template <typename T>
  void operator()(const char* name, const T& field) {
    output << name << ": " << field << "\n";
  }
  void operator()(const char* name, const glm::vec3& field) {
    output << name << ": " << field.r << ", " << field.g << ", " << field.b << "\n";
  }
  void operator()(const char* name, const glm::vec4& field) {
    output << name << ": " << field.r << ", " << field.g << ", " << field.b << ", " << field.a
           << "\n";
  }
  void operator()(const char* name, const int& field, SettingIndexFromName, SettingName toName) {
    output << name << ": " << toName(field) << "\n";
  }
};

} // namespace

void loadViewerSettings(ViewerSettings& settings) {
  std::ifstream input(settingsFilePath());
  if (!input) {
    return;
  }

  std::string line;
  while (std::getline(input, line)) {
    for (size_t i = 0; i < line.size(); ++i) {
      if (line[i] == '#' && (i == 0 || line[i - 1] == ' ' ||
                             line[i - 1] == '\t')) {
        line.resize(i);
        break;
      }
    }
    const auto sep = line.find(':');
    if (sep == std::string::npos) {
      continue;
    }
    const std::string key = trimAscii(line.substr(0, sep));
    const std::string value = trimAscii(line.substr(sep + 1));
    SettingLoader loader{key, value};
    forEachScalarSetting(settings, loader);
    if (loader.matched) {
      continue;
    }
    if (key == "recent_connection") {
      ConnectionEntry entry;
      if (parseConnectionLabel(value, entry)) {
        auto& connections = settings.recentConnections;
        if (std::find_if(connections.begin(), connections.end(),
              [&](const ConnectionEntry& existing) {
                return existing.host == entry.host && existing.port == entry.port;
              }) == connections.end() && connections.size() < 8)
          connections.push_back(entry);
      }
    }
    else if (key == "pane_layout") {
      settings.paneLayout = value;
    }
    else if (key == "pane_endpoint") {
      // "<pane id> <host>:<port>"
      const size_t space = value.find(' ');
      ConnectionEntry entry;
      if (space != std::string::npos && parseConnectionLabel(value.substr(space + 1), entry)) {
        PanePlacement placement;
        placement.pane = static_cast<uint32_t>(std::strtoul(value.substr(0, space).c_str(), nullptr, 10));
        placement.endpoint = entry;
        if (placement.pane != 0) settings.panePlacements.push_back(placement);
      }
    }
    else if (key == "resource_dir") {
      auto& dirs = settings.resourceDirs;
      if (!value.empty() &&
          std::find(dirs.begin(), dirs.end(), value) == dirs.end() &&
          dirs.size() < 24)
        dirs.push_back(value);
    }
  }
  sanitizeViewerSettings(settings);
}

namespace detail {
bool writeSettingsFileAtomically(
    const std::filesystem::path& path,
    const std::function<void(std::ostream&)>& write) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) return false;
  static std::atomic<uint64_t> sequence{0};
  auto temporary = path;
  temporary += ".tmp." + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count()) + "." +
      std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) return false;
  try {
    write(output);
  } catch (...) {
    output.close();
    std::filesystem::remove(temporary, ec);
    return false;
  }
  output.flush();
  const bool written = static_cast<bool>(output);
  output.close();
  if (!written || !output) {
    std::filesystem::remove(temporary, ec);
    return false;
  }
#if defined(_WIN32)
  const bool replaced = MoveFileExW(temporary.c_str(), path.c_str(),
                                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  std::filesystem::rename(temporary, path, ec);
  const bool replaced = !ec;
#endif
  if (!replaced) std::filesystem::remove(temporary, ec);
  return replaced;
}
} // namespace detail

void writeViewerSettings(std::ostream& output, const ViewerSettings& settings) {
  output << "# rayrai TCP viewer settings\n";
  output << std::boolalpha << std::setprecision(6);
  forEachScalarSetting(settings, SettingWriter{output});
  for (const auto& entry : settings.recentConnections) {
    output << "recent_connection: " << formatConnectionLabel(entry) << "\n";
  }
  if (!settings.paneLayout.empty()) {
    output << "pane_layout: " << settings.paneLayout << "\n";
  }
  for (const auto& placement : settings.panePlacements) {
    output << "pane_endpoint: " << placement.pane << " "
           << formatConnectionLabel(placement.endpoint) << "\n";
  }
  for (const auto& dir : settings.resourceDirs) {
    output << "resource_dir: " << dir << "\n";
  }
}

void saveViewerSettings(const ViewerSettings& settings) {
  const auto path = settingsFilePath();
  if (!detail::writeSettingsFileAtomically(path, [&](std::ostream& output) {
        writeViewerSettings(output, settings);
      })) {
    std::cerr << "WARN: Failed to write " << path << "\n";
  }
}

} // namespace raisin::tcp_viewer
