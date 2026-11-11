// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <unistd.h>
#endif

#include "raisim/PathUtils.hpp"

namespace fem_workbench {
inline std::string environmentValue(const char* name) {
#if defined(_WIN32)
  const std::wstring wideName(name,
                              name + std::char_traits<char>::length(name));
  const DWORD size = GetEnvironmentVariableW(wideName.c_str(), nullptr, 0);
  if (size == 0) return {};
  std::wstring value(size, L'\0');
  const DWORD written =
      GetEnvironmentVariableW(wideName.c_str(), value.data(), size);
  if (written == 0 || written >= size) return {};
  value.resize(written);
  return raisim::pathUtf8(std::filesystem::path(value));
#else
  const char* value = std::getenv(name);
  return value ? value : "";
#endif
}
inline std::string findExecutable(const std::string& name,
                                  const std::string& searchPath) {
  if (name.empty()) return {};
  const auto executable = raisim::pathFromUtf8(name);
  const auto usable = [](const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) return false;
#if defined(_WIN32) || defined(__EMSCRIPTEN__)
    return true;
#else
    return access(path.c_str(), X_OK) == 0;
#endif
  };
  const auto resolve = [&](std::filesystem::path path) -> std::string {
#if defined(_WIN32)
    if (path.extension().empty()) path += L".exe";
#endif
    if (!usable(path)) return {};
    return raisim::pathUtf8(std::filesystem::absolute(path));
  };
  if (executable.has_parent_path()) return resolve(executable);
#if defined(_WIN32)
  constexpr char separator = ';';
#else
  constexpr char separator = ':';
#endif
  for (size_t begin = 0;;) {
    const auto end = searchPath.find(separator, begin);
    auto entry = searchPath.substr(begin, end - begin);
#if defined(_WIN32)
    if (entry.size() >= 2 && entry.front() == '"' && entry.back() == '"')
      entry = entry.substr(1, entry.size() - 2);
#endif
    auto found = resolve(raisim::pathFromUtf8(entry) / executable);
    if (!found.empty()) return found;
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return {};
}
inline std::string findBundledMesherExecutable(const char* application) {
  const auto searchPath = environmentValue("PATH");
  const auto app = findExecutable(application, searchPath);
  if (!app.empty()) {
    const auto directory = raisim::pathFromUtf8(app).parent_path();
    for (const auto& path : {directory / "fem-mesher/bin/raisim_fem_mesher",
                             directory / "raisim_fem_mesher"}) {
      const auto found = findExecutable(raisim::pathUtf8(path), "");
      if (!found.empty()) return found;
    }
  }
  return {};
}
inline std::string defaultNativeMesherExecutable(const char* application) {
  const auto requested = environmentValue("RAISIM_FEM_MESHER");
  if (!requested.empty()) return requested;
  const auto bundled = findBundledMesherExecutable(application);
  if (!bundled.empty()) return bundled;
  const auto searchPath = environmentValue("PATH");
  const auto found = findExecutable("raisim_fem_mesher", searchPath);
  return found.empty() ? "raisim_fem_mesher" : found;
}
}  // namespace fem_workbench
