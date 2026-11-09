// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#include "TcpViewerSimulation.hpp"
#include "TcpViewerPathText.hpp"
#include "TcpViewerProcess.hpp"
#include "tinyxml_rai/tinyxml_rai.h"
#include <chrono>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace raisin::tcp_viewer {
namespace fs = std::filesystem;
namespace {
using Clock = std::chrono::steady_clock;
fs::path temporaryDirectory() {
  const auto stamp = Clock::now().time_since_epoch().count();
  for (int i = 0; i < 100; ++i) {
    auto path = fs::temp_directory_path() / ("rayrai-simulation-" + std::to_string(stamp) + "-" + std::to_string(i));
    if (fs::create_directory(path)) return path;
  }
  throw std::runtime_error("cannot create simulation working directory");
}
std::string readFailure(const fs::path& directory) {
  std::ifstream in(directory / "error.txt");
  std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  std::ifstream log(directory / "process.log");
  log.seekg(0, std::ios::end);
  const std::streamoff size = log.tellg();
  if (size > 0) {
    log.seekg(size > 1800 ? size - std::streamoff(1800) : std::streamoff(0));
    text += std::string(std::istreambuf_iterator<char>(log), std::istreambuf_iterator<char>());
  }
  if (text.size() > 2000) text = text.substr(text.size() - 2000);
  return text.empty() ? "simulation process exited before the world could be loaded" : text;
}
} // namespace

DroppedSceneKind classifyDroppedScene(const fs::path& path, std::string& error) {
  error.clear();
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) { error = "File not found: " + utf8PathText(path); return DroppedSceneKind::Invalid; }
  const std::string root = xmlRootElement(path);
  if (root.empty()) { error = "Cannot read scene XML: " + utf8PathText(path); return DroppedSceneKind::Invalid; }
  if (root == "raisim") return DroppedSceneKind::World;
  if (root == "robot" || root == "mujoco") return DroppedSceneKind::Robot;
  error = "Expected a RaiSim world, URDF robot, or MJCF model";
  return DroppedSceneKind::Invalid;
}

std::string xmlRootElement(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return {};
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  raisim::TiXmlDocument doc;
  doc.Parse(text.c_str());
  if (doc.Error() || !doc.RootElement()) return {};
  return doc.RootElement()->Value();
}

struct LocalSimulation::Impl {
  State state = State::Idle;
  fs::path directory, xml;
  std::string error;
  int port = 0;
  Clock::time_point started;
  ChildProcess process;
};

LocalSimulation::LocalSimulation() : impl_(std::make_unique<Impl>()) {}
LocalSimulation::~LocalSimulation() { stop(); }
LocalSimulation::State LocalSimulation::state() const { return impl_->state; }
bool LocalSimulation::active() const { return state() == State::Starting || state() == State::Running; }
int LocalSimulation::port() const { return impl_->port; }
const std::string& LocalSimulation::error() const { return impl_->error; }
const fs::path& LocalSimulation::worldPath() const { return impl_->xml; }

bool LocalSimulation::start(const fs::path& executable, const fs::path& xml,
                            const fs::path& activationKey, std::string& error) {
  if (classifyDroppedScene(xml, error) != DroppedSceneKind::World) {
    if (error.empty()) error = "Simulation requires a RaiSim world XML file";
    return false;
  }
  std::error_code ec;
  if (!fs::is_regular_file(executable, ec)) { error = "Viewer executable not found"; return false; }
  stop();
  auto& p = *impl_;
  try {
    p.directory = temporaryDirectory();
    p.xml = fs::absolute(xml);
    const fs::path key = activationKey.empty() ? fs::path{} : fs::absolute(activationKey);
    if (!p.process.start(executable, {"--xml-simulation-worker", p.xml, p.directory / "ready.txt", key},
                         p.directory / "process.log", error))
      throw std::runtime_error(error);
    p.state = State::Starting;
    p.started = Clock::now();
    error.clear();
    return true;
  } catch (const std::exception& e) {
    error = e.what(); stop(); p.state = State::Failed; p.error = error; return false;
  }
}

void LocalSimulation::poll() {
  auto& p = *impl_;
  if (!active()) return;
  if (!p.process.running()) {
    const auto failure = readFailure(p.directory);
    stop(); p.state = State::Failed; p.error = failure; return;
  }
  if (p.state == State::Starting) {
    int port = 0;
    std::ifstream ready(p.directory / "ready.txt");
    if (ready >> port && port > 0 && port <= 65535) { p.port = port; p.state = State::Running; }
    // Long model imports remain asynchronous but a hung child cannot wait forever.
    else if (Clock::now() - p.started > std::chrono::minutes(5)) {
      stop(); p.state = State::Failed; p.error = "Simulation startup timed out";
    }
  }
}

void LocalSimulation::stop() {
  auto& p = *impl_;
  p.process.stop();
  if (!p.directory.empty()) { std::error_code ignored; fs::remove_all(p.directory, ignored); }
  p.directory.clear(); p.xml.clear(); p.error.clear(); p.port = 0; p.state = State::Idle;
}

fs::path viewerExecutablePath(const char* argv0) {
#if defined(_WIN32)
  std::vector<wchar_t> path(32768);
  const DWORD size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (size && size < path.size()) return fs::path(std::wstring(path.data(), size));
#elif defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::vector<char> path(size);
  if (_NSGetExecutablePath(path.data(), &size) == 0) return fs::weakly_canonical(path.data());
#else
  std::error_code error;
  const auto path = fs::read_symlink("/proc/self/exe", error);
  if (!error) return path;
#endif
  return fs::absolute(argv0 ? argv0 : "rayrai_tcp_viewer");
}
} // namespace raisin::tcp_viewer
