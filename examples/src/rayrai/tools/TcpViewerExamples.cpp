// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "TcpViewerExamples.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>

#include <cstdlib>
#elif !defined(_WIN32)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "TcpViewerProcess.hpp"

namespace raisin::tcp_viewer {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

std::vector<ServerExample> parseServerExampleManifest(const std::string& text) {
  std::vector<ServerExample> examples;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.front() == '#') continue;
    const size_t first = line.find('\t');
    const size_t second = first == std::string::npos ? first : line.find('\t', first + 1);
    if (second == std::string::npos) continue;
    ServerExample example{line.substr(0, first), line.substr(first + 1, second - first - 1),
                          line.substr(second + 1)};
    if (example.name.empty() || example.name.find_first_of("/\\.") != std::string::npos) continue;
    if (example.title.empty()) example.title = example.name;
    examples.push_back(std::move(example));
  }
  std::vector<std::string> categories;
  for (const auto& example : examples)
    if (std::find(categories.begin(), categories.end(), example.category) == categories.end())
      categories.push_back(example.category);
  std::stable_sort(examples.begin(), examples.end(), [&](const ServerExample& a, const ServerExample& b) {
    return std::find(categories.begin(), categories.end(), a.category) <
           std::find(categories.begin(), categories.end(), b.category);
  });
  return examples;
}

std::string stripTerminalEscapes(const std::string& text) {
  std::string plain;
  plain.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '\x1b') {
      plain += text[i];
      continue;
    }
    size_t end = i + 1;
    if (end < text.size() && text[end] == '[') {
      ++end;
      while (end < text.size() && text[end] >= 0x20 && text[end] <= 0x3f) ++end;
    }
    i = end;
  }
  return plain;
}

#if !defined(__EMSCRIPTEN__)

namespace {

// Whether address is an IPv4 address of this computer: only those can be bound.
bool isLocalAddress(const std::string& address) {
  sockaddr_in local{};
  local.sin_family = AF_INET;
  if (inet_pton(AF_INET, address.c_str(), &local.sin_addr) != 1 || local.sin_addr.s_addr == htonl(INADDR_ANY))
    return false;
#if defined(_WIN32)
  WSADATA data;
  if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return false;
  const SOCKET probe = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  const bool bound = probe != INVALID_SOCKET &&
                     bind(probe, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) == 0;
  if (probe != INVALID_SOCKET) closesocket(probe);
  WSACleanup();
#else
  const int probe = socket(AF_INET, SOCK_DGRAM, 0);
  const bool bound = probe >= 0 && bind(probe, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) == 0;
  if (probe >= 0) close(probe);
#endif
  return bound;
}

}  // namespace

struct ExampleLauncher::Impl {
  fs::path directory;
  std::vector<ServerExample> examples;
  std::string source;
  ChildProcess process;
  fs::path log;
  int current = -1;
  State state = State::Idle;
  std::string host;
  int port = 0;
  int exitCode = -1;
  Clock::time_point started;

  fs::path executable(const ServerExample& example) const {
#if defined(_WIN32)
    return directory / (example.name + ".exe");
#else
    return directory / example.name;
#endif
  }
};

ExampleLauncher::ExampleLauncher(const fs::path& directory) : impl_(std::make_unique<Impl>()) {
  auto& p = *impl_;
  p.directory = directory;
  std::ifstream manifest(directory / kServerExampleManifest);
  if (!manifest) return;
  const std::string text{std::istreambuf_iterator<char>(manifest), std::istreambuf_iterator<char>()};
  for (auto& example : parseServerExampleManifest(text)) {
    std::error_code error;
    if (fs::is_regular_file(p.executable(example), error)) p.examples.push_back(std::move(example));
  }
  p.source = p.examples.empty() ? "None of the listed server examples is built yet" : directory.string();
}

ExampleLauncher::~ExampleLauncher() { stop(); }
const std::vector<ServerExample>& ExampleLauncher::examples() const { return impl_->examples; }
const std::string& ExampleLauncher::source() const { return impl_->source; }
ExampleLauncher::State ExampleLauncher::state() const { return impl_->state; }
const ServerExample* ExampleLauncher::example() const {
  return impl_->current < 0 ? nullptr : &impl_->examples[size_t(impl_->current)];
}
const std::string& ExampleLauncher::host() const { return impl_->host; }
int ExampleLauncher::port() const { return impl_->port; }
int64_t ExampleLauncher::pid() const { return impl_->process.pid(); }
int ExampleLauncher::exitCode() const { return impl_->exitCode; }
Clock::duration ExampleLauncher::age() const { return Clock::now() - impl_->started; }
std::string ExampleLauncher::output() const {
  return impl_->log.empty() ? std::string() : stripTerminalEscapes(fileTail(impl_->log, 16384));
}

bool ExampleLauncher::start(const std::string& name, std::string& error) {
  auto& p = *impl_;
  int index = -1;
  for (size_t i = 0; i < p.examples.size(); ++i)
    if (p.examples[i].name == name) index = int(i);
  if (index < 0) {
    error = "Unknown server example: " + name;
    return false;
  }
  stop();
  std::error_code ignored;
  p.log = fs::temp_directory_path(ignored) / ("rayrai-example-" + name + "-" +
                                              std::to_string(Clock::now().time_since_epoch().count()) + ".log");
  if (!p.process.start(p.executable(p.examples[size_t(index)]), {}, p.log, error)) {
    fs::remove(p.log, ignored);
    p.log.clear();
    return false;
  }
  p.current = index;
  p.state = State::Starting;
  p.started = Clock::now();
  return true;
}

void ExampleLauncher::stop() {
  auto& p = *impl_;
  p.process.stop();
  if (!p.log.empty()) {
    std::error_code ignored;
    fs::remove(p.log, ignored);
  }
  p.log.clear();
  p.current = -1;
  p.state = State::Idle;
  p.host.clear();
  p.port = 0;
  p.exitCode = -1;
}

void ExampleLauncher::terminate() {
  poll({});
  auto& p = *impl_;
  if (p.state != State::Starting && p.state != State::Running) return;
  p.process.stop();
  p.state = State::Stopped;
  p.host.clear();
  p.port = 0;
}

void ExampleLauncher::poll(const std::vector<DiscoveredServer>& beacons) {
  auto& p = *impl_;
  if (p.state != State::Starting && p.state != State::Running) return;
  if (!p.process.running()) {
    p.state = State::Exited;
    p.exitCode = p.process.exitCode();
    return;
  }
  if (p.state != State::Starting) return;
  const std::string pid = std::to_string(p.process.pid());
  for (const auto& beacon : beacons) {
    const auto id = beacon.metadata.find("pid");
    if (id == beacon.metadata.end() || id->second != pid) continue;
    p.host = "127.0.0.1";
    p.port = beacon.endpointPort;
    p.state = State::Running;
    return;
  }
}

#else

namespace {
EM_JS_DEPS(rayrai_examples, "$stringToNewUTF8,$UTF8ToString");
// The page's example runtime (web/viewer-examples.mjs) runs each example in its own worker.
EM_JS(char*, pageExampleManifest, (), {
  const examples = Module['rayraiExamples'];
  return stringToNewUTF8(examples ? examples['manifest']() : '');
});
EM_JS(int, pageStartExample, (const char* name), {
  return Module['rayraiExamples']['start'](UTF8ToString(name));
});
EM_JS(void, pageStopExample, (int id), { Module['rayraiExamples']['stop'](id); });
EM_JS(char*, pageExampleStatus, (int id), {
  return stringToNewUTF8(Module['rayraiExamples']['status'](id));
});
EM_JS(char*, pageExampleOutput, (int id), {
  return stringToNewUTF8(Module['rayraiExamples']['output'](id));
});

std::string takeText(char* text) {
  std::string value = text ? text : "";
  std::free(text);
  return value;
}
}  // namespace

struct ExampleLauncher::Impl {
  std::vector<ServerExample> examples;
  std::string source;
  int id = 0;
  int current = -1;
  State state = State::Idle;
  std::string host = "browser";
  int port = 0;
  int exitCode = -1;
  std::string output;
  Clock::time_point started;
};

ExampleLauncher::ExampleLauncher(const fs::path&) : impl_(std::make_unique<Impl>()) {
  impl_->examples = parseServerExampleManifest(takeText(pageExampleManifest()));
  if (!impl_->examples.empty()) impl_->source = "examples built for the browser";
}

ExampleLauncher::~ExampleLauncher() { stop(); }
const std::vector<ServerExample>& ExampleLauncher::examples() const { return impl_->examples; }
const std::string& ExampleLauncher::source() const { return impl_->source; }
ExampleLauncher::State ExampleLauncher::state() const { return impl_->state; }
const ServerExample* ExampleLauncher::example() const {
  return impl_->current < 0 ? nullptr : &impl_->examples[size_t(impl_->current)];
}
const std::string& ExampleLauncher::host() const { return impl_->host; }
int ExampleLauncher::port() const { return impl_->port; }
int64_t ExampleLauncher::pid() const { return impl_->id; }
int ExampleLauncher::exitCode() const { return impl_->exitCode; }
Clock::duration ExampleLauncher::age() const { return Clock::now() - impl_->started; }
std::string ExampleLauncher::output() const {
  return stripTerminalEscapes(impl_->id ? takeText(pageExampleOutput(impl_->id)) : impl_->output);
}

bool ExampleLauncher::start(const std::string& name, std::string& error) {
  auto& p = *impl_;
  int index = -1;
  for (size_t i = 0; i < p.examples.size(); ++i)
    if (p.examples[i].name == name) index = int(i);
  if (index < 0) {
    error = "Unknown server example: " + name;
    return false;
  }
  stop();
  p.id = pageStartExample(name.c_str());
  if (p.id <= 0) {
    p.id = 0;
    error = "The browser could not start " + name;
    return false;
  }
  p.current = index;
  p.state = State::Starting;
  p.started = Clock::now();
  return true;
}

void ExampleLauncher::stop() {
  auto& p = *impl_;
  if (p.id) pageStopExample(p.id);
  p.id = 0;
  p.current = -1;
  p.state = State::Idle;
  p.port = 0;
  p.exitCode = -1;
  p.output.clear();
}

void ExampleLauncher::terminate() {
  poll({});
  auto& p = *impl_;
  if (p.state != State::Starting && p.state != State::Running) return;
  p.output = takeText(pageExampleOutput(p.id));
  pageStopExample(p.id);
  p.id = 0;
  p.state = State::Stopped;
  p.port = 0;
}

void ExampleLauncher::poll(const std::vector<DiscoveredServer>&) {
  auto& p = *impl_;
  if (p.state != State::Starting && p.state != State::Running) return;
  std::istringstream fields(takeText(pageExampleStatus(p.id)));
  std::string state;
  int port = 0, exitCode = -1;
  fields >> state >> port >> exitCode;
  if (state == "exited") {
    p.state = State::Exited;
    p.exitCode = exitCode;
  } else if (state == "running" && port > 0) {
    p.port = port;
    p.state = State::Running;
  }
}

#endif

bool ExampleLauncher::active() const { return state() == State::Starting || state() == State::Running; }

bool ExampleLauncher::conflictsWith(const std::string& host, int port,
                                    const std::vector<DiscoveredServer>& beacons) const {
  return active() && !serves(host, port, beacons);
}

bool ExampleLauncher::serves(const std::string& host, int port,
                             const std::vector<DiscoveredServer>& beacons) const {
  if (state() != State::Running || port != this->port()) return false;
  if (host == this->host()) return true;
#if defined(__EMSCRIPTEN__)
  (void)beacons;
  return false;
#else
  if (!isLocalAddress(host)) return false;
  const std::string id = std::to_string(pid());
  for (const auto& beacon : beacons) {
    const auto beaconPid = beacon.metadata.find("pid");
    if (beacon.endpointHost == host && beacon.endpointPort == port && beaconPid != beacon.metadata.end() &&
        beaconPid->second == id) {
      return true;
    }
  }
  return false;
#endif
}

}  // namespace raisin::tcp_viewer
