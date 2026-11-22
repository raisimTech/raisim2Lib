// This file is part of RaiSim. A valid RaiSim license is required.
#pragma once
#include <charconv>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <thread>

#include "raisim/RaisimServer.hpp"
#include "tendon_scenes.hpp"

namespace raisim_examples::tendons {
struct Options {
  bool help = false;
  int port = 8080;
  std::string key, exportXml;
  Kind kind;
};
inline Options parseOptions(int argc, char** argv, Kind kind, bool rayrai = false) {
  Options options;
  options.kind = kind;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      options.help = true;
      continue;
    }
    const auto separator = arg.find('=');
    const auto name = arg.substr(0, separator);
    if (name != "--activation-key" && name != "--export" && !(rayrai && name == "--scene") && !(!rayrai && name == "--port"))
      throw std::invalid_argument("unknown option: " + arg);
    const std::string value = separator != std::string::npos ? arg.substr(separator + 1) : (++i < argc ? std::string(argv[i]) : std::string());
    if (value.empty()) throw std::invalid_argument("missing value for " + name);
    if (name == "--activation-key")
      options.key = value;
    else if (name == "--export")
      options.exportXml = value;
    else if (name == "--scene")
      options.kind = parseKind(value);
    else {
      int number = 0;
      const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
      if (result.ec != std::errc() || result.ptr != value.data() + value.size() || number < 1 || number > 65535)
        throw std::invalid_argument("invalid value for " + name);
      options.port = number;
    }
  }
  return options;
}
inline void printHelp(bool rayrai = false) {
  std::cout << "--activation-key PATH     Use an explicit RaiSim license; otherwise use normal discovery.\n"
            << "--export PATH             Export the initial scene as native world XML.\n";
  if (rayrai)
    std::cout << "--scene elastic|pulley|coupling|all (default: all)\n";
  else
    std::cout << "--port N                  TCP port (default: 8080). Connect rayrai_tcp_viewer.\n";
}
inline void exportScene(const Scene& scene, const std::string& file) {
  if (file.empty()) return;
  const auto path = std::filesystem::absolute(file);
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  // The directory overload needs a trailing separator for embedded URDF files.
  scene.world->exportToXml(path.parent_path().generic_string() + "/", path.filename().string());
  std::cout << "Exported " << path << '\n';
}
inline volatile std::sig_atomic_t stopRequested = 0;
inline void stopSignal(int) { stopRequested = 1; }
inline int runServerExample(int argc, char** argv, Kind kind) {
  try {
    const auto options = parseOptions(argc, argv, kind);
    if (options.help) {
      printHelp();
      return 0;
    }
    if (!options.key.empty()) World::setActivationKey(options.key);
    Scene scene(kind);
    exportScene(scene, options.exportXml);
    std::signal(SIGINT, stopSignal);
    std::signal(SIGTERM, stopSignal);
    raisim::RaisimServer server(scene.world.get());
    server.launchServer(options.port);
    struct StopServer {
      raisim::RaisimServer& server;
      ~StopServer() { server.killServer(); }
    } guard{server};
    if (kind == Kind::Coupling)
      server.setCameraPositionAndLookAt({3.6, -5.8, 3.6}, {0., 0., 1.5});
    else
      server.setCameraPositionAndLookAt({7., -12., 6.}, {0., 0., 1.5});
    std::cout << "scene=" << kindName(kind) << " Connect rayrai_tcp_viewer to 127.0.0.1:" << server.getPort() << ". Tendons render automatically. Press Ctrl-C to stop.\n" << std::flush;
    if (kind == Kind::Coupling) std::cout << "Orange cable A and turquoise cable B obey delta(LB) = -0.65 * delta(LA).\n" << std::flush;
    while (!stopRequested) {
      const auto next = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
      // Viewer pause/step commands and the control callback share the World lock.
      server.integrateWorldThreadSafe([&] { scene.updateControls(); });
      std::this_thread::sleep_until(next);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace raisim_examples::tendons
