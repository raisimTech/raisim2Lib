#pragma once

#include <iostream>
#include <thread>
#include <chrono>

#include "raisim/RaisimServer.hpp"

namespace raisim_examples {

inline void printRayraiTcpViewerHint(int port = 8080) {
  std::cerr
    << "\n"
    << "[raisim] Tip: if you don't have a visualizer running, start the TCP viewer and connect to 127.0.0.1:" << port << "\n"
    << "        (from your build directory)\n"
    << "          cmake --build . --target rayrai_tcp_viewer\n"
#if defined(_WIN32)
    << "          rayrai_tcp_viewer.exe\n"
#else
    << "          ./rayrai_tcp_viewer\n"
#endif
    << std::endl;
}

inline void warnIfNoClientConnected(const raisim::RaisimServer& server,
                                    int port = 8080,
                                    double waitSeconds = 10.0) {
  // Connection can legitimately happen late (e.g., the client is still loading assets).
  // Don't block the simulation thread; warn once after a grace period if still disconnected.
  std::thread([&server, port, waitSeconds]() {
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + duration_cast<steady_clock::duration>(
      duration<double>(waitSeconds));

    while (steady_clock::now() < deadline) {
      if (server.isConnected())
        return;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (server.isConnected())
      return;

    std::cerr
      << "\n"
      << "[raisim] No visualizer is connected yet (127.0.0.1:" << port << ").\n"
      << "        Start RaisimUnity / RaisimUnreal / rayrai_tcp_viewer and connect to the server.\n";

    // Also show a concrete way to start a local viewer.
    printRayraiTcpViewerHint(port);
  }).detach();
}

}  // namespace raisim_examples
