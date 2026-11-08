// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// A fixed-base Kinova arm folds a cloth in half with its wrist-mounted Robotiq
// 2F-85. The pads hold one corner of the hanging cloth. The arm lays the lower
// half on the table, swings the held corner over onto the corner that touched
// down first, holds the fold, then unfolds the cloth and lifts it back up. The
// world resets and repeats.

#include <chrono>
#include <csignal>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "robotiq_cloth_fold.hpp"

namespace {

volatile std::sig_atomic_t stop_requested = 0;
void RequestStop(int) { stop_requested = 1; }

struct Options {
  bool headless = false;
  bool help = false;
  std::size_t cycles = 0;
  int port = 8080;
  std::string activation_key;
};

unsigned long PositiveInteger(const std::string& text, unsigned long maximum) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("Expected a positive integer, got '" + text + "'");
  }
  const unsigned long number = std::stoul(text);
  if (number == 0 || number > maximum) {
    throw std::invalid_argument("Integer out of range: '" + text + "'");
  }
  return number;
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      options.help = true;
    } else if (arg == "--headless") {
      options.headless = true;
    } else if (arg == "--cycles" || arg == "--port" || arg == "--activation-key") {
      if (++i == argc) {
        throw std::invalid_argument("Missing value for " + arg);
      }
      if (arg == "--cycles") {
        options.cycles = PositiveInteger(argv[i], 1000000);
      } else if (arg == "--port") {
        options.port = static_cast<int>(PositiveInteger(argv[i], 65535));
      } else {
        options.activation_key = argv[i];
      }
    } else {
      throw std::invalid_argument("Unknown option: " + arg);
    }
  }
  if (options.headless && options.cycles == 0) {
    options.cycles = 2;
  }
  return options;
}

void PrintResult(std::size_t cycle, const raisim_examples::cloth_fold::CycleResult& result) {
  std::cout << "Cycle " << cycle << ": " << (result.IsSuccessful() ? "PASS" : "FAIL") << " held=" << result.held << " laid=" << result.laid << " folded=" << result.folded
            << " lifted=" << result.lifted << " corner_gap_m=" << result.corner_gap << " lower_corner_slip_m=" << result.lower_corner_slip
            << " max_mimic_error_rad=" << result.max_mimic_error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  namespace cloth_fold = raisim_examples::cloth_fold;
  try {
    const auto options = ParseOptions(argc, argv);
    if (options.help) {
      std::cout << "Kinova/Robotiq cloth folding\n"
                   "  --headless          Check two cycles without a server\n"
                   "  --cycles N          Stop after N cycles (viewer mode repeats by default)\n"
                   "  --port N            RaisimServer port (default 8080)\n"
                   "  --activation-key P  Optional license path\n";
      return 0;
    }
    raisim::RaiSimMsg::setFatalCallback([] { throw std::runtime_error("RaiSim fatal error"); });
    if (!options.activation_key.empty()) {
      raisim::World::setActivationKey(options.activation_key);
    }
    cloth_fold::Scene scene(exampleRscPath(argv[0], "robotiq_2f85/kinova_robotiq.urdf"));
    std::signal(SIGINT, RequestStop);
    std::signal(SIGTERM, RequestStop);
    std::unique_ptr<raisim::RaisimServer> server;
    if (!options.headless) {
      server = std::make_unique<raisim::RaisimServer>(&scene.world());
      server->launchServer(options.port);
      server->setCameraPositionAndLookAt({1.0, -0.6, 0.65}, {0.42, 0.1, 0.25});
      raisim_examples::warnIfNoClientConnected(*server);
    }

    std::size_t steps = 0;
    std::size_t reported_cycles = 0;
    std::size_t previous_phase = cloth_fold::kPhases.size();
    bool success = true;
    const auto begin = std::chrono::steady_clock::now();
    while (!stop_requested && (options.cycles == 0 || scene.completed_cycles() < options.cycles)) {
      if (server) {
        RS_TIMED_LOOP(static_cast<int>(cloth_fold::kTimeStep * 1e6))
        server->integrateWorldThreadSafe([&scene] { scene.Update(); });
      } else {
        scene.Step();
      }
      if (scene.phase_index() != previous_phase) {
        previous_phase = scene.phase_index();
        std::cout << cloth_fold::kPhases[previous_phase].name << '\n';
      }
      ++steps;
      if (scene.completed_cycles() != reported_cycles) {
        reported_cycles = scene.completed_cycles();
        success &= scene.last_result().IsSuccessful() && scene.last_result().max_mimic_error < 1e-7;
        PrintResult(reported_cycles, scene.last_result());
      }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    if (server) {
      server->killServer();
    }
    if (stop_requested) {
      return 130;
    }
    if (options.headless) {
      std::cout << "cycles=" << scene.completed_cycles() << " steps=" << steps << " elapsed_s=" << seconds << " us_per_step=" << 1e6 * seconds / static_cast<double>(steps)
                << " real_time_factor=" << static_cast<double>(steps) * cloth_fold::kTimeStep / seconds << '\n';
    }
    return success ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "robotiq_cloth_fold: " << error.what() << '\n';
    return 1;
  }
}
