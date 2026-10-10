// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// A fixed-base Kinova arm carries its wrist-mounted Robotiq 2F-85 through
// predefined joint poses. Finger-pad contacts grasp a cube on the blue marker,
// carry it to the green marker, and release it. The world resets and repeats.

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
#include "robotiq_pick_and_place.hpp"

namespace {

volatile std::sig_atomic_t stop_requested = 0;
void RequestStop(int) { stop_requested = 1; }

struct Options {
  bool headless = false;
  bool benchmark = false;
  bool help = false;
  std::size_t cycles = 0;
  int port = 8080;
  std::string activation_key;
};

unsigned long PositiveInteger(const std::string& text, unsigned long maximum) {
  if (text.empty() ||
      text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("Expected a positive integer, got '" + text +
                                "'");
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
    } else if (arg == "--benchmark") {
      options.benchmark = true;
      options.headless = true;
    } else if (arg == "--cycles" || arg == "--port" ||
               arg == "--activation-key") {
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
    options.cycles = options.benchmark ? 10 : 3;
  }
  return options;
}

void PrintResult(std::size_t cycle,
                 const raisim_examples::robotiq::CycleResult& result) {
  std::cout << "Cycle " << cycle << ": "
            << (result.IsSuccessful() ? "PASS" : "FAIL")
            << " lifted=" << result.lifted << " transported=" << result.carried
            << " released=" << result.released
            << " bilateral_grasp=" << result.bilateral_grasp
            << " placement_error_m="
            << (result.placed_position - raisim_examples::robotiq::kDestination)
                   .norm()
            << " max_mimic_error_rad=" << result.max_mimic_error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = ParseOptions(argc, argv);
    if (options.help) {
      std::cout << "Kinova/Robotiq repeating PD pick-and-place\n"
                   "  --headless          Check three cycles without a "
                   "renderer or server\n"
                   "  --benchmark         Time ten unpaced cycles on one "
                   "physics thread\n"
                   "  --cycles N          Stop after N cycles (viewer mode "
                   "repeats by default)\n"
                   "  --port N            RaisimServer port (default 8080)\n"
                   "  --activation-key P  Optional license path\n";
      return 0;
    }
    raisim::RaiSimMsg::setFatalCallback(
        [] { throw std::runtime_error("RaiSim fatal error"); });
    if (!options.activation_key.empty()) {
      raisim::World::setActivationKey(options.activation_key);
    }
    raisim_examples::robotiq::Scene scene(
        exampleRscPath(argv[0], "robotiq_2f85/kinova_robotiq.urdf"));
    std::signal(SIGINT, RequestStop);
    std::signal(SIGTERM, RequestStop);
    std::unique_ptr<raisim::RaisimServer> server;
    if (!options.headless) {
      server = std::make_unique<raisim::RaisimServer>(&scene.world());
      server->launchServer(options.port);
      server->setCameraPositionAndLookAt({1.25, -1.5, 1.15}, {0.23, 0.0, 0.42});
      raisim_examples::warnIfNoClientConnected(*server);
    }

    std::size_t steps = 0;
    std::size_t reported_cycles = 0;
    std::size_t previous_phase = raisim_examples::robotiq::kPhases.size();
    bool success = true;
    const auto begin = std::chrono::steady_clock::now();
    while (!stop_requested &&
           (options.cycles == 0 || scene.completed_cycles() < options.cycles)) {
      if (server) {
        RS_TIMED_LOOP(
            static_cast<int>(raisim_examples::robotiq::kTimeStep * 1e6))
        server->integrateWorldThreadSafe([&scene] { scene.Update(); });
        if (scene.phase_index() != previous_phase) {
          previous_phase = scene.phase_index();
          std::cout << raisim_examples::robotiq::kPhases[previous_phase].name
                    << '\n';
        }
      } else {
        scene.Step();
      }
      ++steps;
      if (scene.completed_cycles() != reported_cycles) {
        reported_cycles = scene.completed_cycles();
        success &= scene.last_result().IsSuccessful() &&
                   scene.last_result().max_mimic_error < 1e-7;
        if (!options.benchmark) {
          PrintResult(reported_cycles, scene.last_result());
        }
      }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    if (server) {
      server->killServer();
    }
    if (stop_requested) {
      return 130;
    }
    if (options.headless) {
      if (options.benchmark) {
        PrintResult(scene.completed_cycles(), scene.last_result());
      }
      std::cout << "cycles=" << scene.completed_cycles() << " steps=" << steps
                << " physics_threads=1 elapsed_s=" << seconds
                << " us_per_step=" << 1e6 * seconds / static_cast<double>(steps)
                << " real_time_factor="
                << static_cast<double>(steps) *
                       raisim_examples::robotiq::kTimeStep / seconds
                << '\n';
    }
    return success ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "robotiq_gripper_mimic: " << error.what() << '\n';
    return 1;
  }
}
