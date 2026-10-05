// Actuators and the motor operating region (EM-MOR) of their geared DC motors, defined in the URDF.
//
// A fixed-base quadruped rig is actuated randomly. rsc/motorOperatingRegion/quadruped_rig.urdf
// puts an actuator on every joint with <actuator> elements that link actuator files: a geared hip
// abduction actuator, and hip and knee actuators whose knee motor also turns with the hip, like
// KAIST Hound. For the motor of each of the twelve actuators, the plots show the operating region
// in the motor's torque-speed plane (green), the
// Box-MOR rectangle of peak torque and no-load speed (dashed), and the operating points of the last
// second. The actuators are commanded with actuator torques from an explicit PD controller, and
// RaiSim clips them to the operating regions. Points outside the region by more than 1% of the peak
// torque are red and counted, so you can check that RaiSim keeps every motor inside its region.
// The controller knows nothing about these limits: it often commands more torque than the motors
// can produce, and RaiSim has to clip it. Untick "Enforce EM-MOR" to see where the same commands
// would go unclipped; the joint velocity limits of the rig (1.5x the no-load speed) keep that run
// bounded. Joint effort limits clip only setGeneralizedForce() and the built-in PD controller, which
// the example does not use.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "rayrai/example_common.hpp"
#include "rayrai_example_compat.hpp"
#include "example_resources.hpp"
#include "raisim/World.hpp"

namespace {

constexpr double kTolerance = 0.01;    // operating points outside by more than this x peak torque

// Random commands: position targets, velocity sweeps beyond the no-load speed, or raw torques.
enum class Mode : int { MIXED = 0, POSITION = 1, VELOCITY = 2, TORQUE = 3 };

// The actuators are commanded with actuator torques from an explicit PD controller: the motor
// operating region clips exactly these torques. Actuator i drives joint i of the rig.
class RandomActuator {
 public:
  explicit RandomActuator(raisim::ArticulatedSystem* robot)
      : robot_(robot), kp_(Eigen::VectorXd::Zero(robot->getDOF())), kd_(kp_), q_(kp_), u_(kp_), tau_(kp_) {}

  Mode mode = Mode::MIXED;
  float period = 0.4f;
  float scale = 1.f;

  // new random targets every period
  void update(double time) {
    if (time < nextChange_) return;
    std::uniform_real_distribution<double> unit(-1., 1.), jitter(0.5, 1.5);
    nextChange_ = time + period * jitter(rng_);
    const Mode active = mode == Mode::MIXED ? static_cast<Mode>(1 + int(rng_() % 3)) : mode;
    kp_.setZero();
    kd_.setZero();
    q_.setZero();
    u_.setZero();
    tau_.setZero();
    for (int i = 0; i < int(robot_->getDOF()); ++i) {
      const bool abduction = i % 3 == 0;  // has position limits, so it gets no velocity sweeps
      const Mode jointMode = abduction && active == Mode::VELOCITY ? Mode::POSITION : active;
      if (jointMode == Mode::POSITION) {
        kp_[i] = abduction ? 40. : 60.;
        kd_[i] = 1.;
        q_[i] = scale * unit(rng_) * (abduction ? 0.5 : 2.5);
      } else if (jointMode == Mode::VELOCITY) {
        // up to 1.5x the no-load speed: the voltage limit and, on reversal, regenerative braking
        kd_[i] = 3.;
        u_[i] = scale * unit(rng_) * 36.;
      } else {
        // up to 1.5x the peak torque at the joint
        tau_[i] = scale * unit(rng_) * 1.5 * 3. * (abduction ? 6. : 10.);
      }
    }
  }

  // the actuator torques of this step
  void apply() {
    const Eigen::VectorXd q = robot_->getGeneralizedCoordinate().e(), u = robot_->getGeneralizedVelocity().e();
    robot_->setActuatorTorques(tau_ + kp_.cwiseProduct(q_ - q) + kd_.cwiseProduct(u_ - u));
  }

  void restart() { nextChange_ = 0.; }

 private:
  raisim::ArticulatedSystem* robot_;
  Eigen::VectorXd kp_, kd_, q_, u_, tau_;
  std::mt19937 rng_{42};
  double nextChange_ = 0.;
};

// Operating points of one motor: the motor speed at the beginning of the step, at which RaiSim
// evaluates the region, and the applied motor torque.
struct Sample {
  float speed = 0.f, torque = 0.f;
  raisim::MotorSaturation saturation = raisim::MotorSaturation::NONE;
  bool outside = false;
};

struct MotorTrace {
  std::vector<Sample> ring = std::vector<Sample>(5000);
  size_t head = 0, count = 0, samples = 0, saturated = 0, outside = 0;
  double maxOutside = 0.;  // [Nm]

  void add(const raisim::ActuatorState& state, const raisim::DcMotorParameters& motor) {
    const double violation = raisim::motorOperatingRegionViolation(motor, state.motorSpeed, state.motorTorque);
    Sample& sample = ring[head];
    sample.speed = float(state.motorSpeed);
    sample.torque = float(state.motorTorque);
    sample.saturation = state.saturation;
    sample.outside = violation > kTolerance * motor.peakTorque;
    head = (head + 1) % ring.size();
    count = std::min(count + 1, ring.size());
    ++samples;
    saturated += state.saturation != raisim::MotorSaturation::NONE;
    outside += sample.outside;
    maxOutside = std::max(maxOutside, violation);
  }

  void reset() { *this = MotorTrace(); }
};

struct SimulationSettings {
  Mode mode = Mode::MIXED;
  float period = 0.4f;
  float scale = 1.f;
  double timeStep = 0.001;
  float busVoltage = 0.f;
  bool enforce = true;
  bool paused = false;
};

// Read-only motor data shared with the visualization.
struct MotorTelemetry {
  std::string name;
  raisim::DcMotorParameters motor;
  double fileVoltage = 0.;
  MotorTrace trace;
};

struct MotorStatistics {
  double worst = 0.;  // fraction of peak torque
  size_t outside = 0, samples = 0;
};

// RaiSim setup, actuator commands, integration, and measurement. No viewer or ImGui calls.
class MotorOperatingRegionSimulation {
 public:
  explicit MotorOperatingRegionSimulation(const std::string& modelPath)
      : world_(createWorld()), robot_(world_->addArticulatedSystem(modelPath)), actuator_(robot_) {
    const auto& names = robot_->getActuatorNames();
    const auto& actuators = robot_->getActuators();
    motors_.reserve(actuators.size());
    for (size_t m = 0; m < actuators.size(); ++m)
      motors_.push_back({names[m], actuators[m].motor, actuators[m].motor.busVoltage, {}});
    settings_.busVoltage = float(actuators[0].motor.busVoltage);
  }

  const std::shared_ptr<raisim::World>& world() const { return world_; }
  const SimulationSettings& settings() const { return settings_; }
  const std::vector<MotorTelemetry>& motors() const { return motors_; }

  void applySettings(const SimulationSettings& settings, bool resetStatistics) {
    if (settings.mode != settings_.mode) {
      actuator_.mode = settings.mode;
      actuator_.restart();
    }
    actuator_.period = settings.period;
    actuator_.scale = settings.scale;
    if (settings.timeStep != settings_.timeStep) {
      world_->setTimeStep(settings.timeStep);
      resetStatistics = true;
    }
    if (settings.busVoltage != settings_.busVoltage) {
      robot_->setBusVoltage(settings.busVoltage);  // all actuators at once
      const auto& actuators = robot_->getActuators();
      for (size_t m = 0; m < motors_.size(); ++m) motors_[m].motor = actuators[m].motor;
      resetStatistics = true;
    }
    if (settings.enforce != settings_.enforce) {
      robot_->setMotorOperatingRegionEnforced(settings.enforce);
      resetStatistics = true;
    }
    settings_ = settings;
    if (resetStatistics) resetTraces();
  }

  // Simulate in real time, with a cap on catch-up after a slow frame.
  void advance(double elapsed) {
    if (!settings_.paused) accumulator_ += std::min(0.05, elapsed);
    while (accumulator_ >= world_->getTimeStep()) {
      step();
      accumulator_ -= world_->getTimeStep();
    }
  }

  MotorStatistics statistics() const {
    MotorStatistics statistics;
    for (const auto& motor : motors_) {
      statistics.worst = std::max(statistics.worst, motor.trace.maxOutside / motor.motor.peakTorque);
      statistics.outside += motor.trace.outside;
      statistics.samples += motor.trace.samples;
    }
    return statistics;
  }

 private:
  static std::shared_ptr<raisim::World> createWorld() {
    auto world = std::make_shared<raisim::World>();
    world->setTimeStep(0.001);
    world->addGround(0., "ground");
    return world;
  }

  void step() {
    actuator_.update(world_->getWorldTime());
    actuator_.apply();
    world_->integrate();
    const auto& states = robot_->getActuatorStates();
    for (size_t m = 0; m < states.size(); ++m) motors_[m].trace.add(states[m], motors_[m].motor);
  }

  void resetTraces() {
    for (auto& motor : motors_) motor.trace.reset();
  }

  std::shared_ptr<raisim::World> world_;
  raisim::ArticulatedSystem* robot_;
  RandomActuator actuator_;
  SimulationSettings settings_;
  std::vector<MotorTelemetry> motors_;
  double accumulator_ = 0.;
};

// Rayrai scene and ImGui presentation.
constexpr double kTrailSeconds = 1.;

ImU32 sampleColor(const Sample& sample) {
  if (sample.outside) return IM_COL32(255, 70, 70, 255);
  if (sample.saturation == raisim::MotorSaturation::PEAK_TORQUE) return IM_COL32(255, 175, 60, 230);
  if (sample.saturation == raisim::MotorSaturation::VOLTAGE) return IM_COL32(205, 125, 255, 230);
  return IM_COL32(110, 175, 255, 200);
}

// fileVoltage fixes the speed axis, so that a lower bus voltage visibly shrinks the region.
void drawMotorPlot(const std::string& name, const raisim::DcMotorParameters& motor,
                   double fileVoltage, const MotorTrace& trace, size_t trail, ImVec2 size) {
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImVec2 lo(origin.x + 4.f, origin.y + ImGui::GetTextLineHeightWithSpacing());
  const ImVec2 hi(origin.x + size.x - 4.f, origin.y + size.y - 4.f);

  const double perVolt = motor.torquePerVolt(), peak = motor.peakTorque;
  const double kv = motor.velocityConstant;
  const double xRange = 1.12 * kv * (std::max(motor.busVoltage, fileVoltage) + peak / perVolt);
  const double yRange = 1.4 * peak;
  const auto toScreen = [&](double speed, double torque) {
    const double x = std::clamp(speed / xRange, -1., 1.), y = std::clamp(torque / yRange, -1., 1.);
    return ImVec2(float(0.5 * (lo.x + hi.x) + 0.5 * x * (hi.x - lo.x)),
                  float(0.5 * (lo.y + hi.y) - 0.5 * y * (hi.y - lo.y)));
  };

  const ImU32 axisColor = IM_COL32(120, 125, 135, 120), labelColor = IM_COL32(150, 155, 165, 255);
  draw->AddRectFilled(lo, hi, IM_COL32(22, 25, 31, 235), 3.f);
  draw->AddLine(toScreen(-xRange, 0.), toScreen(xRange, 0.), axisColor);
  draw->AddLine(toScreen(0., -yRange), toScreen(0., yRange), axisColor);

  // EM-MOR: the bus-voltage band cut by the peak torque, a parallelogram.
  const double volt = motor.busVoltage;
  const double margin = peak / perVolt;
  const ImVec2 region[4] = {
      toScreen(kv * (-volt - margin), peak), toScreen(kv * (volt - margin), peak),
      toScreen(kv * (volt + margin), -peak), toScreen(kv * (-volt + margin), -peak)};
  draw->AddConvexPolyFilled(region, 4, IM_COL32(70, 190, 105, 55));
  draw->AddPolyline(region, 4, IM_COL32(95, 215, 125, 255), ImDrawFlags_Closed, 1.5f);
  // Beyond the overspeed limit, where even full reverse voltage drives more than the peak current,
  // only the peak braking torque is left: the region continues as a line at -/+ peak torque.
  draw->AddLine(region[2], toScreen(xRange, -peak), IM_COL32(95, 215, 125, 255), 2.f);
  draw->AddLine(region[0], toScreen(-xRange, peak), IM_COL32(95, 215, 125, 255), 2.f);

  // Box-MOR: peak torque and no-load speed, dashed.
  const double noLoad = motor.noLoadSpeed();
  const ImVec2 box[4] = {toScreen(-noLoad, peak), toScreen(noLoad, peak), toScreen(noLoad, -peak),
                         toScreen(-noLoad, -peak)};
  for (int side = 0; side < 4; ++side) {
    const ImVec2 a = box[side], b = box[(side + 1) % 4];
    for (float t = 0.f; t < 1.f; t += 0.04f)
      draw->AddLine(ImVec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t),
                    ImVec2(a.x + (b.x - a.x) * (t + 0.02f), a.y + (b.y - a.y) * (t + 0.02f)),
                    IM_COL32(210, 210, 210, 150));
  }

  const size_t shown = std::min(trail, trace.count);
  for (size_t i = 0; i < shown; ++i) {
    const size_t index = (trace.head + trace.ring.size() - shown + i) % trace.ring.size();
    const Sample& sample = trace.ring[index];
    const ImVec2 p = toScreen(sample.speed, sample.torque);
    const float r = sample.outside ? 2.5f : 1.2f;
    draw->AddRectFilled(ImVec2(p.x - r, p.y - r), ImVec2(p.x + r, p.y + r), sampleColor(sample));
  }

  char text[160];
  std::snprintf(text, sizeof(text), "%s  max out %.2f%%  sat %.0f%%", name.c_str(),
                100. * trace.maxOutside / peak,
                trace.samples ? 100. * double(trace.saturated) / double(trace.samples) : 0.);
  draw->AddText(origin, trace.outside ? IM_COL32(255, 110, 110, 255) : IM_COL32(225, 228, 235, 255),
                text);
  std::snprintf(text, sizeof(text), "%.0f rad/s", xRange);
  draw->AddText(ImVec2(hi.x - ImGui::CalcTextSize(text).x - 2.f,
                       hi.y - ImGui::GetTextLineHeight() - 1.f), labelColor, text);
  std::snprintf(text, sizeof(text), "%.1f Nm", yRange);
  draw->AddText(ImVec2(0.5f * (lo.x + hi.x) + 3.f, lo.y + 1.f), labelColor, text);
  ImGui::Dummy(size);
}

void positionCamera(raisin::RayraiWindow& viewer) {
  auto& camera = viewer.getCamera();
  // The plots cover the right of the window; aim so that the rig sits in the free lower left.
  camera.position = glm::vec3(1.64f, -3.55f, 1.57f);
  camera.target = glm::vec3(1.445f, 0.44f, 1.374f);
  const auto direction = glm::normalize(camera.target - camera.position);
  camera.yaw = glm::degrees(std::atan2(direction.y, direction.x));
  camera.pitch = glm::degrees(std::asin(direction.z));
  camera.zoom = 45.f;
  camera.zNear = 0.03f;
  camera.zFar = 60.f;
  camera.setCameraFixedTarget(true);
  camera.setCameraFixedDistance(true);
  camera.update(false);
}

class MotorOperatingRegionVisualization {
 public:
  ~MotorOperatingRegionVisualization() {
    // Rayrai resources must be released while the OpenGL context is still alive.
    viewer_.reset();
    if (initialized_) app_.shutdown();
  }

  bool init(const std::shared_ptr<raisim::World>& world) {
    if (!app_.init("RaiSim motor operating region", 1600, 900)) return false;
    initialized_ = true;
    ImGui::GetIO().IniFilename = nullptr;  // keep ImGui scratch files out of the source tree
    for (auto* object : world->getObjList())
      if (object->getObjectType() == raisim::ObjectType::HALFSPACE)
        static_cast<raisim::Ground*>(object)->setAppearance("checkerboard");
    viewer_ = std::make_shared<raisin::RayraiWindow>(world, 1600, 900);
    viewer_->setRenderQualitySettings(raisin::RayraiWindow::defaultRenderQualitySettings(
        raisin::RayraiWindow::RenderQualityPreset::Balanced));
    raisim_examples::setRayraiBackgroundColorRgb255(*viewer_, {30, 34, 42, 255});
    raisim_examples::addRayraiBasicSceneLights(*viewer_);
    positionCamera(*viewer_);
    return true;
  }

  bool processEvents() {
    app_.processEvents();
    return !app_.quit;
  }

  void render(MotorOperatingRegionSimulation& simulation) {
    app_.beginFrame();
    app_.renderViewer(*viewer_);
    drawControlPanel(simulation);
    drawPlots(simulation);
    app_.endFrame();
  }

 private:
  void drawControlPanel(MotorOperatingRegionSimulation& simulation) {
    SimulationSettings settings = simulation.settings();
    int mode = int(settings.mode);
    int timeStep = settings.timeStep == 0.001 ? 0 : settings.timeStep == 0.0025 ? 1 : 2;
    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.88f);
    ImGui::Begin("Motor operating region", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    if (ImGui::Combo("Actuation", &mode,
                     "Mixed\0Random position targets\0Random velocity sweeps\0Random torques\0")) {
      settings.mode = static_cast<Mode>(mode);
    }
    ImGui::SliderFloat("Command period [s]", &settings.period, 0.05f, 2.f, "%.2f");
    ImGui::SliderFloat("Command scale", &settings.scale, 0.1f, 2.f, "%.2f");
    if (ImGui::Combo("Time step", &timeStep, "1 ms\0002.5 ms\0005 ms\0")) {
      settings.timeStep = timeStep == 0 ? 0.001 : timeStep == 1 ? 0.0025 : 0.005;
    }
    ImGui::SliderFloat("Bus voltage [V]", &settings.busVoltage, 6.f, 36.f, "%.1f");
    ImGui::Checkbox("Enforce EM-MOR", &settings.enforce);
    ImGui::SameLine();
    ImGui::Checkbox("Paused", &settings.paused);
    const bool resetStatistics = ImGui::Button("Reset statistics");
    simulation.applySettings(settings, resetStatistics);
    ImGui::Separator();
    const MotorStatistics statistics = simulation.statistics();
    ImGui::Text("%s: worst point %.2f%% of peak torque outside", settings.enforce ? "EM-MOR" : "unclipped",
                100. * statistics.worst);
    ImGui::TextColored(statistics.outside ? ImVec4(1.f, 0.45f, 0.45f, 1.f) : ImVec4(0.55f, 0.9f, 0.6f, 1.f),
                       "%zu of %zu samples outside by > %.0f%%", statistics.outside, statistics.samples,
                       100. * kTolerance);
    ImGui::TextDisabled("Green: EM-MOR. Dashed: Box-MOR (peak torque, no-load speed).\n"
                        "Points: blue inside, orange at the peak torque,\n"
                        "purple at the voltage limit, red outside.");
    ImGui::End();
  }

  void drawPlots(const MotorOperatingRegionSimulation& simulation) {
    const auto& motors = simulation.motors();
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    const float plotWidth = std::min(980.f, display.x * 0.62f);
    ImGui::SetNextWindowPos(ImVec2(display.x - plotWidth - 10.f, 10.f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(plotWidth, display.y - 20.f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.82f);
    ImGui::Begin("Motor torque [Nm] vs. motor speed [rad/s]", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    const int columns = 3, rows = int((motors.size() + columns - 1) / columns);
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const ImVec2 cell((available.x - (columns - 1) * ImGui::GetStyle().ItemSpacing.x) / columns,
                      (available.y - (rows - 1) * ImGui::GetStyle().ItemSpacing.y) / rows);
    const size_t trail = size_t(kTrailSeconds / simulation.settings().timeStep);
    for (size_t m = 0; m < motors.size(); ++m) {
      if (m % columns != 0) ImGui::SameLine();
      drawMotorPlot(motors[m].name, motors[m].motor, motors[m].fileVoltage, motors[m].trace, trail, cell);
    }
    ImGui::End();
  }

  ExampleApp app_;
  std::shared_ptr<raisin::RayraiWindow> viewer_;
  bool initialized_ = false;
};

}  // namespace

int main(int, char** argv) {
  MotorOperatingRegionSimulation simulation(
      exampleRscPath(argv[0], "motorOperatingRegion/quadruped_rig.urdf"));
  MotorOperatingRegionVisualization visualization;
  if (!visualization.init(simulation.world())) return -1;

  auto previous = std::chrono::steady_clock::now();
  while (visualization.processEvents()) {
    const auto now = std::chrono::steady_clock::now();
    simulation.advance(std::chrono::duration<double>(now - previous).count());
    previous = now;
    visualization.render(simulation);
  }

  return 0;
}
