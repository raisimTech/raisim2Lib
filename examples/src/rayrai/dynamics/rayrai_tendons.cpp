// Interactive versions of the three tendon examples, using automatic rendering.
#include "rayrai/PhysicsSnapshot.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_compat.hpp"
#include "rayrai_example_utils.hpp"
#include "tendon_example_runner.hpp"

using namespace raisim_examples::tendons;
namespace {
void positionCamera(raisin::RayraiWindow& viewer, Kind kind) {
  auto& c = viewer.getCamera();
  // Frame the four stations closely enough to read their cable routes.
  c.position = kind == Kind::All ? glm::vec3(2.4, -10.4, 4.2) : (kind == Kind::Pulley ? glm::vec3(5., -10., 5.) : glm::vec3(3.6, -5.8, 3.6));
  c.target = kind == Kind::All ? glm::vec3(0.5, 0., 1.5) : glm::vec3(0., 0., 1.5);
  const auto direction = glm::normalize(c.target - c.position);
  c.yaw = glm::degrees(std::atan2(direction.y, direction.x));
  c.pitch = glm::degrees(std::asin(direction.z));
  c.zoom = 38.;
  c.zNear = 0.03;
  c.zFar = 80.;
  c.setCameraFixedTarget(true);
  c.setCameraFixedDistance(true);
  c.update(false);
}
void capture(raisin::RayraiWindow& viewer, const std::string& file) {
  const auto& camera = viewer.getCamera();
  if (!raisim_examples::saveTexturePng(viewer.getImageTexture(), camera.rtWidth(), camera.rtHeight(), file))
    throw std::runtime_error("cannot write screenshot: " + file);
  std::cout << "Screenshot " << std::filesystem::path(file) << '\n';
}
int run(const Options& options) {
  if (!options.key.empty()) World::setActivationKey(options.key);
  auto scene = std::make_unique<Scene>(options.kind);
  exportScene(*scene, options.exportXml);
  if (options.headless) return runHeadless(*scene, options.steps);
  ExampleApp app;
  if (!app.init("RaiSim tendon examples", 1280, 720, !options.hidden)) return 77;
  struct Cleanup {
    ExampleApp& app;
    ~Cleanup() { app.shutdown(); }
  } cleanup{app};
  // Keep ImGui scratch files out of the source directory.
  ImGui::GetIO().IniFilename = nullptr;
  // The renderer owns an empty world; physics reaches it only through copied snapshots.
  raisin::PhysicsSnapshotCapture snapshotCapture;
  auto viewer = std::make_shared<raisin::RayraiWindow>(std::make_shared<raisim::World>(), 1280, 720);
  viewer->setRenderQualitySettings(raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::Balanced));
  raisim_examples::setRayraiBackgroundColorRgb255(*viewer, {30, 34, 42, 255});
  raisim_examples::addRayraiBasicSceneLights(*viewer);
  positionCamera(*viewer, options.kind);
  int selected = static_cast<int>(options.kind), frames = 0, steps = 0;
  bool paused = false, oneStep = false;
  double accumulator = 0.;
  float speed = 1.;
  auto previous = std::chrono::steady_clock::now();
  while (!app.quit && (options.frames < 0 || frames < options.frames) && (options.steps < 0 || steps < options.steps)) {
    app.processEvents();
    if (app.quit) break;
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::min(0.05, std::chrono::duration<double>(now - previous).count());
    previous = now;
    if (!paused)
      accumulator += (options.frames >= 0 || options.hidden) ? 0.016 : elapsed * speed;
    else
      accumulator = 0.;
    if (oneStep) {
      accumulator = scene->world->getTimeStep();
      oneStep = false;
    }
    while (accumulator + 1e-12 >= scene->world->getTimeStep() && (options.steps < 0 || steps < options.steps)) {
      scene->step();
      ++steps;
      accumulator -= scene->world->getTimeStep();
    }
    // Capture after the physics batch. No World access is needed while drawing.
    const auto snapshot = snapshotCapture.capture(*scene->world);
    viewer->applyPhysicsSnapshot(snapshot);

    app.beginFrame();
    app.renderViewer(*viewer);
    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.88f);
    ImGui::Begin("Tendons", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    bool reset = ImGui::Combo("Scene", &selected, "Elastic springs\0Wrapped pulleys\0Coupled joints\0All scenes\0");
    ImGui::Checkbox("Paused", &paused);
    ImGui::SameLine();
    if (ImGui::Button("Single step")) {
      paused = true;
      oneStep = true;
    }
    ImGui::SameLine();
    reset = ImGui::Button("Reset") || reset;
    ImGui::SliderFloat("Speed", &speed, 0.1f, 2.f, "%.1fx");
    ImGui::Text("Simulation: %.3f s", scene->world->getWorldTime());
    ImGui::Separator();
    for (const auto& tendon : scene->world->getTendons()) ImGui::Text("%s: length %.3f, tension %.2f", tendon->getName().c_str(), tendon->getLength(), tendon->getTension());
    ImGui::TextUnformatted("Spatial cables render automatically.\nFixed tendons drive the joints without a spatial cable.");
    if (scene->kind == Kind::Coupling || scene->kind == Kind::All) ImGui::TextUnformatted("Orange A / turquoise B: delta(LB) = -0.65 * delta(LA).");
    ImGui::End();
    app.endFrame();
    ++frames;
    if (reset) {
      scene = std::make_unique<Scene>(static_cast<Kind>(selected));
      // The next snapshot replaces the old scene, including its automatic tendon visuals.
      positionCamera(*viewer, scene->kind);
      accumulator = 0.;
      steps = 0;
    }
  }
  size_t segments = 0;
  if (frames > 0) {
    for (const auto& tendon : scene->world->getTendons()) {
      auto visual = viewer->getTendonVisual(tendon->getName());
      if (tendon->getType() == Tendon::Type::Spatial && (!visual || visual->count() == 0)) throw std::runtime_error("missing automatic tendon visualization");
      if (visual) segments += visual->count();
    }
    if (options.frames >= 0 && scene->kind == Kind::Coupling && segments < 2) throw std::runtime_error("coupling example must display both coupled cables");
    if (!options.screenshot.empty()) capture(*viewer, options.screenshot);
    if (gl::glGetError() != gl::GL_NO_ERROR) throw std::runtime_error("OpenGL error in tendon example");
  }
  std::cout << "frames=" << frames << " steps=" << steps << " visual_segments=" << segments << '\n';
  return 0;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const auto options = parseOptions(argc, argv, Kind::All, true);
    if (options.help) {
      printHelp(true);
      return 0;
    }
    return run(options);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
