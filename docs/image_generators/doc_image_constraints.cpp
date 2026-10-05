// Renders the robot figures of the closed-loop and mimic constraint documentation: Minitaur, the
// Strandbeest and the Robotiq 2F-85 gripper, from the models in rsc/.

#include "doc_image_common.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

constexpr int kWidth = 960;
constexpr int kHeight = 600;

struct Renderer {
  std::shared_ptr<raisim::World> world = std::make_shared<raisim::World>();
  std::unique_ptr<raisin::RayraiWindow> window;
  int markers = 0;

  Renderer() {
    window = std::make_unique<raisin::RayraiWindow>(world, kWidth, kHeight);
    const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
    auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
    quality.colorMode = raisin::ViewerColorMode::AcesApprox;
    quality.pbrToneMapping = true;
    doc_image::applyCommonSceneOptions(quality, preset);
    window->setRenderQualitySettings(quality);
    world->addGround(0., "ground")->setAppearance("checkerboard");
  }

  void sphere(const raisim::Vec<3>& at, double radius, const glm::vec4& color) {
    auto visual = window->addVisualSphere("marker" + std::to_string(markers++), radius, color);
    visual->setPosition(at[0], at[1], at[2]);
    visual->setDetectable(true);  // captures draw only detectable visuals
  }

  bool capture(const std::filesystem::path& path, const glm::vec3& eye, const glm::vec3& target, float fov = 45.f) {
    doc_image::setCameraLookAt(window->getCamera(), eye, target, fov);
    raisin::RayraiWindow::RenderOverrides overrides;
    overrides.doShadows = true;
    if (!doc_image::captureScene(*window, kWidth, kHeight, path, 2, overrides)) return false;
    std::printf("doc_image: wrote %s\n", path.string().c_str());
    return true;
  }
};

/// Minitaur at its nominal pose; markers at the four toes, where each leg's loop is closed
bool renderMinitaur(const std::filesystem::path& out, const std::filesystem::path& rsc) {
  Renderer renderer;
  auto* robot = renderer.world->addArticulatedSystem((rsc / "minitaur/minitaur.urdf").string());
  const auto toe = [&](const char* leg) {
    raisim::Vec<3> position;
    robot->getPosition(robot->getBodyIdx(std::string("lower_leg_") + leg + "R_link"), raisim::Vec<3>{0., 0., 0.2},
                       position);
    return position;
  };
  const std::array<const char*, 4> legs{"front_right", "front_left", "back_right", "back_left"};
  // stand it on its toes
  double lowest = 1e9;
  for (const char* leg : legs) lowest = std::min(lowest, toe(leg)[2]);
  Eigen::VectorXd gc = robot->getGeneralizedCoordinate().e();
  gc[2] -= lowest - 0.018;
  robot->setGeneralizedCoordinate(gc);
  for (const char* leg : legs) renderer.sphere(toe(leg), 0.018, glm::vec4(0.95f, 0.45f, 0.1f, 1.f));
  raisim::Vec<3> base;
  robot->getPosition(0, raisim::Vec<3>{0., 0., 0.}, base);
  return renderer.capture(out / "constraints_minitaur.png",
                          {float(base[0]) + 0.55f, float(base[1]) - 0.75f, float(base[2]) + 0.25f},
                          {float(base[0]), float(base[1]), float(base[2]) - 0.08f}, 40.f);
}

/// the Strandbeest walking, a second into the crank's turn
bool renderStrandbeest(const std::filesystem::path& out, const std::filesystem::path& rsc) {
  Renderer renderer;
  renderer.world->setTimeStep(0.002);
  renderer.world->setSleepingEnabled(false);
  auto* beest = renderer.world->addArticulatedSystem((rsc / "strandbeest/strandbeest.urdf").string());
  const size_t crank = beest->getGeneralizedVelocityIndex("joint_crossbar_crank");
  Eigen::VectorXd gain = Eigen::VectorXd::Zero(beest->getDOF()), target = gain;
  gain[crank] = 200.;
  target[crank] = 2. * M_PI;
  beest->setPdGains(Eigen::VectorXd::Zero(beest->getDOF()), gain);
  beest->setPdTarget(beest->getGeneralizedCoordinate().e(), target);
  for (int step = 0; step < 650; ++step) renderer.world->integrate();
  const auto& gc = beest->getGeneralizedCoordinate();
  const glm::vec3 center(float(gc[0]), float(gc[1]), 0.45f);
  return renderer.capture(out / "constraints_strandbeest.png", center + glm::vec3(1.6f, -2.0f, 1.0f), center, 50.f);
}

/// the Robotiq 2F-85 holding a box it has lifted
bool renderGripper(const std::filesystem::path& out, const std::filesystem::path& rsc) {
  Renderer renderer;
  auto& world = *renderer.world;
  world.setTimeStep(0.001);
  world.setSleepingEnabled(false);
  auto* gripper = world.addArticulatedSystem((rsc / "robotiq_2f85/robotiq_2f85_mimic.urdf").string());
  for (auto& body : gripper->getCollisionBodies()) body.setMaterial("pad");
  auto* box = world.addBox(0.04, 0.04, 0.06, 0.1, "box");
  box->setPosition(0., 0., 0.03);
  box->setAppearance("0.95,0.55,0.15,1.0");
  world.setMaterialPairProp("pad", "box", 1.0, 0.0, 0.001);
  world.setMaterialPairProp("ground", "box", 0.8, 0.0, 0.001);
  const size_t lift = gripper->getGeneralizedVelocityIndex("lift");
  const size_t knuckle = gripper->getGeneralizedVelocityIndex("robotiq_85_left_knuckle_joint");
  Eigen::VectorXd gc = gripper->getGeneralizedCoordinate().e();
  gc[lift] = 0.17;
  gripper->setGeneralizedCoordinate(gc);
  Eigen::VectorXd p = Eigen::VectorXd::Zero(gripper->getDOF()), d = p, target = p;
  p[lift] = 5000.;
  d[lift] = 200.;
  p[knuckle] = 10.;
  d[knuckle] = 0.2;
  gripper->setPdGains(p, d);
  // close for a second, then lift by 0.12 m over two seconds and hold
  for (int step = 0; step < 4000; ++step) {
    const double t = step * 1e-3;
    const double s = std::clamp((t - 1.) / 2., 0., 1.);
    target[knuckle] = std::min(t, 1.) * 0.75;
    target[lift] = 0.17 + 0.12 * s * s * (3. - 2. * s);
    gripper->setPdTarget(target, Eigen::VectorXd::Zero(gripper->getDOF()));
    world.integrate();
  }
  const glm::vec3 center(0.f, 0.f, float(box->getPosition()[2]) + 0.02f);
  return renderer.capture(out / "constraints_gripper.png", center + glm::vec3(0.26f, -0.3f, 0.12f), center, 40.f);
}

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  // the images go to rsc/docs/image/rayrai; the models are in rsc/ (or the second argument)
  const auto rsc = argc >= 3 ? std::filesystem::path(argv[2])
                             : std::filesystem::weakly_canonical(outputDir / ".." / ".." / "..");
  if (std::filesystem::exists(rsc / "activation.raisim"))
    raisim::World::setActivationKey((rsc / "activation.raisim").string());
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_constraints")) doc_image::finishAndExit(1);
  const bool ok = renderMinitaur(outputDir, rsc) && renderStrandbeest(outputDir, rsc) && renderGripper(outputDir, rsc);
  doc_image::finishAndExit(ok ? 0 : 1);
}
