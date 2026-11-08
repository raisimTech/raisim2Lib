// Renders the figures of the fluid pages: the fluid pool example after its bodies came to rest, each at
// the depth and in the orientation its buoyancy holds it; a raft half a wave long with crates on
// RaiSim's own waves, drawn by rayrai where the physics moves them, as the raft spans a crest and a
// trough; and, stroboscopically (faint to solid), a plate and a ball falling into water and a floating
// capsule yawing as it glides; and ANYmal knocked over and carried off by a stream (fluid_stream.hpp).
#include "doc_image_common.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "fluid_pool.hpp"
#include "fluid_stream.hpp"

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr double kPi = 3.14159265358979323846;

std::shared_ptr<raisin::RayraiWindow> makeRenderer(raisim::World& world) {
  std::shared_ptr<raisim::World> shared(&world, [](raisim::World*) {});
  auto renderer = std::make_shared<raisin::RayraiWindow>(shared, kWidth, kHeight);
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::High);
  doc_image::applyCommonSceneOptions(quality, raisin::RayraiWindow::RenderQualityPreset::High);
  quality.reflectiveGround = false;
  quality.depthOfFieldEnabled = false;
  renderer->setRenderQualitySettings(quality);
  return renderer;
}

bool renderBuoyancy(const std::filesystem::path& output) {
  namespace pool = raisim_examples::fluid_pool;
  pool::Scene scene;
  while (scene.world().getWorldTime() < 12.0) scene.step();
  auto renderer = makeRenderer(scene.world());
  auto water = raisin::defaultWaterBody(raisin::WaterPreset::Pool);
  water.center = {0.0, 0.0, pool::kSurface};
  water.halfExtents = {static_cast<float>(pool::kHalfLength), static_cast<float>(pool::kHalfWidth)};
  water.depth = static_cast<float>(pool::kSurface);
  renderer->setWaterBody("pool", water);
  doc_image::setCameraLookAt(renderer->getCamera(), {0.9f, -2.6f, 2.5f}, {-0.3f, 0.3f, 0.65f}, 50.0f);
  return doc_image::captureScene(*renderer, kWidth, kHeight, output);
}

bool renderRaft(const std::filesystem::path& output) {
  constexpr double kWavelength = 8.0, kAmplitude = 0.4;
  raisim::World world;
  world.setTimeStep(0.002);
  raisim::FluidVolume sea;
  sea.medium = raisim::FluidMedium::seawater();
  sea.waves.push_back({{1.0, 0.0}, kWavelength, kAmplitude, 0.0});
  world.addFluidVolume(sea);
  world.setAmbientFluid(raisim::FluidMedium::air());
  auto* raft = world.addBox(0.5 * kWavelength, 1.6, 0.3, 400.0 * 0.5 * kWavelength * 1.6 * 0.3);
  raft->setPosition(0.0, 0.0, 0.0);
  raft->setAppearance("0.62,0.47,0.3,1");
  // a row of crates in front of the raft over one wave: their heights trace its profile
  for (int i = 0; i < 8; ++i) {
    const double side = 0.4;
    auto* crate = world.addBox(side, side, side, 500.0 * side * side * side);
    crate->setPosition(-3.5 + kWavelength * i / 8.0, -2.2, 0.0);
    crate->setAppearance("0.55,0.36,0.18,1");
  }
  // until the raft's middle sits on a falling node: its ends on a crest and a trough
  const double k = 2.0 * kPi / kWavelength, omega = std::sqrt(9.81 * k);
  const auto phase = [&] { return std::remainder(k * raft->getPosition()[0] - omega * world.getWorldTime(), 2.0 * kPi); };
  while (world.getWorldTime() < 4.0 || std::abs(phase() - 0.5 * kPi) > 0.02) world.integrate();
  auto renderer = makeRenderer(world);
  renderer->setWaterBody("sea", raisin::waterBodyForFluidVolume(sea, raisin::WaterPreset::Ocean));
  renderer->setWaterTimeSeconds(world.getWorldTime());
  const auto& center = raft->getPosition();
  doc_image::setCameraLookAt(renderer->getCamera(),
                             {static_cast<float>(center[0]), static_cast<float>(center[1]) - 9.0f, 1.3f},
                             {static_cast<float>(center[0]), static_cast<float>(center[1]) - 1.0f, -0.1f}, 45.0f);
  return doc_image::captureScene(*renderer, kWidth, kHeight, output);
}

// A pose of a body at one moment, for a stroboscopic figure.
struct Pose {
  Eigen::Vector3d position;
  Eigen::Vector4d quaternion;
};

Pose poseOf(raisim::SingleBodyObject& body) {
  const auto q = body.getQuaternion();
  return {body.getPosition(), Eigen::Vector4d(q[0], q[1], q[2], q[3])};
}

// Draws a body at each recorded pose, from faint to solid; make draws one copy by name and colour.
template <typename Make>
void drawStrobe(const std::vector<Pose>& poses, const glm::vec3& color, const Make& make) {
  for (size_t i = 0; i < poses.size(); ++i) {
    const float shade = poses.size() > 1 ? float(i) / float(poses.size() - 1) : 1.f;
    const glm::vec4 rgba(color * (0.55f + 0.45f * shade), 0.25f + 0.75f * shade);
    auto copy = make(i, rgba);
    copy->setPosition(poses[i].position.x(), poses[i].position.y(), poses[i].position.z());
    copy->setOrientation(poses[i].quaternion[0], poses[i].quaternion[1], poses[i].quaternion[2], poses[i].quaternion[3]);
    copy->setDetectable(true);
  }
}

raisim::FluidVolume stillWater() {
  raisim::FluidVolume water;
  water.medium = raisim::FluidMedium::water();
  return water;
}

// A renderer of a world that holds only a floor under a pool of clear water, for the strobes.
std::shared_ptr<raisin::RayraiWindow> poolRenderer(raisim::World& world, double depth,
                                                   raisin::WaterPreset preset = raisin::WaterPreset::Pool) {
  world.addGround(-depth)->setAppearance("checkerboard");
  auto renderer = makeRenderer(world);
  auto water = raisin::defaultWaterBody(preset);
  water.center = {0.0, 0.0, 0.0};
  water.halfExtents = {6.0f, 6.0f};
  water.depth = static_cast<float>(depth);
  renderer->setWaterBody("pool", water);
  return renderer;
}

// A plate and a ball of the same density dropped from the same height: the plate meets the water
// with its whole face and hands it its momentum at once; the ball plunges.
bool renderEntry(const std::filesystem::path& output) {
  raisim::World world;
  world.setTimeStep(0.001);
  world.addFluidVolume(stillWater());
  world.setAmbientFluid(raisim::FluidMedium::air());
  auto* plate = world.addBox(0.8, 0.8, 0.05, 300.0 * 0.8 * 0.8 * 0.05);
  plate->setPosition(-0.6, 0.0, 1.0);
  auto* ball = world.addSphere(0.15, 300.0 * 4.0 / 3.0 * kPi * 0.15 * 0.15 * 0.15);
  ball->setPosition(0.8, 0.0, 1.0);
  std::vector<Pose> plates, balls;
  for (int step = 0; step <= 750; ++step) {
    if (step % 50 == 0) {
      plates.push_back(poseOf(*plate));
      balls.push_back(poseOf(*ball));
    }
    world.integrate();
  }
  raisim::World stage;
  auto renderer = poolRenderer(stage, 0.8);
  drawStrobe(plates, glm::vec3(0.62f, 0.47f, 0.3f), [&](size_t i, const glm::vec4& rgba) {
    return renderer->addVisualBox("plate" + std::to_string(i), 0.8, 0.8, 0.05, rgba);
  });
  drawStrobe(balls, glm::vec3(0.9f, 0.2f, 0.12f), [&](size_t i, const glm::vec4& rgba) {
    return renderer->addVisualSphere("ball" + std::to_string(i), 0.15, rgba);
  });
  doc_image::setCameraLookAt(renderer->getCamera(), {0.1f, -3.4f, 0.3f}, {0.1f, 0.0f, 0.3f}, 50.0f);
  return doc_image::captureScene(*renderer, kWidth, kHeight, output);
}

// A floating capsule shoved along at an angle to its axis: the water it carries yaws it (the Munk
// moment) while the drag slows it and bends its path.
bool renderMunk(const std::filesystem::path& output) {
  raisim::World world;
  world.setTimeStep(0.002);
  world.addFluidVolume(stillWater());
  world.setAmbientFluid(raisim::FluidMedium::air());
  const double radius = 0.1, length = 1.4;
  auto* log = world.addCapsule(radius, length, 500.0 * kPi * radius * radius * (length + 4.0 / 3.0 * radius));
  const Eigen::Matrix3d lying =
      (Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(0.5 * kPi, Eigen::Vector3d::UnitY()))
          .toRotationMatrix();
  log->setPosition(-1.5, -0.6, 0.0);
  log->setOrientation(lying);
  log->setLinearVelocity({2.5, 0.0, 0.0});
  std::vector<Pose> poses;
  for (int step = 0; step <= 1200; ++step) {
    if (step % 80 == 0) poses.push_back(poseOf(*log));
    world.integrate();
  }
  raisim::World stage;
  auto renderer = poolRenderer(stage, 2.0, raisin::WaterPreset::CalmLake);
  drawStrobe(poses, glm::vec3(0.62f, 0.47f, 0.3f), [&](size_t i, const glm::vec4& rgba) {
    return renderer->addVisualCapsule("log" + std::to_string(i), radius, length, rgba);
  });
  const Eigen::Vector3d middle = 0.5 * (poses.front().position + poses.back().position);
  doc_image::setCameraLookAt(renderer->getCamera(),
                             {static_cast<float>(middle.x()) - 0.2f, static_cast<float>(middle.y()) - 1.6f, 4.6f},
                             {static_cast<float>(middle.x()), static_cast<float>(middle.y()), 0.0f}, 50.0f);
  return doc_image::captureScene(*renderer, kWidth, kHeight, output);
}

// ANYmal across a stream of 2 m/s, every half second from standing: the water tips it over and carries
// it off downstream.
bool renderStreamRobot(const std::filesystem::path& output, const std::string& urdf) {
  namespace stream = raisim_examples::fluid_stream;
  constexpr double kSpeed = 2.0;
  stream::RobotScene scene(urdf, kSpeed);
  std::vector<Eigen::VectorXd> poses;
  for (int step = 0; step <= 1750; ++step) {
    if (step % 250 == 0) poses.push_back(scene.robot()->getGeneralizedCoordinate().e());
    scene.step(kSpeed);
  }
  raisim::World stage;
  stream::addChannel(stage, stream::RobotScene::kDepth, 0.0);
  for (const auto& pose : poses) stage.addArticulatedSystem(urdf)->setGeneralizedCoordinate(pose);
  auto renderer = makeRenderer(stage);
  auto water = raisin::defaultWaterBody(raisin::WaterPreset::Pool);
  water.center = {0.5 * (stream::kStart + stream::kEnd), 0.0, stream::RobotScene::kDepth};
  water.halfExtents = {static_cast<float>(0.5 * (stream::kEnd - stream::kStart)), static_cast<float>(stream::kHalfWidth)};
  water.depth = static_cast<float>(stream::RobotScene::kDepth);
  renderer->setWaterBody("stream", water);
  const float middle = static_cast<float>(0.5 * (poses.front()[0] + poses.back()[0]));
  doc_image::setCameraLookAt(renderer->getCamera(), {middle - 0.6f, -4.6f, 3.6f}, {middle, 0.0f, 0.15f}, 55.0f);
  return doc_image::captureScene(*renderer, kWidth, kHeight, output);
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
  if (!gl.init("doc_image_fluids")) doc_image::finishAndExit(1);
  const bool ok = renderBuoyancy(outputDir / "fluid_buoyancy.png") && renderRaft(outputDir / "fluid_raft_waves.png") &&
                  renderEntry(outputDir / "fluid_entry.png") && renderMunk(outputDir / "fluid_munk.png") &&
                  renderStreamRobot(outputDir / "fluid_stream_anymal.png",
                                    std::filesystem::canonical(rsc / "anymal" / "urdf" / "anymal.urdf").string());
  if (!ok) std::fprintf(stderr, "doc_image_fluids: capture failed\n");
  doc_image::finishAndExit(ok ? 0 : 1);
}
