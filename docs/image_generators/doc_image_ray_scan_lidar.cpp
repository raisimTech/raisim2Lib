// Records ray_scan_lidar's Husky, terrain, and range-colored ray hits for the front-page gallery.
// PNG frames are composited into a GIF by CMake using ffmpeg.

#include "doc_image_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {

constexpr int kWidth = 960;
constexpr int kHeight = 600;
constexpr int kFrames = 100;
constexpr int kStepsPerFrame = 40;  // 25 frames per second at a 1 ms physics step
constexpr int kScanRows = 40;
constexpr int kScanColumns = 50;
const double kPi = std::acos(-1.);

size_t updateScan(raisim::World& world, const raisim::ArticulatedSystem& robot,
                  raisin::InstancedVisuals& scans) {
  raisim::Vec<3> position;
  raisim::Mat<3, 3> orientation;
  robot.getFramePosition("imu_joint", position);
  robot.getFrameOrientation("imu_joint", orientation);
  size_t hits = 0;
  for (int i = 0; i < kScanRows; ++i) {
    for (int j = 0; j < kScanColumns; ++j) {
      const double yaw = j * kPi / kScanColumns * 0.6 - 0.3 * kPi;
      const double pitch = -(i * 0.3 / kScanRows) + 0.2;
      const double normInv = 1. / std::sqrt(pitch * pitch + 1.);
      const Eigen::Vector3d localDirection(std::cos(yaw) * normInv, std::sin(yaw) * normInv,
                                           -pitch * normInv);
      const Eigen::Vector3d direction = orientation.e() * localDirection;
      const auto& collisions = world.rayTest(position.e(), direction, 30.);
      const size_t index = size_t(i * kScanColumns + j);
      if (collisions.size() > 0) {
        const auto& hit = collisions[0].getPosition();
        scans.setPosition(index, glm::vec3(float(hit[0]), float(hit[1]), float(hit[2])));
        scans.setColorWeight(index, std::min(float((hit - position.e()).norm()) / 15.f, 1.f));
        ++hits;
      } else {
        scans.setPosition(index, glm::vec3(0.f, 0.f, 100.f));
      }
    }
  }
  return hits;
}

void positionCamera(raisin::Camera& camera, const raisim::ArticulatedSystem& robot) {
  raisim::Vec<3> position;
  raisim::Mat<3, 3> orientation;
  robot.getFramePosition("imu_joint", position);
  robot.getFrameOrientation("imu_joint", orientation);
  // imu_joint faces backward relative to the chassis. Frame the sensor's forward scan,
  // keeping the camera level as the robot rolls and pitches over the terrain.
  Eigen::Vector3d forward = orientation.e().col(0);
  forward[2] = 0.;
  forward.normalize();
  const Eigen::Vector3d side = Eigen::Vector3d::UnitZ().cross(forward);
  const Eigen::Vector3d eye = position.e() - 1.5 * forward - 2.5 * side + 1.4 * Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d target = position.e() + 1.5 * forward - 0.3 * Eigen::Vector3d::UnitZ();
  const auto toGlm = [](const Eigen::Vector3d& v) { return glm::vec3(float(v[0]), float(v[1]), float(v[2])); };
  doc_image::setCameraLookAt(camera, toGlm(eye), toGlm(target), 55.f);
}

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  const auto rsc = argc >= 3 ? std::filesystem::path(argv[2])
                             : std::filesystem::weakly_canonical(outputDir / ".." / ".." / "..");
  const char* framesOverride = std::getenv("RAISIM_DOC_RAY_SCAN_LIDAR_FRAMES_DIR");
  const auto framesDir = framesOverride ? std::filesystem::path(framesOverride)
                                       : outputDir / "ray_scan_lidar_frames";
  std::filesystem::create_directories(framesDir);
  if (std::filesystem::exists(rsc / "activation.raisim"))
    raisim::World::setActivationKey((rsc / "activation.raisim").string());

  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_ray_scan_lidar")) doc_image::finishAndExit(1);

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(0.001);
  raisim::TerrainProperties terrain;
  terrain.frequency = 0.2;
  terrain.zScale = 2.;
  terrain.xSize = terrain.ySize = 70.;
  terrain.xSamples = terrain.ySamples = 70;
  terrain.fractalOctaves = 3;
  terrain.fractalLacunarity = 2.;
  terrain.fractalGain = 0.25;
  world->addHeightMap(0., 0., terrain)->setAppearance("soil2");
  auto* robot = world->addArticulatedSystem((rsc / "husky/husky.urdf").string());
  robot->setName("smb");
  Eigen::VectorXd gc = Eigen::VectorXd::Zero(robot->getGeneralizedCoordinateDim());
  gc.head<7>() << 0., 0., 2., 1., 0., 0., 0.;
  robot->setGeneralizedCoordinate(gc);
  robot->setGeneralizedVelocity(Eigen::VectorXd::Zero(robot->getDOF()));
  Eigen::VectorXd damping = Eigen::VectorXd::Zero(robot->getDOF());
  damping.tail<4>().setConstant(1.);
  robot->setJointDamping(damping);
  const auto step = [&] {
    world->integrate();
    robot->setGeneralizedForce({0., 0., 0., 0., 0., 0., -20., -20., -20., -20.});
    gc = robot->getGeneralizedCoordinate().e();
    if (std::fabs(gc[0]) > 35. || std::fabs(gc[1]) > 35.) {
      gc.head<7>() << 0., 0., 2., 1., 0., 0., 0.;
      robot->setState(gc, Eigen::VectorXd::Zero(robot->getDOF()));
    }
  };
  for (int i = 0; i < 1000; ++i) step();

  raisin::RayraiWindow renderer(world, kWidth, kHeight);
  const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrToneMapping = true;
  quality.pbrExposure = 0.8f;
  quality.mainLightDirection = glm::normalize(glm::vec3(-0.6f, -0.5f, -0.8f));
  doc_image::applyCommonSceneOptions(quality, preset);
  renderer.setRenderQualitySettings(quality);
  auto scans = renderer.addInstancedVisuals("scan points", raisim::Shape::Box, glm::vec3(0.05f),
                                           glm::vec4(1.f, 0.f, 0.f, 1.f), glm::vec4(0.f, 1.f, 0.f, 1.f));
  scans->resize(kScanRows * kScanColumns);
  scans->setDetectable(true);
  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;

  for (int frame = 0; frame < kFrames; ++frame) {
    if (!gc.allFinite()) {
      std::fprintf(stderr, "doc_image: Husky state is not finite at frame %d\n", frame);
      doc_image::finishAndExit(1);
    }
    const size_t hits = updateScan(*world, *robot, *scans);
    positionCamera(renderer.getCamera(), *robot);
    char name[64];
    std::snprintf(name, sizeof(name), "frame_%03d.png", frame);
    if (frame == 0) {
      if (!doc_image::captureScene(renderer, kWidth, kHeight, framesDir / name, 2, overrides))
        doc_image::finishAndExit(1);
    } else {
      renderer.update(kWidth, kHeight, /*save=*/false, /*saveW=*/0, /*saveH=*/0, /*headless=*/true);
      auto capture = renderer.captureSupersampledRgba(renderer.getCamera(), 2, overrides);
      if (capture.rgba.empty() || !doc_image::writePng(framesDir / name, capture.width, capture.height,
                                                     std::move(capture.rgba), /*flipVertical=*/false))
        doc_image::finishAndExit(1);
    }
    for (int i = 0; i < kStepsPerFrame; ++i) step();
    if ((frame + 1) % 25 == 0) {
      std::printf("doc_image: lidar frame %d/%d, %zu ray hits\n", frame + 1, kFrames, hits);
      std::fflush(stdout);
    }
  }
  std::printf("doc_image: wrote %d lidar frames into %s\n", kFrames, framesDir.string().c_str());
  doc_image::finishAndExit(0);
}
