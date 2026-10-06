// Rotated height maps for the HeightMap page: a rough map tilted into a slope
// and a bumpy map stood up as a wall at its foot. Bodies released on the slope
// slide and roll down and come to rest against the wall; one ball's descent is
// shown stroboscopically (light to dark).

#include "doc_image_common.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr double kTimestep = 0.001;

std::vector<double> roughHeights(size_t xSamples, size_t ySamples, double amplitude,
                                 double phase) {
  std::vector<double> heights;
  heights.reserve(xSamples * ySamples);
  for (size_t y = 0; y < ySamples; ++y)
    for (size_t x = 0; x < xSamples; ++x)
      heights.push_back(amplitude * (std::sin(0.55 * double(x) + phase) *
                                         std::cos(0.45 * double(y)) +
                                     0.4 * std::sin(1.7 * double(x + 2 * y))));
  return heights;
}

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_heightmap_rotation")) doc_image::finishAndExit(1);

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(kTimestep);
  world->addGround(-0.6, "ground")->setAppearance("checkerboard");

  // The slope: rotated 20 degrees about y, so it descends toward +x.
  auto* slope = world->addHeightMap(41, 25, 6.0, 3.6, 0.0, 0.0,
                                    roughHeights(41, 25, 0.035, 0.0), "terrain");
  slope->setPosition(0.0, 0.0, 1.0);
  slope->setOrientation(
      Eigen::Matrix3d(Eigen::AngleAxisd(20.0 * M_PI / 180.0, Eigen::Vector3d::UnitY())));
  slope->setAppearance("0.55, 0.62, 0.42, 1.0");

  // The wall: rotated -90 degrees about y, so its surface faces -x, up the slope.
  const double wallX = 2.95;
  auto* wall = world->addHeightMap(17, 25, 1.6, 3.6, 0.0, 0.0,
                                   roughHeights(17, 25, 0.05, 1.3), "terrain");
  wall->setPosition(wallX, 0.0, -0.05);
  wall->setOrientation(
      Eigen::Matrix3d(Eigen::AngleAxisd(-0.5 * M_PI, Eigen::Vector3d::UnitY())));
  wall->setAppearance("0.62, 0.52, 0.42, 1.0");

  // Bodies released along the top of the slope.
  const Eigen::Matrix3d slopeRotation = slope->getRotationMatrix();
  auto onSlope = [&](double x, double y, double clearance) {
    return Eigen::Vector3d(slope->getPosition() +
                           slopeRotation * Eigen::Vector3d(x, y, clearance));
  };
  auto place = [&](raisim::SingleBodyObject* object, const Eigen::Vector3d& p,
                   const char* appearance) {
    object->setPosition(p.x(), p.y(), p.z());
    object->setOrientation(Eigen::Matrix3d(slopeRotation));
    object->setAppearance(appearance);
  };
  auto* ball = world->addSphere(0.16, 1.0, "body");
  place(ball, onSlope(-2.4, 0.0, 0.25), "0.95, 0.42, 0.12, 1.0");
  place(world->addBox(0.4, 0.3, 0.2, 1.0, "body"), onSlope(-1.5, -1.1, 0.2),
        "0.20, 0.45, 0.85, 1.0");
  place(world->addCylinder(0.15, 0.4, 1.0, "body"), onSlope(-2.1, 1.0, 0.25),
        "0.80, 0.20, 0.30, 1.0");
  place(world->addCapsule(0.1, 0.35, 1.0, "body"), onSlope(-0.4, 1.0, 0.3),
        "0.95, 0.80, 0.20, 1.0");
  place(world->addSphere(0.12, 1.0, "body"), onSlope(-0.9, -1.3, 0.2),
        "0.30, 0.75, 0.55, 1.0");
  world->setMaterialPairProp("terrain", "body", 0.35, 0.1, 0.1);

  // Simulate, recording the ball's path.
  std::vector<Eigen::Vector3d> trail;
  const int kSteps = 2600;
  for (int step = 0; step < kSteps; ++step) {
    if (step % 130 == 0) trail.push_back(ball->getPosition());
    world->integrate();
  }

  raisin::RayraiWindow renderer(world, kWidth, kHeight);
  const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrToneMapping = true;
  doc_image::applyCommonSceneOptions(quality, preset);
  renderer.setRenderQualitySettings(quality);

  for (size_t i = 0; i < trail.size(); ++i) {
    const float shade = 0.25f + 0.75f * float(i) / float(std::max<size_t>(1, trail.size() - 1));
    auto ghost = renderer.addVisualSphere("trail" + std::to_string(i), 0.16,
                                          glm::vec4(0.95f, 0.42f, 0.12f, 0.15f + 0.45f * shade));
    ghost->setPosition(trail[i].x(), trail[i].y(), trail[i].z());
    ghost->setDetectable(true);
  }

  doc_image::setCameraLookAt(renderer.getCamera(), glm::vec3(0.5f, -5.0f, 2.9f),
                             glm::vec3(0.35f, 0.1f, 0.55f), /*horizontalFovDeg=*/58.0f);
  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;
  overrides.drawCoordinateFrames = false;
  const bool ok = doc_image::captureScene(renderer, kWidth, kHeight,
                                          outputDir / "heightmap_rotation.png", 2, overrides);
  doc_image::finishAndExit(ok ? 0 : 1);
}
