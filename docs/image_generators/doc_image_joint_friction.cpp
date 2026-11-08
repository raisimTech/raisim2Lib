// Renders spherical_joint_friction.png for the Joint Damping and Friction page: a spherical pendulum
// with joint friction, released on a conical swing. The bob's positions (light to dark) spiral down,
// and the bob comes to rest off the vertical, inside the ring where gravity's torque equals the
// friction.

#include "doc_image_common.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 960;
constexpr int kHeight = 600;
constexpr double kPivotHeight = 1.5, kLength = 0.5, kMass = 1., kFriction = 0.6, kGravity = 9.81;

std::string pendulumUrdf() {
  const std::string inertia = "<inertia ixx=\"0.0004\" ixy=\"0\" ixz=\"0\" iyy=\"0.0004\" iyz=\"0\" izz=\"0.0004\"/>";
  return "<robot name=\"pendulum\"><link name=\"world\"/><joint name=\"mount\" type=\"fixed\"><parent link=\"world\"/>"
         "<child link=\"socket\"/><origin xyz=\"0 0 " + std::to_string(kPivotHeight) + "\"/></joint>"
         "<link name=\"socket\"><inertial><mass value=\"1\"/>" + inertia + "</inertial><visual><geometry>"
         "<sphere radius=\"0.03\"/></geometry><material name=\"joint_friction_socket\"><color rgba=\"0.25 0.26 0.28 1\"/>"
         "</material></visual></link><joint name=\"ball\" type=\"spherical\"><parent link=\"socket\"/><child link=\"bob\"/>"
         "<dynamics friction=\"" + std::to_string(kFriction) + "\"/></joint><link name=\"bob\"><inertial><origin xyz=\"0 0 -" +
         std::to_string(kLength) + "\"/><mass value=\"" + std::to_string(kMass) + "\"/>" + inertia + "</inertial>"
         "<visual><origin xyz=\"0 0 -" + std::to_string(0.5 * kLength) + "\"/><geometry><cylinder radius=\"0.006\" length=\"" +
         std::to_string(kLength) + "\"/></geometry><material name=\"joint_friction_rod\"><color rgba=\"0.78 0.79 0.82 1\"/>"
         "</material></visual><visual><origin xyz=\"0 0 -" + std::to_string(kLength) + "\"/><geometry><sphere radius=\"0.025\"/>"
         "</geometry><material name=\"joint_friction_bob\"><color rgba=\"0.95 0.42 0.12 1\"/></material></visual></link></robot>";
}

Eigen::Vector3d bobPosition(raisim::ArticulatedSystem* pendulum) {
  raisim::Mat<3, 3> orientation;
  pendulum->getBodyOrientation(1, orientation);
  return Eigen::Vector3d(0., 0., kPivotHeight) + orientation.e() * Eigen::Vector3d(0., 0., -kLength);
}

void addSegment(raisin::RayraiWindow& window, const std::string& name, const Eigen::Vector3d& from,
                const Eigen::Vector3d& to, double radius, const glm::vec4& color) {
  const Eigen::Vector3d center = 0.5 * (from + to);
  const Eigen::Quaterniond rotation = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), to - from);
  auto segment = window.addVisualCylinder(name, radius, (to - from).norm(), color);
  segment->setPosition(center.x(), center.y(), center.z());
  segment->setOrientation(rotation.w(), rotation.x(), rotation.y(), rotation.z());
  segment->setDetectable(true);
}

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_joint_friction")) doc_image::finishAndExit(1);

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(0.001);
  world->setSleepingEnabled(false);
  world->addGround(0., "ground")->setAppearance("checkerboard");
  auto* pendulum = world->addArticulatedSystem(pendulumUrdf());

  // tilted 1.1 rad about x, turning about the vertical at half the rate of a steady conical swing
  const double tilt = 1.1;
  const Eigen::Quaterniond start(Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX()));
  const double rate = 0.5 * std::sqrt(kGravity / (kLength * std::cos(tilt)));
  Eigen::VectorXd gc(4);
  gc << start.w(), start.x(), start.y(), start.z();
  pendulum->setState(gc, start.toRotationMatrix().transpose() * Eigen::Vector3d(0., 0., rate));

  std::vector<Eigen::Vector3d> trail;
  for (int step = 0; step < 5000; ++step) {
    if (step % 40 == 0) trail.push_back(bobPosition(pendulum));
    world->integrate();
  }
  if (pendulum->getGeneralizedVelocity().e().norm() > 1e-9) {
    std::fprintf(stderr, "doc_image_joint_friction: the pendulum did not come to rest\n");
    doc_image::finishAndExit(1);
  }
  const Eigen::Vector3d pivot(0., 0., kPivotHeight), rest = bobPosition(pendulum);
  const double cone = std::asin(kFriction / (kMass * kGravity * kLength));
  const double restTilt = std::acos((pivot - rest).z() / kLength);
  std::printf("doc_image_joint_friction: friction cone %.2f deg, rest tilt %.2f deg\n", cone * 180. / M_PI,
              restTilt * 180. / M_PI);

  // a gallows that holds the socket
  for (const auto& [position, size] : {std::pair{Eigen::Vector3d(0., 0.4, kPivotHeight + 0.045), Eigen::Vector3d(0.05, 0.85, 0.05)},
                                       std::pair{Eigen::Vector3d(0., 0.8, 0.5 * (kPivotHeight + 0.07)), Eigen::Vector3d(0.05, 0.05, kPivotHeight + 0.07)}}) {
    auto* beam = world->addBox(size.x(), size.y(), size.z(), 1.);
    beam->setBodyType(raisim::BodyType::STATIC);
    beam->setPosition(position.x(), position.y(), position.z());
    beam->setAppearance("0.36 0.30 0.24 1");
  }

  raisin::RayraiWindow renderer(world, kWidth, kHeight);
  const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrToneMapping = true;
  doc_image::applyCommonSceneOptions(quality, preset);
  renderer.setRenderQualitySettings(quality);

  for (size_t i = 0; i < trail.size(); ++i) {
    const float age = float(i) / float(std::max<size_t>(1, trail.size() - 1));
    auto ghost = renderer.addVisualSphere("joint_friction_trail" + std::to_string(i), 0.012,
                                          glm::vec4(0.98f - 0.45f * age, 0.80f - 0.55f * age, 0.55f - 0.50f * age, 1.f));
    ghost->setPosition(trail[i].x(), trail[i].y(), trail[i].z());
    ghost->setDetectable(true);
  }
  // the friction cone: where the bob can rest, gravity's torque m g d sin(tilt) is at most the friction
  constexpr int kRing = 72;
  const auto ringPoint = [&](int k) {
    const double azimuth = 2. * M_PI * k / kRing;
    return Eigen::Vector3d(pivot + kLength * Eigen::Vector3d(std::sin(cone) * std::cos(azimuth),
                                                             std::sin(cone) * std::sin(azimuth), -std::cos(cone)));
  };
  for (int k = 0; k < kRing; ++k)
    addSegment(renderer, "joint_friction_cone" + std::to_string(k), ringPoint(k), ringPoint(k + 1), 0.0025,
               glm::vec4(0.15f, 0.55f, 0.95f, 1.f));
  addSegment(renderer, "joint_friction_vertical", pivot, pivot - Eigen::Vector3d(0., 0., kLength + 0.08), 0.0015,
             glm::vec4(0.15f, 0.55f, 0.95f, 1.f));

  const glm::vec3 target(0.f, 0.08f, float(kPivotHeight - 0.36));
  doc_image::setCameraLookAt(renderer.getCamera(), target + glm::vec3(0.35f, -1.25f, 0.85f), target, 40.f);
  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;
  overrides.drawCoordinateFrames = false;
  const bool ok = doc_image::captureScene(renderer, kWidth, kHeight, outputDir / "spherical_joint_friction.png", 2, overrides);
  doc_image::finishAndExit(ok ? 0 : 1);
}
