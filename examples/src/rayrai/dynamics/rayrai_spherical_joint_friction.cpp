// Coulomb friction of spherical joints: three spherical pendulums are released on the same conical
// swing, with no friction, a little and a lot (URDF <dynamics friction>). Without friction the swing
// goes on; with friction it spirals down and stops off the vertical, anywhere inside the ring where
// gravity's torque equals the friction (the friction cone). Each bob leaves a trail, and the
// pendulums are released again every 12 seconds.
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "raisim/World.hpp"
#include "rayrai/PhysicsSnapshot.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_compat.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kLength = 0.5, kMass = 1.0, kPivotHeight = 1.5, kGravity = 9.81, kTilt = 1.1;
constexpr double kFrictions[3] = {0.0, 0.2, 0.6};
constexpr float kColors[3][3] = {{0.25f, 0.55f, 0.95f}, {0.95f, 0.6f, 0.15f}, {0.85f, 0.2f, 0.2f}};
constexpr int kTrail = 90;

// A bob of kMass hanging kLength below a spherical joint at height kPivotHeight above x, with the
// friction kFrictions[index] [Nm].
std::string pendulumUrdf(int index, double x) {
  const std::string id = std::to_string(index), color = std::to_string(kColors[index][0]) + " " +
                                                        std::to_string(kColors[index][1]) + " " +
                                                        std::to_string(kColors[index][2]) + " 1";
  return "<robot name=\"pendulum" + id + "\"><link name=\"world\"/>"
         "<joint name=\"mount\" type=\"fixed\"><parent link=\"world\"/><child link=\"socket\"/>"
         "<origin xyz=\"" + std::to_string(x) + " 0 " + std::to_string(kPivotHeight) + "\"/></joint>"
         "<link name=\"socket\"><inertial><mass value=\"1\"/><inertia ixx=\"0.01\" ixy=\"0\" ixz=\"0\" iyy=\"0.01\" "
         "iyz=\"0\" izz=\"0.01\"/></inertial><visual><geometry><sphere radius=\"0.03\"/></geometry>"
         "<material name=\"socket" + id + "\"><color rgba=\"0.25 0.26 0.28 1\"/></material></visual></link>"
         "<joint name=\"ball\" type=\"spherical\"><parent link=\"socket\"/><child link=\"bob\"/>"
         "<dynamics friction=\"" + std::to_string(kFrictions[index]) + "\"/></joint>"
         "<link name=\"bob\"><inertial><origin xyz=\"0 0 -0.5\"/><mass value=\"1\"/><inertia ixx=\"0.0004\" "
         "ixy=\"0\" ixz=\"0\" iyy=\"0.0004\" iyz=\"0\" izz=\"0.0004\"/></inertial>"
         "<visual><origin xyz=\"0 0 -0.25\"/><geometry><cylinder radius=\"0.006\" length=\"0.5\"/></geometry>"
         "<material name=\"rod" + id + "\"><color rgba=\"0.78 0.79 0.82 1\"/></material></visual>"
         "<visual><origin xyz=\"0 0 -0.5\"/><geometry><sphere radius=\"0.03\"/></geometry>"
         "<material name=\"bob" + id + "\"><color rgba=\"" + color + "\"/></material></visual></link></robot>";
}

// Tilted kTilt about x and turning about the vertical at half the rate of a steady conical swing.
void release(raisim::ArticulatedSystem* pendulum) {
  const Eigen::Quaterniond tilt(Eigen::AngleAxisd(kTilt, Eigen::Vector3d::UnitX()));
  const double rate = 0.5 * std::sqrt(kGravity / (kLength * std::cos(kTilt)));
  Eigen::VectorXd gc(4);
  gc << tilt.w(), tilt.x(), tilt.y(), tilt.z();
  // the joint velocity is the bob's angular velocity in its own frame
  pendulum->setState(gc, tilt.toRotationMatrix().transpose() * Eigen::Vector3d(0.0, 0.0, rate));
}

Eigen::Vector3d bobPosition(raisim::ArticulatedSystem* pendulum) {
  const size_t bob = pendulum->getBodyIdx("bob");
  raisim::Vec<3> pivot;
  raisim::Mat<3, 3> orientation;
  pendulum->getBodyPosition(bob, pivot);
  pendulum->getBodyOrientation(bob, orientation);
  return pivot.e() + orientation.e() * Eigen::Vector3d(0.0, 0.0, -kLength);
}

}  // namespace

int main() {
  ExampleApp app;
  if (!app.init("rayrai_spherical_joint_friction", 1280, 720)) return -1;

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(0.001);
  // A slow swing would otherwise be put to sleep, and frozen, before friction stops it.
  world->setSleepingEnabled(false);
  world->addGround()->setAppearance("checkerboard");
  std::vector<raisim::ArticulatedSystem*> pendulums;
  for (int i = 0; i < 3; ++i) pendulums.push_back(world->addArticulatedSystem(pendulumUrdf(i, 1.2 * (i - 1))));
  // a beam on two posts holds the three sockets
  auto* beam = world->addBox(3.2, 0.06, 0.06, 1.0);
  beam->setPosition(0.0, 0.0, kPivotHeight + 0.06);
  for (const double x : {-1.6, 1.6}) {
    auto* post = world->addBox(0.06, 0.06, kPivotHeight + 0.09, 1.0);
    post->setPosition(x, 0.0, 0.5 * (kPivotHeight + 0.09));
    post->setBodyType(raisim::BodyType::STATIC);
    post->setAppearance("0.36,0.3,0.24,1");
  }
  beam->setBodyType(raisim::BodyType::STATIC);
  beam->setAppearance("0.36,0.3,0.24,1");

  raisin::PhysicsSnapshotCapture snapshotCapture;
  auto viewer = std::make_shared<raisin::RayraiWindow>(std::make_shared<raisim::World>(), 1280, 720);
  viewer->setRenderQualitySettings(
    raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::High));
  raisim_examples::setRayraiBackgroundColorRgb255(*viewer, {150, 170, 190, 255});
  raisim_examples::addRayraiBasicSceneLights(*viewer);

  // The friction cone of each pendulum with friction: the bob can rest where gravity's torque,
  // m g L sin(tilt), is at most the friction, inside a ring of that tilt.
  for (int i = 1; i < 3; ++i) {
    const double cone = std::asin(kFrictions[i] / (kMass * kGravity * kLength));
    const Eigen::Vector3d pivot(1.2 * (i - 1), 0.0, kPivotHeight);
    constexpr int kSegments = 72;
    for (int k = 0; k < kSegments; ++k) {
      const auto point = [&](int s) {
        const double azimuth = 2.0 * kPi * s / kSegments;
        return Eigen::Vector3d(pivot + kLength * Eigen::Vector3d(std::sin(cone) * std::cos(azimuth),
                                                                 std::sin(cone) * std::sin(azimuth), -std::cos(cone)));
      };
      const Eigen::Vector3d from = point(k), to = point(k + 1), middle = 0.5 * (from + to);
      const Eigen::Quaterniond turn = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), to - from);
      auto segment = viewer->addVisualCylinder("cone" + std::to_string(i) + "_" + std::to_string(k), 0.003,
                                               (to - from).norm(), kColors[i][0], kColors[i][1], kColors[i][2], 1.f);
      segment->setPosition(middle.x(), middle.y(), middle.z());
      segment->setOrientation(turn.w(), turn.x(), turn.y(), turn.z());
    }
    std::printf("friction %.1f Nm: the bob can rest up to %.1f degrees off the vertical\n", kFrictions[i],
                cone * 180.0 / kPi);
  }

  // trails: the bobs' recent positions, newest first, as shrinking spheres
  std::vector<std::vector<std::shared_ptr<raisin::Visuals>>> trails(3);
  std::vector<std::vector<Eigen::Vector3d>> history(3);
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < kTrail; ++k)
      trails[i].push_back(viewer->addVisualSphere("trail" + std::to_string(i) + "_" + std::to_string(k),
                                                  0.014 * (1.0 - 0.8 * k / kTrail), kColors[i][0], kColors[i][1],
                                                  kColors[i][2], 1.f));

  auto& camera = viewer->getCamera();
  camera.target = {0.0f, 0.0f, 1.0f};
  camera.position = {0.4f, -2.45f, 1.85f};
  camera.yaw = 99.3f;
  camera.pitch = -18.9f;
  camera.setCameraFixedTarget(true);
  camera.setCameraFixedDistance(true);

  // steps each pendulum has been at rest, and whether that was reported
  std::vector<int> restSteps(3, 0);
  std::vector<bool> reported(3, false);
  int step = 0;
  while (!app.quit) {
    app.processEvents();
    if (app.quit) break;

    // Every 12 seconds the pendulums are released again.
    if (step % 12000 == 0) {
      for (int i = 0; i < 3; ++i) {
        release(pendulums[i]);
        history[i].assign(kTrail, bobPosition(pendulums[i]));
        restSteps[i] = 0;
        reported[i] = false;
      }
    }
    world->integrate();

    // A joint held by friction keeps only round-off of its velocity. At a turning point of the swing
    // friction can stop it for a step even where gravity pulls harder; it has come to rest when it
    // stays stopped (half a second here).
    for (int i = 1; i < 3; ++i) {
      restSteps[i] = pendulums[i]->getGeneralizedVelocity().e().norm() < 1e-9 ? restSteps[i] + 1 : 0;
      if (reported[i] || restSteps[i] < 500) continue;
      reported[i] = true;
      const Eigen::Vector3d down = bobPosition(pendulums[i]) - Eigen::Vector3d(1.2 * (i - 1), 0.0, kPivotHeight);
      std::printf("friction %.1f Nm: at rest after %.1f s, %.1f degrees off the vertical\n", kFrictions[i],
                  (step % 12000 - 499) * world->getTimeStep(), std::acos(-down.z() / kLength) * 180.0 / kPi);
    }

    // Draw every 16th step, 62.5 frames a second of simulated time.
    if (step % 16 == 0) {
      for (int i = 0; i < 3; ++i) {
        history[i].insert(history[i].begin(), bobPosition(pendulums[i]));
        history[i].pop_back();
        for (int k = 0; k < kTrail; ++k) trails[i][k]->setPosition(history[i][k].x(), history[i][k].y(), history[i][k].z());
      }
      viewer->applyPhysicsSnapshot(snapshotCapture.capture(*world));
      app.beginFrame();
      app.renderViewer(*viewer);
      app.endFrame();
    }
    ++step;
  }

  viewer.reset();
  app.shutdown();
  return 0;
}
