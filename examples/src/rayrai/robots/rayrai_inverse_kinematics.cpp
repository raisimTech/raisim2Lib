// Inverse kinematics with the mouse: a Kinova arm with a Robotiq gripper
// reaches for a target that you move around its workbench.
//
// Every frame, ArticulatedSystem::solveIK() finds the joint angles that put the
// gripper's grasp frame on the target, starting from the previous solution.
// The arm itself is simulated: its joint PD targets follow that solution at a
// limited speed, so it pushes the cubes it touches and can pick them up
// (include/arm_ik_demo.hpp).
//
//   Left-drag the orange ball   move the target in the plane facing the camera
//   Wheel over the ball         move the target away from or towards the camera
//   Right-drag                  turn the target (while orientation is followed)
//   Left-drag elsewhere, wheel  orbit and zoom the camera
//   G / R                       close or open the gripper / reset the scene
//
// --tour moves the target along a scripted path and --frames=N exits after N
// frames; they exist for the documentation's clip and the tests.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "arm_ik_demo.hpp"
#include "example_resources.hpp"
#include "rayrai/PhysicsSnapshot.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_compat.hpp"

namespace {

using raisim_examples::arm_ik::Ray;
using raisim_examples::arm_ik::Scene;

constexpr double kHandleRadius = 0.025;
// The handle can be grabbed this many pixels outside its outline.
constexpr double kGrabSlackPixels = 10.;
// Target travel per mouse-wheel notch [m].
constexpr double kWheelStep = 0.02;
constexpr double kRadiansPerPixel = 0.01;
constexpr const char* kMethods[] = {"dls", "pinv", "svd", "transpose"};

Eigen::Vector3d toEigen(const glm::vec3& v) { return {v.x, v.y, v.z}; }

glm::quat toGlm(const Eigen::Matrix3d& rotation) {
  const Eigen::Quaterniond q(rotation);
  return {float(q.w()), float(q.x()), float(q.y()), float(q.z())};
}

/// The mouse pointer over the full-window viewer image.
using Pointer = ExampleApp::Viewport;

/// World-space ray through a pixel of the viewer image.
Ray pointerRay(const raisin::Camera& camera, const Pointer& pointer) {
  const glm::vec4 viewport(0.f, 0.f, float(pointer.width), float(pointer.height));
  const glm::mat4 view = camera.getViewMatrix(), projection = camera.getProjectionMatrix();
  // OpenGL window coordinates grow upwards.
  const glm::vec2 pixel(float(pointer.x), float(pointer.height - pointer.y));
  const glm::vec3 nearPoint = glm::unProject(glm::vec3(pixel, 0.f), view, projection, viewport);
  const glm::vec3 farPoint = glm::unProject(glm::vec3(pixel, 1.f), view, projection, viewport);
  return {toEigen(nearPoint), toEigen(farPoint - nearPoint).normalized()};
}

/// Moves and turns the scene's target with the mouse.
class TargetControl {
 public:
  /// Applies this frame's mouse input to the target. Returns whether the
  /// camera may use the pointer: not while the handle is hovered or held.
  bool update(const Pointer& pointer, const raisin::Camera& camera, Scene& scene) {
    const ImGuiIO& io = ImGui::GetIO();
    touched_ = false;
    if (!io.MouseDown[ImGuiMouseButton_Left]) drag_.End();
    if (!io.MouseDown[ImGuiMouseButton_Right]) turning_ = false;
    // Outside the window, ImGui has no pointer position to cast a ray from.
    if (pointer.width <= 0 || pointer.height <= 0 || !ImGui::IsMousePosValid()) {
      hovered_ = false;
      return true;
    }
    const Ray ray = pointerRay(camera, pointer);
    const Eigen::Vector3d view = toEigen(camera.front).normalized();
    Eigen::Vector3d position = scene.target_position();
    Eigen::Matrix3d orientation = scene.target_orientation();
    // A small ball is easy to miss, so accept a few pixels around it.
    const double depth = std::max(1e-3, (position - toEigen(camera.position)).dot(view));
    const double metresPerPixel = 2. * depth * std::tan(glm::radians(camera.zoom) * 0.5) / pointer.height;
    const double grabRadius = kHandleRadius + kGrabSlackPixels * metresPerPixel;
    // While the camera orbits, the handle passing under the pointer stays inactive.
    const bool orbiting = io.MouseDown[ImGuiMouseButton_Left] && !io.MouseClicked[ImGuiMouseButton_Left] && !drag_.active();
    hovered_ = pointer.hovered && !orbiting && raisim_examples::arm_ik::DistanceToRay(ray, position) <= grabRadius;

    if (hovered_ && io.MouseClicked[ImGuiMouseButton_Left]) drag_.Begin(ray, position, view, grabRadius);
    if (pointer.hovered && io.MouseClicked[ImGuiMouseButton_Right]) turning_ = true;

    const double push = drag_.active() || hovered_ ? kWheelStep * io.MouseWheel : 0.;
    if (drag_.active()) {
      drag_.Push(push);
      position = drag_.Move(ray, position);
    } else {
      position += push * view;
    }
    const bool turn = turning_ && scene.track_orientation() && (io.MouseDelta.x != 0.f || io.MouseDelta.y != 0.f);
    if (turn) {
      orientation = raisim_examples::arm_ik::RotateByDrag(orientation, io.MouseDelta.x, io.MouseDelta.y, toEigen(camera.right), toEigen(camera.up),
                                                          kRadiansPerPixel);
    }
    touched_ = drag_.active() || turn || push != 0.;
    if (touched_) scene.SetTarget(position, orientation);
    return !drag_.active() && !hovered_;
  }

  /// The handle is under the pointer or held.
  bool hot() const { return hovered_ || drag_.active(); }
  /// The user moved or turned the target this frame.
  bool touched() const { return touched_; }

 private:
  raisim_examples::arm_ik::TargetDrag drag_;
  bool hovered_ = false, turning_ = false, touched_ = false;
};

void drawPanel(Scene& scene, bool& tour, double& tourTime, int& method, bool& reset) {
  const auto& ik = scene.ik_result();
  Eigen::Vector3d hand;
  Eigen::Matrix3d handOrientation;
  scene.GraspPose(hand, handOrientation);
  ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.88f);
  ImGui::Begin("Inverse kinematics", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  if (ik.success) {
    ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1.f), "IK converged");
  } else {
    ImGui::TextColored(ImVec4(1.f, 0.4f, 0.35f, 1.f), "Out of reach: best pose %.0f mm, %.1f deg off", 1e3 * ik.bestPositionResidual,
                       glm::degrees(ik.bestOrientationResidual));
  }
  ImGui::Text("Gripper %.1f mm from the target", 1e3 * (hand - scene.target_position()).norm());
  ImGui::Separator();
  bool trackOrientation = scene.track_orientation();
  if (ImGui::Checkbox("Follow the target's orientation", &trackOrientation)) scene.SetTrackOrientation(trackOrientation);
  if (ImGui::Combo("Method", &method, "Damped least squares\0Pseudo-inverse\0SVD\0Jacobian transpose\0")) scene.ik_options().inverseMethod = kMethods[method];
  if (method == 0) {
    float damping = float(scene.ik_options().damping);
    if (ImGui::SliderFloat("Damping", &damping, 0.001f, 0.2f, "%.3f", ImGuiSliderFlags_Logarithmic)) scene.ik_options().damping = damping;
  }
  if (ImGui::Button(scene.gripper_closed() ? "Open gripper (G)" : "Close gripper (G)")) scene.SetGripperClosed(!scene.gripper_closed());
  ImGui::SameLine();
  reset = ImGui::Button("Reset (R)") || reset;
  ImGui::SameLine();
  if (ImGui::Checkbox("Tour", &tour) && tour) tourTime = 0.;
  if (ImGui::CollapsingHeader("Joint angles")) {
    ImGui::TextDisabled("joint   IK solution   arm");
    const auto& q = scene.robot()->getGeneralizedCoordinate();
    for (std::size_t i = 0; i < 6; ++i) {
      const std::size_t index = scene.arm_index(i);
      ImGui::Text("  %zu    %8.1f deg  %6.1f deg", i + 1, glm::degrees(scene.ik_solution()[index]), glm::degrees(q[index]));
    }
  }
  ImGui::Separator();
  ImGui::TextDisabled("Left-drag the orange ball: move the target\n"
                      "Wheel over the ball: push it away or pull it closer\n"
                      "Right-drag: turn the target\n"
                      "Left-drag elsewhere / wheel: orbit / zoom");
  ImGui::End();
}

}  // namespace

int main(int argc, char* argv[]) {
  bool tour = false;
  int frames = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--tour") {
      tour = true;
    } else if (arg.rfind("--frames=", 0) == 0 && std::atoi(arg.c_str() + 9) > 0) {
      frames = std::atoi(arg.c_str() + 9);
    } else {
      std::cerr << "usage: " << argv[0] << " [--tour] [--frames=N]\n";
      return 2;
    }
  }
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));
  Scene scene(exampleRscPath(argv[0], "robotiq_2f85/kinova_robotiq.urdf"));

  ExampleApp app;
  if (!app.init("RaiSim inverse kinematics", 1280, 720)) return 77;
  // Keep ImGui scratch files out of the working directory.
  ImGui::GetIO().IniFilename = nullptr;
  int status = 0;
  {
    // The renderer owns an empty world; physics reaches it only through copied snapshots.
    raisin::PhysicsSnapshotCapture snapshotCapture;
    raisin::RayraiWindow viewer(std::make_shared<raisim::World>(), 1280, 720);
    viewer.setRenderQualitySettings(raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::Balanced));
    raisim_examples::setRayraiBackgroundColorRgb255(viewer, {30, 34, 42, 255});
    raisim_examples::addRayraiBasicSceneLights(viewer);
    // The camera orbits the target's home, from the side of the bench where
    // the arm's base shows.
    auto& camera = viewer.getCamera();
    const Eigen::Vector3d& home = scene.target_position();
    camera.target = glm::vec3(home.x(), home.y(), home.z());
    camera.position = camera.target + glm::vec3(0.5f, -1.35f, 0.8f);
    const glm::vec3 direction = glm::normalize(camera.target - camera.position);
    camera.yaw = glm::degrees(std::atan2(direction.y, direction.x));
    camera.pitch = glm::degrees(std::asin(direction.z));
    camera.zoom = 40.f;
    camera.setCameraFixedTarget(true);
    camera.setCameraFixedDistance(true);
    camera.update(false);

    auto handle = viewer.addVisualSphere("ik_target", kHandleRadius, 1.f, 0.45f, 0.05f, 1.f);
    auto axes = viewer.addCoordinateFrame("ik_target_axes");
    axes->frameSize = 0.1;
    axes->poses.resize(1);

    TargetControl control;
    double tourTime = 0., accumulator = 0.;
    int method = 0, frame = 0;
    auto previous = std::chrono::steady_clock::now();
    while (!app.quit && (frames < 0 || frame < frames)) {
      app.processEvents();
      if (app.quit) break;
      app.beginFrame();
      bool reset = false;
      if (!ImGui::GetIO().WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_G, false)) scene.SetGripperClosed(!scene.gripper_closed());
        reset = ImGui::IsKeyPressed(ImGuiKey_R, false);
      }

      // The mouse moves the target before the arm reaches for it.
      const Pointer pointer = app.beginViewport();
      const bool cameraUsesPointer = control.update(pointer, viewer.getCamera(), scene);
      if (control.touched()) tour = false;

      const auto now = std::chrono::steady_clock::now();
      accumulator += frames >= 0 ? 1. / 60. : std::min(0.05, std::chrono::duration<double>(now - previous).count());
      previous = now;
      const int steps = int(accumulator / raisim_examples::arm_ik::kTimeStep);
      accumulator -= steps * raisim_examples::arm_ik::kTimeStep;
      if (tour) {
        tourTime += steps * raisim_examples::arm_ik::kTimeStep;
        Eigen::Vector3d position;
        Eigen::Matrix3d orientation;
        scene.TourPose(tourTime, position, orientation);
        scene.SetTarget(position, orientation);
      }
      scene.SolveIk();
      for (int i = 0; i < steps; ++i) scene.Step();

      viewer.applyPhysicsSnapshot(snapshotCapture.capture(scene.world()));
      const Eigen::Vector3d& target = scene.target_position();
      handle->setPosition(target.x(), target.y(), target.z());
      const glm::vec3 color = scene.ik_result().success ? glm::vec3(1.f, 0.45f, 0.05f) : glm::vec3(0.95f, 0.1f, 0.1f);
      handle->setColor(glm::vec4(control.hot() ? glm::mix(color, glm::vec3(1.f), 0.45f) : color, 1.f));
      axes->enable(scene.track_orientation());
      axes->poses[0].position = glm::vec3(target.x(), target.y(), target.z());
      axes->poses[0].quaternion = toGlm(scene.target_orientation());
      // Clicks would make the camera follow the clicked body; this demo orbits the bench.
      app.endViewport(viewer, pointer, cameraUsesPointer, false);

      drawPanel(scene, tour, tourTime, method, reset);
      app.endFrame();
      if (reset) {
        scene.Reset();
        tour = false;
        accumulator = 0.;
      }
      ++frame;
    }
    if (frames >= 0) {
      Eigen::Vector3d hand;
      Eigen::Matrix3d orientation;
      scene.GraspPose(hand, orientation);
      const Eigen::Vector3d& target = scene.target_position();
      const Eigen::Quaterniond turn(scene.target_orientation());
      std::cout << "frames=" << frame << " ik_converged=" << scene.ik_result().success << " gripper_to_target_mm=" << 1e3 * (hand - target).norm() << "\ntarget=" << target.x()
                << ',' << target.y() << ',' << target.z() << " target_quaternion=" << turn.w() << ',' << turn.x() << ',' << turn.y() << ',' << turn.z()
                << "\ncamera=" << camera.position.x << ',' << camera.position.y << ',' << camera.position.z << " camera_yaw=" << camera.yaw
                << " camera_pitch=" << camera.pitch << '\n';
      if (gl::glGetError() != gl::GL_NO_ERROR) {
        std::cerr << "OpenGL error in the inverse kinematics example\n";
        status = 1;
      }
    }
  }
  app.shutdown();
  return status;
}
