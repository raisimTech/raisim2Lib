// Records the existing granular-media and tendon-pulley scenes for the home gallery.
#include "doc_image_common.hpp"
#include "tendon_scenes.hpp"
#include <rayrai/helper.hpp>
#include <raisim/object/granular/GranularSystem.hpp>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <random>

namespace {
constexpr int kWidth = 640, kHeight = 400, kFrames = 60;  // Four seconds at 15 fps.
constexpr float kPi = 3.14159265359f;

void record(raisin::RayraiWindow& viewer, const std::filesystem::path& frames,
            const glm::vec3& eye, const glm::vec3& target, float fov,
            const std::function<void()>& step, const std::function<void()>& sync,
            float orbitAmplitude = .10f) {
  viewer.setAsyncMeshLoadingEnabled(false);
  // Use an independent capture camera so viewer updates cannot change its framing.
  raisin::Camera camera;
  doc_image::setCameraLookAt(camera, eye, target, fov);
  camera.aspect = float(kWidth) / kHeight;
  camera.ensureRenderTargets(kWidth, kHeight);
  sync();
  viewer.generateWeatherSkyEnvironment(128, 32, true);
  viewer.updateObjectLists();
  viewer.renderWithExternalCamera(camera, {});
  const char* previewFrames = std::getenv("RAISIM_DOC_GALLERY_FRAME_COUNT");
  const int frameCount = previewFrames ? std::clamp(std::atoi(previewFrames), 1, kFrames) : kFrames;
  for (int frame = 0; frame < frameCount; ++frame) {
    const float angle = orbitAmplitude * std::sin(2.f * kPi * frame / kFrames);
    const auto offset = eye - target;
    doc_image::setCameraLookAt(camera, target + glm::vec3(
        std::cos(angle) * offset.x - std::sin(angle) * offset.y,
        std::sin(angle) * offset.x + std::cos(angle) * offset.y, offset.z), target, fov);
    sync();
    // External captures synchronize the world too; avoid rendering each frame twice.
    auto capture = viewer.captureSupersampledRgba(camera, 1, {});
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%03d.png", frame);
    if (capture.rgba.empty() || !doc_image::writePng(frames / name, capture.width, capture.height,
                                                    std::move(capture.rgba), false))
      doc_image::finishAndExit(1);
    // Alternate 66 and 67 physics steps to preserve real-time playback at 15 fps.
    const int steps = (frame+1)*4000/kFrames - frame*4000/kFrames;
    for (int i = 0; i < steps; ++i) step();
    if ((frame + 1) % 5 == 0 || frame + 1 == frameCount) {
      std::printf("doc_image: %s frame %d/%d\n", DOC_IMAGE_SCENE, frame + 1, frameCount);
      std::fflush(stdout);
    }
  }
}

// Large, sharply changing joint targets produce repeatable kicks through the sand.
class GranularJointNoise {
 public:
  GranularJointNoise(raisim::ArticulatedSystem& robot, const Eigen::VectorXd& nominal,
                     const Eigen::VectorXd& velocity)
      : robot_(robot), target_(nominal), nominal_(nominal), velocity_(velocity) {}

  void step(double dt) {
    if (time_ >= nextSample_) {
      noise_.setZero();
      if (kicking_) {
        // Plant the foot again before the next leg kicks.
        nextSample_ += .18;
      } else {
        constexpr int legOrder[] = {1, 3, 0, 2};
        const int leg = legOrder[kick_++ % 4];
        const double swing = distribution_(rng_);
        noise_[3*leg] = .28 * distribution_(rng_);
        noise_[3*leg+1] = std::copysign(.95 * (.65 + .35 * std::abs(swing)), swing);
        // Fold the knee substantially, then drive it back into the sand.
        noise_[3*leg+2] = (leg < 2 ? -1. : 1.) * (1.15 + .25 * distribution_(rng_));
        nextSample_ += .30;
      }
      kicking_ = !kicking_;
    }
    // A short 8 ms transition keeps the target changes sharp.
    offset_ += (1. - std::exp(-dt / .008)) * (noise_ - offset_);
    target_.tail<12>() = nominal_.tail<12>() + offset_;
    robot_.setPdTarget(target_, velocity_);
    time_ += dt;
  }

 private:
  raisim::ArticulatedSystem& robot_;
  Eigen::VectorXd target_, nominal_, velocity_;
  Eigen::Matrix<double, 12, 1> noise_ = Eigen::Matrix<double, 12, 1>::Zero();
  Eigen::Matrix<double, 12, 1> offset_ = Eigen::Matrix<double, 12, 1>::Zero();
  std::mt19937 rng_{42};
  std::uniform_real_distribution<double> distribution_{-1., 1.};
  double time_ = 0., nextSample_ = 0.;
  int kick_ = 0;
  bool kicking_ = false;
};

void pulleys(const std::filesystem::path& frames) {
  raisim_examples::tendons::Scene scene(raisim_examples::tendons::Kind::Pulley);
  // Record one complete drive period after the initial ramp has settled.
  for (int i = 0; i < 8000; ++i) scene.step();
  raisin::RayraiWindow viewer(scene.world, kWidth, kHeight);
  auto quality = raisin::RenderQualitySettings::preset(raisin::RenderQualityPreset::Balanced);
  doc_image::applyCommonSceneOptions(quality, raisin::RenderQualityPreset::Balanced);
  quality.reflectiveGround = false;
  quality.shadowResolution = 2048;
  quality.viewerMsaaSamples = 2;
  viewer.setRenderQualitySettings(quality);
  record(viewer, frames, {5.f,-10.f,5.f}, {0.f,0.f,1.5f}, 38.f,
         [&] { scene.step(); }, [] {});
  scene.verify(12000);
}

void granular(const std::filesystem::path& frames, const std::filesystem::path& rsc) {
  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(0.001);
  world->setSleepingEnabled(false);
  // Same bed dimensions, particle layout, material and PD gains as granular_media.
  constexpr double length = 1.8, width = 1.25, wallHeight = 0.38, thickness = 0.04;
  const auto box = [&](double x, double y, double z, double sx, double sy, double sz) {
    auto* body = world->addBox(sx, sy, sz, 1.);
    body->setBodyType(raisim::BodyType::STATIC);
    body->setPosition(x, y, z);
    body->setAppearance("0.2,0.35,0.7,0.5");
  };
  box(0,0,-thickness/2,length+2*thickness,width+2*thickness,thickness);
  box(-(length+thickness)/2,0,wallHeight/2,thickness,width+2*thickness,wallHeight);
  box((length+thickness)/2,0,wallHeight/2,thickness,width+2*thickness,wallHeight);
  box(0,-(width+thickness)/2,wallHeight/2,length,thickness,wallHeight);
  box(0,(width+thickness)/2,wallHeight/2,length,thickness,wallHeight);
  constexpr double radius = 0.022, spacing = 0.047;
  const int nx = int((length - 2*radius - radius/2)/spacing) + 1;
  const int ny = int((width - 2*radius - radius/2)/spacing) + 1;
  std::vector<raisim::Vec<3>> positions;
  std::vector<double> radii;
  for (int z = 0; z < 7; ++z) for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
    positions.push_back({spacing*(x-(nx-1)*.5)+(((y+z)&1)?radius/2:0),
                         spacing*(y-(ny-1)*.5)+((z&1)?radius/2:0), radius+spacing*z});
    radii.push_back(radius);
  }
  raisim::GranularSystem::Material material;
  material.density = 1600.;
  material.normalStiffness = 3.5e4;
  material.normalDamping = 35.;
  material.tangentialStiffness = 1.75e4;
  material.tangentialDamping = 8.;
  material.friction = .85;
  material.rollingFriction = .03;
  material.substeps = 8;
  material.maxSpeed = 10.;
  material.maxAngularSpeed = 100.;
  auto* grains = world->addGranularParticles(positions, radii, material);
  for (int i = 0; i < nx*ny; ++i) grains->setParticleFixed(i, true);
  for (int i = 0; i < 500; ++i) world->integrate();
  double surface = 0.;
  for (const auto& p : grains->getPositions()) surface = std::max(surface, p[2]+radius);
  auto* robot = world->addArticulatedSystem((rsc / "anymal/urdf/anymal.urdf").string());
  Eigen::VectorXd gc = Eigen::VectorXd::Zero(robot->getGeneralizedCoordinateDim());
  Eigen::VectorXd gv = Eigen::VectorXd::Zero(robot->getDOF()), pgain = gv, dgain = gv;
  gc.head<7>() << 0.,0.,.54+surface+.02,1.,0.,0.,0.;
  gc.tail<12>() << .03,.4,-.8,-.03,.4,-.8,.03,-.4,.8,-.03,-.4,.8;
  pgain.tail<12>().setConstant(200.);
  dgain.tail<12>().setConstant(10.);
  robot->setState(gc, gv);
  robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  robot->setPdGains(pgain, dgain);
  robot->setPdTarget(gc, gv);
  robot->setGeneralizedForce(gv);
  for (int i = 0; i < 250; ++i) world->integrate();
  GranularJointNoise jointNoise(*robot, gc, gv);
  const auto initialPositions = grains->getPositions();
  const auto step = [&] {
    jointNoise.step(world->getTimeStep());
    world->integrate();
  };
  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  auto quality = raisin::RenderQualitySettings::preset(raisin::RenderQualityPreset::High);
  doc_image::applyCommonSceneOptions(quality, raisin::RenderQualityPreset::High);
  quality.shadowResolution = 2048;
  quality.viewerMsaaSamples = 2;
  viewer.setRenderQualitySettings(quality);
  // Grains cover only a few pixels at gallery resolution; use a smooth, smaller sphere mesh.
  auto particleMeshes = raisin::GenMeshSphere(1.f, 6, 12);
  for (auto& mesh : *particleMeshes)
    mesh->material = raisin::Material::pbr("grain", glm::vec4(1.f), 0.f, .55f);
  auto particles = viewer.addInstancedVisuals("granular particles", particleMeshes,
      glm::vec3(radius), {0.92f,0.70f,0.34f,1.f}, {0.16f,0.45f,0.95f,1.f}, false);
  particles->resize(grains->getNumParticles());
  particles->setDetectable(true);
  const auto sync = [&] {
    const auto& p = grains->getPositions();
    const auto& v = grains->getVelocities();
    if (!robot->getGeneralizedCoordinate().e().allFinite())
      throw std::runtime_error("non-finite robot state");
    for (size_t i = 0; i < p.size(); ++i) {
      if (!p[i].e().allFinite())
        throw std::runtime_error("non-finite granular simulation state");
      particles->setPosition(i, {float(p[i][0]),float(p[i][1]),float(p[i][2])});
      particles->setColorWeight(i, std::min(float(v[i].norm()), 1.f));
    }
  };
  // Start close and keep exactly the same camera pose and field of view throughout.
  record(viewer, frames, {2.1f,-2.55f,1.8f}, {0.f,0.f,.14f}, 30.f, step, sync, 0.f);
  size_t disturbed = 0;
  double maxDisplacement = 0.;
  const auto& finalPositions = grains->getPositions();
  for (size_t i = nx*ny; i < finalPositions.size(); ++i) {
    const double displacement = (finalPositions[i].e() - initialPositions[i].e()).norm();
    disturbed += displacement > radius;
    maxDisplacement = std::max(maxDisplacement, displacement);
  }
  std::printf("doc_image: %zu free grains displaced by over one radius; max displacement %.3f m; robot z %.3f m\n",
              disturbed, maxDisplacement, robot->getGeneralizedCoordinate()[2]);
  sync();
}
}  // namespace

int main(int argc, char** argv) {
  const auto output = doc_image::resolveOutputDir(argc, argv);
  const auto rsc = argc >= 3 ? std::filesystem::path(argv[2]) :
      std::filesystem::weakly_canonical(output / "../../..");
  const char* override = std::getenv("RAISIM_DOC_GALLERY_FRAMES_DIR");
  const auto frames = override ? std::filesystem::path(override) : output / (std::string(DOC_IMAGE_SCENE)+"_frames");
  std::filesystem::create_directories(frames);
  if (std::filesystem::exists(rsc / "activation.raisim"))
    raisim::World::setActivationKey((rsc / "activation.raisim").string());
  doc_image::OffscreenContext gl;
  if (!gl.init(DOC_IMAGE_SCENE)) doc_image::finishAndExit(1);
  try {
    if (std::string(DOC_IMAGE_SCENE) == "tendon_pulleys") pulleys(frames);
    else granular(frames, rsc);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "doc_image: %s\n", e.what());
    doc_image::finishAndExit(1);
  }
  doc_image::finishAndExit(0);
}
