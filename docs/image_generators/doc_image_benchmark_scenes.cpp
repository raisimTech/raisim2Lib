// Renders the scenes of the benchmark page (sections/Benchmark.rst): the spring chains, ANYmal
// standing, ANYmal falling, ANYmal on the fractal height map and the primitive grid. Each scene is
// built as in its benchmark and simulated for a moment; scenes without a ground are posed in a
// second world, which adds a floor for reference and is never integrated. The Strandbeest figure of
// that page is constraints_strandbeest.png from doc_image_constraints.

#include "doc_image_common.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 960;
constexpr int kHeight = 600;

struct Renderer {
  std::shared_ptr<raisim::World> world;
  std::unique_ptr<raisin::RayraiWindow> window;

  explicit Renderer(std::shared_ptr<raisim::World> scene) : world(std::move(scene)) {
    window = std::make_unique<raisin::RayraiWindow>(world, kWidth, kHeight);
    const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
    auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
    quality.colorMode = raisin::ViewerColorMode::AcesApprox;
    quality.pbrToneMapping = true;
    doc_image::applyCommonSceneOptions(quality, preset);
    window->setRenderQualitySettings(quality);
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

/// ANYmal's nominal stance and joint PD gains, as in the ANYmal benchmarks
void standAnymal(raisim::ArticulatedSystem* anymal, double height) {
  Eigen::VectorXd gc(19), gv = Eigen::VectorXd::Zero(18), p = gv, d = gv;
  gc << 0., 0., height, 1., 0., 0., 0., 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
  p.tail(12).setConstant(200.);
  d.tail(12).setConstant(10.);
  anymal->setState(gc, gv);
  anymal->setPdGains(p, d);
  anymal->setPdTarget(gc, gv);
}

/// a spring chain hanging from a fixed point 10 m up, 0.3 s into curling up under its spring rest
/// angles; there
/// is no ground in the benchmark, so the floor of the figure is only for reference
bool renderChain(const std::filesystem::path& out, const std::filesystem::path& rsc, int links) {
  const auto urdf = (rsc / "chain" / ("robot_springed_" + std::to_string(links) + ".urdf")).string();
  raisim::World simulation;
  simulation.setTimeStep(0.001);
  auto* chain = simulation.addArticulatedSystem(urdf);
  for (int step = 0; step < 300; ++step) simulation.integrate();

  // the chain's spheres have no material; give each its own (URDF materials are global by name)
  std::ifstream file(urdf);
  std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  int link = 0;
  for (size_t at = text.find("</geometry>"); at != std::string::npos; at = text.find("</geometry>", at + 1)) {
    const double s = double(link) / std::max(1, links - 1);
    char material[160];
    std::snprintf(material, sizeof(material), "<material name=\"chain%d_%d\"><color rgba=\"%.3f %.3f %.3f 1\"/></material>",
                  links, link++, 0.95 - 0.75 * s, 0.55 - 0.1 * s, 0.15 + 0.6 * s);
    text.insert(at + std::string("</geometry>").size(), material);
  }

  auto scene = std::make_shared<raisim::World>();
  scene->addGround(0., "ground")->setAppearance("checkerboard");
  auto* pose = scene->addArticulatedSystem(text);
  pose->setState(chain->getGeneralizedCoordinate().e(), chain->getGeneralizedVelocity().e());
  // frame the links where the springs have curled them
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> bodies;
  for (size_t body = 0; body < chain->getNumberOfJoints(); ++body) {
    raisim::Vec<3> position;
    chain->getPosition(body, raisim::Vec<3>{0., 0., -0.12}, position);
    bodies.push_back(position.e());
    center += position.e() / double(chain->getNumberOfJoints());
  }
  double radius = 0.;
  for (const auto& body : bodies) radius = std::max(radius, (body - center).norm());
  const glm::vec3 target{float(center[0]), float(center[1]), float(center[2])};
  const glm::vec3 direction = glm::normalize(glm::vec3(1.f, -1.2f, 0.35f));
  Renderer renderer(scene);
  return renderer.capture(out / ("benchmark_chain" + std::to_string(links) + ".png"),
                          target + direction * float(3.4 * radius + 0.6), target, 40.f);
}

bool renderAnymalStanding(const std::filesystem::path& out, const std::filesystem::path& rsc) {
  auto scene = std::make_shared<raisim::World>();
  scene->setTimeStep(0.002);
  scene->setDefaultMaterial(0.8, 0.0, 0.01);
  scene->addGround(0., "ground")->setAppearance("checkerboard");
  auto* anymal = scene->addArticulatedSystem((rsc / "anymal/urdf/anymal.urdf").string());
  standAnymal(anymal, 0.54);
  for (int step = 0; step < 500; ++step) scene->integrate();
  Renderer renderer(scene);
  return renderer.capture(out / "benchmark_anymal_standing.png", {1.5f, -1.8f, 0.9f}, {0.f, 0.f, 0.3f}, 40.f);
}

/// the falling ANYmal 0.2 s after its release; there is no ground in the benchmark
bool renderAnymalFalling(const std::filesystem::path& out, const std::filesystem::path& rsc) {
  const auto urdf = (rsc / "anymal/urdf/anymal.urdf").string();
  raisim::World simulation;
  simulation.setTimeStep(0.002);
  auto* anymal = simulation.addArticulatedSystem(urdf);
  standAnymal(anymal, 1.2);
  for (int step = 0; step < 100; ++step) simulation.integrate();

  auto scene = std::make_shared<raisim::World>();
  scene->addGround(0., "ground")->setAppearance("checkerboard");
  auto* pose = scene->addArticulatedSystem(urdf);
  pose->setState(anymal->getGeneralizedCoordinate().e(), anymal->getGeneralizedVelocity().e());
  const auto& gc = anymal->getGeneralizedCoordinate();
  const glm::vec3 center(float(gc[0]), float(gc[1]), float(gc[2]) - 0.25f);
  Renderer renderer(scene);
  return renderer.capture(out / "benchmark_anymal_falling.png", center + glm::vec3(1.5f, -1.8f, 0.5f), center, 40.f);
}

bool renderHeightmapAnymal(const std::filesystem::path& out, const std::filesystem::path& rsc) {
  auto scene = std::make_shared<raisim::World>();
  scene->setTimeStep(0.002);
  raisim::TerrainProperties terrain;
  terrain.frequency = 0.2;
  terrain.zScale = 3.0;
  terrain.xSize = 20.0;
  terrain.ySize = 20.0;
  terrain.xSamples = 100;
  terrain.ySamples = 100;
  terrain.fractalOctaves = 3;
  terrain.fractalLacunarity = 2.0;
  terrain.fractalGain = 0.25;
  terrain.heightOffset = -1;
  scene->addHeightMap(0.0, 0.0, terrain)->setAppearance("0.55,0.5,0.42,1");
  auto* anymal = scene->addArticulatedSystem((rsc / "anymal/urdf/anymal.urdf").string());
  standAnymal(anymal, 1.5);
  Eigen::VectorXd gc = anymal->getGeneralizedCoordinate().e();
  gc[0] = -1.;  // where the benchmark drops it
  anymal->setGeneralizedCoordinate(gc);
  for (int step = 0; step < 1500; ++step) scene->integrate();
  const auto& base = anymal->getGeneralizedCoordinate();
  const glm::vec3 center(float(base[0]), float(base[1]), float(base[2]) - 0.3f);
  Renderer renderer(scene);
  return renderer.capture(out / "benchmark_heightmap_anymal.png", center + glm::vec3(2.6f, -3.2f, 1.7f), center, 45.f);
}

/// 16 boxes and 16 spheres, a sphere dropped onto each box, on a 4 x 4 grid
bool renderPrimitives(const std::filesystem::path& out) {
  auto scene = std::make_shared<raisim::World>();
  scene->setTimeStep(0.002);
  scene->addGround(0., "ground")->setAppearance("checkerboard");
  for (int ix = 0; ix < 4; ++ix) {
    for (int iy = 0; iy < 4; ++iy) {
      const double x = (ix - 1.5) * 0.6, y = (iy - 1.5) * 0.6;
      auto* box = scene->addBox(0.4, 0.4, 0.4, 1.0);
      box->setPosition(x, y, 0.6);
      box->setAppearance("0.18,0.46,0.74,1");
      auto* sphere = scene->addSphere(0.15, 1.0);
      sphere->setPosition(x, y, 1.2);
      sphere->setAppearance("0.92,0.62,0.2,1");
    }
  }
  for (int step = 0; step < 500; ++step) scene->integrate();
  Renderer renderer(scene);
  return renderer.capture(out / "benchmark_primitive_speed.png", {2.4f, -2.9f, 1.9f}, {0.f, 0.f, 0.25f}, 45.f);
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
  if (!gl.init("doc_image_benchmark_scenes")) doc_image::finishAndExit(1);
  const bool ok = renderChain(outputDir, rsc, 10) && renderChain(outputDir, rsc, 20) &&
                  renderAnymalStanding(outputDir, rsc) && renderAnymalFalling(outputDir, rsc) &&
                  renderHeightmapAnymal(outputDir, rsc) && renderPrimitives(outputDir);
  doc_image::finishAndExit(ok ? 0 : 1);
}
