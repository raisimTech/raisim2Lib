// Renders the scenes of the benchmark page (sections/Benchmark.md): the spring chains, ANYmal
// standing, ANYmal falling, terrain LiDAR, primitive rays and deformables. Simulation scenes use
// their benchmark parameters at representative poses; ray scenes show their query geometry.
// Scenes without a ground are posed in a
// second world, which adds a floor for reference and is never integrated. The Strandbeest figure of
// that page is constraints_strandbeest.png from doc_image_constraints.

#include "doc_image_common.hpp"

#include <Eigen/Geometry>
#include <rayrai/PointCloud.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
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

  bool capture(const std::filesystem::path& path, const glm::vec3& eye, const glm::vec3& target,
               float fov = 45.f, float farPlane = 80.f) {
    doc_image::setCameraLookAt(window->getCamera(), eye, target, fov);
    window->getCamera().farPlane = window->getCamera().zFar = farPlane;
    window->getCamera().update(false);
    raisin::RayraiWindow::RenderOverrides overrides;
    overrides.doShadows = true;
    if (!doc_image::captureScene(*window, kWidth, kHeight, path, 2, overrides)) return false;
    std::printf("doc_image: wrote %s\n", path.string().c_str());
    return true;
  }
};

/// ANYmal's nominal stance and joint PD gains, as in the ANYmal benchmarks
void standAnymal(raisim::ArticulatedSystem* anymal, double height, double pGain = 200., double dGain = 10.) {
  Eigen::VectorXd gc(19), gv = Eigen::VectorXd::Zero(18), p = gv, d = gv;
  gc << 0., 0., height, 1., 0., 0., 0., 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
  p.tail(12).setConstant(pGain);
  d.tail(12).setConstant(dGain);
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

void addRaySegment(Renderer& renderer, const std::string& name, const Eigen::Vector3d& from,
                   const Eigen::Vector3d& to, double radius, const glm::vec4& color) {
  const Eigen::Vector3d delta = to - from;
  const Eigen::Vector3d center = 0.5 * (from + to);
  const Eigen::Quaterniond rotation = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), delta);
  auto ray = renderer.window->addVisualCylinder(name, radius, delta.norm(), color);
  ray->setPosition(center.x(), center.y(), center.z());
  ray->setOrientation(rotation.w(), rotation.x(), rotation.y(), rotation.z());
  ray->setMaterialOverride(raisin::Material::unlitColor(name, color));
  ray->setCastsShadows(false);
  ray->setDetectable(true);
}

bool renderHeightmapLidar(const std::filesystem::path& out, const std::filesystem::path& rsc) {
  auto scene = std::make_shared<raisim::World>();
  scene->setSleepingEnabled(false);
  scene->setTimeStep(0.005);
  raisim::TerrainProperties terrain;
  terrain.frequency = 0.2;
  terrain.zScale = 1.0;
  terrain.xSize = terrain.ySize = 200.0;
  terrain.xSamples = terrain.ySamples = 400;
  terrain.fractalOctaves = 3;
  terrain.fractalLacunarity = 2.0;
  terrain.fractalGain = 0.25;
  terrain.heightOffset = -1.0;
  auto* heightmap = scene->addHeightMap(0.0, 0.0, terrain);
  heightmap->setAppearance("0.55,0.5,0.42,1");
  auto* anymal = scene->addArticulatedSystem((rsc / "anymal_c/urdf/anymal_sensored.urdf").string());
  standAnymal(anymal, heightmap->getHeight(0.0, 0.0) + 0.6, 100.0, 1.0);
  for (auto& sensorSet : anymal->getSensorSets()) {
    for (auto& sensor : sensorSet->getSensors()) sensor->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);
  }
  for (int step = 0; step < 200; ++step) scene->integrate();
  auto* lidarSet = anymal->getSensorSet("lidar_link");
  auto* lidar = lidarSet ? lidarSet->getSensor<raisim::SpinningLidar>("lidar") : nullptr;
  if (!lidar) throw std::runtime_error("Benchmark ANYmal C has no LiDAR sensor");
  lidar->update(*scene);

  // Accumulate one complete revolution while holding the settled robot fixed, as in the timed run.
  std::vector<Eigen::Vector3d> hits;
  for (int scan = 0; scan < 20; ++scan) {
    scene->setWorldTime(scene->getWorldTime() + scene->getTimeStep());
    lidar->update(*scene);
    for (const auto& point : lidar->getScan()) {
      const raisim::Vec<3> worldPoint = lidar->getOrientation() * point + lidar->getPosition();
      hits.push_back(worldPoint.e());
    }
  }
  if (hits.empty()) throw std::runtime_error("Benchmark LiDAR did not produce any terrain hits");
  Renderer renderer(scene);
  auto cloud = renderer.window->addPointCloud("benchmark_lidar_hits");
  cloud->setDetectable(true);
  cloud->pointSize = 3.0f;
  for (const auto& hit : hits) cloud->positions.emplace_back(hit.x(), hit.y(), hit.z() + 0.015);
  cloud->colors.assign(hits.size(), glm::vec4(0.1f, 0.95f, 0.4f, 1.f));
  cloud->updatePointBuffer();
  for (size_t i = 0; i < hits.size(); i += std::max<size_t>(1, hits.size() / 16)) {
    addRaySegment(renderer, "lidar_ray_" + std::to_string(i), lidar->getPosition().e(), hits[i],
                  0.006, glm::vec4(0.1f, 0.9f, 0.4f, 1.f));
  }
  const auto& gc = anymal->getGeneralizedCoordinate();
  const glm::vec3 target(float(gc[0]), float(gc[1]), float(gc[2]) - 0.2f);
  std::printf("doc_image: heightmap_lidar samples=400x400 scan_hits=%zu\n", hits.size());
  return renderer.capture(out / "benchmark_heightmap_lidar.png", target + glm::vec3(4.5f, -5.5f, 3.2f), target, 50.f);
}

bool renderPrimitiveRays(const std::filesystem::path& out) {
  auto scene = std::make_shared<raisim::World>();
  scene->setSleepingEnabled(false);
  constexpr int kObjects = 10000;
  constexpr double kSpacing = 2.5;
  constexpr const char* kColors[] = {
      "0.95,0.6,0.15,1", "0.15,0.45,0.8,1", "0.8,0.25,0.3,1", "0.3,0.65,0.35,1"};
  for (int index = 0; index < kObjects; ++index) {
    raisim::SingleBodyObject* object = nullptr;
    switch (index % 4) {
      case 0: object = scene->addSphere(0.45, 1.0); break;
      case 1: object = scene->addBox(0.8, 0.9, 1.0, 1.0); break;
      case 2: object = scene->addCapsule(0.3, 0.8, 1.0); break;
      case 3: object = scene->addCylinder(0.38, 1.1, 1.0); break;
    }
    object->setPosition((index % 100 - 49.5) * kSpacing, (index / 100 - 49.5) * kSpacing, 0.0);
    object->setBodyType(raisim::BodyType::STATIC);
    object->setAppearance(kColors[index % 4]);
  }
  Renderer renderer(scene);
  // The floor and a few hit/miss traces are visual aids, without collision geometry.
  auto floor = renderer.window->addVisualBox("ray_field_reference_floor", 252.0, 252.0, 0.03,
                                             glm::vec4(0.38f, 0.4f, 0.43f, 1.f));
  floor->setPosition(0.0, 0.0, -0.85);
  floor->setDetectable(true);
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 5; ++column) {
      const Eigen::Vector3d origin((column - 49.5) * kSpacing, (row - 49.5) * kSpacing, 5.0);
      const auto& contacts = scene->rayTest(origin, Eigen::Vector3d(0.0, 0.0, -1.0), 10.0);
      if (contacts.size() != 1) throw std::runtime_error("Primitive benchmark illustration ray missed its target");
      const std::string id = std::to_string(5 * row + column);
      addRaySegment(renderer, "primitive_hit_" + id, origin, contacts[0].getPosition(), 0.015,
                    glm::vec4(0.15f, 0.9f, 0.55f, 1.f));
      const Eigen::Vector3d missOrigin = origin + Eigen::Vector3d(1.25, 1.25, 0.0);
      if (scene->rayTest(missOrigin, Eigen::Vector3d(0.0, 0.0, -1.0), 10.0).size() != 0)
        throw std::runtime_error("Primitive benchmark illustration miss ray hit an object");
      addRaySegment(renderer, "primitive_miss_" + id, missOrigin,
                    {missOrigin.x(), missOrigin.y(), -0.7}, 0.015, glm::vec4(1.f, 0.3f, 0.2f, 1.f));
    }
  }
  std::printf("doc_image: primitive_ray_scaling objects=%d hit_traces=20 miss_traces=20\n", kObjects);
  return renderer.capture(out / "benchmark_primitive_ray_scaling.png", {-122.f, -137.f, 8.f},
                          {-110.f, -105.f, 0.f}, 48.f, 500.f);
}

bool renderCloth(const std::filesystem::path& out) {
  auto scene = std::make_shared<raisim::World>();
  scene->setSleepingEnabled(false);
  scene->setTimeStep(0.001);
  scene->addGround(0.0)->setAppearance("checkerboard");
  auto* sphere = scene->addSphere(0.25, 1.0);
  sphere->setBodyType(raisim::BodyType::STATIC);
  sphere->setPosition(0.0, 0.0, 0.25);
  sphere->setAppearance("0.15,0.4,0.75,1");
  std::vector<raisim::Vec<3>> vertices;
  std::vector<raisim::DeformableObject::Triangle> triangles;
  constexpr int kGrid = 32;
  for (int y = 0; y < kGrid; ++y) {
    for (int x = 0; x < kGrid; ++x) vertices.push_back({(x - 15.5) * 0.035, (y - 15.5) * 0.035, 0.85});
  }
  for (int y = 0; y + 1 < kGrid; ++y) {
    for (int x = 0; x + 1 < kGrid; ++x) {
      const unsigned int i = static_cast<unsigned int>(y * kGrid + x);
      triangles.push_back({i, i + 1, i + kGrid + 1});
      triangles.push_back({i, i + kGrid + 1, i + kGrid});
    }
  }
  raisim::DeformableObject::Material material;
  material.totalMass = 10.24;
  material.distanceCompliance = 1.0e-5;
  material.bendCompliance = -1.0;
  material.airDrag = 5.0;
  material.collisionRadius = 0.004;
  material.iterations = 6;
  material.substeps = 1;
  auto* cloth = scene->addDeformableCloth(vertices, triangles, {}, material);
  for (int step = 0; step < 450; ++step) scene->integrate();
  if (cloth->getNumParticles() != 1024 || cloth->getTriangles().size() != 1922)
    throw std::runtime_error("Cloth illustration does not match the recorded benchmark grid");
  std::printf("doc_image: deformable_cloth particles=%zu triangles=%zu\n",
              cloth->getNumParticles(), cloth->getTriangles().size());
  Renderer renderer(scene);
  return renderer.capture(out / "benchmark_deformable_cloth.png", {1.15f, -1.4f, 0.9f}, {0.f, 0.f, 0.28f}, 45.f);
}

void writeCubeSurface(const std::filesystem::path& path, const Eigen::Matrix3d& rotation) {
  constexpr int kGrid = 5;
  std::array<int, kGrid * kGrid * kGrid> ids;
  ids.fill(-1);
  auto index = [](int x, int y, int z) { return (z * kGrid + y) * kGrid + x; };
  std::ofstream file(path);
  if (!file) throw std::runtime_error("Cannot write temporary benchmark cube mesh");
  int count = 0;
  for (int z = 0; z < kGrid; ++z) {
    for (int y = 0; y < kGrid; ++y) {
      for (int x = 0; x < kGrid; ++x) {
        if (x != 0 && x != 4 && y != 0 && y != 4 && z != 0 && z != 4) continue;
        const Eigen::Vector3d point = rotation * Eigen::Vector3d(-0.15 + 0.075 * x, -0.15 + 0.075 * y, -0.15 + 0.075 * z);
        ids[index(x, y, z)] = ++count;
        file << "v " << point.x() << " " << point.y() << " " << point.z() << "\n";
      }
    }
  }
  auto vertex = [&](int x, int y, int z) { return ids[index(x, y, z)]; };
  auto quad = [&](int a, int b, int c, int d) {
    file << "f " << a << " " << c << " " << b << "\nf " << a << " " << d << " " << c << "\n";
  };
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      quad(vertex(a, b, 0), vertex(a + 1, b, 0), vertex(a + 1, b + 1, 0), vertex(a, b + 1, 0));
      quad(vertex(a, b, 4), vertex(a, b + 1, 4), vertex(a + 1, b + 1, 4), vertex(a + 1, b, 4));
      quad(vertex(a, 0, b), vertex(a, 0, b + 1), vertex(a + 1, 0, b + 1), vertex(a + 1, 0, b));
      quad(vertex(a, 4, b), vertex(a + 1, 4, b), vertex(a + 1, 4, b + 1), vertex(a, 4, b + 1));
      quad(vertex(0, a, b), vertex(0, a + 1, b), vertex(0, a + 1, b + 1), vertex(0, a, b + 1));
      quad(vertex(4, a, b), vertex(4, a, b + 1), vertex(4, a + 1, b + 1), vertex(4, a + 1, b));
    }
  }
  if (!file) throw std::runtime_error("Cannot finish temporary benchmark cube mesh");
}

bool renderCubeStack(const std::filesystem::path& out) {
  auto scene = std::make_shared<raisim::World>();
  scene->setSleepingEnabled(false);
  scene->setTimeStep(0.001);
  scene->addGround(0.0, "ground")->setAppearance("checkerboard");
  scene->setMaterialPairProp("ground", "deformable_cube", 0.9, 0.75, 0.02);
  scene->setMaterialPairProp("deformable_cube", "deformable_cube", 0.9, 0.65, 0.02);
  constexpr double kRadius = 0.06525;
  raisim::DeformableObject::Material material;
  material.totalMass = 1.0;
  material.distanceCompliance = 7.5e-5;
  material.airDrag = 4.0;
  material.collisionRadius = kRadius;
  material.iterations = 6;
  material.substeps = 1;
  raisim::DeformableObject::MeshBuildOptions build;
  build.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Surface;
  build.particles.spacing = 1.5 * 0.075;
  build.internalStruts.mode = raisim::DeformableObject::InternalStrutOptions::Mode::PairsWithinRadius;
  build.internalStruts.radius = 2.05 * 0.075;

  struct TemporaryMeshes {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("raisim_doc_benchmark_cubes_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryMeshes() { std::filesystem::create_directories(directory); }
    ~TemporaryMeshes() {
      std::error_code error;
      std::filesystem::remove_all(directory, error);
    }
  } meshes;
  double top = 0.20;
  size_t particles = 0;
  std::vector<raisim::DeformableObject*> cubes;
  for (int i = 0; i < 8; ++i) {
    const double roll = 0.17 * (i + 1), pitch = 0.11 * (i + 2), yaw = 0.23 * (i + 3);
    const Eigen::Matrix3d rotation = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const double halfHeight = 0.15 * (std::abs(std::sin(pitch)) + std::abs(std::cos(pitch) * std::sin(roll)) +
                                    std::abs(std::cos(pitch) * std::cos(roll)));
    const double z = top + halfHeight;
    top = z + halfHeight + 2.0 * kRadius + 0.02;
    const auto mesh = meshes.directory / ("cube_" + std::to_string(i) + ".obj");
    writeCubeSurface(mesh, rotation);
    auto* cube = scene->addDeformableObject(mesh.string(), material, build, {}, "deformable_cube");
    cube->setPositionOffset({i % 2 == 0 ? -0.035 : 0.035, (i / 2) % 2 == 0 ? -0.025 : 0.025, z});
    cube->setName("benchmark_cube_" + std::to_string(i));
    cubes.push_back(cube);
    particles += cube->getNumParticles();
  }
  if (particles != 784) throw std::runtime_error("Cube illustration does not match the benchmark particle count");
  for (int step = 0; step < 400; ++step) scene->integrate();
  Eigen::Vector3d minimum = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d maximum = -minimum;
  for (const auto* cube : cubes) {
    for (size_t particle = 0; particle < cube->getNumParticles(); ++particle) {
      const Eigen::Vector3d position = cube->getVisualPosition(particle).e();
      minimum = minimum.cwiseMin(position);
      maximum = maximum.cwiseMax(position);
    }
  }
  const Eigen::Vector3d center = 0.5 * (minimum + maximum);
  const glm::vec3 target(float(center.x()), float(center.y()), float(center.z()));
  const float distance = float(2.2 * (maximum - minimum).norm());
  const glm::vec3 eye = target + distance * glm::normalize(glm::vec3(1.f, -1.25f, 0.3f));
  std::printf("doc_image: deformable_cube_stack cubes=8 particles=%zu\n", particles);
  Renderer renderer(scene);
  return renderer.capture(out / "benchmark_deformable_cube_stack.png", eye, target, 42.f);
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
                  renderHeightmapAnymal(outputDir, rsc) && renderPrimitives(outputDir) &&
                  renderHeightmapLidar(outputDir, rsc) && renderPrimitiveRays(outputDir) &&
                  renderCloth(outputDir) && renderCubeStack(outputDir);
  doc_image::finishAndExit(ok ? 0 : 1);
}
