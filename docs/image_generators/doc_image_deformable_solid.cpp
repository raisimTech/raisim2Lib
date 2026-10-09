// Elastic solids for the DeformableObject page: two soft rubber cubes, each squeezed by the same
// heavy plate. The left one has Poisson's ratio 0 and only shortens; the right one has 0.45 and
// bulges sideways as it shortens, keeping its volume. Wire frames mark the rest shape.

#include "doc_image_common.hpp"

#include <cstdio>
#include <fstream>
#include <string>

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr double kSize = 0.3;

void writeCubeObj(const std::filesystem::path& path, double size) {
  const double h = 0.5 * size;
  std::ofstream out(path);
  out << "v " << -h << " " << -h << " " << -h << "\nv " << h << " " << -h << " " << -h << "\n";
  out << "v " << h << " " << h << " " << -h << "\nv " << -h << " " << h << " " << -h << "\n";
  out << "v " << -h << " " << -h << " " << h << "\nv " << h << " " << -h << " " << h << "\n";
  out << "v " << h << " " << h << " " << h << "\nv " << -h << " " << h << " " << h << "\n";
  out << "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\n";
  out << "f 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n";
}

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_deformable_solid")) doc_image::finishAndExit(1);

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(0.001);
  world->addGround(0.0, "ground")->setAppearance("checkerboard");
  world->setMaterialPairProp("ground", "rubber", 0.9, 0.0, 0.0);
  world->setMaterialPairProp("plate", "rubber", 0.9, 0.0, 0.0);

  const auto obj = std::filesystem::temp_directory_path() / "doc_image_deformable_solid_cube.obj";
  writeCubeObj(obj, kSize);
  raisim::DeformableObject::MeshBuildOptions build;
  build.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Filled;
  build.particles.spacing = 0.025;

  // seen from -x, +y is on the left
  const double poisson[2] = {0.0, 0.45};
  const double side[2] = {0.36, -0.36};
  for (int k = 0; k < 2; ++k) {
    raisim::DeformableObject::Material material;
    material.totalMass = 2.0;
    material.youngsModulus = 2.0e4;
    material.poissonRatio = poisson[k];
    material.damping = 0.003;
    material.airDrag = 2.0;
    material.iterations = 10;
    material.substeps = 2;
    auto* cube = world->addDeformableObject(obj.string(), material, build, {}, "rubber");
    cube->setPositionOffset({0.0, side[k], 0.5 * kSize + 0.001});

    auto* plate = world->addBox(0.34, 0.34, 0.015, 15.0, "plate");
    plate->setPosition(0.0, side[k], kSize + 0.0125);
    plate->setAppearance("0.30, 0.34, 0.40, 1.0");
  }
  std::filesystem::remove(obj);
  for (int step = 0; step < 3000; ++step) world->integrate();

  raisin::RayraiWindow renderer(world, kWidth, kHeight);
  const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrToneMapping = true;
  doc_image::applyCommonSceneOptions(quality, preset);
  renderer.setRenderQualitySettings(quality);

  // the rest shape of each cube: its twelve edges
  constexpr double kWire = 0.004;
  const double h = 0.5 * kSize;
  int edge = 0;
  for (int k = 0; k < 2; ++k) {
    for (int axis = 0; axis < 3; ++axis) {
      for (const double a : {-h, h}) {
        for (const double b : {-h, h}) {
          double extent[3] = {kWire, kWire, kWire};
          extent[axis] = kSize + kWire;
          double center[3];
          center[axis] = 0.0;
          center[(axis + 1) % 3] = a;
          center[(axis + 2) % 3] = b;
          auto wire = renderer.addVisualBox("rest" + std::to_string(edge++), extent[0], extent[1], extent[2],
                                            glm::vec4(0.92f, 0.94f, 1.0f, 1.0f));
          wire->setPosition(center[0], side[k] + center[1], h + center[2]);
          wire->setDetectable(true);
        }
      }
    }
  }

  doc_image::setCameraLookAt(renderer.getCamera(), glm::vec3(-1.5f, -0.05f, 0.42f),
                             glm::vec3(0.0f, 0.0f, 0.15f), /*horizontalFovDeg=*/42.0f);
  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;
  overrides.drawCoordinateFrames = false;
  const bool ok = doc_image::captureScene(renderer, kWidth, kHeight,
                                          outputDir / "deformable_solid_poisson.png", 2, overrides);
  doc_image::finishAndExit(ok ? 0 : 1);
}
