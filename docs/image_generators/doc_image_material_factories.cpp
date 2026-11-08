// Generates rayrai_materials_factories.png — five primitives demonstrating
// the visual range produced by Material::pbr / unlitColor / simpleColor /
// foliage / defaultGround when applied to imported assets.
//
// Note: captureSupersampledRgba in the installed rayrai package does not
// render `addVisualSphere` / `addVisualBox` custom visuals reliably in the
// offscreen-only path we use here, so this image uses raisim::World
// primitives with setAppearance() to keep the demonstration self-contained.
// The Material factory API is illustrated in code inside Rayrai.md; this
// image is a visual companion to that snippet.

#include "doc_image_common.hpp"

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 480;

struct Slot {
  const char* name;
  const char* appearance;  // "r,g,b,a" string interpreted by raisim/rayrai.
};

// Match the colour and rough material intent of each Material factory.
const Slot kSlots[] = {
    {"pbr",     "0.95,0.43,0.12,1"},  // Material::pbr — saturated orange
    {"unlit",   "0.92,0.92,0.96,1"},  // Material::unlitColor — flat white
    {"simple",  "0.22,0.78,0.36,1"},  // Material::simpleColor — green
    {"foliage", "0.32,0.58,0.21,1"},  // Material::foliage — leaf green
    {"ground",  "0.55,0.55,0.55,1"},  // Material::defaultGround — mid-grey
};

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_material_factories")) return 1;

  auto world = std::make_shared<raisim::World>();
  raisin::RayraiWindow renderer(world, kWidth, kHeight);

  const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrToneMapping = true;
  quality.bloomEnabled = false;
  doc_image::applyCommonSceneOptions(quality, preset);
  renderer.setRenderQualitySettings(quality);

  world->addGround();

  constexpr double spacing = 0.9;
  constexpr double radius = 0.3;
  const double startX = -spacing * (static_cast<double>(std::size(kSlots)) - 1) / 2.0;
  for (size_t i = 0; i < std::size(kSlots); ++i) {
    auto* sphere = world->addSphere(radius, 1.0);
    sphere->setPosition(startX + spacing * static_cast<double>(i), 0.0, radius);
    sphere->setBodyType(raisim::BodyType::STATIC);
    sphere->setAppearance(kSlots[i].appearance);
  }

  doc_image::setCameraLookAt(renderer.getCamera(),
                             glm::vec3(0.0f, -2.8f, 0.55f),
                             glm::vec3(0.0f, 0.0f, static_cast<float>(radius)),
                             /*horizontalFovDeg=*/75.0f);

  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;
  overrides.drawVisualizationObjects = true;

  const auto path = outputDir / "rayrai_materials_factories.png";
  if (!doc_image::captureScene(renderer, kWidth, kHeight, path,
                               /*supersampleScale=*/2, overrides))
    doc_image::finishAndExit(1);
  std::printf("doc_image: wrote %s\n", path.string().c_str());
  doc_image::finishAndExit(0);
}
