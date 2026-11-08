// Generates rayrai_quality_<preset>.png for Fast / Balanced / High / Ultra.
//
// Used by docs/sections/Rayrai.md to illustrate "Render-quality controls".

#include "doc_image_common.hpp"

namespace {

constexpr int kWidth = 960;
constexpr int kHeight = 540;

struct PresetSpec {
  const char* fileSuffix;
  raisin::RayraiWindow::RenderQualityPreset preset;
};

const PresetSpec kPresets[] = {
    {"fast",     raisin::RayraiWindow::RenderQualityPreset::Fast},
    {"balanced", raisin::RayraiWindow::RenderQualityPreset::Balanced},
    {"high",     raisin::RayraiWindow::RenderQualityPreset::High},
    {"ultra",    raisin::RayraiWindow::RenderQualityPreset::Ultra},
};

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_quality_presets")) return 1;

  for (const auto& spec : kPresets) {
    auto world = std::make_shared<raisim::World>();
    raisin::RayraiWindow renderer(world, kWidth, kHeight);

    auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(spec.preset);
    doc_image::applyCommonSceneOptions(quality, spec.preset);
    renderer.setRenderQualitySettings(quality);

    doc_image::buildStarterScene(*world, renderer);

    const auto path = outputDir / (std::string("rayrai_quality_") + spec.fileSuffix + ".png");
    if (!doc_image::captureScene(renderer, kWidth, kHeight, path))
      doc_image::finishAndExit(1);
    std::printf("doc_image: wrote %s\n", path.string().c_str());
  }
  doc_image::finishAndExit(0);
}
