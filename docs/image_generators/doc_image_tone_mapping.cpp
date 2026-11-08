// Generates rayrai_tonemap_<mode>.png for FastLinear / ACES / UnrealPreview /
// Filmic / AgX. The only setting that varies between images is
// `colorMode` — everything else is whatever the High preset ships with.
// If the comparison reads as too bright/dim/saturated, the fix belongs in
// the rayrai library's default High preset, not in this generator.
//
// Used by docs/sections/Rayrai.md "Tone mapping, exposure, and color grading".

#include "doc_image_common.hpp"

namespace {

constexpr int kWidth = 960;
constexpr int kHeight = 540;

struct ToneMapSpec {
  const char* fileSuffix;
  raisin::ViewerColorMode mode;
};

const ToneMapSpec kModes[] = {
    {"fast_linear",     raisin::ViewerColorMode::FastLinear},
    {"aces",            raisin::ViewerColorMode::AcesApprox},
    {"unreal_preview",  raisin::ViewerColorMode::UnrealPreviewApprox},
    {"filmic",          raisin::ViewerColorMode::FilmicApprox},
    {"agx",             raisin::ViewerColorMode::AgXApprox},
};

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_tone_mapping")) doc_image::finishAndExit(1);

  for (const auto& spec : kModes) {
    auto world = std::make_shared<raisim::World>();
    raisin::RayraiWindow renderer(world, kWidth, kHeight);

    const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
    auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
    // The only variable across the five images.
    quality.colorMode = spec.mode;
    doc_image::applyCommonSceneOptions(quality, preset);
    renderer.setRenderQualitySettings(quality);

    doc_image::buildStarterScene(*world, renderer);

    const auto path = outputDir / (std::string("rayrai_tonemap_") + spec.fileSuffix + ".png");
    if (!doc_image::captureScene(renderer, kWidth, kHeight, path))
      doc_image::finishAndExit(1);
    std::printf("doc_image: wrote %s\n", path.string().c_str());
  }
  doc_image::finishAndExit(0);
}
