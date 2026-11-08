// Generates rayrai_weather_<preset>.png for the most visually distinct
// weather presets.
//
// Used by docs/sections/Rayrai.md "Weather, scene effects, and post-process".

#include "doc_image_common.hpp"

namespace {

constexpr int kWidth = 960;
constexpr int kHeight = 540;

struct WeatherSpec {
  const char* fileSuffix;
  raisin::WeatherPreset preset;
  float timeOfDayHours;
};

const WeatherSpec kPresets[] = {
    {"clear",        raisin::WeatherPreset::Clear,       12.0f},
    {"overcast",     raisin::WeatherPreset::Overcast,    13.5f},
    {"rain",         raisin::WeatherPreset::Rain,        15.5f},
    {"snow",         raisin::WeatherPreset::Snow,        11.0f},
    {"storm",        raisin::WeatherPreset::Storm,        9.0f},
    {"night_clear",  raisin::WeatherPreset::NightClear,  21.5f},
};

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_weather_presets")) return 1;

  for (const auto& spec : kPresets) {
    auto world = std::make_shared<raisim::World>();
    raisin::RayraiWindow renderer(world, kWidth, kHeight);

    const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
    auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
    quality.colorMode = raisin::ViewerColorMode::AcesApprox;
    quality.pbrToneMapping = true;
    doc_image::applyCommonSceneOptions(quality, preset);
    renderer.setRenderQualitySettings(quality);

    auto weather = raisin::RayraiWindow::defaultWeatherSettings(spec.preset);
    weather.enabled = true;
    weather.timeOfDayHours = spec.timeOfDayHours;
    weather.affectSensors = false;
    renderer.setWeatherSettings(weather);

    // Run the weather state forward a little so particle systems, sky LUTs
    // and wetness ramps are stable before the capture.
    for (int i = 0; i < 60; ++i) {
      renderer.updateWeather(1.0 / 60.0);
    }

    doc_image::buildStarterScene(*world, renderer);

    const auto path = outputDir / (std::string("rayrai_weather_") + spec.fileSuffix + ".png");
    if (!doc_image::captureScene(renderer, kWidth, kHeight, path))
      doc_image::finishAndExit(1);
    std::printf("doc_image: wrote %s\n", path.string().c_str());
  }
  doc_image::finishAndExit(0);
}
