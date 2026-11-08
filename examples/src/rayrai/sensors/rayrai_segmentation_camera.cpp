// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
// Compare RGB and segmentation from the same camera and physics snapshot.
// Run normally for the interactive app, or use --screenshot comparison.png to save and exit.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "raisim/World.hpp"
#include "raisim/sensors/RGBSensor.hpp"
#include "raisim/sensors/SegmentationSensor.hpp"
#include "rayrai/GlStateGuard.hpp"
#include "rayrai/PhysicsSnapshot.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_compat.hpp"
#include "rayrai_example_utils.hpp"

namespace {
constexpr int kWidth = 640, kHeight = 400;
using Segmentation = raisim::SegmentationCamera;

// An integer segmentation texture cannot be displayed directly by ImGui.
// Upload the colorized IDs as RGBA8, preserving the renderer's GL state.
class SegmentationPreview {
 public:
  ~SegmentationPreview() { if (texture_) gl::glDeleteTextures(1, &texture_); }
  SegmentationPreview() = default;
  SegmentationPreview(const SegmentationPreview&) = delete;
  SegmentationPreview& operator=(const SegmentationPreview&) = delete;

  void upload(const std::vector<unsigned char>& rgba) {
    const raisin::ScopedPixelUnpackBufferZero unpackBuffer;
    const std::array<gl::GLenum, 4> fields = {gl::GL_UNPACK_ALIGNMENT,
      gl::GL_UNPACK_ROW_LENGTH, gl::GL_UNPACK_SKIP_ROWS, gl::GL_UNPACK_SKIP_PIXELS};
    const std::array<gl::GLint, 4> values = {1, 0, 0, 0};
    std::array<gl::GLint, 4> previous{};
    gl::GLint binding = 0;
    gl::glGetIntegerv(gl::GL_TEXTURE_BINDING_2D, &binding);
    for (size_t i = 0; i < fields.size(); ++i) {
      gl::glGetIntegerv(fields[i], &previous[i]);
      gl::glPixelStorei(fields[i], values[i]);
    }
    const bool created = texture_ == 0;
    if (created) gl::glGenTextures(1, &texture_);
    gl::glBindTexture(gl::GL_TEXTURE_2D, texture_);
    if (created) {
      // Nearest sampling keeps class boundaries sharp when the window is resized.
      gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MIN_FILTER, gl::GL_NEAREST);
      gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MAG_FILTER, gl::GL_NEAREST);
      gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_S, gl::GL_CLAMP_TO_EDGE);
      gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_T, gl::GL_CLAMP_TO_EDGE);
      gl::glTexImage2D(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA8, kWidth, kHeight, 0,
                     gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
    } else {
      gl::glTexSubImage2D(gl::GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight,
                        gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
    }
    gl::glBindTexture(gl::GL_TEXTURE_2D, static_cast<gl::GLuint>(binding));
    for (size_t i = 0; i < fields.size(); ++i) gl::glPixelStorei(fields[i], previous[i]);
  }

  unsigned int texture() const { return texture_; }

 private:
  unsigned int texture_ = 0;
};

void legendEntry(const char* name, int32_t id) {
  const auto color = Segmentation::colorOf(id);
  ImGui::ColorButton(name, ImVec4(color[0] / 255.f, color[1] / 255.f, color[2] / 255.f, 1.f),
                    ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14));
  ImGui::SameLine();
  ImGui::TextUnformatted(name);
  ImGui::SameLine(0, 18);
}

// Both readbacks use top-left row order. RGB readback is BGRA; PNG expects RGBA.
void saveComparison(raisin::Camera& camera, const std::vector<unsigned char>& segmentation,
                    const std::filesystem::path& path) {
  std::vector<char> bgra(size_t(kWidth) * kHeight * 4);
  camera.readSceneColorBgra(bgra.data(), bgra.size());
  std::vector<unsigned char> combined(size_t(kWidth) * kHeight * 8);
  for (int y = 0; y < kHeight; ++y) {
    const size_t src = size_t(y) * kWidth * 4;
    const size_t dst = size_t(y) * kWidth * 8;
    for (int x = 0; x < kWidth; ++x) {
      const size_t pixel = size_t(x) * 4;
      combined[dst + pixel] = static_cast<unsigned char>(bgra[src + pixel + 2]);
      combined[dst + pixel + 1] = static_cast<unsigned char>(bgra[src + pixel + 1]);
      combined[dst + pixel + 2] = static_cast<unsigned char>(bgra[src + pixel]);
      combined[dst + pixel + 3] = 255;
    }
    std::copy_n(segmentation.data() + src, kWidth * 4, combined.data() + dst + kWidth * 4);
  }
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  if (!stbi_write_png(path.string().c_str(), kWidth * 2, kHeight, 4, combined.data(), kWidth * 8))
    throw std::runtime_error("Cannot write " + path.string());
  std::cout << "Saved " << path << " (RGB left, segmentation right)\n";
}

void run(ExampleApp& app, const std::filesystem::path& screenshot) {
  int mode = 0;
  auto world = std::make_shared<raisim::World>();
  auto ground = world->addGround();
  ground->setName("Ground");
  ground->setAppearance("0.36,0.40,0.44,1.0");
  ground->setSegmentationLabel(world->addSegmentationClass("Ground"));
  const int32_t boxClass = world->addSegmentationClass("Box");
  const int32_t sphereClass = world->addSegmentationClass("Sphere");
  const int32_t cylinderClass = world->addSegmentationClass("Cylinder");

  const auto configure = [](auto* body, const char* name, int32_t label,
                            const char* appearance, double x, double y, double z) {
    body->setName(name);
    body->setBodyType(raisim::BodyType::STATIC);
    body->setPosition(x, y, z);
    body->setAppearance(appearance);
    body->setSegmentationLabel(label);
  };
  auto box1 = world->addBox(0.9, 0.9, 1.3, 1.0);
  configure(box1, "Box A", boxClass, "0.90,0.24,0.12,1.0", -1.4, 0.25, 0.65);
  auto box2 = world->addBox(0.7, 0.7, 0.7, 1.0);
  configure(box2, "Box B", boxClass, "0.15,0.55,0.95,1.0", -0.4, -0.45, 0.35);
  auto sphere = world->addSphere(0.55, 1.0);
  configure(sphere, "Sphere", sphereClass, "0.95,0.72,0.12,1.0", 0.65, 0.15, 0.55);
  auto cylinder = world->addCylinder(0.4, 1.5, 1.0);
  configure(cylinder, "Cylinder", cylinderClass, "0.22,0.75,0.40,1.0", 1.8, 0.35, 0.75);
  const std::array<raisim::Object*, 5> objects = {ground, box1, box2, sphere, cylinder};

  // The renderer consumes a copy of the world, as in the other sensor examples.
  raisin::PhysicsSnapshotCapture capture;
  raisin::RayraiWindow viewer(std::make_shared<raisim::World>(), kWidth, kHeight);
  viewer.setRenderQualitySettings(raisin::RayraiWindow::defaultRenderQualitySettings(
    raisin::RayraiWindow::RenderQualityPreset::Balanced));
  raisim_examples::setRayraiBackgroundColorRgb255(viewer, {30, 34, 42, 255});
  viewer.setFogDensity(0.f);
  viewer.applyPhysicsSnapshot(capture.capture(*world));

  raisim::RGBCamera::RGBCameraProperties properties;
  properties.width = kWidth;
  properties.height = kHeight;
  properties.clipNear = 0.05;
  properties.clipFar = 50.0;
  properties.hFOV = 1.05;
  raisin::Camera camera(properties);
  camera.ensureSegmentationRenderTarget();
  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.drawCoordinateFrames = false;
  overrides.drawPointClouds = false;
  overrides.drawVisualizationObjects = false;
  overrides.drawWeatherEffects = false;
  // Disable temporal jitter and upscaling so both passes use the same pixel rays.
  overrides.allowTemporalAa = false;
  overrides.allowViewerUpscale = false;
  overrides.allowViewerMsaa = false;

  std::vector<int32_t> objectIds(size_t(kWidth) * kHeight);
  std::vector<int32_t> labels(objectIds.size());
  std::vector<unsigned char> rgba;
  SegmentationPreview preview;
  float azimuth = -80.f;
  bool rotate = false;
  int frame = 0;
  while (!app.quit) {
    app.processEvents();
    if (app.quit) break;
    app.beginFrame();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("RGB and Segmentation", nullptr,
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
    ImGui::TextUnformatted("RGB + segmentation  |  Same camera, same scene, 640 x 400");
    ImGui::SetNextItemWidth(210);
    ImGui::Combo("Color by", &mode, "Semantic class\0Object instance\0");
    ImGui::SameLine();
    ImGui::Checkbox("Rotate camera", &rotate);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(240);
    ImGui::SliderFloat("Camera angle", &azimuth, -180.f, 180.f, "%.0f deg");
    if (rotate) {
      azimuth += 15.f * std::min(ImGui::GetIO().DeltaTime, 0.1f);
      if (azimuth > 180.f) azimuth -= 360.f;
    }
    if (mode == 0) {
      const auto& classes = world->getSegmentationClassNames();
      for (size_t id = 1; id < classes.size(); ++id)
        legendEntry(classes[id].c_str(), static_cast<int32_t>(id));
    } else {
      for (const auto* object : objects)
        legendEntry(object->getName().c_str(), static_cast<int32_t>(object->getIndexInWorld()));
    }
    ImGui::TextUnformatted("Black: background");
    ImGui::TextUnformatted(mode == 0 ? "Both boxes share one class color." : "Each object has its own color.");
    ImGui::Separator();

    const float angle = glm::radians(azimuth);
    camera.position = {6.f * std::cos(angle), 6.f * std::sin(angle), 3.2f};
    camera.target = {0.1f, 0.f, 0.65f};
    camera.front = glm::normalize(camera.target - camera.position);
    camera.right = glm::normalize(glm::cross(camera.front, glm::vec3(0, 0, 1)));
    camera.up = camera.worldUp = glm::normalize(glm::cross(camera.right, camera.front));

    // Reuse one camera: identical pose, calibration, resolution and clipping planes.
    viewer.renderWithExternalCamera(camera, overrides);
    viewer.renderSegmentation(camera);
    // nullptr skips body IDs; instance IDs and semantic labels remain available separately.
    camera.readSegmentation(objectIds.data(), nullptr, labels.data(), objectIds.size());
    Segmentation::colorize(mode == 0 ? labels : objectIds, rgba, {0, 0, 0, 255});
    preview.upload(rgba);

    if (ImGui::BeginTable("Images", 2, ImGuiTableFlags_SizingStretchSame)) {
      for (int column = 0; column < 2; ++column) {
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(column == 0 ? "RGB camera" : "Segmentation camera");
        ImGui::BeginChild(column == 0 ? "RGB" : "Segmentation");
        const ImVec2 size = raisim_examples::fitToAspect(ImGui::GetContentRegionAvail(), float(kWidth) / kHeight);
        // Rendered RGB has GL's bottom-left origin. The uploaded preview is top-left.
        const auto texture = column == 0 ? camera.getFinalTexture() : preview.texture();
        ImGui::Image((ImTextureID)(intptr_t)texture, size,
                     ImVec2(0, column == 0 ? 1 : 0), ImVec2(1, column == 0 ? 0 : 1));
        ImGui::EndChild();
      }
      ImGui::EndTable();
    }
    ImGui::End();
    // External-camera passes leave an offscreen framebuffer bound.
    gl::glBindFramebuffer(gl::GL_FRAMEBUFFER, 0);
    int width = 0, height = 0;
    SDL_GL_GetDrawableSize(app.window, &width, &height);
    gl::glViewport(0, 0, width, height);
    gl::glClearColor(0.06f, 0.07f, 0.09f, 1.f);
    gl::glClear(gl::GL_COLOR_BUFFER_BIT);
    app.endFrame();
    if (!screenshot.empty() && ++frame == 3) {
      saveComparison(camera, rgba, screenshot);
      break;
    }
  }
  // GL resources above are destroyed while ExampleApp's context is still current.
}
}  // namespace

int main(int argc, char* argv[]) {
  std::filesystem::path screenshot;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--screenshot" && i + 1 < argc) screenshot = argv[++i];
    else {
      const bool help = arg == "--help";
      std::cout << "Usage: " << argv[0] << " [--screenshot comparison.png]\n";
      return help ? 0 : 1;
    }
  }
  ExampleApp app;
  if (!app.init("Rayrai RGB and Segmentation", 1320, 600, screenshot.empty())) return 1;
  ImGui::GetIO().IniFilename = nullptr;
  int result = 0;
  try {
    run(app, screenshot);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    result = 1;
  }
  app.shutdown();
  return result;
}
