// The photoreal warehouse seen through one camera: the rendered image on the left and its
// segmentation on the right. Drag on the left half to move the camera.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <glbinding/gl/gl.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "example_resources.hpp"
#include "raisim/World.hpp"
#include "raisim/sensors/SegmentationSensor.hpp"
#include "rayrai/RsceneVisuals.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_loading_progress.hpp"
#include "stb/stb_image_write.h"

namespace {

// Segmentation class of a scene path: the first keyword its last component contains. Carton loads
// are named by their size, e.g. 290x170x190_1.
std::string classOf(const std::string& path) {
  static const std::vector<std::pair<std::string, std::string>> kKeywords = {
    {"anymal", "robot"}, {"palletjack", "pallet jack"}, {"forklift", "forklift"}, {"pallet", "pallet"},
    {"carton", "carton"}, {"crate", "crate"}, {"barrel", "barrel"}, {"shelf", "shelf"},
    {"racking", "racking"}, {"extinguisher", "extinguisher"}, {"propane", "gas tank"}, {"cone", "cone"},
    {"camera", "camera"}, {"sign", "sign"}, {"desk", "desk"}, {"cart", "cart"}, {"truck", "cart"},
    {"floor", "floor"}, {"yard", "floor"}, {"door", "door"}};
  std::string name = path.substr(path.find_last_of('/') + 1);
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if (!name.empty() && std::isdigit(static_cast<unsigned char>(name.front()))) return "carton";
  for (const auto& [keyword, label] : kKeywords)
    if (name.find(keyword) != std::string::npos) return label;
  return "building";
}

raisim::ArticulatedSystem* addStandingAnymal(raisim::World& world, const std::string& urdf) {
  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto* anymal = world.addArticulatedSystem(urdf, "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1),
                                            options);
  anymal->setName("anymal");
  const double yaw = -0.6;
  Eigen::VectorXd gc(anymal->getGeneralizedCoordinateDim()), gv(anymal->getDOF());
  gc << 4.4, 0.3, 0.56, std::cos(yaw / 2), 0, 0, std::sin(yaw / 2), 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4,
      0.8, -0.03, -0.4, 0.8;
  gv.setZero();
  Eigen::VectorXd pGain = Eigen::VectorXd::Zero(anymal->getDOF()), dGain = pGain;
  pGain.tail(12).setConstant(200.0);
  dGain.tail(12).setConstant(10.0);
  anymal->setState(gc, gv);
  anymal->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  anymal->setPdGains(pGain, dGain);
  anymal->setPdTarget(gc, gv);
  return anymal;
}

// World objects carry their labels themselves. Most of a photoreal scene is drawn by rayrai visuals
// instead (visual-only meshes, and the visual meshes of .rasset props whose colliders are hidden),
// so they get ids here: a prop's collider body gives the world index, and a visual-only mesh keeps
// SegmentationCamera::kNotWorldObject.
void labelSceneVisuals(raisim::World& world, raisin::RayraiWindow& viewer) {
  const auto& scene = raisin::rsceneVisuals(viewer);
  for (const auto& visual : scene.objects) {
    const int32_t label = world.addSegmentationClass(classOf(visual->getName()));
    if (const auto* body = world.getObject(visual->getName()))
      visual->setSegmentationIds(int32_t(body->getIndexInWorld()), 0, label);
    else
      visual->setSegmentationLabel(label);
  }
  for (const auto& batch : scene.instancedVisuals)
    batch->setSegmentationLabel(world.addSegmentationClass(classOf(batch->getName())));
}

void uploadRgba(unsigned int& texture, int width, int height, const std::vector<unsigned char>& rgba) {
  if (texture == 0) {
    gl::glGenTextures(1, &texture);
    gl::glBindTexture(gl::GL_TEXTURE_2D, texture);
    gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MIN_FILTER, gl::GL_NEAREST);
    gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MAG_FILTER, gl::GL_NEAREST);
  }
  gl::glBindTexture(gl::GL_TEXTURE_2D, texture);
  gl::glTexImage2D(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA8, width, height, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE,
                   rgba.data());
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
}

// The rendered image (bottom row first in the texture) beside the segmentation colors (top row first).
bool saveSideBySide(raisin::Camera& camera, const std::vector<unsigned char>& segmentation,
                    const std::filesystem::path& path) {
  const int width = camera.rtWidth(), height = camera.rtHeight();
  const size_t row = size_t(width) * 4;
  std::vector<unsigned char> color(row * height), image(2 * row * height);
  gl::glBindTexture(gl::GL_TEXTURE_2D, camera.getFinalTexture());
  gl::glGetTexImage(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, color.data());
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
  for (int y = 0; y < height; ++y) {
    std::copy_n(&color[(height - 1 - y) * row], row, &image[2 * y * row]);
    std::copy_n(&segmentation[y * row], row, &image[2 * y * row + row]);
  }
  return stbi_write_png(path.string().c_str(), 2 * width, height, 4, image.data(), int(2 * row)) != 0;
}

ImVec4 swatch(int32_t id) {
  const auto c = raisim::SegmentationCamera::colorOf(id);
  return ImVec4(c[0] / 255.f, c[1] / 255.f, c[2] / 255.f, 1.f);
}

}  // namespace

int main(int argc, char* argv[]) {
  std::filesystem::path scene = exampleRscPath(argv[0], "warehouse/rayrai_warehouse.rscene"), screenshot;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--scene" && i + 1 < argc) {
      scene = argv[++i];
    } else if (option == "--screenshot" && i + 1 < argc) {
      screenshot = argv[++i];
    } else {
      std::cerr << "Usage: rayrai_segmentation_camera [--scene FILE.rscene] [--screenshot PNG]\n";
      return 2;
    }
  }
  auto world = std::make_shared<raisim::World>(scene.string());
  auto* anymal = addStandingAnymal(*world, exampleRscPath(argv[0], "anymal_c/urdf/anymal.urdf"));
  for (auto* object : world->getObjList())
    object->setSegmentationLabel(world->addSegmentationClass(classOf(object->getName())));
  const int32_t legs = world->addSegmentationClass("robot leg");
  for (const auto& body : anymal->getBodyNames())
    if (body.find("THIGH") != std::string::npos || body.find("SHANK") != std::string::npos)
      anymal->setBodySegmentationLabel(body, legs);

  ExampleApp app;
  if (!app.init("RaiSim segmentation camera", 1600, 700, screenshot.empty())) return 1;
  ImGui::GetIO().IniFilename = nullptr;
  int status = 0;
  {
    raisin::RayraiWindow viewer(world, 800, 700);
    viewer.setAsyncMeshLoadingEnabled(true);
    labelSceneVisuals(*world, viewer);
    RayraiLoadingProgress loading("Loading warehouse assets");
    std::vector<int32_t> objects, bodies, labels;
    std::vector<unsigned char> rgba;
    unsigned int segmentationTexture = 0;
    int showObjects = 0;
    int settleFrames = 60;

    while (!app.quit) {
      app.processEvents();
      if (app.quit) break;
      loading.update(viewer.pendingAsyncMeshLoadCount());
      if (!loading.visible())
        for (int step = 0; step < 8; ++step) world->integrate();

      app.beginFrame();
      auto view = app.beginViewport();
      const ImVec2 half(view.size.x * 0.5f, view.size.y);
      const ImVec2 right(view.origin.x + half.x, view.origin.y);
      view.hovered = view.hovered && ImGui::GetIO().MousePos.x < right.x;
      viewer.update(std::max(1, view.width / 2), view.height, view.hovered, int(view.x), int(view.y), false);
      auto* drawList = ImGui::GetWindowDrawList();
      drawList->AddImage((ImTextureID)(intptr_t)viewer.getImageTexture(), view.origin,
                         ImVec2(view.origin.x + half.x, view.origin.y + half.y), ImVec2(0, 1), ImVec2(1, 0));

      // The same camera, segmented. Visuals are drawn too: they hold most of the scene.
      auto& camera = viewer.getCamera();
      viewer.renderSegmentation(camera, true, false);
      const int width = camera.rtWidth(), height = camera.rtHeight();
      const size_t pixels = size_t(width) * size_t(height);
      objects.resize(pixels);
      bodies.resize(pixels);
      labels.resize(pixels);
      camera.readSegmentation(objects.data(), bodies.data(), labels.data(), pixels);
      raisim::SegmentationCamera::colorize(showObjects ? objects : labels, rgba, {0, 0, 0, 255});
      uploadRgba(segmentationTexture, width, height, rgba);
      drawList->AddImage((ImTextureID)(intptr_t)segmentationTexture, right, ImVec2(right.x + half.x, right.y + half.y));

      const ImVec2 mouse = ImGui::GetIO().MousePos;
      if (mouse.x >= right.x && mouse.x < right.x + half.x && mouse.y >= right.y && mouse.y < right.y + half.y) {
        const int col = std::clamp(int((mouse.x - right.x) / half.x * width), 0, width - 1);
        const int row = std::clamp(int((mouse.y - right.y) / half.y * height), 0, height - 1);
        const size_t pixel = size_t(row) * size_t(width) + size_t(col);
        if (labels[pixel] < 0) {
          ImGui::SetTooltip("nothing");
        } else {
          const int32_t object = objects[pixel];
          const std::string owner = object >= 0 ? world->getObject(size_t(object))->getName() : "visual-only mesh";
          ImGui::SetTooltip("%s\nobject %d (%s), body %d", world->getSegmentationClassName(labels[pixel]).c_str(),
                            object, owner.c_str(), bodies[pixel]);
        }
      }
      ImGui::End();
      ImGui::PopStyleVar();

      ImGui::SetNextWindowPos(ImVec2(right.x + 10.f, right.y + 10.f), ImGuiCond_FirstUseEver);
      ImGui::Begin("Segmentation", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::RadioButton("classes", &showObjects, 0);
      ImGui::SameLine();
      ImGui::RadioButton("objects", &showObjects, 1);
      if (!showObjects) {
        const auto& names = world->getSegmentationClassNames();
        for (size_t label = 1; label < names.size(); ++label) {
          ImGui::ColorButton(names[label].c_str(), swatch(int32_t(label)), ImGuiColorEditFlags_NoTooltip,
                             ImVec2(12, 12));
          ImGui::SameLine();
          ImGui::TextUnformatted(names[label].c_str());
        }
      }
      ImGui::End();
      if (screenshot.empty()) loading.draw();
      app.endFrame();
      if (!screenshot.empty() && !loading.visible() && --settleFrames == 0) {
        status = saveSideBySide(camera, rgba, screenshot) ? 0 : 1;
        break;
      }
    }
    if (segmentationTexture != 0) gl::glDeleteTextures(1, &segmentationTexture);
  }  // Release graphics resources before closing the OpenGL context.
  app.shutdown();
  return status;
}
