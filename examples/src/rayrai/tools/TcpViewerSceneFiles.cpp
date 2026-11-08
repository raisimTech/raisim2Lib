// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerSceneFiles.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "rayrai/RaisimTcpCommon.hpp"
#include "rayrai/RayraiWindow.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/Water.hpp"

namespace raisin
{
namespace tcp_viewer
{

namespace
{

constexpr const char* kFluidWaterPrefix = "raisim_fluid_volume_";

SceneSync::Options syncOptions(std::filesystem::path cacheRoot) {
  SceneSync::Options options;
  options.cacheRoot = std::move(cacheRoot);
  return options;
}

} // namespace

ViewerSceneFiles::ViewerSceneFiles(std::filesystem::path cacheRoot)
  : sync_(syncOptions(std::move(cacheRoot))) {}

void ViewerSceneFiles::setResourceDirs(const std::vector<std::string>& dirs) {
  sync_.setResourceDirs({dirs.begin(), dirs.end()});
}

void ViewerSceneFiles::reset(RayraiWindow& viewer, RemoteScene& scene) {
  fluidVolumes_.clear();
  remove(viewer);
  viewer.setWaterTimeSeconds(0.0);
  scene.setServerPathMap({});
  sync_.reset();
  errorDismissed_ = false;
}

bool ViewerSceneFiles::update(RayraiWindow& viewer, RemoteScene& scene) {
  return meshLoading_.update(viewer, [&] { return apply(viewer, scene); });
}

bool ViewerSceneFiles::apply(RayraiWindow& viewer, RemoteScene& scene) {
  sync_.poll();
  if (sync_.state() != SceneSync::State::Ready) return false;
  // The sync starts over when the server stops and resumes sharing; names must stay unique.
  remove(viewer);
  // The scene's render settings replace the removed ones right away.
  viewerSettingsReset_ = false;
  touched_ = true;
  try {
    // The server's scene is not the viewer's world scene, so it is applied here. On failure,
    // applyRscene() removes the visuals it created before it throws.
    visuals_ = rayrai_internal::applyRscene(*sync_.scene(), viewer,
                                            rsceneRenderSettings(*sync_.scene()));
  } catch (const std::exception& error) {
    remove(viewer);
    sync_.markFailed(std::string("The scene could not be shown: ") + error.what());
    return false;
  }
  // Bodies the server streams name their mesh by its server path.
  scene.setServerPathMap(sync_.pathMap());
  sync_.markApplied();
  applied_ = true;
  showFluidVolumes(viewer);
  return true;
}

void ViewerSceneFiles::setFluidVolumes(RayraiWindow& viewer, std::vector<StreamedFluidVolume> volumes) {
  // the server repeats them every update; only a change is drawn again
  if (volumes == fluidVolumes_) return;
  fluidVolumes_ = std::move(volumes);
  showFluidVolumes(viewer);
}

void receiveServerControls(std::vector<ServerControlView>& shown, const std::vector<ServerControl>& sent) {
  bool same = shown.size() == sent.size();
  for (size_t i = 0; same && i < sent.size(); ++i)
    same = shown[i].control.name == sent[i].name && shown[i].control.min == sent[i].min &&
           shown[i].control.max == sent[i].max;
  if (!same) {
    shown.clear();
    for (const auto& control : sent) shown.push_back({control, control.value});
    return;
  }
  for (size_t i = 0; i < sent.size(); ++i) shown[i].serverValue = sent[i].value;
}

std::vector<bool> heldServerControls(size_t count, int dragged, const std::vector<ClientRequest>& pending) {
  std::vector<bool> held(count, false);
  if (dragged >= 0 && static_cast<size_t>(dragged) < count) held[static_cast<size_t>(dragged)] = true;
  for (const auto& request : pending)
    if (request.type == ClientRequestType::CR_SET_CONTROL && request.controlIndex >= 0 &&
        static_cast<size_t>(request.controlIndex) < count)
      held[static_cast<size_t>(request.controlIndex)] = true;
  return held;
}

void settleServerControls(std::vector<ServerControlView>& shown, const std::vector<bool>& held) {
  for (size_t i = 0; i < shown.size(); ++i)
    if (!(i < held.size() && held[i])) shown[i].control.value = shown[i].serverValue;
}

void applySceneStateCommands(const std::vector<ViewerCommand>& commands, ViewerSceneFiles& files,
                             RayraiWindow& viewer) {
  for (const auto& command : commands)
    if (command.type == ViewerCommandType::SetFluidVolumes) files.setFluidVolumes(viewer, command.fluidVolumes);
}

void ViewerSceneFiles::showFluidVolumes(RayraiWindow& viewer) {
  // an applied scene with water of its own hides the server's; only the volumes that changed are drawn
  // again
  const size_t count = applied_ && !visuals_.waterBodies.empty() ? 0 : fluidVolumes_.size();
  for (size_t i = count; i < fluidWaterShown_.size(); ++i)
    if (fluidWaterShown_[i]) viewer.removeWaterBody(kFluidWaterPrefix + std::to_string(i));
  fluidWaterShown_.resize(count);
  for (size_t i = 0; i < count; ++i) {
    const auto& streamed = fluidVolumes_[i];
    const auto& box = streamed.box;
    auto& shown = fluidWaterShown_[i];
    if (shown && *shown == streamed) continue;
    const std::string name = kFluidWaterPrefix + std::to_string(i);
    if (shown) viewer.removeWaterBody(name);
    shown.reset();
    // the server's values are checked as a world checks a volume; one it could not hold is not drawn
    raisim::FluidVolume volume;
    volume.min = {box[0], box[1]};
    volume.max = {box[2], box[3]};
    volume.surfaceHeight = box[4];
    volume.bottomHeight = box[5];
    for (const auto& wave : streamed.waves)
      volume.waves.push_back({{wave[0], wave[1]}, wave[2], wave[3], wave[4]});
    if (!raisim::invalidFluidVolume(volume).empty()) continue;
    const bool bounded = std::isfinite(box[0]) && std::isfinite(box[1]) && std::isfinite(box[2]) &&
                         std::isfinite(box[3]);
    try {
      viewer.setWaterBody(name, waterBodyForFluidVolume(volume, bounded ? WaterPreset::Pool : WaterPreset::Ocean));
      shown = streamed;
    } catch (const std::invalid_argument& error) {
      std::cerr << "[rayrai_tcp_viewer] a fluid volume of the server is not drawn: " << error.what() << "\n";
    }
  }
}

bool ViewerSceneFiles::settled(const RayraiWindow& viewer) const {
  switch (sync_.state()) {
    case SceneSync::State::Idle:
    case SceneSync::State::Declined:
    case SceneSync::State::Failed:
      return true;
    case SceneSync::State::Applied:
      return viewer.pendingAsyncMeshLoadCount() == 0;
    default:
      return false;
  }
}

void ViewerSceneFiles::remove(RayraiWindow& viewer) {
  for (const auto& visual : visuals_.objects)
    if (visual) viewer.removeVisualObject(visual->getName());
  for (const auto* batches : {&visuals_.instancedVisuals, &visuals_.foliage})
    for (const auto& batch : *batches)
      if (batch) viewer.removeInstancedVisuals(batch->getName());
  for (const auto& name : visuals_.waterBodies) viewer.removeWaterBody(name);
  visuals_ = {};
  applied_ = false;
  showFluidVolumes(viewer);
  meshLoading_.hide();
  if (!touched_) return;
  touched_ = false;
  viewerSettingsReset_ = true;
  // What applyRscene() set on the viewer; the pane re-applies its own settings afterwards.
  viewer.clearAdditionalLights();
  viewer.clearLocalFogVolumes();
  viewer.clearProjectedDecals();
  viewer.clearIrradianceVolumes();
  viewer.clearReflectionProbes();
  viewer.clearEnvironmentBackground();
  viewer.clearWeatherSkyEnvironment();
  viewer.setHeightmapPatternResourcePath("");
  viewer.setHeightmapNormalResourcePath("");
  viewer.setHeightmapHeightResourcePath("");
  viewer.clearHeightmapSplatLayer();
}

void ViewerSceneFiles::drawOverlay(const ImVec2& paneOrigin, const ImVec2& paneSize, int paneId,
                                   float panelRight) {
  drawSceneOverlay(sync_, errorDismissed_, meshLoading_, paneOrigin, paneSize, paneId, panelRight);
}

} // namespace tcp_viewer
} // namespace raisin
