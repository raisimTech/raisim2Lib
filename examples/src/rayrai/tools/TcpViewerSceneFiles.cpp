// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerSceneFiles.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <exception>
#include <utility>

#include "rayrai/RaisimTcpCommon.hpp"
#include "rayrai/RayraiWindow.hpp"
#include "rayrai/Visuals.hpp"

namespace raisin
{
namespace tcp_viewer
{

namespace
{

std::string megabytes(uint64_t bytes) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
  return text;
}

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
  remove(viewer);
  scene.setServerPathMap({});
  sync_.reset();
  errorDismissed_ = false;
}

bool ViewerSceneFiles::update(RayraiWindow& viewer, RemoteScene& scene) {
  sync_.poll();
  if (sync_.state() != SceneSync::State::Ready) return false;
  // The sync starts over when the server stops and resumes sharing; names must stay unique.
  remove(viewer);
  // The scene's render settings replace the removed ones right away.
  viewerSettingsReset_ = false;
  touched_ = true;
  try {
    // On failure, applyRscene() removes the visuals it created before it throws.
    visuals_ = applyRscene(*sync_.scene(), viewer);
  } catch (const std::exception& error) {
    remove(viewer);
    sync_.markFailed(std::string("The scene could not be shown: ") + error.what());
    return false;
  }
  // Bodies the server streams name their mesh by its server path.
  scene.setServerPathMap(sync_.pathMap());
  sync_.markApplied();
  applied_ = true;
  return true;
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
  visuals_ = {};
  applied_ = false;
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

void ViewerSceneFiles::drawOverlay(const ImVec2& paneOrigin, const ImVec2& paneSize, int paneId) {
  using State = SceneSync::State;
  const State state = sync_.state();
  const bool prompt = state == State::AwaitingConsent;
  const bool busy = state == State::Downloading || state == State::Finishing;
  const bool failed = state == State::Failed && !errorDismissed_;
  if (!prompt && !busy && !failed && state != State::Declined) return;

  const float margin = ImGui::GetFontSize();
  const float width = std::min(paneSize.x - 2.0f * margin, ImGui::GetFontSize() * 30.0f);
  ImGui::SetNextWindowPos(ImVec2(paneOrigin.x + paneSize.x * 0.5f, paneOrigin.y + margin),
                          ImGuiCond_Always, ImVec2(0.5f, 0.0f));
  ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(std::max(width, 1.0f), FLT_MAX));
  ImGui::SetNextWindowBgAlpha(state == State::Declined ? 0.45f : 0.85f);
  const std::string name = "Scene files##pane" + std::to_string(paneId);
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
    ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoMove;
  if (!ImGui::Begin(name.c_str(), nullptr, kFlags)) {
    ImGui::End();
    return;
  }
  ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + std::max(width, 1.0f));
  const std::string scene = sync_.sceneName().empty() ? "the scene" : sync_.sceneName();
  if (prompt) {
    ImGui::TextUnformatted("Download the server's scene?");
    ImGui::Separator();
    ImGui::Text("%s offers %zu file%s (%s). Accept to check local files and download missing resources.", scene.c_str(),
                sync_.missingFileCount(), sync_.missingFileCount() == 1 ? "" : "s",
                megabytes(sync_.missingBytes()).c_str());
    ImGui::TextDisabled("They are saved in %s", raisim::server::scene_files::pathUtf8(sync_.copyDirectory()).c_str());
    ImGui::TextDisabled("Add a resource directory that holds them to skip the download.");
    if (ImGui::Button("Download")) sync_.acceptDownload();
    ImGui::SameLine();
    if (ImGui::Button("Not now")) sync_.declineDownload();
  } else if (state == State::Downloading) {
    const uint64_t total = std::max<uint64_t>(1, sync_.missingBytes());
    ImGui::Text("Downloading %s", scene.c_str());
    const std::string progress =
      megabytes(sync_.downloadedBytes()) + " / " + megabytes(sync_.missingBytes());
    ImGui::ProgressBar(static_cast<float>(static_cast<double>(sync_.downloadedBytes()) / total),
                       ImVec2(width, 0.0f), progress.c_str());
    if (ImGui::Button("Cancel")) sync_.cancelDownload();
  } else if (state == State::Finishing) {
    ImGui::Text("Loading %s...", scene.c_str());
  } else if (failed) {
    ImGui::TextUnformatted("Scene files");
    ImGui::Separator();
    ImGui::TextUnformatted(sync_.error().c_str());
    if (ImGui::Button("Dismiss")) errorDismissed_ = true;
  } else {
    // Declined: the scene stays without its visuals until the user asks for it.
    ImGui::Text("%s: visuals not downloaded", scene.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Download")) sync_.acceptDownload();
  }
  ImGui::PopTextWrapPos();
  ImGui::End();
}

} // namespace tcp_viewer
} // namespace raisin
