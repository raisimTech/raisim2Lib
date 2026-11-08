// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#pragma once

#include <array>
#include <cfloat>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <imgui/imgui.h>

#include "rayrai/RaisimTcpCommon.hpp"
#include "rayrai/RsceneVisuals.hpp"
#include "rayrai/TcpSceneSync.hpp"
#include "TcpViewerSceneLoading.hpp"

namespace raisin
{
class RayraiWindow;

namespace tcp_viewer
{
class RemoteScene;

/**
 * The server's RaiSim Engine scene in one viewer pane: finding or downloading its files
 * (SceneSync), the prompt that asks before a download, and the scene's visuals, lights and
 * render settings once applied. The server streams the physics bodies; the scene adds what
 * rayrai draws for it locally (raisin::rayrai_internal::applyRscene()). The scene's meshes load
 * in the background while the pane shows their progress (SceneMeshLoading).
 */
class ViewerSceneFiles {
 public:
  /// @param cacheRoot Scene directory (SceneSync::Options::cacheRoot); empty uses
  ///        defaultSceneDirectory(), ~/.raisim/scenes.
  explicit ViewerSceneFiles(std::filesystem::path cacheRoot = {});

  /// The viewer's resource directories, searched for the scene's files.
  void setResourceDirs(const std::vector<std::string>& dirs);
  /// Disconnect: removes the applied scene and the streamed water from @p viewer, forgets them, and
  /// sets its water time back to 0.
  void reset(RayraiWindow& viewer, RemoteScene& scene);

  /// Scene-file block of a live response (not of a replayed one) from @p server. A server on
  /// this computer, such as a --simulate child, names this computer's files, so they load in
  /// place without a prompt (SceneSync::setServerPathsTrusted()). The socket's peer address
  /// decides it (TcpClient::peerIsLoopback()), not the host name the user typed.
  void receive(SceneFileBlock&& block, const TcpClient& server) {
    sync_.setServerPathsTrusted(server.peerIsLoopback());
    sync_.receive(std::move(block));
  }
  /// Requests for the next update.
  void appendRequests(std::vector<ClientRequest>& requests) { sync_.appendRequests(requests); }
  /// The server's volumes of liquid (ViewerCommandType::SetFluidVolumes). The pane draws their water
  /// while no applied scene draws water of its own.
  void setFluidVolumes(RayraiWindow& viewer, std::vector<StreamedFluidVolume> volumes);
  /**
   * Advances the background work, applies a ready scene and loads its meshes; call it once
   * per frame.
   * @return True in the frame the scene was applied.
   */
  bool update(RayraiWindow& viewer, RemoteScene& scene);

  /// True while the scene's render settings, lights and camera are in effect; the viewer
  /// then does not drive the main light and camera lens from its own settings.
  bool applied() const { return applied_; }
  /// True once after the scene's lights and render settings were removed from the viewer;
  /// the pane then re-applies its own.
  bool takeViewerSettingsReset() {
    const bool reset = viewerSettingsReset_;
    viewerSettingsReset_ = false;
    return reset;
  }
  const SceneSync& sync() const { return sync_; }
  /// False while the scene is on its way (the server lists its files, the viewer checks,
  /// downloads, loads or applies them, or its meshes load); a screenshot waits for it.
  /// True without a scene, at the download prompt, after a decline or a failure.
  bool settled(const RayraiWindow& viewer) const;

  /// Draws the download prompt or progress, or the progress of the scene's meshes, at the top
  /// of the pane (drawSceneOverlay()), clear of a panel along its left edge that ends at
  /// @p panelRight, such as the viewer's left overlay.
  void drawOverlay(const ImVec2& paneOrigin, const ImVec2& paneSize, int paneId,
                   float panelRight = -FLT_MAX);

 private:
  /// Applies a ready scene. @return True when it was applied.
  bool apply(RayraiWindow& viewer, RemoteScene& scene);
  void remove(RayraiWindow& viewer);
  /// Draws the water of fluidVolumes_, or none while the applied scene has water.
  void showFluidVolumes(RayraiWindow& viewer);

  SceneSync sync_;
  RsceneVisuals visuals_;
  bool applied_ = false;
  /// applyRscene() ran, possibly failing part way; the viewer holds scene state.
  bool touched_ = false;
  bool viewerSettingsReset_ = false;
  bool errorDismissed_ = false;
  SceneMeshLoading meshLoading_;
  std::vector<StreamedFluidVolume> fluidVolumes_;
  /// The volume each water body on the viewer (kFluidWaterPrefix and the volume's index) shows.
  std::vector<std::optional<StreamedFluidVolume>> fluidWaterShown_;
};

/// Applies the commands that carry scene state, the server's fluid volumes, to the pane.
void applySceneStateCommands(const std::vector<ViewerCommand>& commands, ViewerSceneFiles& files,
                             RayraiWindow& viewer);

/// A slider of the server as a pane shows it, with the value the server last sent.
struct ServerControlView {
  ServerControl control;
  double serverValue = 0.0;
};

/// Takes the sliders the server sent (ViewerCommandType::SetControls) into those a pane shows: a new
/// set of sliders replaces them, otherwise each keeps the server's value.
void receiveServerControls(std::vector<ServerControlView>& shown, const std::vector<ServerControl>& sent);

/// Which sliders the user holds: the one dragged (-1 for none) and those with a value of the user's
/// not yet sent (a CR_SET_CONTROL in @p pending).
std::vector<bool> heldServerControls(size_t count, int dragged, const std::vector<ClientRequest>& pending);

/// Shows the server's value on each slider the user does not hold. Called while no update is on its
/// way, the reply to the user's last value has come, so the slider shows what the server kept of it.
void settleServerControls(std::vector<ServerControlView>& shown, const std::vector<bool>& held);

/// The commands of one response: the scene state reaches the pane also when a session replays, and
/// @p live acts on the other commands only for a live response.
template <typename LiveAction>
void dispatchViewerCommands(const std::vector<ViewerCommand>& commands, bool replayed, ViewerSceneFiles& files,
                            RayraiWindow& viewer, LiveAction&& live) {
  applySceneStateCommands(commands, files, viewer);
  if (replayed) return;
  for (const auto& command : commands) live(command);
}

} // namespace tcp_viewer
} // namespace raisin
