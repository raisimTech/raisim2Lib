// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// Tank Survival: a tank battle game on RaiSim and Rayrai, in five missions and an endless survival mode.
//
// Every round is a RaiSim projectile: gravity and the drag of the wind-blown air bend its path, it
// hands its momentum to what it hits, and high explosives set off blast waves that push, flip and
// wreck tanks, crates and barrels and dig craters. A hit is resolved against the armor of the zone it
// strikes (front, side, rear or roof), the angle it meets the plate at and the round's remaining
// penetration: shallow hits ricochet, side hits can break tracks, rear hits can set off the ammunition.
// Enemies differ: fast scouts, medium tanks, heavy tanks, tank destroyers, mortar carriers and a
// super-heavy boss. The player has armor-piercing and high-explosive shells, a machine gun, guided
// missiles, artillery strikes and smoke. The same source builds for the desktop and, with the RaiSim
// WebAssembly SDK, for the browser.

#include "hud.hpp"

#if !defined(__EMSCRIPTEN__)
#include <cstring>
#include <glbinding/gl/gl.h>
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#else
#include <emscripten/emscripten.h>

// In the browser the best scores outlive the page, in its local storage (where the browser allows it).
EM_JS(int, storedBest, (int level), {
  try {
    const score = Number(localStorage.getItem('raisim-tank-survival-best-' + level));
    return Number.isInteger(score) && score > 0 && score <= 2147483647 ? score : 0;
  } catch (error) {
    return 0;
  }
});
EM_JS(void, storeBest, (int level, int score), {
  try { localStorage.setItem('raisim-tank-survival-best-' + level, String(score)); } catch (error) {}
});
// SDL keeps its relative mouse mode when the browser ends the pointer lock (Esc); the page knows.
EM_JS(int, pointerLocked, (), { return document.pointerLockElement ? 1 : 0; });
#endif

namespace tank {
namespace {

struct Options {
  int frames = 0;
  std::string screenshot;
  int level = 0;
  bool autopilot = false;
  int quality = int(raisin::RenderQualityPreset::High);
  bool qualityGiven = false;
  unsigned seed = 0;
  // A fixed camera for screenshots: its position and the point it looks at.
  std::optional<std::array<float, 6>> view;
};

#if !defined(__EMSCRIPTEN__)
// The whole window with its overlays, read from the back buffer before it is swapped.
bool saveWindow(SDL_Window* window, const std::string& path) {
  int width = 0, height = 0;
  SDL_GL_GetDrawableSize(window, &width, &height);
  const size_t stride = size_t(width) * 4;
  std::vector<unsigned char> pixels(stride * height), flipped(stride * height);
  gl::glBindFramebuffer(gl::GL_READ_FRAMEBUFFER, 0);
  gl::glReadBuffer(gl::GL_BACK);
  gl::glPixelStorei(gl::GL_PACK_ALIGNMENT, 1);
  gl::glReadPixels(0, 0, width, height, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, pixels.data());
  for (int y = 0; y < height; ++y) std::memcpy(&flipped[y * stride], &pixels[(height - 1 - y) * stride], stride);
  return stbi_write_png(path.c_str(), width, height, 4, flipped.data(), int(stride)) != 0;
}
#endif

// The mouse captured to turn the camera, as in other tank games. In the browser it is the page's pointer
// lock, which the browser grants at a click and ends at Esc; a device that never grants it (a touch
// screen) keeps aiming where it points.
struct MouseCapture {
  bool requested = false, held = false, everHeld = false, unavailable = false;
  double waited = 0.0;

  void request() {
    if (requested || unavailable) return;
    requested = SDL_SetRelativeMouseMode(SDL_TRUE) == 0;
    unavailable = !requested;
    waited = 0.0;
  }
  void release() {
    if (requested) SDL_SetRelativeMouseMode(SDL_FALSE);
    requested = held = false;
  }
  // Follows the lock; returns true when one held was taken away. A browser grants the lock when the click
  // ends, so a request waits while the button is down.
  bool update(double dt, bool pressing) {
    if (!requested) return false;
#if defined(__EMSCRIPTEN__)
    const bool locked = pointerLocked() != 0;
#else
    const bool locked = SDL_GetRelativeMouseMode() == SDL_TRUE;
#endif
    if (locked) {
      held = everHeld = true;
      return false;
    }
    if (held) {
      release();
      return true;
    }
    if (!pressing) waited += dt;
    if (waited > 1.5) {
      release();
      unavailable = !everHeld;
    }
    return false;
  }
};

Vec3 viewDirection(const OrbitHeading& orbit) {
  return {std::cos(orbit.heading) * std::cos(orbit.pitch), std::sin(orbit.heading) * std::cos(orbit.pitch),
          std::sin(orbit.pitch)};
}

// How close and how far the camera can be from the player's tank (m).
constexpr float kClosest = 12.0f, kFarthest = 70.0f;

void setOrbitDistance(raisin::Camera& camera, float distance) {
  const float yaw = glm::radians(camera.yaw), pitch = glm::radians(camera.pitch);
  const glm::vec3 front(std::cos(yaw) * std::cos(pitch), std::sin(yaw) * std::cos(pitch), std::sin(pitch));
  camera.setCameraFixedDistance(false);
  camera.position = camera.target - front * distance;
  camera.setCameraFixedDistance(true);
}

// Plays the player's tank for tests and demonstrations: picks a target and the weapon for it, keeps its
// distance, collects what it needs and calls artillery on groups.
bool autopilot(Game& game, Tank& hero, double& throttle, double& turn) {
  const Vec3 at = hero.position();
  const Tank* target = nullptr;
  for (const auto& enemy : game.enemies())
    if (enemy->system && (!target || (enemy->position() - at).norm() < (target->position() - at).norm())) target = enemy.get();
  const auto wanted = [&](const Pickup& pickup) {
    if (pickup.type == Bonus::Repair) return game.playerHealth() < 160.0;
    if (pickup.type == Bonus::Ammo) return game.ammo(WeaponId::AP) < 20;
    return true;
  };
  const Pickup* supply = nullptr;
  for (const auto& pickup : game.pickups())
    if (wanted(pickup) && (!supply || (pickup.position - at).norm() < (supply->position - at).norm())) supply = &pickup;
  Vec3 goal = supply ? supply->position : Vec3(25.0 * std::cos(0.15 * game.now()), 25.0 * std::sin(0.15 * game.now()), 0.0);
  bool trigger = false;
  if (target) {
    const Vec3 away = at - target->position();
    if (away.head<2>().norm() < 16.0) goal = at + Vec3(away.x(), away.y(), 0.0).normalized() * 12.0;
    const Role role = target->spec->role;
    WeaponId choice = role == Role::Scout ? WeaponId::MG : WeaponId::AP;
    if ((role == Role::Heavy || role == Role::Boss || role == Role::Destroyer) && game.ammo(WeaponId::Missile) > 0)
      choice = WeaponId::Missile;
    if (choice == WeaponId::AP && game.ammo(WeaponId::AP) == 0) choice = WeaponId::HE;
    int near = 0;
    for (const auto& enemy : game.enemies())
      if (enemy->system && (enemy->position() - target->position()).norm() < 14.0) ++near;
    if (near >= 2 && game.ammo(WeaponId::Artillery) > 0 && (target->position() - at).norm() > 30.0) choice = WeaponId::Artillery;
    if (game.selected() != choice) game.select(choice);
    const double speed = choice == WeaponId::HE ? weapon(WeaponId::HE).speed : weapon(WeaponId::AP).speed;
    const double flight = (target->position() - at).norm() / speed;
    const Vec3 lead = target->position() + target->velocity() * flight + Vec3(0, 0, 0.6 * target->scale());
    game.setAim(lead);
    const double error = hero.aimAt(lead, speed, true);
    trigger = (target->position() - at).norm() > 9.0 && error < (choice == WeaponId::MG ? 0.06 : 0.02);
    if (game.playerHealth() < 120.0 && game.smokeCharges() > 0) game.smoke();
  }
  const double turnError = wrapAngle(std::atan2(goal.y() - at.y(), goal.x() - at.x()) - hero.heading());
  throttle = std::max(0.25, std::cos(turnError));
  turn = 1.8 * turnError;
  return trigger;
}

int run(int argc, char* argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--frames" && i + 1 < argc) options.frames = std::stoi(argv[++i]);
    else if (option == "--level" && i + 1 < argc) options.level = std::clamp(std::stoi(argv[++i]) - 1, 0, int(levels().size()) - 1);
    else if (option == "--autopilot") options.autopilot = true;
    else if (option == "--quality" && i + 1 < argc) {
      options.quality = std::clamp(std::stoi(argv[++i]), 0, 3);
      options.qualityGiven = true;
    }
    else if (option == "--seed" && i + 1 < argc) options.seed = unsigned(std::stoul(argv[++i]));
    else if (option == "--screenshot" && i + 1 < argc) options.screenshot = argv[++i];
    else if (option == "--view" && i + 6 < argc) {
      std::array<float, 6> view{};
      for (float& value : view) value = std::stof(argv[++i]);
      options.view = view;
    }
    else {
      std::cerr << "Usage: tank_survival [--level 1-" << levels().size()
                << "] [--autopilot] [--frames N] [--quality 0-3] [--seed N] [--screenshot file.png]"
                << " [--view x y z targetX targetY targetZ]\n";
      return 2;
    }
  }
  const bool scripted = options.frames > 0;
  const bool playing = scripted || options.autopilot;
  if (!options.seed) options.seed = scripted ? 7u : unsigned(std::chrono::steady_clock::now().time_since_epoch().count());

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(kTimeStep);

  ExampleApp app;
  if (!app.init("RaiSim Tank Survival", 1600, 900)) return 1;
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  const std::string fontFile = std::string(TANK_SURVIVAL_ASSETS) + "/fonts/roboto/Roboto-SemiBold.ttf";
  Fonts fonts;
  // Latin-1 and the infinity sign that marks the endless mission.
  static const ImWchar kGlyphs[] = {0x0020, 0x00FF, 0x221E, 0x221E, 0};
  fonts.text = io.Fonts->AddFontFromFileTTF(fontFile.c_str(), 19.0f);
  fonts.title = io.Fonts->AddFontFromFileTTF(fontFile.c_str(), 46.0f, nullptr, kGlyphs);
  fonts.heading = io.Fonts->AddFontFromFileTTF(fontFile.c_str(), 24.0f);
  fonts.small = io.Fonts->AddFontFromFileTTF(fontFile.c_str(), 15.0f);
  if (!fonts.text) fonts.text = fonts.title = fonts.heading = fonts.small = io.Fonts->AddFontDefault();
  ImGuiStyle& style = ImGui::GetStyle();
  style.FrameRounding = 6.0f;
  style.Colors[ImGuiCol_Button] = ImVec4(0.22f, 0.26f, 0.16f, 0.95f);
  style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.36f, 0.42f, 0.22f, 1.0f);
  style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.5f, 0.56f, 0.28f, 1.0f);
  style.Colors[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.15f, 0.1f, 0.95f);
  style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.26f, 0.16f, 1.0f);
  style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.3f, 0.35f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_PopupBg] = ImVec4(0.06f, 0.07f, 0.05f, 0.97f);
  style.Colors[ImGuiCol_Header] = ImVec4(0.3f, 0.35f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.36f, 0.42f, 0.22f, 1.0f);
  style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.5f, 0.56f, 0.28f, 1.0f);
  int status = 0;
  // Draws one frame of the loading screen, so that the page shows it between the stages of loading.
  const auto showLoading = [&](double fraction, const char* what) {
    app.processEvents();
    app.beginFrame();
    drawLoading(fonts, fraction, what);
    app.endFrame();
  };
  showLoading(0.02, "Starting the renderer");
  {
    Viewer viewer(world, 1600, 900);
    viewer.setKeyboardInputEnabled(false);
    showLoading(0.1, "Preparing the explosion shaders");
    viewer.prepareExplosionEffects();
    auto& camera = viewer.getCamera();
    camera.nearPlane = 0.1f;
    camera.farPlane = 1600.0f;
    camera.pitch = -19.0f;
    camera.setCameraFixedTarget(true);
    const float fieldOfView = camera.zoom;
    float distance = 24.0f, sightBlend = 0.0f, clearDistance = distance;
    int quality = options.quality;

    Game game(world, viewer, TANK_SURVIVAL_ASSETS, options.seed);
    game.load(options.level, playing, quality, [&](double fraction, const char* what) { showLoading(0.2 + 0.6 * fraction, what); });
    showLoading(0.85, "Compiling shaders (faster on your next visit)");
    // Rayrai draws with its core PBR program whenever no high-fidelity mesh is in view; compiled now, it does
    // not stall the game the first time that happens (0.2 s in the browser).
    for (const char* program : {"pbrMeshCore", "pbrMeshCoreDepthPrepass"}) viewer.compileShaderByName(program);
    std::vector<int> best(levels().size(), 0);
#if defined(__EMSCRIPTEN__)
    for (size_t i = 0; i < best.size(); ++i) best[i] = storedBest(int(i));
#endif
    bool paused = false, help = true, helpOpen = false, helpShown = false;
    const Tank* locked = nullptr;
    double lockedBorn = 0.0;
    const Tank* tracked = nullptr;
    double trackedBorn = 0.0;
    MouseCapture capture;
    bool captureClick = false;
    double sinceLockLost = 1e9;
    SniperSight sniper;
    ClickOrHold rightButton;
    double magnification = 1.0;
    std::mt19937 shakeNoise(1);
    auto last = std::chrono::steady_clock::now();
    int frame = 0;
    Screen screen;
    double endedFor = 0.0;
    OrbitHeading orbit;
    double physicsTime = 0.0, gameTime = 0.0, renderTime = 0.0;
    int physicsSteps = 0;
    using Clock = std::chrono::steady_clock;
    const auto seconds = [](Clock::time_point from) { return std::chrono::duration<double>(Clock::now() - from).count(); };
    camera.target = glm::vec3(0.0f, 0.0f, 2.0f);
    // Before a level is first played at a quality, it is drawn all around the player behind the loading screen
    // once its models are in, with an explosion flashing beside it (the world's clock stands still), so that
    // the renderer makes the pipelines the level needs then: in the browser, each takes 50-100 ms the first
    // time it is drawn, a stall when something first comes into view or the first shell lands.
    std::vector<std::pair<int, int>> warmed;
    const auto warmUp = [&] {
      const Tank* hero = game.playerTank();
      const ImGuiIO& io = ImGui::GetIO();
      const int width = int(io.DisplaySize.x * io.DisplayFramebufferScale.x);
      const int height = int(io.DisplaySize.y * io.DisplayFramebufferScale.y);
      const std::pair<int, int> key(game.level(), quality);
      if (!hero || !hero->system || width <= 0 || height <= 0 || std::find(warmed.begin(), warmed.end(), key) != warmed.end())
        return;
      warmed.push_back(key);
      const glm::vec3 position = camera.position, front = camera.front, target = camera.target, up = camera.up,
                      right = camera.right;
      const float yaw = camera.yaw, pitch = camera.pitch;
      const Vec3 p = hero->position();
      camera.target = glm::vec3(float(p.x()), float(p.y()), float(p.z() + 2.6));
      camera.pitch = -19.0f;
      viewer.addExplosionEffect({camera.target + glm::vec3(4.0f, 0.0f, -2.0f), 1.0f, true});
      viewer.addFireEffect({camera.target + glm::vec3(-4.0f, 0.0f, -2.0f), 1.0f, 2.0f});
      const auto start = Clock::now();
      bool wrecksShown = false;
      for (int k = 0, left = 6; left > 0; ++k) {
        // The level's meshes have loaded by now; the wrecks are then drawn from every side.
        if (!wrecksShown && viewer.pendingAsyncMeshLoadCount() == 0) {
          game.showWreckModels(p);
          wrecksShown = true;
          left = 6;
        }
        camera.yaw = 60.0f * float(k % 6);
        setOrbitDistance(camera, distance);
        game.updateScenery(camera.position);
        viewer.update(width, height, false, 0, 0, false);
        showLoading(std::min(0.99, 0.9 + 0.015 * k), "Preparing the level");
        left = viewer.pendingAsyncMeshLoadCount() > 0 && seconds(start) < 6.0 ? 6 : left - 1;
      }
      viewer.clearExplosionEffects();
      game.hideWreckModels();
      camera.position = position;
      camera.front = front;
      camera.target = target;
      camera.up = up;
      camera.right = right;
      camera.yaw = yaw;
      camera.pitch = pitch;
    };
    bool warmUpPending = playing;
    // A mission starts with the camera behind the hull.
    const auto startBehindHull = [&] {
      if (const Tank* hero = game.playerTank(); hero && hero->system) orbit.place(hero->heading());
    };
    if (playing) startBehindHull();
    HoldToConfirm restartKey;
    QualityTuner tuner;
    tuner.active = !playing && !options.qualityGiven;
    std::string qualityNote;

    while (!app.quit) {
      app.processEvents();
      if (app.quit) break;
      if (warmUpPending) {
        warmUpPending = false;
        warmUp();
      }
      const auto clockNow = std::chrono::steady_clock::now();
      const double elapsedSinceLast = std::chrono::duration<double>(clockNow - last).count();
      const double frameTime = scripted ? 1.0 / 60 : std::min(elapsedSinceLast, 1.0 / 15);
      // Keys held to confirm count the time on the clock, a stall at most a quarter of a second.
      const double holdTime = scripted ? frameTime : std::min(elapsedSinceLast, 0.25);
      last = clockNow;

      app.beginFrame();
      const auto viewport = app.beginViewport();
      screen.origin = viewport.origin;
      screen.size = viewport.size;
      screen.viewProjection = camera.getProjectionMatrix() * camera.getViewMatrix();
      screen.inverse = glm::inverse(screen.viewProjection);

      // The browser ends a pointer lock at Esc and may pass the key on as well: losing the lock pauses, and
      // the key pressed with it does nothing more.
      const bool lockLost = capture.update(elapsedSinceLast, ImGui::IsMouseDown(ImGuiMouseButton_Left));
      sinceLockLost = lockLost ? 0.0 : sinceLockLost + elapsedSinceLast;
      if (!scripted && game.phase() == Phase::Playing) {
        if (lockLost)
          paused = true;
        else if (!helpOpen && sinceLockLost > 0.3 &&
                 (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_P, false)))
          paused = !paused;
        else if (!paused && ImGui::IsKeyPressed(ImGuiKey_H, false))
          helpOpen = !helpOpen;
      }
      const bool halted = paused || helpOpen;
      const bool live = game.phase() == Phase::Playing && !halted;
      if (capture.requested && (!live || options.autopilot)) capture.release();
      if (game.phase() != Phase::Playing) sniper.active = false;
      int mouseX = 0, mouseY = 0;
      SDL_GetRelativeMouseState(&mouseX, &mouseY);
      double throttle = 0.0, turn = 0.0;
      bool trigger = false, autoAim = false;
      if (!scripted) {
        if (!live) restartKey.update(false, holdTime);
        if (live) {
          const Uint8* keys = SDL_GetKeyboardState(nullptr);
          throttle = (keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP] ? 1.0 : 0.0) - (keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN] ? 1.0 : 0.0);
          turn = (keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_LEFT] ? 1.0 : 0.0) - (keys[SDL_SCANCODE_D] || keys[SDL_SCANCODE_RIGHT] ? 1.0 : 0.0);
          // A click in the view captures the mouse; the button fires once it is held and pressed again (at once
          // where the mouse cannot be held).
          if (!capture.requested && !capture.unavailable && viewport.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            capture.request();
            captureClick = true;
          }
          if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) captureClick = false;
          const bool mouseFires = !captureClick && (capture.held || (capture.unavailable && viewport.hovered));
          trigger = keys[SDL_SCANCODE_SPACE] || (mouseFires && ImGui::IsMouseDown(ImGuiMouseButton_Left));
          autoAim = rightButton.update(capture.held && ImGui::IsMouseDown(ImGuiMouseButton_Right), holdTime);
          if (capture.held && (ImGui::IsKeyPressed(ImGuiKey_LeftShift, false) || ImGui::IsKeyPressed(ImGuiKey_RightShift, false)))
            sniper.toggle();
          for (int i = 0; i < kWeaponCount; ++i)
            if (ImGui::IsKeyPressed(ImGuiKey(int(ImGuiKey_1) + i), false)) game.select(WeaponId(i));
          if (ImGui::IsKeyPressed(ImGuiKey_Q, false)) game.cycle(-1);
          if (ImGui::IsKeyPressed(ImGuiKey_E, false)) game.cycle(1);
          if (ImGui::IsKeyPressed(ImGuiKey_F, false)) game.smoke();
          if (ImGui::IsKeyPressed(ImGuiKey_C, false)) orbit.behindHull = true;
          // Restarting takes a held R: E, the next weapon, is beside it.
          if (restartKey.update(ImGui::IsKeyDown(ImGuiKey_R), holdTime)) {
            tracked = nullptr;
            sniper.active = false;
            game.load(game.level(), true, quality);
            warmUpPending = true;
            startBehindHull();
          }
          if (throttle != 0.0 || turn != 0.0) help = help && game.elapsed() < 25.0;
        }
        if (capture.held) {
          if (live) sniper.scroll(io.MouseWheel, distance, kClosest, kFarthest);
          orbit.look(mouseX, mouseY, 1.0 / sniper.magnification());
        } else if (viewport.hovered) {
          distance = std::clamp(distance - 2.0f * io.MouseWheel, kClosest, kFarthest);
          if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) orbit.drag(io.MouseDelta.x);
        }
      }

      // The view turns before the sight aims along it.
      if (const Tank* hero = game.playerTank(); hero && hero->system && game.phase() != Phase::Menu)
        orbit.update(hero->heading(), frameTime, options.autopilot);
      locked = nullptr;
      bool onEnemy = false;
      if (Tank* hero = game.player(); hero && !halted) {
        if (options.autopilot) {
          trigger = autopilot(game, *hero, throttle, turn);
        } else if (capture.held || viewport.hovered) {
          const Vec3 cameraAt(camera.position.x, camera.position.y, camera.position.z);
          const WeaponId selected = game.selected();
          const bool gunRound = selected == WeaponId::AP || selected == WeaponId::HE || selected == WeaponId::MG;
          const double shellSpeed = weapon(gunRound ? selected : WeaponId::AP).speed;
          std::vector<ImVec2> points;
          std::vector<const Tank*> enemies;
          for (const auto& enemy : game.enemies()) {
            ImVec2 at;
            if (!enemy->alive() || !screen.project(enemy->position() + Vec3(0, 0, 0.6 * enemy->scale()), at)) continue;
            points.push_back(at);
            enemies.push_back(enemy.get());
          }
          Vec3 aim;
          if (capture.held) {
            // The sight in the middle of the view aims exactly at what its ray meets first: a vehicle, a wall,
            // a tree or the ground. While the right button is down the gun keeps its aim; a click locks the
            // aim on the enemy nearest the sight (or lets the lock go) and leads it.
            const ImVec2 middle(screen.origin.x + 0.5f * screen.size.x, screen.origin.y + 0.5f * screen.size.y);
            // From where the camera is about to look along the turned view, which the sight is drawn on.
            const Vec3 ray = viewDirection(orbit);
            const Vec3 focus(camera.target.x, camera.target.y, camera.target.z);
            const Vec3 eye = focus - double(glm::length(camera.position - camera.target)) * ray;
            Vec3 struck;
            bool vehicle = false;
            const bool hit = game.sweep(eye, ray, 1500.0, struck, vehicle);
            onEnemy = hit && vehicle;
            aim = rightButton.down() ? game.aim() : hit ? struck : Vec3(eye + 1500.0 * ray);
            if (autoAim) {
              const int nearest = nearestWithin(points, middle, 80.0f);
              tracked = nearest >= 0 && enemies[size_t(nearest)] != tracked ? enemies[size_t(nearest)] : nullptr;
              trackedBorn = tracked ? tracked->born : 0.0;
            }
            // A vehicle made where a destroyed one was is another vehicle: the lock goes with the first.
            if (tracked && std::none_of(game.enemies().begin(), game.enemies().end(), [&](const auto& enemy) {
                  return enemy.get() == tracked && enemy->alive() && enemy->born == trackedBorn;
                }))
              tracked = nullptr;
            locked = tracked;
          } else {
            // Where the mouse cannot be captured (a touch screen), the sight is the cursor and locks on the
            // enemy nearest it.
            tracked = nullptr;
            aim = game.groundHit(cameraAt, screen.ray(io.MousePos, camera.position)) + Vec3(0, 0, 0.4);
            if (const int nearest = nearestWithin(points, io.MousePos, 56.0f); nearest >= 0) locked = enemies[size_t(nearest)];
          }
          if (locked) {
            lockedBorn = locked->born;
            const Vec3 center = locked->position() + Vec3(0, 0, 0.6 * locked->scale());
            aim = center + locked->velocity() * ((center - hero->position()).norm() / shellSpeed);
          }
          game.setAim(aim);
          hero->aimAt(aim, shellSpeed, true);
        }
        hero->command(throttle, turn, game.overdrive(), game.now());
        if (trigger) game.fire();
      }

      if (!halted) {
        const int steps = std::clamp(int(std::round(frameTime / kTimeStep)), 1, 20);
        const auto physicsStart = Clock::now();
        for (int i = 0; i < steps; ++i) game.step();
        physicsTime += seconds(physicsStart);
        physicsSteps += steps;
        const auto gameStart = Clock::now();
        game.update(frameTime);
        gameTime += seconds(gameStart);
      }
      const auto renderStart = Clock::now();

      // The sniper sight looks from just above the gun and magnifies the view.
      const bool sighting = capture.held && sniper.active;
      sightBlend = std::clamp(sightBlend + float(frameTime / 0.15) * (sighting ? 1.0f : -1.0f), 0.0f, 1.0f);
      magnification = std::exp(std::log(magnification) +
                               (std::log(sighting ? sniper.magnification() : 1.0) - std::log(magnification)) *
                                   std::min(1.0, frameTime * 14.0));
      if (const Tank* hero = game.playerTank(); hero && hero->system && game.phase() != Phase::Menu) {
        // The camera turns about a point above the tank, higher the farther it is, so that the tank sits below
        // the sight and the view passes over what is around it.
        const Vec3 p = hero->position();
        glm::vec3 focus(float(p.x()), float(p.y()), float(p.z() + 2.6 + 0.12 * distance));
        if (sightBlend > 0.0f) {
          const Vec3 gun = hero->gunPivot() + Vec3(0.0, 0.0, 1.1 * hero->scale());
          focus = glm::mix(focus, glm::vec3(float(gun.x()), float(gun.y()), float(gun.z())), sightBlend);
        }
        camera.target = glm::mix(camera.target, focus, float(std::min(1.0, frameTime * (8.0 + 40.0 * sightBlend))));
        camera.pitch = float(orbit.pitch * 180.0 / kPi);
      } else {
        const glm::vec3 center(0.0f, 0.0f, 3.0f);
        camera.target = glm::mix(camera.target, center, float(std::min(1.0, frameTime * 2.0)));
        orbit.heading += frameTime * 0.08;
        camera.pitch = game.phase() == Phase::Menu ? -16.0f : -22.0f;
      }
      if (game.shake() > 0.01) {
        std::normal_distribution<float> noise(0.0f, float(0.35 * game.shake()));
        camera.target += glm::vec3(noise(shakeNoise), noise(shakeNoise), noise(shakeNoise));
      }
      camera.yaw = float(orbit.heading * 180.0 / kPi);
      float orbitDistance = game.phase() == Phase::Menu ? 75.0f : distance * (1.0f - sightBlend) + 0.05f * sightBlend;
      camera.zoom = float(fieldOfView / magnification);
      if (options.view) {
        const auto& v = *options.view;
        const glm::vec3 eye(v[0], v[1], v[2]), target(v[3], v[4], v[5]), d = target - eye;
        camera.target = target;
        orbit.heading = std::atan2(d.y, d.x);
        camera.yaw = float(orbit.heading * 180.0 / kPi);
        camera.pitch = float(std::atan2(d.z, std::hypot(d.x, d.y)) * 180.0 / kPi);
        orbitDistance = glm::length(d);
      }
      // Walls, buildings and the ground between the tank and the camera bring the camera in front of them at
      // once; a tree crown it would sit in brings it out of the leaves more gently. It moves back out
      // smoothly once they are out of the way, and comes closer rather than looking elsewhere, so that the
      // sight keeps pointing where the mouse put it.
      if (!options.view && game.phase() != Phase::Menu) {
        constexpr double kWallMargin = 1.2;
        setOrbitDistance(camera, orbitDistance);
        const glm::vec3 back = camera.position - camera.target;
        const float length = glm::length(back);
        float walls = orbitDistance, reach = orbitDistance;
        if (length > 1e-3f) {
          const Vec3 from(camera.target.x, camera.target.y, camera.target.z);
          const Vec3 direction(back.x / length, back.y / length, back.z / length);
          walls = std::max(0.05f, float(game.wallReach(from, direction, orbitDistance + kWallMargin) - kWallMargin));
          reach = std::min(walls, float(game.crownReach(from, direction, orbitDistance)));
        }
        const float rate = reach < clearDistance ? 10.0f : 3.0f;
        clearDistance = walls < clearDistance ? walls : clearDistance + (reach - clearDistance) * std::min(1.0f, float(frameTime) * rate);
        orbitDistance = std::min(orbitDistance, clearDistance);
      } else {
        clearDistance = orbitDistance;
      }
      setOrbitDistance(camera, orbitDistance);
      while (!options.view && orbitDistance > 2.0f &&
             camera.position.z < game.height(camera.position.x, camera.position.y) + 1.8) {
        orbitDistance *= 0.97f;
        setOrbitDistance(camera, orbitDistance);
      }
      game.updateScenery(camera.position);

      app.endViewport(viewer, viewport, false, false);
      ImGui::PushFont(fonts.text);
      if (locked && std::none_of(game.enemies().begin(), game.enemies().end(), [&](const auto& enemy) {
            return enemy.get() == locked && enemy->born == lockedBorn;
          }))
        locked = nullptr;
      if (game.phase() != Phase::Menu) {
        Sight sight;
        sight.centered = capture.held;
        sight.at = capture.held ? ImVec2(screen.origin.x + 0.5f * screen.size.x, screen.origin.y + 0.5f * screen.size.y)
                                : io.MousePos;
        sight.onEnemy = onEnemy;
        sight.offerMouse = live && !capture.requested && !capture.unavailable && !options.autopilot && !scripted;
        sight.magnification = sightBlend > 0.5f ? sniper.magnification() : 1.0;
        drawHud(game, screen, fonts, orbit.heading, help && !options.autopilot, locked, halted, sight);
      }
      if (!scripted && game.phase() == Phase::Playing && !halted && !capture.requested && drawHelpButton(screen))
        helpOpen = true;
      if (!scripted && game.phase() == Phase::Playing && !halted && restartKey.fraction() > 0.0)
        drawHold(screen, fonts, "Hold R to restart", restartKey.fraction());
      if (const size_t pending = viewer.pendingAsyncMeshLoadCount(); pending > 0) {
        ImDrawList* draw = ImGui::GetForegroundDrawList();
        const ImVec2 at(viewport.origin.x + 18.0f, viewport.origin.y + viewport.size.y - 34.0f);
        text(draw, fonts.text, 17.0f, at, IM_COL32(230, 220, 190, 230), "Loading models: " + std::to_string(pending) + " left");
      }
      MenuResult choice;
      if (scripted || helpOpen || game.phase() != Phase::Menu || viewer.pendingAsyncMeshLoadCount() > 0) tuner.restart();
      if (!scripted) {
        if (helpOpen) {
          if (drawHelp(screen, fonts)) helpOpen = false;
        } else if (game.phase() == Phase::Menu) {
          const int before = quality;
          choice = drawMainMenu(screen, fonts, best, quality, qualityNote.empty() ? nullptr : qualityNote.c_str());
          if (choice.qualityPicked) {
            tuner.active = false;
            qualityNote.clear();
          } else if (viewer.pendingAsyncMeshLoadCount() == 0) {
            quality = tuner.update(quality, frameTime);
            if (quality != before) {
              const char* names[] = {"Fast", "Balanced", "High", "Ultra"};
              qualityNote = std::string("Render quality lowered to ") + names[quality] + " for a smoother game";
            }
          }
          if (quality != before) game.load(game.level(), false, quality);
        } else if (paused) {
          choice = drawPauseMenu(screen, fonts);
        } else if (game.phase() == Phase::Victory || game.phase() == Phase::Defeat) {
          endedFor += frameTime;
          if (int& record = best[size_t(game.level())]; game.score() > record) {
            record = game.score();
#if defined(__EMSCRIPTEN__)
            storeBest(game.level(), record);
#endif
          }
          if (endedFor > 1.5) choice = drawResult(screen, fonts, game);
        }
      }
      ImGui::PopFont();
      switch (choice.choice) {
        case Choice::Play:
        case Choice::Next:
          game.load(choice.level, true, quality);
          warmUpPending = true;
          startBehindHull();
          paused = false;
          endedFor = 0.0;
          // Before the first mission, the help comes up with the game held still.
          helpOpen = !helpShown;
          helpShown = true;
          break;
        case Choice::Help:
          helpOpen = true;
          helpShown = true;
          break;
        case Choice::Retry:
          game.load(game.level(), true, quality);
          warmUpPending = true;
          startBehindHull();
          paused = false;
          endedFor = 0.0;
          break;
        case Choice::Menu:
          game.load(game.level(), false, quality);
          paused = false;
          endedFor = 0.0;
          break;
        case Choice::Resume:
          paused = false;
          break;
        case Choice::None:
          break;
      }
#if !defined(__EMSCRIPTEN__)
      if (scripted && frame + 1 >= options.frames && !options.screenshot.empty()) {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (!saveWindow(app.window, options.screenshot)) status = 1;
        SDL_GL_SwapWindow(app.window);
      } else {
        app.endFrame();
      }
#else
      app.endFrame();
#endif
      renderTime += seconds(renderStart);
      ++frame;
      if (scripted && frame >= options.frames) {
        const Stats& stats = game.stats();
        std::printf("tank_survival level=%d phase=%d frames=%d time=%.2f wave=%d kills=%d score=%d armor=%.0f shots=%d "
                    "hits=%d penetrations=%d trees_felled=%zu\n",
                    game.level() + 1, int(game.phase()), frame, game.elapsed(), game.wave(), stats.kills, game.score(),
                    game.playerHealth(), stats.shots, stats.hits, stats.penetrations, game.treesFelled());
        std::printf("tank_survival timing: physics %.3f ms/frame (%.1f us/step), game %.3f ms/frame, render %.3f ms/frame\n",
                    1e3 * physicsTime / frame, 1e6 * physicsTime / std::max(1, physicsSteps), 1e3 * gameTime / frame,
                    1e3 * renderTime / frame);
        std::fflush(stdout);
        break;
      }
    }
  }
  app.shutdown();
  return status;
}

}  // namespace
}  // namespace tank

int main(int argc, char* argv[]) {
  try {
    return tank::run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
