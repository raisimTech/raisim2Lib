// Copyright (c) 2025 Raion Robotics Inc.
// All rights reserved.

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <SDL.h>

#include <glbinding/glbinding.h>
#include <glbinding/gl/gl.h>
#include <imgui/imgui.h>

#include "imgui/backend/imgui_impl_opengl3.h"
#include "imgui/backend/imgui_impl_sdl2.h"

#include <algorithm>
#include <atomic>
#include <clocale>
#include <csignal>
#include <iostream>
#include <locale>
#include <utility>

#include "raisim/PathUtils.hpp"
#include "rayrai/TcpViewerApp.hpp"
#include "rayrai/sdl_hints.hpp"

#if defined(__EMSCRIPTEN__)
#include "webgpu/Backend.hpp"
#endif

namespace {

// SDL's own SIGINT/SIGTERM handlers may call SDL_Quit() inside the handler, which
// races the normal shutdown; they are disabled and a signal requests a graceful exit.
std::atomic<bool> gSignalQuit{false};

void handleSignalQuit(int) {
  gSignalQuit.store(true, std::memory_order_relaxed);
}

bool isFullscreen(SDL_Window* window) {
  return (SDL_GetWindowFlags(window) & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP)) != 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::setlocale(LC_ALL, "C");
  std::locale::global(std::locale::classic());
  if (raisin::tcp_viewer::isSimulationWorkerCommand(argc, argv)) {
    return raisin::tcp_viewer::runSimulationWorker(argc, argv);
  }

  const char* program = argc > 0 ? argv[0] : "rayrai_tcp_viewer";
  raisin::tcp_viewer::ViewerAppConfig config;
  if (!raisin::tcp_viewer::parseViewerOptions(argc, argv, config.options)) {
    raisin::tcp_viewer::printViewerUsage(program);
    return 2;
  }
  if (config.options.printHelp) {
    raisin::tcp_viewer::printViewerUsage(program);
    return 0;
  }
  config.simulationWorker = raisin::tcp_viewer::currentExecutablePath(argc > 0 ? argv[0] : nullptr);

  raisin::rayrai::sdl::setWindowsDpiAwarenessHint();
  SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
  std::signal(SIGINT, handleSignalQuit);
#if defined(SIGTERM)
  std::signal(SIGTERM, handleSignalQuit);
#endif
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
    std::cerr << "FATAL ERROR: Failed to initialize SDL: " << SDL_GetError() << "\n";
    return -1;
  }

#if defined(__EMSCRIPTEN__)
  // The page's canvas draws through WebGPU, so the window must not create a WebGL context.
  constexpr Uint32 kWindowGraphicsFlag = 0;
#else
  constexpr Uint32 kWindowGraphicsFlag = SDL_WINDOW_OPENGL;
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  auto setContextVersion = [](int major, int minor) {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, major);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minor);
  };
  setContextVersion(4, 3);
#endif

  const raisin::tcp_viewer::ViewerOptions& options = config.options;
  SDL_Window* window = SDL_CreateWindow("Rayrai Raisim TCP Viewer", SDL_WINDOWPOS_CENTERED,
    SDL_WINDOWPOS_CENTERED, options.windowWidth, options.windowHeight,
    SDL_WindowFlags(kWindowGraphicsFlag | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI |
                    (options.fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0)));
  if (!window) {
    std::cerr << "FATAL ERROR: Failed to create SDL window: " << SDL_GetError() << "\n";
    SDL_Quit();
    return -1;
  }
  if (!options.fullscreen) {
    SDL_SetWindowBordered(window, SDL_TRUE);
  }

#if defined(__EMSCRIPTEN__)
  SDL_GLContext context = nullptr;
  raisin::webgpu::initialize();
#else
  SDL_GLContext context = SDL_GL_CreateContext(window);
  if (!context) {
    SDL_ClearError();
    setContextVersion(3, 3);
    context = SDL_GL_CreateContext(window);
  }
  if (!context) {
    std::cerr << "FATAL ERROR: Failed to create OpenGL context: " << SDL_GetError() << "\n";
    SDL_DestroyWindow(window);
    SDL_Quit();
    return -1;
  }
  SDL_GL_MakeCurrent(window, context);
  SDL_GL_SetSwapInterval(1);
  glbinding::initialize(
    [](const char* name) {
      return reinterpret_cast<glbinding::ProcAddress>(SDL_GL_GetProcAddress(name));
    },
    false);
#endif

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
#if defined(__EMSCRIPTEN__)
  ImGui_ImplSDL2_InitForOther(window);
#else
  ImGui_ImplSDL2_InitForOpenGL(window, context);
#endif
  ImGui_ImplOpenGL3_Init("#version 330");
#if defined(__EMSCRIPTEN__)
  raisin::webgpu::CanvasTarget canvasTarget;
#endif

  config.window.resize = [window](int width, int height) { SDL_SetWindowSize(window, width, height); };
  config.window.isFullscreen = [window] { return isFullscreen(window); };
  config.window.setFullscreen = [window](bool fullscreen) {
    SDL_SetWindowFullscreen(window, fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
  };
  raisin::tcp_viewer::ViewerApp app(std::move(config));
  int exitCode = 1;
  if (app.init()) {
    bool windowClosed = false;
    while (!windowClosed && !app.quitRequested() && !gSignalQuit.load(std::memory_order_relaxed)) {
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if (event.type == SDL_QUIT) windowClosed = true;
        if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
            event.window.windowID == SDL_GetWindowID(window)) {
          windowClosed = true;
        }
        if (event.type == SDL_DROPFILE && event.drop.file) {
          app.openFile(raisim::pathFromUtf8(event.drop.file));
          SDL_free(event.drop.file);
        }
      }

      int fbW = 0;
      int fbH = 0;
      SDL_GL_GetDrawableSize(window, &fbW, &fbH);
      int winW = 0;
      int winH = 0;
      SDL_GetWindowSize(window, &winW, &winH);
      app.beginFrame(ImVec2(static_cast<float>(fbW), static_cast<float>(fbH)),
                     ImVec2(static_cast<float>(winW), static_cast<float>(winH)));
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplSDL2_NewFrame();
      ImGui::NewFrame();
      app.drawFrame();
      ImGui::Render();
#if defined(__EMSCRIPTEN__)
      canvasTarget.bind(std::max(1, fbW), std::max(1, fbH));
#endif
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#if defined(__EMSCRIPTEN__)
      canvasTarget.present();
      raisin::webgpu::waitForAnimationFrame();
#else
      SDL_GL_SwapWindow(window);
#endif
    }
    exitCode = app.exitCode();
  }
  app.shutdown();

#if defined(__EMSCRIPTEN__)
  canvasTarget.release();
#endif
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

#if defined(__EMSCRIPTEN__)
  raisin::webgpu::shutdown();
#else
  SDL_GL_DeleteContext(context);
#endif
  SDL_DestroyWindow(window);
  SDL_Quit();
  return exitCode;
}
