//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include "VideoRecorder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include <SDL.h>
#include <glbinding/gl/gl.h>
#include <glm/glm.hpp>

#include "raisim/World.hpp"
#include "rayrai/RayraiWindow.hpp"

#if defined(_WIN32)
#define RAISIMGYM_POPEN _popen
#define RAISIMGYM_PCLOSE _pclose
#else
#include <sys/wait.h>
#define RAISIMGYM_POPEN popen
#define RAISIMGYM_PCLOSE pclose
#endif

namespace raisim {

namespace {

constexpr int kWidth = 1280, kHeight = 720;
constexpr double kFramesPerSecond = 30.;
/// camera position relative to the followed body, and its follow time constant
const glm::vec3 kCameraOffset(0.8f, -2.6f, 1.1f);
constexpr double kFollowTimeConstant = 0.1;

std::string findFfmpeg() {
  const char* path = std::getenv("PATH");
  if (path == nullptr) return {};
#if defined(_WIN32)
  const char separator = ';';
  const char* name = "ffmpeg.exe";
#else
  const char separator = ':';
  const char* name = "ffmpeg";
#endif
  std::string directories(path);
  for (size_t begin = 0; begin <= directories.size();) {
    size_t end = directories.find(separator, begin);
    if (end == std::string::npos) end = directories.size();
    if (end > begin) {
      const std::filesystem::path candidate =
          std::filesystem::path(directories.substr(begin, end - begin)) / name;
      std::error_code ec;
      if (std::filesystem::is_regular_file(candidate, ec)) return candidate.string();
    }
    begin = end + 1;
  }
  return {};
}

std::string quoted(const std::string& text) {
#if defined(_WIN32)
  return "\"" + text + "\"";
#else
  std::string result = "'";
  for (char c : text) result += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return result + "'";
#endif
}

/// Whether an OpenGL context can be created, checked without rayrai's fatal errors.
bool openGlAvailable(std::string& status) {
  // Leave SIGINT to Python, as rayrai's own SDL initialization does.
  SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
  if ((SDL_WasInit(SDL_INIT_VIDEO) & SDL_INIT_VIDEO) == 0 &&
      SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
    const std::string reason = SDL_GetError();
    // Without a display, SDL's offscreen driver can still render through EGL.
    SDL_SetHint(SDL_HINT_VIDEODRIVER, "offscreen");
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
      status = "no display for OpenGL rendering (" + reason + ")";
      return false;
    }
  }
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_Window* window = SDL_CreateWindow("raisimgym_gl_probe", SDL_WINDOWPOS_UNDEFINED,
      SDL_WINDOWPOS_UNDEFINED, 1, 1, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
  if (window == nullptr) {
    status = std::string("cannot create an OpenGL window (") + SDL_GetError() + ")";
    return false;
  }
  SDL_GLContext context = SDL_GL_CreateContext(window);
  if (context == nullptr) status = std::string("no OpenGL 3.3 context (") + SDL_GetError() + ")";
  else SDL_GL_DeleteContext(context);
  SDL_DestroyWindow(window);
  return context != nullptr;
}

void lookAt(raisin::Camera& camera, const glm::vec3& eye, const glm::vec3& target) {
  camera.position = eye;
  camera.front = glm::normalize(target - eye);
  camera.worldUp = camera.up = glm::vec3(0.f, 0.f, 1.f);
  camera.yaw = glm::degrees(std::atan2(camera.front.y, camera.front.x));
  camera.pitch = glm::degrees(std::asin(std::clamp(camera.front.z, -1.f, 1.f)));
  camera.zoom = 50.f;
  camera.horizontalFovRad = glm::radians(50.f);
  camera.nearPlane = camera.zNear = 0.05f;
  camera.farPlane = camera.zFar = 200.f;
  camera.aspect = float(kWidth) / float(kHeight);
  camera.update(false);
}

}  // namespace

struct VideoRecorder::Impl {
  SDL_Window* window = nullptr;
  SDL_GLContext context = nullptr;
  std::unique_ptr<raisin::RayraiWindow> renderer;
  World* world = nullptr;

  std::FILE* ffmpeg = nullptr;
  std::string path, status;
  bool writeFailed = false;
  size_t frames = 0;
  double nextFrameTime = 0., lastWorldTime = 0., lastRenderTime = 0.;
  std::vector<unsigned char> rgba, row;
  glm::vec3 focus{0.f};
  bool focusValid = false;

  ~Impl() {
    if (ffmpeg != nullptr) RAISIMGYM_PCLOSE(ffmpeg);
    if (renderer) {
      raisin::RayraiWindow::makeOffscreenContextCurrent(window, context);
      renderer.reset();
    }
    if (context != nullptr) SDL_GL_DeleteContext(context);
    if (window != nullptr) SDL_DestroyWindow(window);
  }

  bool ensureRenderer(World& target) {
    if (context == nullptr) {
      if (!openGlAvailable(status)) return false;
      raisin::RayraiWindow::createOffscreenGlContext(window, context, "raisimgym_video");
    }
    if (renderer && world == &target) return true;
    renderer = std::make_unique<raisin::RayraiWindow>(target, kWidth, kHeight);
    // The rayrai TCP viewer's default look.
    renderer->setRenderQualityPreset(raisin::RenderQualityPreset::Balanced);
    renderer->setKeyboardInputEnabled(false);
    renderer->setKeyboardShortcutsEnabled(false);
    world = &target;
    return true;
  }

  /// The base of the first articulated system, or the origin.
  glm::vec3 followedPosition() const {
    for (auto* object : world->getObjList())
      if (object->getObjectType() == ObjectType::ARTICULATED_SYSTEM) {
        const auto position = static_cast<ArticulatedSystem*>(object)->getBasePosition();
        return {float(position[0]), float(position[1]), float(position[2])};
      }
    return {0.f, 0.f, 0.5f};
  }

  void render() {
    const double time = world->getWorldTime();
    const glm::vec3 target = followedPosition();
    // Snap on the first frame and when a reset moves the body far away; follow smoothly otherwise.
    if (!focusValid || glm::distance(target, focus) > 3.f) {
      focus = target;
      focusValid = true;
    } else {
      const double blend = 1. - std::exp(-std::max(0., time - lastRenderTime) / kFollowTimeConstant);
      focus += (target - focus) * float(blend);
    }
    lastRenderTime = time;

    auto& camera = renderer->getCamera();
    lookAt(camera, focus + kCameraOffset, focus);
    renderer->update(kWidth, kHeight, false, 0, 0, false);

    const int width = camera.rtWidth(), height = camera.rtHeight();
    rgba.resize(size_t(width) * size_t(height) * 4u);
    gl::GLint packAlignment = 4;
    gl::glGetIntegerv(gl::GL_PACK_ALIGNMENT, &packAlignment);
    gl::glBindTexture(gl::GL_TEXTURE_2D, camera.getFinalTexture());
    gl::glPixelStorei(gl::GL_PACK_ALIGNMENT, 1);
    gl::glGetTexImage(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
    gl::glPixelStorei(gl::GL_PACK_ALIGNMENT, packAlignment);
    gl::glBindTexture(gl::GL_TEXTURE_2D, 0);

    // OpenGL rows start at the bottom; video rows start at the top.
    const size_t stride = size_t(width) * 4u;
    row.resize(stride);
    for (int y = 0; y < height / 2; ++y) {
      unsigned char* top = rgba.data() + size_t(y) * stride;
      unsigned char* bottom = rgba.data() + size_t(height - 1 - y) * stride;
      std::copy(top, top + stride, row.data());
      std::copy(bottom, bottom + stride, top);
      std::copy(row.data(), row.data() + stride, bottom);
    }
  }
};

VideoRecorder::VideoRecorder() : impl_(std::make_unique<Impl>()) {}
VideoRecorder::~VideoRecorder() = default;

bool VideoRecorder::start(World& world, const std::string& path) {
  auto& d = *impl_;
  if (d.ffmpeg != nullptr) stop();

  const std::string ffmpeg = findFfmpeg();
  if (ffmpeg.empty()) {
    d.status = "ffmpeg is not on PATH";
    return false;
  }
  if (!d.ensureRenderer(world)) return false;

  const std::string command = quoted(ffmpeg) +
      " -loglevel error -y -f rawvideo -pix_fmt rgba -s " + std::to_string(kWidth) + "x" +
      std::to_string(kHeight) + " -r " + std::to_string(int(kFramesPerSecond)) +
      " -i - -an -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart " +
      quoted(path);
#if defined(_WIN32)
  d.ffmpeg = RAISIMGYM_POPEN(command.c_str(), "wb");
#else
  d.ffmpeg = RAISIMGYM_POPEN(command.c_str(), "w");
#endif
  if (d.ffmpeg == nullptr) {
    d.status = "cannot start ffmpeg";
    return false;
  }

  d.path = path;
  d.frames = 0;
  d.writeFailed = false;
  d.focusValid = false;
  d.nextFrameTime = d.lastWorldTime = d.lastRenderTime = world.getWorldTime();
  d.status = "recording " + path;
  // Let shadows and temporal effects settle before the first frame.
  for (int i = 0; i < 3; ++i) d.render();
  capture();
  return true;
}

void VideoRecorder::capture() {
  auto& d = *impl_;
  if (d.ffmpeg == nullptr || d.writeFailed) return;
  const double time = d.world->getWorldTime();
  if (time < d.lastWorldTime) d.nextFrameTime = time;  // the world clock was reset
  d.lastWorldTime = time;
  if (time + 1e-9 < d.nextFrameTime) return;

  d.render();
  // Repeat the frame when a control step spans more than one video frame.
  for (; d.nextFrameTime <= time + 1e-9; d.nextFrameTime += 1. / kFramesPerSecond) {
    if (std::fwrite(d.rgba.data(), 1, d.rgba.size(), d.ffmpeg) != d.rgba.size()) {
      d.writeFailed = true;
      return;
    }
    ++d.frames;
  }
}

bool VideoRecorder::stop() {
  auto& d = *impl_;
  if (d.ffmpeg == nullptr) {
    d.status = "no video is being recorded";
    return false;
  }
  const int closeStatus = RAISIMGYM_PCLOSE(d.ffmpeg);
  d.ffmpeg = nullptr;
#if defined(_WIN32)
  const bool ffmpegFailed = closeStatus != 0;
#else
  const bool ffmpegFailed = !WIFEXITED(closeStatus) || WEXITSTATUS(closeStatus) != 0;
#endif
  if (d.writeFailed || ffmpegFailed) {
    d.status = "ffmpeg failed while writing " + d.path;
    return false;
  }
  if (d.frames == 0) {
    d.status = "no frames were captured for " + d.path;
    return false;
  }
  d.status = "wrote " + std::to_string(d.frames) + " frames to " + d.path;
  return true;
}

bool VideoRecorder::isRecording() const { return impl_->ffmpeg != nullptr; }

const std::string& VideoRecorder::status() const { return impl_->status; }

}
