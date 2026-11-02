#include "runtime.hpp"
#include <algorithm>
#include <cmath>
#include <glbinding/gl/gl.h>

namespace rayrai_py {
thread_local State *active_weather_callback = nullptr;
namespace {
std::recursive_mutex registry_mutex;
std::vector<std::weak_ptr<State>> states;
thread_local std::weak_ptr<State> active;
std::optional<Window::ThreadingMode> process_mode;
bool owns_video = false;
size_t window_count = 0;
thread_local unsigned resource_cleanup_depth = 0;
thread_local unsigned guard_depth = 0;
thread_local bool draining_states = false;
std::unordered_map<raisim::World*,std::thread::id> user_world_owners;
struct ResourceCleanup {
  ResourceCleanup() { ++resource_cleanup_depth; }
  ~ResourceCleanup() { --resource_cleanup_depth; }
};
// Objects collected on a foreign Python thread are disposed on the owner at
// its next API call or shutdown. Never destroy a GL context on that thread.
auto &deferred_states() {
  static auto *queue = new std::vector<State *>;
  return *queue;
}

void fail_sdl(const char *action) {
  throw std::runtime_error(std::string(action) + ": " + SDL_GetError());
}

void drain_deferred_states() {
  if (draining_states || resource_cleanup_depth || active_weather_callback) return;
  draining_states=true;
  struct Reset { ~Reset(){draining_states=false;} } reset;
  std::vector<State *> batch;
  {
    std::lock_guard lock(registry_mutex);
    for (auto it = deferred_states().begin(); it != deferred_states().end();) {
      if ((*it)->owner == std::this_thread::get_id()) {
        batch.push_back(*it);
        it = deferred_states().erase(it);
      } else
        ++it;
    }
  }
  // Native destruction can release Python objects. Never keep the registry
  // locked while their finalizers execute arbitrary Python code.
  for (auto *value : batch)
    delete value;
}

struct PreviousContext {
  SDL_Window* window=SDL_GL_GetCurrentWindow();
  SDL_GLContext context=SDL_GL_GetCurrentContext();
  ImGuiContext* imgui=ImGui::GetCurrentContext();
  std::weak_ptr<State> previous=active;
  bool managed=false;
  bool enabled=true;
  SDL_GLContext closing;
  PreviousContext(SDL_GLContext current,ImGuiContext* closing_imgui):closing(current) {
    if(imgui==closing_imgui) imgui=nullptr;
    std::lock_guard lock(registry_mutex);
    for(auto& weak:states) if(auto state=weak.lock();state && state->context==context) {
      managed=true; previous=state; break;
    }
  }
  ~PreviousContext() noexcept {
    if(!enabled) return;
    try {
      if (managed) {
        if(auto state=previous.lock();state && !state->closed && state->context!=closing) {
          state->make_current(false); return;
        }
      } else if(context && context!=closing) {
        if(SDL_GL_MakeCurrent(window,context)==0) {
          Window::makeOffscreenContextCurrent(window,context);
          ImGui::SetCurrentContext(imgui); active.reset(); return;
        }
      }
      SDL_GL_MakeCurrent(nullptr,nullptr);
      ImGui::SetCurrentContext(nullptr); active.reset();
    } catch (...) {
      SDL_GL_MakeCurrent(nullptr,nullptr);
      ImGui::SetCurrentContext(nullptr); active.reset();
    }
  }
  void dismiss() noexcept {enabled=false;}
};
} // namespace

void enter_guard(){++guard_depth;}
void leave_guard() noexcept {--guard_depth;}
bool outermost_guard(){return guard_depth==0 && !active_weather_callback;}
bool caller_holds_world(raisim::World* world) {
  std::lock_guard lock(registry_mutex);
  auto found=user_world_owners.find(world);
  return found!=user_world_owners.end() && found->second==std::this_thread::get_id();
}

std::shared_ptr<State> current_state(bool required) {
  auto state =
      resource_cleanup_depth ? std::shared_ptr<State>() : active.lock();
  if (state && state->closed)
    state.reset();
  if (!state && required)
    throw std::runtime_error("Create or make current a Rayrai viewer first.");
  return state;
}

void State::check_thread() const {
  if (owner != std::this_thread::get_id())
    throw std::runtime_error("Use and close Rayrai objects on the thread that "
                             "created their viewer.");
}

void State::make_current(bool flush_disposals) {
  check_thread();
  if (resource_cleanup_depth)
    throw std::runtime_error(
        "Rayrai contexts cannot be used during resource finalization.");
  if (closed || !window || !context)
    throw std::runtime_error("The Rayrai viewer is closed.");
  if (flush_disposals && outermost_guard()) drain_deferred_states();
  if (SDL_GL_GetCurrentContext() != context) {
    if (SDL_GL_MakeCurrent(window, context) != 0)
      fail_sdl("Cannot make the Rayrai context current");
    Window::makeOffscreenContextCurrent(window, context);
  }
  ImGui::SetCurrentContext(imgui);
  active = shared_from_this();
  if (flush_disposals && outermost_guard())
    drain_disposals();
}

void State::drain_disposals() {
  ResourceCleanup cleanup;
  for (;;) {
    std::vector<std::shared_ptr<void>> batch;
    {
      std::lock_guard lock(disposal_mutex);
      batch.swap(pending_disposal);
    }
    if (batch.empty())
      return;
    batch.clear();
  }
}

std::shared_ptr<State>
State::create(std::shared_ptr<raisim::World> world, int width, int height,
              bool visible, const std::string &title,
              Window::ThreadingMode mode, bool shader_cache,
              const std::string &shader_cache_directory,
              unsigned shader_threads, bool log_shader_cache) {
  if (active_weather_callback)
    throw std::runtime_error(
        "Create Rayrai viewers after the callback returns.");
  if (resource_cleanup_depth)
    throw std::runtime_error(
        "Rayrai viewers cannot be created during resource finalization.");
  if (outermost_guard()) drain_deferred_states();
  if (!world)
    throw std::invalid_argument("world must be a live raisimpy.World");
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("Render dimensions must be positive.");
  bool require_main_thread=visible;
#ifdef __APPLE__
  require_main_thread=true;
#endif
  if(require_main_thread) {
    auto threading=py::module_::import_("threading");
    if(!threading.attr("current_thread")().is(threading.attr("main_thread")()))
      throw std::runtime_error("Create visible Rayrai viewers on Python's main thread (all viewers on macOS).");
  }
  PreviousContext previous(nullptr,nullptr);
  std::lock_guard global(registry_mutex);
  std::erase_if(states,[](auto& weak){return weak.expired();});
  if (process_mode && *process_mode != mode)
    throw std::invalid_argument(
        "All Rayrai viewers in a process must use the same threading mode.");
  auto state = std::shared_ptr<State>(new State, [](State *value) {
    if (value->closed ||
        (value->owner == std::this_thread::get_id() && !resource_cleanup_depth))
      delete value;
    else {
      std::lock_guard lock(registry_mutex);
      deferred_states().push_back(value);
    }
  });
  state->world = std::move(world);
  state->width = width;
  state->height = height;
  state->visible = visible;
  state->owner = std::this_thread::get_id();
  raisim::ScopedRaiSimFatalCallback fatal([] {
    throw std::runtime_error(
        "Rayrai initialization failed; see the RaiSim diagnostic above.");
  });
  try {
    SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
    if (!(SDL_WasInit(SDL_INIT_VIDEO) & SDL_INIT_VIDEO)) {
      if (SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
        fail_sdl("SDL initialization failed");
      owns_video = true;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    auto flags =
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    if (!visible)
      flags |= SDL_WINDOW_HIDDEN;
    state->window =
        SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED,
                         SDL_WINDOWPOS_CENTERED, width, height, flags);
    if (!state->window)
      fail_sdl("SDL window creation failed");
    ++window_count;
    state->context = SDL_GL_CreateContext(state->window);
    if (!state->context) {
      SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
      SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
      state->context = SDL_GL_CreateContext(state->window);
    }
    if (!state->context)
      fail_sdl("OpenGL context creation failed");
    if (SDL_GL_MakeCurrent(state->window, state->context) != 0)
      fail_sdl("OpenGL activation failed");
    Window::makeOffscreenContextCurrent(state->window, state->context);
#ifdef RAISIMPY_RAYRAI_TEST_FAULTS
    if(const char* fault=std::getenv("RAISIMPY_RAYRAI_FAIL_CREATE");fault && *fault=='1')
      throw std::runtime_error("Injected viewer creation failure");
#endif
    gl::GLint texture_units = 0, max_size = 0;
    gl::glGetIntegerv(gl::GL_MAX_TEXTURE_IMAGE_UNITS, &texture_units);
    gl::glGetIntegerv(gl::GL_MAX_TEXTURE_SIZE, &max_size);
    const auto capability_error =
        raisin::rayrai_internal::rendererCapabilityError(texture_units);
    if (!capability_error.empty())
      throw std::runtime_error(capability_error);
    if (width > max_size || height > max_size)
      throw std::invalid_argument("Resolution exceeds GL_MAX_TEXTURE_SIZE.");
    SDL_GL_SetSwapInterval(visible ? 1 : 0);
    IMGUI_CHECKVERSION();
    state->imgui = ImGui::CreateContext();
    ImGui::SetCurrentContext(state->imgui);
    ImGui::GetIO().IniFilename = nullptr;
    if (!ImGui_ImplSDL2_InitForOpenGL(state->window, state->context))
      throw std::runtime_error("ImGui SDL initialization failed.");
    if (!ImGui_ImplOpenGL3_Init("#version 330"))
      throw std::runtime_error("ImGui OpenGL initialization failed.");
    state->renderer = std::make_unique<Window>(
        state->world, width, height, mode, shader_threads, shader_cache,
        shader_cache_directory, log_shader_cache);
    process_mode = mode;
    states.push_back(state);
    active = state;
    previous.dismiss();
    return state;
  } catch (...) {
    state->close();
    throw;
  }
}

std::shared_ptr<Record> State::record(void *pointer,
                                      std::shared_ptr<void> owned) {
  if (!pointer)
    return {};
  if (++registrations % 256 == 0) {
    std::erase_if(records,[](auto& weak){return weak.expired();});
    std::erase_if(by_pointer,[](auto& entry){return entry.second.expired();});
  }
  if (auto existing = by_pointer[pointer].lock();
      existing && existing->pointer) {
    if (owned && !existing->owned)
      existing->owned = std::move(owned);
    return existing;
  }
  auto entry = std::make_shared<Record>();
  entry->state = shared_from_this();
  entry->pointer = pointer;
  entry->owned = std::move(owned);
  by_pointer[pointer] = entry;
  records.push_back(entry);
  return entry;
}

void State::reconcile_target() {
  if(target_visual && renderer->getTargetVisual()!=target_visual->pointer) target_visual.reset();
}

void State::lock_user_world() {
  check_thread();
  if(active_weather_callback)
    throw std::runtime_error("Acquire the World lock after the Rayrai callback returns.");
  if(closed) throw std::runtime_error("The Rayrai viewer is closed.");
  if(user_locked_world || caller_holds_world(world.get()))
    throw std::runtime_error("This thread already holds the World's user lock.");
  auto captured=world;
  std::unique_lock<raisim::World> acquired(*captured,std::defer_lock);
  if(!acquired.try_lock()) {
    {py::gil_scoped_release release;acquired.lock();}
    make_current(false);
  }
#ifdef RAISIMPY_RAYRAI_TEST_FAULTS
  if(const char* fault=std::getenv("RAISIMPY_RAYRAI_FAIL_LOCK");fault && *fault=='1')
    throw std::runtime_error("Injected World lock publication failure");
#endif
  std::lock_guard lock(registry_mutex);
  user_world_owners[captured.get()]=owner;
  user_locked_world=std::move(captured); unlock_after_close=false;
  acquired.release();
}

void State::unlock_user_world() {
  check_thread();
  if(!user_locked_world) {
    if(unlock_after_close){unlock_after_close=false;return;}
    throw std::runtime_error("This viewer has no user-held World lock.");
  }
  auto captured=std::move(user_locked_world);
  {std::lock_guard lock(registry_mutex);user_world_owners.erase(captured.get());}
  captured->unlock();
}

void State::invalidate_borrowed() {
  for (auto &weak : records) {
    if (auto record = weak.lock(); record && record->world_dependent)
      record->pointer = nullptr;
  }
}

void State::dispose(std::shared_ptr<void> value) noexcept {
  if (!value)
    return;
  // GC must not silently select this viewer's context. Dispose during its
  // next guarded call or close, with exactly that context already current.
  std::lock_guard lock(disposal_mutex);
  pending_disposal.push_back(std::move(value));
}

Record::~Record() {
  if (owned) {
    if (auto parent = state.lock())
      parent->dispose(std::move(owned));
  }
}

void State::close() {
  check_thread();
  if (closed)
    return;
  if (resource_cleanup_depth)
    throw std::runtime_error(
        "Rayrai viewers cannot be closed during resource finalization.");
  if (active_weather_callback == this)
    throw std::runtime_error("Use requestClose() inside a Rayrai callback and "
                             "close the viewer after rendering returns.");
  // Releasing a World mutex requires no GL context. Do it before any context
  // activation can fail, including when close() runs from the State deleter.
  if(user_locked_world){unlock_user_world();unlock_after_close=true;}
  std::lock_guard lock(mutex);
  if (closed)
    return;
  PreviousContext previous(context,imgui);
#ifdef RAISIMPY_RAYRAI_TEST_FAULTS
  if(const char* fault=std::getenv("RAISIMPY_RAYRAI_FAIL_CLOSE");fault && *fault=='1')
    throw std::runtime_error("Injected viewer close failure");
#endif
  if (context && window) {
    if (SDL_GL_MakeCurrent(window, context) != 0)
      fail_sdl("Cannot activate the context for cleanup");
    Window::makeOffscreenContextCurrent(window, context);
  }
  if (imgui)
    ImGui::SetCurrentContext(imgui);
  // Reject reentrant API calls before releasing any Python-owned callbacks or
  // materials. A finalizer must never render through a partially destroyed SDK
  // object.
  closed = true;
  quit = true;
  ResourceCleanup cleanup;
  if (renderer) {
    renderer->setWeatherThunderCallback({});
    renderer->setTargetVisual(nullptr);
  }
  if (callback_viewer) {
    auto object=callback_viewer();
    if(!object.is_none()) {
      auto attrs=object.attr("__dict__");
      if(PyDict_GetItemString(attrs.ptr(),"_rayrai_thunder_callback"))
        PyDict_DelItemString(attrs.ptr(),"_rayrai_thunder_callback");
    }
    callback_viewer.reset();
  }
  target_visual.reset();
  for (auto it = records.rbegin(); it != records.rend(); ++it) {
    if (auto record = it->lock()) {
      record->pointer = nullptr;
      record->resolve = {};
      record->owned.reset();
    }
  }
  by_pointer.clear();
  drain_disposals();
  renderer.reset();
  drain_disposals();
  if (imgui) {
    ImGui::SetCurrentContext(imgui);
    if (ImGui::GetIO().BackendRendererUserData)
      ImGui_ImplOpenGL3_Shutdown();
    if (ImGui::GetIO().BackendPlatformUserData)
      ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext(imgui);
    imgui = nullptr;
  }
  {
    std::lock_guard global(registry_mutex);
    if (context) {
      SDL_GL_DeleteContext(context);
      context = nullptr;
    }
    if (window) {
      SDL_DestroyWindow(window);
      window = nullptr;
      --window_count;
    }
    if (window_count == 0 && owns_video) {
      SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_TIMER);
      owns_video = false;
    }
  }
  world.reset();
}

State::~State() {
  if (!closed) {
    // The Python wrapper defers foreign-thread destruction before its final
    // shared_ptr is dropped. This is the normal owning-thread cleanup path.
    try {
      close();
    } catch (...) {
      if(user_locked_world) {
        try {unlock_user_world();} catch (...) {}
      }
      renderer.release();
    }
  }
}

void close_all() {
  std::vector<std::shared_ptr<State>> live;
  {
    std::lock_guard lock(registry_mutex);
    for (auto &weak : states)
      if (auto state = weak.lock();
          state && state->owner == std::this_thread::get_id())
        live.push_back(std::move(state));
  }
  for (auto &state : live)
    state->close();
  drain_deferred_states();
}

void State::poll_events() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    auto destination = shared_from_this();
    Uint32 event_window = 0;
    switch (event.type) {
    case SDL_WINDOWEVENT:
      event_window = event.window.windowID;
      break;
    case SDL_KEYDOWN:
    case SDL_KEYUP:
      event_window = event.key.windowID;
      break;
    case SDL_TEXTINPUT:
      event_window = event.text.windowID;
      break;
    case SDL_MOUSEMOTION:
      event_window = event.motion.windowID;
      break;
    case SDL_MOUSEBUTTONDOWN:
    case SDL_MOUSEBUTTONUP:
      event_window = event.button.windowID;
      break;
    case SDL_MOUSEWHEEL:
      event_window = event.wheel.windowID;
      break;
    }
    if (event_window) {
      std::lock_guard lock(registry_mutex);
      for (const auto &weak : states) {
        if (auto candidate = weak.lock();
            candidate && candidate->window &&
            SDL_GetWindowID(candidate->window) == event_window) {
          destination = candidate;
          break;
        }
      }
    }
    if (destination->imgui) {
      ImGui::SetCurrentContext(destination->imgui);
      ImGui_ImplSDL2_ProcessEvent(&event);
    }
    if (event.type == SDL_QUIT || (event.type == SDL_WINDOWEVENT &&
                                   event.window.event == SDL_WINDOWEVENT_CLOSE))
      destination->quit = true;
  }
  ImGui::SetCurrentContext(imgui);
  // Dispose viewers whose final Python reference was released on another
  // thread.
  drain_deferred_states();
  make_current();
}

bool State::render(bool process_events) {
  if (active_weather_callback)
    throw std::runtime_error(
        "Defer rendering until the Rayrai callback returns.");
  Guard guard(shared_from_this());
  if (process_events)
    poll_events();
  if (quit)
    return false;
  if (!visible) {
    WorldLock world_lock(shared_from_this());
    renderer->update(width, height, false, 0, 0, false);
    return true;
  }
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplSDL2_NewFrame();
  ImGui::NewFrame();
  struct FrameScope {
    bool begun=false,style=false,ended=false;
    ~FrameScope() {
      if(begun) ImGui::End();
      if(style) ImGui::PopStyleVar();
      if(!ended) ImGui::EndFrame();
    }
  } frame;
  int logical_w, logical_h, w, h;
  SDL_GetWindowSize(window, &logical_w, &logical_h);
  SDL_GL_GetDrawableSize(window, &w, &h);
  if (w <= 0 || h <= 0) {
    return true;
  }
  width = w;
  height = h;
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(float(logical_w), float(logical_h)));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  frame.style=true;
  ImGui::Begin("Rayrai viewport", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_NoBringToFrontOnFocus);
  frame.begun=true;
#ifdef RAISIMPY_RAYRAI_TEST_FAULTS
  if(const char* fault=std::getenv("RAISIMPY_RAYRAI_FAIL_FRAME");fault && *fault=='1')
    throw std::runtime_error("Injected visible-frame failure");
#endif
  const bool hovering = ImGui::IsWindowHovered();
  const auto mouse = ImGui::GetIO().MousePos;
  // ImGui uses -FLT_MAX for an unavailable mouse position. Clamp in floating
  // point before converting; converting that sentinel directly to int is UB.
  const auto pixel = [](float position, int logical, int drawable) {
    if (!std::isfinite(position) || position <= 0)
      return 0;
    const double scaled = double(position) * drawable / std::max(1, logical);
    return int(std::clamp(scaled, 0., double(drawable - 1)));
  };
  const int x = pixel(mouse.x, logical_w, w);
  const int y = pixel(mouse.y, logical_h, h);
  {
    WorldLock world_lock(shared_from_this());
    renderer->update(w, h, hovering, x, y, true);
  }
  ImGui::Image((ImTextureID)(intptr_t)renderer->getImageTexture(),
               ImVec2(float(logical_w), float(logical_h)), ImVec2(0, 1),
               ImVec2(1, 0));
  ImGui::End();
  frame.begun=false;
  ImGui::PopStyleVar();
  frame.style=false;
  if (!sliders.empty()) {
    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls");
    frame.begun=true;
    for (auto &slider : sliders)
      ImGui::SliderFloat(slider.name.c_str(), &slider.value, slider.min,
                         slider.max);
    ImGui::End();
    frame.begun=false;
  }
  ImGui::Render();
  frame.ended=true;
  gl::glBindFramebuffer(gl::GL_FRAMEBUFFER, 0);
  gl::glViewport(0, 0, w, h);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  SDL_GL_SwapWindow(window);
  return true;
}

void bind_runtime(py::module_ &m) {
  py::class_<Viewer>(m, "RayraiWindow",py::dynamic_attr(),py::is_weak_referenceable());
  m.attr("Viewer") = m.attr("RayraiWindow");
  m.attr("RayraiViewer") = m.attr("RayraiWindow");
}

} // namespace rayrai_py
