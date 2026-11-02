#pragma once

#include "glm_casters.hpp"
#include "rayrai/RayraiWindow.hpp"
#include "rayrai/example_common.hpp"
#include <SDL.h>
#include <functional>
#include <imgui/imgui.h>
#include <mutex>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/variant.h>
#include <optional>
#include <thread>
#include <unordered_map>

namespace rayrai_py {
namespace py = nanobind;
using Window = raisin::RayraiWindow;

struct State;
extern thread_local State *active_weather_callback;
struct Record {
  std::weak_ptr<State> state;
  std::shared_ptr<void> owned;
  void *pointer = nullptr;
  std::function<void *()> resolve;
  bool world_dependent = false;
  ~Record();
};

struct State : std::enable_shared_from_this<State> {
  std::recursive_mutex mutex;
  SDL_Window *window = nullptr;
  SDL_GLContext context = nullptr;
  ImGuiContext *imgui = nullptr;
  std::unique_ptr<Window> renderer;
  std::shared_ptr<raisim::World> world;
  std::thread::id owner;
  int width = 0, height = 0;
  bool closed = false, quit = false, visible = false;
  std::unordered_map<void *, std::weak_ptr<Record>> by_pointer;
  std::vector<std::weak_ptr<Record>> records;
  std::shared_ptr<Record> target_visual;
  py::object callback_viewer; // weakref only; the callable lives in Python's GC-visible dict
  std::shared_ptr<raisim::World> user_locked_world;
  bool unlock_after_close = false;
  size_t registrations = 0;
  std::mutex disposal_mutex;
  std::vector<std::shared_ptr<void>> pending_disposal;
  struct Slider {
    std::string name;
    float min, max, value;
  };
  std::vector<Slider> sliders;

  static std::shared_ptr<State>
  create(std::shared_ptr<raisim::World> world, int width, int height,
         bool visible, const std::string &title, Window::ThreadingMode mode,
         bool shader_cache, const std::string &shader_cache_directory,
         unsigned shader_threads = 1, bool log_shader_cache = false);
  ~State();
  void check_thread() const;
  void make_current(bool flush_disposals = true);
  void drain_disposals();
  void close();
  void poll_events();
  bool render(bool process_events = true);
  void dispose(std::shared_ptr<void> value) noexcept;
  std::shared_ptr<Record> record(void *pointer,
                                 std::shared_ptr<void> owned = {});
  void invalidate_borrowed();
  void reconcile_target();
  void lock_user_world();
  void unlock_user_world();
};

std::shared_ptr<State> current_state(bool required = false);
void close_all();
bool caller_holds_world(raisim::World* world);
void enter_guard();
void leave_guard() noexcept;
bool outermost_guard();

struct Guard {
  std::shared_ptr<State> state;
  std::unique_lock<std::recursive_mutex> lock;
  raisim::ScopedRaiSimFatalCallback fatal;
  explicit Guard(std::shared_ptr<State> state, bool required = true)
      : state(std::move(state)), fatal([] {
          throw std::runtime_error(
              "Rayrai reported an error; see the RaiSim diagnostic above.");
        }) {
    if (this->state) {
      // Reject foreign threads before waiting for a mutex held by an owner
      // that may itself be waiting for Python work on that foreign thread.
      this->state->check_thread();
      lock = std::unique_lock<std::recursive_mutex>(this->state->mutex);
      this->state->make_current();
    } else if (required)
      throw std::runtime_error("A live Rayrai viewer/context is required.");
    enter_guard();
  }
  ~Guard() { if (state && !state->closed) state->reconcile_target(); leave_guard(); }
};

struct WorldLock {
  std::shared_ptr<raisim::World> world;
  std::unique_lock<raisim::World> lock;
  explicit WorldLock(const std::shared_ptr<State> &state)
      : world(state->world), lock(*world, std::defer_lock) {
    if (active_weather_callback)
      throw std::runtime_error("Defer rendering and World operations until the "
                               "Rayrai callback returns.");
    if (caller_holds_world(world.get())) return;
    if (!lock.try_lock()) {
      {
        py::gil_scoped_release release;
        lock.lock();
      }
      // Another Python thread may have used a different ImGui context while
      // the owner waited. Avoid running resource finalizers under the World
      // lock.
      state->make_current(false);
    }
  }
};

struct WorldPairLock {
  std::shared_ptr<raisim::World> first, second;
  std::unique_lock<raisim::World> a, b;
  WorldPairLock(const std::shared_ptr<State>& state, std::shared_ptr<raisim::World> other)
      : first(state->world), second(std::move(other)), a(*first,std::defer_lock) {
    if (!second) throw std::invalid_argument("world must not be None");
    if (active_weather_callback) throw std::runtime_error("Swap Worlds after the callback returns.");
    const bool need_a=!caller_holds_world(first.get());
    const bool need_b=second!=first && !caller_holds_world(second.get());
    if (need_b) b=std::unique_lock<raisim::World>(*second,std::defer_lock);
    if (need_a && need_b) {
      {py::gil_scoped_release release; std::lock(a,b);}
      state->make_current(false);
    } else if (need_a || need_b) {
      auto& lock=need_a?a:b;
      if (!lock.try_lock()) {
        {py::gil_scoped_release release; lock.lock();}
        state->make_current(false);
      }
    }
  }
};

template <typename T> struct Handle {
  std::shared_ptr<State> state;
  std::shared_ptr<Record> record;
  T &get() const {
    if (!state || state->closed || !record || !record->pointer)
      throw std::runtime_error(
          "This Rayrai object belongs to a closed or invalidated viewer.");
    state->check_thread();
    void *p = record->resolve ? record->resolve() : record->pointer;
    if (!p)
      throw std::runtime_error("The rendered object no longer exists.");
    return *static_cast<T *>(p);
  }
};

using Viewer = Handle<Window>;
using TextureHandle = Handle<const unsigned int>;
struct ModelHandle {
  std::shared_ptr<State> state;
  std::shared_ptr<Record> record;
};

template <typename T>
Handle<T> borrow(const std::shared_ptr<State> &state, T *pointer) {
  if (!pointer)
    return {};
  if (!state || state->closed)
    throw std::runtime_error("A live Rayrai viewer is required for this handle.");
  return {state, state->record(const_cast<std::remove_const_t<T> *>(pointer))};
}

template <typename T>
Handle<std::remove_const_t<T>> own(const std::shared_ptr<State> &state,
                                   std::shared_ptr<T> pointer) {
  if (!pointer)
    return {};
  if (!state || state->closed)
    throw std::runtime_error("A live Rayrai viewer is required for this handle.");
  auto mutable_pointer =
      std::const_pointer_cast<std::remove_const_t<T>>(pointer);
  return {state, state->record(mutable_pointer.get(), mutable_pointer)};
}

template <typename T, typename... Args>
Handle<T> construct(const Viewer &viewer, Args &&...args) {
  Guard guard(viewer.state);
  return own(viewer.state,
             std::shared_ptr<T>(new T(std::forward<Args>(args)...)));
}

template <typename T>
py::class_<T> bound_class(py::handle scope, const char *name) {
  return py::borrow<py::class_<T>>(scope.attr(name));
}

void bind_runtime(py::module_ &module);
void bind_manual(py::module_ &module);
void declare_generated(py::module_ &module);
void define_generated(py::module_ &module);

} // namespace rayrai_py
