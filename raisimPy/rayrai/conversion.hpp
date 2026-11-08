#pragma once
#include "rayrai/MeshPreprocessCache.hpp"
#include "runtime.hpp"
#include "values.hpp"
#include <span>
#include <sstream>

namespace rayrai_py {
template <class T> inline constexpr bool gpu_type = false;
template <class T> inline constexpr bool value_type = false;
#include "generated/traits.inc"

template <class T> using Bare = std::remove_cv_t<std::remove_reference_t<T>>;
template <class T> struct MapBase {
  using type = T;
};
template<> struct MapBase<Window::RenderOverrides> { using type=RenderOverridesValue; };
template <> struct MapBase<std::shared_ptr<const unsigned int>> {
  using type = TextureHandle;
};
template <class T>
  requires gpu_type<T>
struct MapBase<T> {
  using type = Handle<T>;
};
template <class T>
  requires gpu_type<std::remove_const_t<T>>
struct MapBase<T *> {
  using type = std::optional<Handle<std::remove_const_t<T>>>;
};
template <class T>
  requires gpu_type<std::remove_const_t<T>>
struct MapBase<std::shared_ptr<T>> {
  using type = Handle<std::remove_const_t<T>>;
};
template <class T> struct MapBase<std::vector<T>> {
  using type = std::vector<typename MapBase<T>::type>;
};
template <class T, size_t N> struct MapBase<std::array<T, N>> {
  using type = std::array<typename MapBase<T>::type, N>;
};
template <class T> struct MapBase<std::shared_ptr<std::vector<T>>> {
  using type = std::vector<typename MapBase<T>::type>;
};
template <class T> struct MapBase<std::shared_ptr<const std::vector<T>>> {
  using type = std::vector<typename MapBase<T>::type>;
};
template <class T, size_t N> struct MapBase<std::span<T, N>> {
  using type = std::vector<typename MapBase<std::remove_const_t<T>>::type>;
};
template <class T> using Mapped = typename MapBase<Bare<T>>::type;
template <class T>
using Arg = std::conditional_t<std::is_same_v<Bare<T>, Mapped<T>> &&
                                   !std::is_rvalue_reference_v<T>,
                               T, Mapped<T>>;

inline void same_state(const std::shared_ptr<State> &state,
                       const std::shared_ptr<State> &other) {
  if (!other || other->closed)
    throw std::runtime_error("The Rayrai argument belongs to a closed viewer.");
  if (!state || state != other)
    throw std::invalid_argument(
        "Rayrai objects must belong to the same viewer/context.");
}
template <class T> struct IsVector : std::false_type {};
template <class T> struct IsVector<std::vector<T>> : std::true_type {
  using value_type = T;
};
template <class T> struct IsShared : std::false_type {};
template <class T> struct IsShared<std::shared_ptr<T>> : std::true_type {
  using value_type = T;
};
template <class T> struct IsArray : std::false_type {};
template <class T, size_t N> struct IsArray<std::array<T, N>> : std::true_type {
  using value_type = T;
};
template <class T> struct IsSpan : std::false_type {};
template <class T, size_t N> struct IsSpan<std::span<T, N>> : std::true_type {
  using value_type = T;
};

template <class S> struct SpanStorage;
template <class T, class A>
decltype(auto) unwrap(A &value, const std::shared_ptr<State> &state) {
  using B = Bare<T>;
  if constexpr (std::is_same_v<B,Window::RenderOverrides>) return value.native(state);
  else if constexpr (gpu_type<B>) {
    same_state(state, value.state);
    return value.get();
  } else if constexpr (std::is_pointer_v<B> &&
                       gpu_type<
                           std::remove_const_t<std::remove_pointer_t<B>>>) {
    if (!value)
      return static_cast<B>(nullptr);
    same_state(state, value->state);
    return static_cast<B>(&value->get());
  } else if constexpr (IsShared<B>::value) {
    using V = std::remove_const_t<typename IsShared<B>::value_type>;
    if constexpr (std::is_same_v<B, std::shared_ptr<const unsigned int>>) {
      same_state(state, value.state);
      value.get();
      return std::shared_ptr<const unsigned int>(value.record->owned,
                                                 &value.get());
    } else if constexpr (gpu_type<std::remove_const_t<V>>) {
      same_state(state, value.state);
      value.get();
      if (value.record->owned)
        return std::shared_ptr<V>(value.record->owned, &value.get());
      return std::shared_ptr<V>(value.state, &value.get());
    } else if constexpr (IsVector<V>::value) {
      V out;
      for (auto &item : value)
        out.push_back(unwrap<typename V::value_type>(item, state));
      return std::make_shared<V>(std::move(out));
    } else
      return (value);
  } else if constexpr (IsVector<B>::value) {
    if constexpr (std::is_same_v<B, Mapped<B>>)
      return (value);
    else {
      B out;
      for (auto &item : value)
        out.push_back(unwrap<typename B::value_type>(item, state));
      return out;
    }
  } else if constexpr (IsArray<B>::value) {
    if constexpr (std::is_same_v<B, Mapped<B>>)
      return (value);
    else {
      B out;
      for (size_t i = 0; i < value.size(); ++i)
        out[i] = unwrap<typename B::value_type>(value[i], state);
      return out;
    }
  } else if constexpr (IsSpan<B>::value)
    return SpanStorage<B>(value, state);
  else
    return (value);
}

template <class S> struct SpanStorage {
  using V = std::remove_const_t<typename S::element_type>;
  std::vector<V> data;
  template <class A>
  SpanStorage(A &value, const std::shared_ptr<State> &state) {
    for (auto &item : value)
      data.push_back(unwrap<V>(item, state));
  }
  operator S() const { return S(data); }
};

template <class T>
py::object wrap(const std::shared_ptr<State> &state, T &&value) {
  using B = Bare<T>;
  if constexpr (std::is_same_v<B,Window::RenderOverrides>)
    return py::cast(RenderOverridesValue(value,state));
  else if constexpr (gpu_type<B>) {
    // Reference returns belong to the viewer; non-reference GPU returns are
    // constructed by generated bindings directly, without moving a resource.
    return py::cast(borrow(state, const_cast<B *>(&value)));
  } else if constexpr (std::is_pointer_v<B> &&
                       gpu_type<
                           std::remove_const_t<std::remove_pointer_t<B>>>) {
    if (!value)
      return py::none();
    using V = std::remove_const_t<std::remove_pointer_t<B>>;
    if (!state || state->closed)
      throw std::runtime_error("A live Rayrai viewer is required for this handle.");
    if constexpr (std::is_same_v<V, raisin::Visuals> || std::is_same_v<V, raisin::InstancedVisuals>) {
      auto existing = state->by_pointer.find(const_cast<V*>(value));
      if (existing != state->by_pointer.end()) {
        if (auto record=existing->second.lock(); record && record->pointer && record->owned)
          return py::cast(Handle<V>{state,std::move(record)});
      }
      if constexpr (std::is_same_v<V,raisin::Visuals>) {
        auto owned=state->renderer->findVisualObject(value->getName());
        if (owned.get()==value) return py::cast(own(state,std::move(owned)));
      } else {
        auto owned=state->renderer->findInstancedVisuals(value->getName());
        if (owned.get()==value) return py::cast(own(state,std::move(owned)));
      }
      throw std::runtime_error("The returned visual has no live shared owner.");
    } else return py::cast(borrow(state, const_cast<V *>(value)));
  } else if constexpr (std::is_pointer_v<B> &&
                       std::is_base_of_v<
                           raisim::Object,
                           std::remove_const_t<std::remove_pointer_t<B>>>) {
    if (!value)
      return py::none();
    if (!state || !state->world)
      throw std::runtime_error("An object identity requires its live World.");
    const auto &objects = state->world->getObjList();
    if (std::find(objects.begin(), objects.end(), value) == objects.end())
      throw std::runtime_error(
          "The source object is no longer in this viewer's World.");
    return py::cast(value, py::rv_policy::reference_internal,
                    py::cast(state->world));
  } else if constexpr (IsShared<B>::value) {
    if (!value)
      return py::none();
    using V = std::remove_const_t<typename IsShared<B>::value_type>;
    if constexpr (std::is_same_v<B, std::shared_ptr<const unsigned int>>)
      return py::cast(TextureHandle{state, own(state, value).record});
    else if constexpr (gpu_type<std::remove_const_t<V>>)
      return py::cast(own(state, value));
    else if constexpr (IsVector<V>::value)
      return wrap(state, *value);
    else
      return py::cast(value);
  } else if constexpr (IsVector<B>::value || IsArray<B>::value) {
    py::list result;
    for (auto &item : value)
      result.append(wrap(state, item));
    return result;
  } else if constexpr (std::is_pointer_v<B> &&
                       value_type<
                           std::remove_const_t<std::remove_pointer_t<B>>>) {
    if (!value)
      return py::none();
    return py::cast(*value, py::rv_policy::copy);
  } else
    return py::cast(std::forward<T>(value), py::rv_policy::copy);
}

template <class T> void default_constructor(py::class_<T> &cls) {
  if constexpr (std::is_default_constructible_v<T>)
    cls.def(py::init<>());
}
// File timestamps have an implementation-specific epoch and representation
// (including 128-bit counts on some standard libraries). Expose Unix ns.
template<class Time> struct FileTimeConverter {
  using Clock=typename Time::clock;
  using Nanoseconds=std::chrono::nanoseconds;
  static auto epoch_offset() {
    static const auto offset=std::chrono::system_clock::now().time_since_epoch()-
        std::chrono::duration_cast<std::chrono::system_clock::duration>(Clock::now().time_since_epoch());
    return offset;
  }
  static int64_t to_python(Time value) {
    if constexpr(requires {Clock::to_sys(value);})
      return std::chrono::duration_cast<Nanoseconds>(Clock::to_sys(value).time_since_epoch()).count();
    else
      return std::chrono::duration_cast<Nanoseconds>(value.time_since_epoch()+epoch_offset()).count();
  }
  static Time from_python(int64_t value) {
    auto unix_time=std::chrono::sys_time<Nanoseconds>(Nanoseconds(value));
    if constexpr(requires {Clock::from_sys(unix_time);})
      return std::chrono::time_point_cast<typename Time::duration>(Clock::from_sys(unix_time));
    else
      return Time(std::chrono::duration_cast<typename Time::duration>(unix_time.time_since_epoch()-epoch_offset()));
  }
};
template <class C, class V>
void field(py::class_<C> &cls, const char *name, V C::*member) {
  if constexpr (!std::is_same_v<V, Mapped<V>>) {
    cls.def_prop_rw(
        name, [member](C &self) { return wrap(current_state(), self.*member); },
        [member](C &self, Mapped<V> value) {
          self.*member = unwrap<V>(value, current_state());
        });
  } else if constexpr (std::is_same_v<V, std::filesystem::file_time_type>) {
    cls.def_prop_rw(
        name,
        [member](C &self) { return FileTimeConverter<V>::to_python(self.*member); },
        [member](C &self, int64_t value) {
          self.*member = FileTimeConverter<V>::from_python(value);
        });
  } else if constexpr(std::is_same_v<V,const char*>) {
    cls.def_ro(name,member);
  } else
    cls.def_rw(name, member);
}

template<class V> void field(py::class_<RenderOverridesValue>& cls,const char* name,V Window::RenderOverrides::*member) {
  if constexpr(std::is_same_v<V,Shader*>) {
    size_t index=member==&Window::RenderOverrides::mesh?0:member==&Window::RenderOverrides::pointcloud?1:member==&Window::RenderOverrides::picking?2:3;
    cls.def_prop_rw(name,[index](RenderOverridesValue& self){return self.shaders[index];},
      [index](RenderOverridesValue& self,std::optional<Handle<Shader>> value){if(value)value->get();self.shaders[index]=std::move(value);},
      py::for_setter(py::arg("value").none()));
  } else {
    cls.def_prop_rw(name,[member](RenderOverridesValue& self){return wrap({},self.data.*member);},
      [member](RenderOverridesValue& self,const V& value){self.data.*member=value;});
  }
}

template <class C, class V>
void field(py::class_<Handle<C>> &cls, const char *name, V C::*member) {
  cls.def_prop_rw(
      name,
      [member](Handle<C> &self) {
        Guard guard(self.state);
        auto &value = self.get().*member;
        return wrap(self.state, value);
      },
      [member](Handle<C> &self, Mapped<V> value) {
        Guard guard(self.state);
        self.get().*member = unwrap<V>(value, self.state);
      });
}
} // namespace rayrai_py
