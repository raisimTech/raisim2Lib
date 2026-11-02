#pragma once
#include "runtime.hpp"
#include <array>

namespace rayrai_py {
// A value record can own GPU handles without exposing dangling native pointers.
// It stays constructible without a GL context; validation happens at each render.
struct RenderOverridesValue {
  Window::RenderOverrides data;
  std::array<std::optional<Handle<Shader>>, 4> shaders;

  RenderOverridesValue() = default;
  RenderOverridesValue(const Window::RenderOverrides& source,
                       const std::shared_ptr<State>& state) : data(source) {
    Shader* pointers[]{source.mesh,source.pointcloud,source.picking,source.post};
    for(size_t i=0;i<4;++i) if(pointers[i]) shaders[i]=borrow(state,pointers[i]);
    data.mesh=data.pointcloud=data.picking=data.post=nullptr;
  }
  Window::RenderOverrides native(const std::shared_ptr<State>& state) const {
    auto result=data;
    Shader** pointers[]{&result.mesh,&result.pointcloud,&result.picking,&result.post};
    for(size_t i=0;i<4;++i) if(shaders[i]) {
      if(!state || state!=shaders[i]->state)
        throw std::invalid_argument("Override shaders must belong to the rendering viewer.");
      *pointers[i]=&shaders[i]->get();
    }
    return result;
  }
};
}
