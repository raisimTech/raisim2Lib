#include "conversion.hpp"
#include <algorithm>
#include <cmath>
#include <glbinding/gl/gl.h>
#include <png/png.h>

namespace rayrai_py {
extern thread_local State *active_weather_callback;
namespace {
struct CallbackScope {
  std::shared_ptr<State> state;
  State *previous;
  bool restored=false;
  explicit CallbackScope(std::shared_ptr<State> owner)
      : state(std::move(owner)), previous(active_weather_callback) {
    active_weather_callback = state.get();
  }
  void restore() {
    active_weather_callback = previous;
    if (state && !state->closed)
      state->make_current(false);
    restored=true;
  }
  ~CallbackScope() noexcept { if(!restored) try{restore();}catch(...){} }
};
struct ModelLoadFuture {
  std::shared_future<raisin::assimp::AsyncModelLoadResult> future;
  std::shared_ptr<raisin::assimp::AsyncModelLoadResult> result;
};
template <class T>
py::object array_copy(const std::vector<T> &data,
                      std::initializer_list<size_t> shape) {
  auto *result = new T[data.size()];
  std::copy(data.begin(), data.end(), result);
  return py::steal<py::object>(py::detail::owned_array(result, shape));
}
py::object rgba_texture(unsigned texture, bool flip) {
  if (!texture)
    throw std::runtime_error("Render a frame before reading its image.");
  using namespace gl;
  GLint previous = 0, pack = 0, alignment = 0, row = 0, skip_rows = 0,
        skip_pixels = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
  glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &pack);
  glGetIntegerv(GL_PACK_ALIGNMENT, &alignment);
  glGetIntegerv(GL_PACK_ROW_LENGTH, &row);
  glGetIntegerv(GL_PACK_SKIP_ROWS, &skip_rows);
  glGetIntegerv(GL_PACK_SKIP_PIXELS, &skip_pixels);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ROW_LENGTH, 0);
  glPixelStorei(GL_PACK_SKIP_ROWS, 0);
  glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
  glBindTexture(GL_TEXTURE_2D, texture);
  GLint w = 0, h = 0;
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
  std::vector<unsigned char> data(size_t(std::max(w, 0)) *
                                  size_t(std::max(h, 0)) * 4);
  if (!data.empty())
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
  glBindTexture(GL_TEXTURE_2D, previous);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pack);
  glPixelStorei(GL_PACK_ALIGNMENT, alignment);
  glPixelStorei(GL_PACK_ROW_LENGTH, row);
  glPixelStorei(GL_PACK_SKIP_ROWS, skip_rows);
  glPixelStorei(GL_PACK_SKIP_PIXELS, skip_pixels);
  if (w <= 0 || h <= 0)
    throw std::runtime_error("The image texture has no storage.");
  if (flip)
    for (int y = 0; y < h / 2; ++y)
      std::swap_ranges(data.begin() + size_t(y) * w * 4,
                       data.begin() + size_t(y + 1) * w * 4,
                       data.begin() + size_t(h - 1 - y) * w * 4);
  return array_copy(data, {size_t(h), size_t(w), 4});
}
void valid_resolution(int w, int h) {
  gl::GLint max = 0;
  gl::glGetIntegerv(gl::GL_MAX_TEXTURE_SIZE, &max);
  if (w <= 0 || h <= 0 || w > max || h > max)
    throw std::invalid_argument(
        "Resolution must be positive and no larger than GL_MAX_TEXTURE_SIZE.");
}
} // namespace
void bind_manual(py::module_ &m) {
  auto c = bound_class<Viewer>(m, "RayraiWindow");
  c.def(
      "__init__",
      [](Viewer *self, std::shared_ptr<raisim::World> world, int width,
         int height, bool visible, const std::string &title,
         Window::ThreadingMode mode, bool shaderCache,
         const std::string &cacheDirectory, unsigned compileThreads,
         bool logShaderCache) {
        auto state = State::create(world, width, height, visible, title, mode,
                                   shaderCache, cacheDirectory, compileThreads,
                                   logShaderCache);
        new (self) Viewer(borrow(state, state->renderer.get()));
      },
      py::arg("world"), py::arg("width") = 800, py::arg("height") = 600,
      py::arg("visible") = true, py::arg("title") = "RaiSim Rayrai",
      py::arg("threadingMode") = Window::ThreadingMode::MultiThread,
      py::arg("shaderBinaryCacheEnabled") = true,
      py::arg("shaderBinaryCacheDirectory") = "",
      py::arg("shaderCompileThreadCount") = 1u,
      py::arg("logShaderBinaryCacheOperations") = false,
      "Create an in-process renderer. visible=False creates a hidden context "
      "for image capture. Use and close on this thread.");
  c.def(
      "render", [](Viewer &s, bool events) { return s.state->render(events); },
      py::arg("processEvents") = true);
  c.def("pollEvents", [](Viewer &s) {
    Guard g(s.state);
    s.state->poll_events();
    return !s.state->quit;
  });
  c.def("shouldClose", [](Viewer &s) {
    Guard g(s.state);
    return s.state->quit;
  });
  c.def("requestClose", [](Viewer &s) {
    Guard g(s.state);
    s.state->quit = true;
  });
  c.def("close", [](Viewer &s) { s.state->close(); });
  c.def("swapWorld",[](Viewer& self,std::shared_ptr<raisim::World> world) {
    Guard guard(self.state);
    WorldPairLock worlds(self.state,world);
    self.get().swapWorld(world);
    self.state->invalidate_borrowed();
    self.state->target_visual.reset();
    self.state->world=std::move(world);
  },py::arg("world"));
  c.def("setTargetVisual", [](Viewer& self, std::optional<Handle<raisin::Visuals>> visual) {
    Guard guard(self.state);
    if (visual) same_state(self.state,visual->state);
    self.get().setTargetVisual(visual ? &visual->get() : nullptr);
    self.state->target_visual=visual ? visual->record : nullptr;
  },py::arg("visual")=py::none());
  c.def(
      "invalidateExternalFrameCache",
      [](Viewer &s, bool invalidateStaticSceneLayer) {
        Guard g(s.state);
#if RAISIMPY_HAS_LAYER_CACHE_INVALIDATION
        s.get().invalidateExternalFrameCache(invalidateStaticSceneLayer);
#else
        s.get().invalidateExternalFrameCache();
#endif
      },
      py::arg("invalidateStaticSceneLayer") = true,
      "Invalidate cached frames. SDKs without layer-selective invalidation "
      "clear both layers.");
  c.def("makeCurrent", [](Viewer &s) { Guard g(s.state); });
  c.def("makeOffscreenContextCurrent", [](Viewer &s) { Guard g(s.state); });
  c.def("detachOffscreenGlContext", [](Viewer &s) {
    Guard g(s.state);
    if (SDL_GL_MakeCurrent(s.state->window, nullptr))
      throw std::runtime_error(SDL_GetError());
  });
  c.def_static(
      "createOffscreenGlContext",
      [](std::shared_ptr<raisim::World> w, int width, int height,
         const std::string &title) {
        auto state =
            State::create(w, width, height, false, title,
                          Window::ThreadingMode::MultiThread, true, "");
        return borrow(state, state->renderer.get());
      },
      py::arg("world"), py::arg("width") = 800, py::arg("height") = 600,
      py::arg("title") = "rayrai_offscreen");
  c.def_prop_ro("closed", [](Viewer &s) { return s.state->closed; });
  c.def_prop_ro("world", [](Viewer &s) {
    Guard g(s.state);
    return s.state->world;
  });
  c.def_prop_ro("visible", [](Viewer &s) {
    Guard g(s.state);
    return s.state->visible;
  });
  c.def_prop_ro("size", [](Viewer &s) {
    Guard g(s.state);
    return py::make_tuple(s.state->width, s.state->height);
  });
  c.def(
      "resize",
      [](Viewer &s, int w, int h) {
        Guard g(s.state);
        valid_resolution(w, h);
        SDL_SetWindowSize(s.state->window, w, h);
        s.state->width = w;
        s.state->height = h;
        s.get().getCamera().resizeRenderTargets(w, h);
      },
      py::arg("width"), py::arg("height"));
  c.def(
      "setVisible",
      [](Viewer &s, bool visible) {
        Guard g(s.state);
        if(visible) {
          auto threading=py::module_::import_("threading");
          if(!threading.attr("current_thread")().is(threading.attr("main_thread")()))
            throw std::runtime_error("Show visible Rayrai viewers on Python's main thread.");
        }
        s.state->visible = visible;
        if (visible)
          SDL_ShowWindow(s.state->window);
        else
          SDL_HideWindow(s.state->window);
      },
      py::arg("visible"));
  c.def(
      "setTitle",
      [](Viewer &s, const std::string &title) {
        Guard g(s.state);
        SDL_SetWindowTitle(s.state->window, title.c_str());
      },
      py::arg("title"));
  c.def(
      "readRgba",
      [](Viewer &s, bool flip) {
        Guard g(s.state);
        return rgba_texture(s.get().getImageTexture(), flip);
      },
      py::arg("flipVertical") = true,
      "Return an owned uint8 image with shape (height, width, 4), after "
      "post-processing.");
  c.def(
      "saveScreenshot",
      [](Viewer &s, const std::filesystem::path &path) {
        Guard g(s.state);
        auto a = rgba_texture(s.get().getImageTexture(), true);
        auto pixels =
            py::cast<py::ndarray<py::numpy, const uint8_t, py::c_contig>>(a);
        png_image image{};
        image.version = PNG_IMAGE_VERSION;
        image.width = png_uint_32(pixels.shape(1));
        image.height = png_uint_32(pixels.shape(0));
        image.format = PNG_FORMAT_RGBA;
        if (!png_image_write_to_file(&image, path.string().c_str(), 0,
                                     pixels.data(), 0, nullptr)) {
          std::string message = image.message;
          png_image_free(&image);
          throw std::runtime_error("Cannot save PNG: " + message);
        }
        png_image_free(&image);
      },
      py::arg("path"));
  c.def(
      "__enter__",
      [](Viewer &s) -> Viewer & {
        Guard g(s.state);
        return s;
      },
      py::rv_policy::reference_internal);
  c.def(
      "__exit__",
      [](Viewer &s, py::handle, py::handle, py::handle) { s.state->close(); },
      py::arg("exc_type").none(), py::arg("exc_value").none(),
      py::arg("traceback").none());
  // Python-friendly controls also let the procedural-world examples use the
  // same controller through the in-process and TCP renderers.
  c.def(
      "focusOn",
      [](Viewer &s, raisim::Object *target) {
        Guard g(s.state);
        WorldLock lock(s.state);
        s.get().getCamera().setTargetObject(target);
      },
      py::arg("object").none());
  c.def("integrateWorldThreadSafe", [](Viewer &s) {
    Guard g(s.state);
    WorldLock lock(s.state);
    s.state->world->integrate();
  });
  c.def("lockVisualizationServerMutex", [](Viewer &s) {
    Guard g(s.state);
    s.state->lock_user_world();
  });
  c.def("unlockVisualizationServerMutex", [](Viewer &s) {
    s.state->unlock_user_world();
  });
  c.def(
      "addSlider",
      [](Viewer &s, const std::string &name, float low, float high,
         float value) {
        Guard g(s.state);
        if (!std::isfinite(low) || !std::isfinite(high) ||
            !std::isfinite(value) || low >= high || value < low || value > high)
          throw std::invalid_argument("Invalid slider range/value.");
        for (auto &x : s.state->sliders)
          if (x.name == name)
            throw std::invalid_argument("Duplicate slider name.");
        s.state->sliders.push_back({name, low, high, value});
        return s.state->sliders.size() - 1;
      },
      py::arg("name"), py::arg("min"), py::arg("max"), py::arg("value"));
  c.def(
      "getSlider",
      [](Viewer &s, size_t index) {
        Guard g(s.state);
        if (index >= s.state->sliders.size())
          throw std::out_of_range("Unknown slider index.");
        return s.state->sliders[index].value;
      },
      py::arg("index"));
  c.def(
      "getSliderValue",
      [](Viewer &s, const std::string &name) {
        Guard g(s.state);
        for (auto &x : s.state->sliders)
          if (x.name == name)
            return x.value;
        throw std::invalid_argument("Unknown slider: " + name);
      },
      py::arg("name"));
  c.def(
      "setSliderValue",
      [](Viewer &s, const std::string &name, float v) {
        Guard g(s.state);
        for (auto &x : s.state->sliders)
          if (x.name == name) {
            if (!std::isfinite(v) || v < x.min || v > x.max)
              throw std::invalid_argument("Slider value outside its range.");
            x.value = v;
            return;
          }
        throw std::invalid_argument("Unknown slider: " + name);
      },
      py::arg("name"), py::arg("value"));
  c.def_static(
      "updateSingleChannelTexture",
      [](unsigned existing,
         py::ndarray<py::numpy, const uint8_t, py::c_contig> data) {
        Guard g(current_state(true));
        if (data.ndim() != 2)
          throw std::invalid_argument(
              "data must be a contiguous uint8 (height,width) array.");
        valid_resolution(int(data.shape(1)), int(data.shape(0)));
        return Window::updateSingleChannelTexture(
            existing, int(data.shape(1)), int(data.shape(0)), data.data());
      },
      py::arg("existingTextureId"), py::arg("data").noconvert());
  py::class_<Handle<ImFont>>(m, "Font");
  c.def("getExternalFont", [](Viewer &s) -> py::object {
    Guard g(s.state);
    auto *p = s.get().getExternalFont();
    return p ? py::cast(borrow(s.state, p)) : py::none();
  });
  c.def(
      "setExternalFont",
      [](Viewer &s, std::optional<Handle<ImFont>> font) {
        Guard g(s.state);
        if (font)
          same_state(s.state, font->state);
        s.get().setExternalFont(font ? &font->get() : nullptr);
      },
      py::arg("font") = py::none());
  c.def(
      "loadFont",
      [](Viewer &s, const std::string &path, float size) {
        Guard g(s.state);
        if (!std::filesystem::is_regular_file(path) || !std::isfinite(size) ||
            size <= 0)
          throw std::invalid_argument(
              "Expected a readable font path and positive size.");
        auto *font =
            ImGui::GetIO().Fonts->AddFontFromFileTTF(path.c_str(), size);
        if (!font)
          throw std::runtime_error("Could not load the font.");
        ImGui_ImplOpenGL3_DestroyFontsTexture();
        if(!ImGui_ImplOpenGL3_CreateFontsTexture())
          throw std::runtime_error("Could not rebuild the font atlas.");
        return borrow(s.state, font);
      },
      py::arg("path"), py::arg("size") = 16.f);
  c.def(
      "debugObjectFor",
      [](Viewer &s, raisim::Object *object) -> py::object {
        Guard g(s.state);
        WorldLock lock(s.state);
        auto *p = s.get().debugObjectFor(object);
        if (!p)
          return py::none();
        Handle<raisin::RaisimObject> h{s.state, std::make_shared<Record>()};
        h.record->state = s.state;
        h.record->world_dependent = true;
        h.record->pointer = const_cast<raisin::RaisimObject *>(p);
        s.state->records.push_back(h.record);
        auto id = object->getId();
        std::weak_ptr<State> weak = s.state;
        h.record->resolve = [weak, id]() -> void * {
          auto st = weak.lock();
          if (!st || st->closed)
            return nullptr;
          auto *o = st->world->getObjectById(id);
          if (!o)
            return nullptr;
          return const_cast<raisin::RaisimObject *>(
              st->renderer->debugObjectFor(o));
        };
        return py::cast(h);
      },
      py::arg("object"));

  auto cam = bound_class<Handle<raisin::Camera>>(m, "Camera");
  cam.def(
      "readRgba",
      [](Handle<raisin::Camera> &s, bool flip) {
        Guard g(s.state);
        s.get().resolveSceneFboIfNeeded();
        return rgba_texture(s.get().getFinalTexture(), flip);
      },
      py::arg("flipVertical") = true);
  cam.def(
      "readSceneColorBgra",
      [](Handle<raisin::Camera> &s, bool flip) {
        Guard g(s.state);
        auto &c = s.get();
        int w = c.rtWidth(), h = c.rtHeight();
        valid_resolution(w, h);
        std::vector<unsigned char> data(size_t(w) * h * 4);
        c.readSceneColorBgra(reinterpret_cast<char *>(data.data()), data.size(),
                             flip);
        return array_copy(data, {size_t(h), size_t(w), 4});
      },
      py::arg("flipVertical") = true);
  cam.def(
      "readLinearDepth",
      [](Handle<raisin::Camera> &s, bool flip) {
        Guard g(s.state);
        auto &c = s.get();
        int w = c.rtWidth(), h = c.rtHeight();
        valid_resolution(w, h);
        std::vector<float> data(size_t(w) * h);
        c.readLinearDepth(data.data(), data.size(), flip);
        return array_copy(data, {size_t(h), size_t(w)});
      },
      py::arg("flipVertical") = true);
  cam.def(
      "readSceneColorRgbaAsync",
      [](Handle<raisin::Camera> &s, bool flip) -> py::object {
        Guard g(s.state);
        std::vector<unsigned char> data;
        if (!s.get().readSceneColorRgbaAsync(data, flip))
          return py::none();
        return array_copy(
            data, {size_t(s.get().rtHeight()), size_t(s.get().rtWidth()), 4});
      },
      py::arg("flipVertical") = true,
      "Return uint8 RGBA or None while the readback ring is pending.");
  cam.def(
      "readLinearDepthAsync",
      [](Handle<raisin::Camera> &s, bool flip) -> py::object {
        Guard g(s.state);
        std::vector<float> data;
        if (!s.get().readLinearDepthAsync(data, flip))
          return py::none();
        return array_copy(
            data, {size_t(s.get().rtHeight()), size_t(s.get().rtWidth())});
      },
      py::arg("flipVertical") = true,
      "Return float32 depth or None while the readback ring is pending.");
  cam.def(
      "getRawImage",
      [](Handle<raisin::Camera> &s, raisim::RGBCamera &sensor,
         raisin::Camera::SensorStorageMode mode, bool flip) -> py::object {
        Guard g(s.state);
        if (mode == raisin::Camera::SensorStorageMode::SENSOR_SETTER) {
          s.get().getRawImage(sensor, mode, nullptr, 0, flip);
          return py::none();
        }
        int w = s.get().rtWidth(), h = s.get().rtHeight();
        valid_resolution(w, h);
        std::vector<unsigned char> data(size_t(w) * h * 4);
        s.get().getRawImage(sensor, mode, reinterpret_cast<char *>(data.data()),
                            data.size(), flip);
        return array_copy(data, {size_t(h), size_t(w), 4});
      },
      py::arg("sensor"),
      py::arg("mode") = raisin::Camera::SensorStorageMode::SENSOR_SETTER,
      py::arg("flipVertical") = true);
  cam.def(
      "getRawImage",
      [](Handle<raisin::Camera> &s, raisim::DepthCamera &sensor,
         raisin::Camera::SensorStorageMode mode, bool flip) -> py::object {
        Guard g(s.state);
        if (mode == raisin::Camera::SensorStorageMode::SENSOR_SETTER) {
          s.get().getRawImage(sensor, mode, nullptr, 0, flip);
          return py::none();
        }
        int w = s.get().rtWidth(), h = s.get().rtHeight();
        valid_resolution(w, h);
        std::vector<float> data(size_t(w) * h);
        s.get().getRawImage(sensor, mode, data.data(), data.size(), flip);
        return array_copy(data, {size_t(h), size_t(w)});
      },
      py::arg("sensor"),
      py::arg("mode") = raisin::Camera::SensorStorageMode::SENSOR_SETTER,
      py::arg("flipVertical") = true);
  cam.def(
      "readSegmentation",
      [](Handle<raisin::Camera> &s, bool flip) {
        Guard g(s.state);
        auto &c = s.get();
        int w = c.rtWidth(), h = c.rtHeight();
        valid_resolution(w, h);
        const size_t pixels = size_t(w) * h;
        std::vector<int32_t> objects(pixels), bodies(pixels), labels(pixels);
        c.readSegmentation(objects.data(), bodies.data(), labels.data(), pixels, flip);
        return py::make_tuple(array_copy(objects, {size_t(h), size_t(w)}),
                              array_copy(bodies, {size_t(h), size_t(w)}),
                              array_copy(labels, {size_t(h), size_t(w)}));
      },
      py::arg("flipVertical") = true,
      "Return int32 (objects, bodies, labels) images of the last renderSegmentation().");
  cam.def(
      "getRawImage",
      [](Handle<raisin::Camera> &s, raisim::SegmentationCamera &sensor,
         raisin::Camera::SensorStorageMode mode, bool flip) -> py::object {
        Guard g(s.state);
        if (mode == raisin::Camera::SensorStorageMode::SENSOR_SETTER) {
          s.get().getRawImage(sensor, mode, nullptr, nullptr, nullptr, 0, flip);
          return py::none();
        }
        int w = s.get().rtWidth(), h = s.get().rtHeight();
        valid_resolution(w, h);
        const size_t pixels = size_t(w) * h;
        std::vector<int32_t> objects(pixels), bodies(pixels), labels(pixels);
        s.get().getRawImage(sensor, mode, objects.data(), bodies.data(), labels.data(), pixels,
                            flip);
        return py::make_tuple(array_copy(objects, {size_t(h), size_t(w)}),
                              array_copy(bodies, {size_t(h), size_t(w)}),
                              array_copy(labels, {size_t(h), size_t(w)}));
      },
      py::arg("sensor"),
      py::arg("mode") = raisin::Camera::SensorStorageMode::SENSOR_SETTER,
      py::arg("flipVertical") = true);
  auto inst =
      bound_class<Handle<raisin::InstancedVisuals>>(m, "InstancedVisuals");
  inst.def(
      "addInstances",
      [](Handle<raisin::InstancedVisuals> &s,
         const std::vector<raisin::InstancedVisuals::InstanceSpec> &instances) {
        Guard g(s.state);
        s.get().addInstances(instances.data(), instances.size());
      },
      py::arg("instances"));
  py::class_<TextureHandle>(m, "TextureHandle")
      .def_prop_ro("id", [](TextureHandle &self) {
        Guard g(self.state);
        return self.get();
      });
  py::class_<ModelHandle>(m, "ModelHandle");
  c.def(
      "modelHandle",
      [](Viewer &self, const std::string &path) -> py::object {
        Guard g(self.state);
        auto value = self.get().modelHandle(path);
        if (!value)
          return py::none();
        auto mutableValue = std::const_pointer_cast<void>(value);
        return py::cast(ModelHandle{
            self.state, self.state->record(mutableValue.get(), mutableValue)});
      },
      py::arg("path"));
  auto assimp = py::borrow<py::module_>(m.attr("assimp"));
  py::class_<ModelLoadFuture>(assimp, "ModelLoadFuture")
      .def("done",
           [](ModelLoadFuture &f) {
             return bool(f.result) ||
                    f.future.wait_for(std::chrono::seconds(0)) ==
                        std::future_status::ready;
           })
      .def("result", [](ModelLoadFuture &f) {
        if (!f.result) {
          {
            py::gil_scoped_release release;
            f.future.wait();
          }
          if (!f.result)
            f.result = std::make_shared<raisin::assimp::AsyncModelLoadResult>(
                f.future.get());
        }
        return f.result;
      });
  assimp.def(
      "loadModelSceneAsync",
      [](const std::string &path) {
        auto future = std::make_shared<ModelLoadFuture>();
        future->future = raisin::assimp::loadModelSceneAsync(path).share();
        return future;
      },
      py::arg("path"));
  using LoadingScope = Window::SynchronousMeshLoadingScope;
  auto loading = bound_class<Handle<LoadingScope>>(
      m.attr("RayraiWindow"), "SynchronousMeshLoadingScope");
  loading.def(
      "__enter__",
      [](Handle<LoadingScope> &s) -> Handle<LoadingScope> & {
        Guard guard(s.state);
        s.get();
        return s;
      },
      py::rv_policy::reference_internal);
  loading.def(
      "__exit__",
      [](Handle<LoadingScope> &s, py::handle, py::handle, py::handle) {
        Guard guard(s.state);
        s.record->pointer = nullptr;
        s.record->owned.reset();
      },
      py::arg("exc_type").none(), py::arg("exc_value").none(),
      py::arg("traceback").none());
  c.def("synchronousMeshLoading", [](Viewer &s) {
    Guard guard(s.state);
    return own(s.state, std::make_shared<LoadingScope>(s.get()));
  });
  c.def(
      "setWeatherThunderCallback",
      [](Viewer &self, py::object callback) {
        Guard guard(self.state);
        auto object=py::find(&self);
        auto attrs=object.attr("__dict__");
        if (callback.is_none()) {
          self.get().setWeatherThunderCallback({});
          if(PyDict_GetItemString(attrs.ptr(),"_rayrai_thunder_callback"))
            PyDict_DelItemString(attrs.ptr(),"_rayrai_thunder_callback");
          self.state->callback_viewer.reset();
          return;
        }
        if (!PyCallable_Check(callback.ptr()))
          throw py::type_error("callback must be callable or None");
        attrs["_rayrai_thunder_callback"]=callback;
        self.state->callback_viewer=py::weakref(object);
        self.get().setWeatherThunderCallback(
            [owner = std::weak_ptr<State>(self.state)](
                const raisin::WeatherDiagnostics &diagnostics) {
              py::gil_scoped_acquire gil;
              auto state = owner.lock();
              if (!state || state->closed)
                return;
              if(!state->callback_viewer) return;
              auto object=state->callback_viewer();
              if(object.is_none())return;
              auto attrs=object.attr("__dict__");
              PyObject* value=PyDict_GetItemString(attrs.ptr(),"_rayrai_thunder_callback");
              if(!value) return;
              auto callback=py::borrow<py::object>(value);
              CallbackScope scope(state);
              callback(py::cast(diagnostics, py::rv_policy::copy));
              scope.restore();
            });
      },
      py::arg("callback").none());
  c.def("importVisualScene",[](Viewer& self,const std::string& name,const std::string& path,
          const Window::ImportVisualSceneOptions& options,Window::SceneImportReport* report) {
    Guard guard(self.state);
    auto copy=options;copy.report=report;
    return own(self.state,self.get().importVisualScene(name,path,copy));
  },py::arg("name"),py::arg("sceneFile"),py::arg("opts"),py::arg("report").none()=py::none());
  m.def("syncRaisimCameraPose",[](raisim::RGBCamera& sensor,Handle<raisin::Camera>& camera) {
    Guard guard(camera.state);WorldLock lock(camera.state);
    raisin::syncRaisimCameraPose(sensor,camera.get());
  },py::arg("sensorCam"),py::arg("renderCam"));
  m.def("syncRaisimCameraPose",[](raisim::DepthCamera& sensor,Handle<raisin::Camera>& camera) {
    Guard guard(camera.state);WorldLock lock(camera.state);
    raisin::syncRaisimCameraPose(sensor,camera.get());
  },py::arg("sensorCam"),py::arg("renderCam"));
  m.def("closeAll", &close_all);
  m.attr("__version__") = py::module_::import_("raisimpy").attr("__version__");
}
} // namespace rayrai_py
