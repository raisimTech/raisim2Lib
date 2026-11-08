#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "EditorWorld.hpp"
#include "rayrai/WebRenderer.hpp"

namespace {
raisim::web::EditorWorld editor;
std::unique_ptr<raisin::WebRenderer> renderer;
std::vector<std::array<float, 3>> colors;
int lastSelection = -2;
double accumulated = 0.;
std::string error, result, renderText;

template <class Action>
int guarded(Action action) {
  error.clear();
  try {
    action();
    return 1;
  } catch (const std::exception& exception) {
    error = exception.what();
    return 0;
  } catch (...) {
    error = "The editor could not complete this action";
    return 0;
  }
}
void render(int selected) {
  if (!renderer) throw std::logic_error("Load an editor scene first");
  if (selected < -1 || selected >= int(editor.colors().size()))
    throw std::invalid_argument("Invalid object selection");
  if (selected != lastSelection) {
    colors = editor.colors();
    if (selected >= 0) colors[size_t(selected)] = {1.f, .72f, .26f};
    lastSelection = selected;
  }
  renderer->render(editor.world(), colors);
}
}
extern "C" {
int editor_load(const char* text, size_t length) {
  return guarded([&] {
    if (!text || !length || length > raisim::web::EditorWorld::kMaxTextBytes)
      throw std::invalid_argument("Invalid scene text");
    editor.load(std::string_view(text, length));
    if (!renderer) renderer = std::make_unique<raisin::WebRenderer>();
    renderer->clearScene();
    lastSelection = -2;
    accumulated = 0.;
    render(-1);
  });
}
double editor_frame(double seconds, int advance, int selected) {
  if (!guarded([&] {
        if (!std::isfinite(seconds))
          throw std::invalid_argument("Invalid frame interval");
        if (advance) {
          accumulated =
              std::min(.024, accumulated + std::clamp(seconds, 0., .05));
          for (int i = 0; i < 12 && accumulated >= .002; ++i) {
            editor.step(1);
            accumulated -= .002;
          }
        } else {
          accumulated = 0.;
        }
        render(selected);
      }))
    return -1.;
  return editor.world().getWorldTime();
}
double editor_step(int count, int selected) {
  if (!guarded([&] {
        editor.step(count);
        accumulated = 0.;
        render(selected);
      }))
    return -1.;
  return editor.world().getWorldTime();
}
int editor_orbit(double horizontal, double vertical, double zoom) {
  return guarded([&] {
    if (!renderer || !std::isfinite(horizontal) || !std::isfinite(vertical) ||
        !std::isfinite(zoom))
      throw std::invalid_argument("Invalid camera action");
    renderer->orbit(horizontal, vertical, zoom);
  });
}
int editor_move(size_t index, double x, double y, double z) {
  return guarded([&] { editor.move(index, x, y, z); });
}
const char* editor_pick(double x, double y, double height, int hitObjects) {
  if (!guarded([&] {
        if (!renderer || !std::isfinite(height) || std::abs(height) > 20. ||
            (hitObjects != 0 && hitObjects != 1))
          throw std::invalid_argument("Invalid pick plane");
        const auto ray = renderer->screenRay(x, y);
        const auto& o = ray.origin;
        const auto& d = ray.direction;
        int objectIndex = -1;
        if (hitObjects) {
          const auto& hits = editor.world().rayTest({o[0], o[1], o[2]},
                                                    {d[0], d[1], d[2]}, 100.);
          if (hits.size())
            objectIndex = int(hits[0].getObject()->getIndexInWorld());
        }
        std::ostringstream output;
        output.imbue(std::locale::classic());
        output << std::setprecision(17) << "{\"index\":";
        output << objectIndex;
        output << ",\"point\":";
        const double t = std::abs(d[2]) > 1e-8 ? (height - o[2]) / d[2] : -1.;
        if (t >= 0. && t <= 100.)
          output << '[' << o[0] + t * d[0] << ',' << o[1] + t * d[1] << ','
                 << height << ']';
        else
          output << "null";
        output << '}';
        result = output.str();
      }))
    return nullptr;
  return result.c_str();
}
int editor_write_file(const char* path, size_t pathLength, const char* data, size_t length) {
  return guarded([&] {
    if (!path || (!data && length)) throw std::invalid_argument("Invalid robot file");
    editor.writeFile(std::string_view(path, pathLength), std::string_view(data, length));
  });
}
const char* editor_state() {
  if (!guarded([&] { result = editor.poses(); })) return nullptr;
  return result.c_str();
}
// Render settings: the JSON schema (schema != 0) or the current values; empty before a renderer exists.
const char* editor_render_settings(int schema) {
  try {
    renderText = !renderer ? std::string() : schema ? renderer->renderSettingSchema() : renderer->renderSettings();
  } catch (const std::exception&) {
    renderText.clear();
  }
  return renderText.c_str();
}
// Applies a preset (0-3, or -1 for none) and then a JSON object of settings; returns "" or the error.
const char* editor_render_apply(const char* json, size_t length, int preset) {
  try {
    if (!renderer) throw std::logic_error("The renderer is not ready yet");
    if (preset >= 0) renderer->useRenderPreset(preset);
    if (length) renderer->applyRenderSettings(std::string_view(json, length));
    renderText.clear();
  } catch (const std::exception& exception) {
    renderText = exception.what();
  }
  return renderText.c_str();
}
const char* editor_error() { return error.c_str(); }
void editor_dispose() {
  renderer.reset();
  editor = raisim::web::EditorWorld();
  colors.clear();
  lastSelection = -2;
  accumulated = 0.;
  result.clear();
  error.clear();
}
}
