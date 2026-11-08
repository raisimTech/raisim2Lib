// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#include "nanobind_helpers.hpp"
#if RAISIMPY_HAS_SEGMENTATION
#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include <nanobind/stl/array.h>

#include "raisim/World.hpp"
#include "raisim/sensors/SegmentationSensor.hpp"

namespace py = nanobind;

namespace {

using raisim::SegmentationCamera;
using IdImage = py::ndarray<py::numpy, int32_t, py::c_contig>;
using IdInput = py::ndarray<const int32_t, py::c_contig, py::device::cpu>;
using ColorImage = py::ndarray<py::numpy, uint8_t, py::c_contig>;

template <class T, class Array>
Array ownedArray(const std::vector<T>& values, const std::vector<size_t>& shape) {
  auto* data = new T[values.size()];
  std::copy(values.begin(), values.end(), data);
  py::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return Array(data, shape.size(), shape.data(), owner);
}

std::optional<IdImage> idImage(const SegmentationCamera& camera, const std::vector<int32_t>& ids) {
  if (ids.empty()) return std::nullopt;
  const auto& properties = camera.getProperties();
  return ownedArray<int32_t, IdImage>(ids, {size_t(properties.height), size_t(properties.width)});
}

std::vector<int32_t> idVector(const IdInput& image) {
  return std::vector<int32_t>(image.data(), image.data() + image.size());
}

}  // namespace

void init_segmentation(py::module_& m) {
  using Properties = SegmentationCamera::SegmentationCameraProperties;
  using Type = SegmentationCamera::SegmentationType;
  py::enum_<Type>(m, "SegmentationType", py::is_arithmetic())
      .value("SEMANTIC", Type::SEMANTIC)
      .value("INSTANCE", Type::INSTANCE)
      .value("PANOPTIC", Type::PANOPTIC);

  py::class_<Properties>(m, "SegmentationCameraProperties")
      .def(py::init<>())
      .def_rw("name", &Properties::name)
      .def_rw("full_name", &Properties::full_name)
      .def_rw("width", &Properties::width)
      .def_rw("height", &Properties::height)
      .def_rw("xOffset", &Properties::xOffset)
      .def_rw("yOffset", &Properties::yOffset)
      .def_rw("clipNear", &Properties::clipNear)
      .def_rw("clipFar", &Properties::clipFar)
      .def_rw("hFOV", &Properties::hFOV)
      .def_rw("lens", &Properties::lens)
      .def_rw("segmentationType", &Properties::segmentationType)
      .def_static("stringToSegmentationType", &Properties::stringToSegmentationType);

  py::class_<SegmentationCamera, raisim::Sensor> camera(m, "SegmentationCamera");
  camera.def_static("getStaticType", &SegmentationCamera::getType)
      .def("getProperties", &SegmentationCamera::getProperties, py::rv_policy::copy)
      .def("setProperties", &SegmentationCamera::setProperties, py::arg("properties"))
      .def("update", &SegmentationCamera::update, py::arg("world"))
      .def("getObjectImage", [](const SegmentationCamera& self) {
        return idImage(self, self.getObjectImage());
      }, "World index of the object in each pixel as a (height, width) int32 array; None for SEMANTIC.")
      .def("getBodyImage", [](const SegmentationCamera& self) {
        return idImage(self, self.getBodyImage());
      }, "Body index within the object in each pixel as a (height, width) int32 array; None for SEMANTIC.")
      .def("getLabelImage", [](const SegmentationCamera& self) {
        return idImage(self, self.getLabelImage());
      }, "Segmentation label in each pixel as a (height, width) int32 array; None for INSTANCE.")
      .def("setObjectImage", [](SegmentationCamera& self, const IdInput& image) {
        self.setObjectImage(idVector(image));
      }, py::arg("image"))
      .def("setBodyImage", [](SegmentationCamera& self, const IdInput& image) {
        self.setBodyImage(idVector(image));
      }, py::arg("image"))
      .def("setLabelImage", [](SegmentationCamera& self, const IdInput& image) {
        self.setLabelImage(idVector(image));
      }, py::arg("image"))
      .def_static("hasInstanceImages", &SegmentationCamera::hasInstanceImages, py::arg("type"))
      .def_static("hasLabelImage", &SegmentationCamera::hasLabelImage, py::arg("type"))
      .def_static("imageCount", &SegmentationCamera::imageCount, py::arg("type"))
      .def_static("colorOf", &SegmentationCamera::colorOf, py::arg("id"))
      .def_static("colorize", [](const IdInput& ids, const std::array<unsigned char, 4>& background) {
        std::vector<unsigned char> rgba;
        SegmentationCamera::colorize(idVector(ids), rgba, background);
        std::vector<size_t> shape(ids.shape_ptr(), ids.shape_ptr() + ids.ndim());
        shape.push_back(4);
        return ownedArray<unsigned char, ColorImage>(rgba, shape);
      }, py::arg("ids"), py::arg("background") = std::array<unsigned char, 4>{0, 0, 0, 0},
      "RGBA uint8 colors of an id image (shape + (4,)); negative ids get background.");
  camera.attr("BACKGROUND") = SegmentationCamera::kBackground;
  camera.attr("NOT_WORLD_OBJECT") = SegmentationCamera::kNotWorldObject;
}

void init_segmentation_world(py::class_<raisim::World>& world) {
  world.def("addSegmentationClass", &raisim::World::addSegmentationClass, py::arg("name"))
      .def("getSegmentationClassLabel", &raisim::World::getSegmentationClassLabel, py::arg("name"))
      .def("getSegmentationClassName", &raisim::World::getSegmentationClassName, py::arg("label"))
      .def("getSegmentationClassNames", &raisim::World::getSegmentationClassNames)
      .def("setSegmentationClassName", &raisim::World::setSegmentationClassName,
           py::arg("label"), py::arg("name"));
}

void init_segmentation_articulated(py::class_<raisim::ArticulatedSystem, raisim::Object>& system) {
  using raisim::ArticulatedSystem;
  system.def("setBodySegmentationLabel",
             py::overload_cast<size_t, int32_t>(&ArticulatedSystem::setBodySegmentationLabel),
             py::arg("bodyIdx"), py::arg("label"))
      .def("setBodySegmentationLabel",
           py::overload_cast<const std::string&, int32_t>(&ArticulatedSystem::setBodySegmentationLabel),
           py::arg("body"), py::arg("label"))
      .def("getBodySegmentationLabel", &ArticulatedSystem::getBodySegmentationLabel, py::arg("bodyIdx"))
      .def("clearBodySegmentationLabels", &ArticulatedSystem::clearBodySegmentationLabels);
}
#endif
