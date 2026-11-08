/**
 * Python wrappers for raisim.sensors using nanobind.
 */

#include "nanobind_helpers.hpp"

#include <algorithm>
#include <array>
#include <sstream>
#include <vector>

#include <nanobind/stl/array.h>

#include "raisim/World.hpp"
#include "raisim/sensors/SensorSet.hpp"
#include "raisim/sensors/InertialMeasurementUnit.hpp"
#include "raisim/sensors/RGBSensor.hpp"
#include "raisim/sensors/DepthSensor.hpp"
#include "raisim/sensors/SegmentationSensor.hpp"
#include "raisim/sensors/SpinningLidar.hpp"

#include "converter.hpp"

namespace py = nanobind;

namespace {

template <typename Allocator>
NDArray vec3_list_to_ndarray(const std::vector<raisim::Vec<3>, Allocator> &data) {
    NDArray array = make_ndarray_2d(data.size(), 3);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < 3; ++j)
            *ndarray_mutable_data(array, i, j) = data[i][j];
    }
    return array;
}

std::vector<raisim::Vec<3>> ndarray_to_vec3_list(const NDArray &array) {
    if (array.ndim() != 2 || array.shape(1) != 3) {
        std::ostringstream s;
        s << "error: expecting array shape (N, 3), but got ("
          << array.shape(0) << ", " << (array.ndim() > 1 ? array.shape(1) : 0)
          << ").";
        throw std::domain_error(s.str());
    }

    const size_t count = array.shape(0);
    std::vector<raisim::Vec<3>> out(count);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < 3; ++j)
            out[i][j] = *ndarray_data(array, i, j);
    }
    return out;
}

// A segmentation image as an (H, W) int32 array, or None for an image the camera does not fill.
py::object segmentation_image(const raisim::SegmentationCamera &camera, const std::vector<int32_t> &image) {
    if (image.empty()) return py::none();
    const auto &properties = camera.getProperties();
    auto *data = new int32_t[image.size()];
    std::copy(image.begin(), image.end(), data);
    py::capsule owner(data, [](void *p) noexcept { delete[] static_cast<int32_t *>(p); });
    return py::cast(py::ndarray<py::numpy, int32_t, py::c_contig>(
        data, {size_t(properties.height), size_t(properties.width)}, owner));
}

std::vector<int32_t> segmentation_ids(const py::ndarray<int32_t, py::c_contig, py::device::cpu> &ids) {
    return std::vector<int32_t>(ids.data(), ids.data() + ids.size());
}

}  // namespace

void init_wrench_sensors(py::module_& m);
void init_segmentation(py::module_& m);

void init_sensors(py::module_ &m) {
    py::enum_<raisim::Sensor::Type>(m, "SensorType", py::is_arithmetic())
        .value("UNKNOWN", raisim::Sensor::Type::UNKNOWN)
        .value("RGB", raisim::Sensor::Type::RGB)
        .value("DEPTH", raisim::Sensor::Type::DEPTH)
        .value("IMU", raisim::Sensor::Type::IMU)
        .value("SPINNING_LIDAR", raisim::Sensor::Type::SPINNING_LIDAR)
        .value("SEGMENTATION", raisim::Sensor::Type::SEGMENTATION)
#if RAISIMPY_HAS_WRENCH_SENSORS
        .value("FORCE", raisim::Sensor::Type::FORCE)
        .value("TACTILE", raisim::Sensor::Type::TACTILE)
#endif
#if RAISIMPY_HAS_SEGMENTATION
        .value("SEGMENTATION", raisim::Sensor::Type::SEGMENTATION)
#endif
        ;

    py::enum_<raisim::Sensor::MeasurementSource>(m, "SensorMeasurementSource", py::is_arithmetic())
        .value("RAISIM", raisim::Sensor::MeasurementSource::RAISIM)
        .value("MANUAL", raisim::Sensor::MeasurementSource::MANUAL);

    py::class_<raisim::Sensor>(m, "Sensor")
        .def("getPosition", [](raisim::Sensor &self) { return convert_vec_to_np(self.getPosition()); })
        .def("getOrientation", [](raisim::Sensor &self) { return convert_mat_to_np(self.getOrientation()); })
        .def("getFramePosition", [](raisim::Sensor &self) { return convert_vec_to_np(self.getFramePosition()); })
        .def("getFrameOrientation", [](raisim::Sensor &self) { return convert_mat_to_np(self.getFrameOrientation()); })
        .def("getPosInSensorFrame", [](raisim::Sensor &self) { return convert_vec_to_np(self.getPosInSensorFrame()); })
        .def("getOriInSensorFrame", [](raisim::Sensor &self) { return convert_mat_to_np(self.getOriInSensorFrame()); })
        .def("getName", &raisim::Sensor::getName)
        .def("getFullName", &raisim::Sensor::getFullName)
        .def("getSensorSetModel", &raisim::Sensor::getSensorSetModel)
        .def("getSerialNumber", &raisim::Sensor::getSerialNumber)
        .def("setSerialNumber", &raisim::Sensor::setSerialNumber)
        .def("getType", &raisim::Sensor::getType)
        .def("getUpdateRate", &raisim::Sensor::getUpdateRate)
        .def("getUpdateTimeStamp", &raisim::Sensor::getUpdateTimeStamp)
        .def("setUpdateRate", &raisim::Sensor::setUpdateRate)
        .def("setUpdateTimeStamp", &raisim::Sensor::setUpdateTimeStamp)
        .def("updatePose", &raisim::Sensor::updatePose)
        .def("getMeasurementSource", &raisim::Sensor::getMeasurementSource)
        .def("setMeasurementSource", &raisim::Sensor::setMeasurementSource)
        .def("getFrameId", &raisim::Sensor::getFrameId)
        .def("lockMutex", &raisim::Sensor::lockMutex)
        .def("unlockMutex", &raisim::Sensor::unlockMutex)
        .def("lock", &raisim::Sensor::lock)
        .def("unlock", &raisim::Sensor::unlock);

    py::class_<raisim::SensorSet>(m, "SensorSet")
        .def("getSensors", [](raisim::SensorSet &self) { return self.getSensors(); },
             py::rv_policy::reference_internal)
        .def("getSensor", &raisim::SensorSet::getSensorRawPtr, py::rv_policy::reference_internal)
        .def("getModel", &raisim::SensorSet::getModel)
        .def("getSerialNumber", &raisim::SensorSet::getSerialNumber)
        .def("getName", &raisim::SensorSet::getName);

    py::enum_<raisim::CameraProjectionType>(m, "CameraProjectionType")
        .value("PINHOLE", raisim::CameraProjectionType::PINHOLE)
        .value("FISHEYE_EQUIDISTANT", raisim::CameraProjectionType::FISHEYE_EQUIDISTANT);
    py::class_<raisim::CameraIntrinsics>(m, "CameraIntrinsics")
        .def(py::init<>())
        .def_rw("fx", &raisim::CameraIntrinsics::fx).def_rw("fy", &raisim::CameraIntrinsics::fy)
        .def_rw("cx", &raisim::CameraIntrinsics::cx).def_rw("cy", &raisim::CameraIntrinsics::cy)
        .def("hasExplicitFocalLength", &raisim::CameraIntrinsics::hasExplicitFocalLength)
        .def("hasExplicitPrincipalPoint", &raisim::CameraIntrinsics::hasExplicitPrincipalPoint);
    py::class_<raisim::CameraFisheyeDistortion>(m, "CameraFisheyeDistortion")
        .def(py::init<>()).def_rw("k1", &raisim::CameraFisheyeDistortion::k1)
        .def_rw("k2", &raisim::CameraFisheyeDistortion::k2)
        .def_rw("k3", &raisim::CameraFisheyeDistortion::k3)
        .def_rw("k4", &raisim::CameraFisheyeDistortion::k4);
    py::class_<raisim::CameraLensModel>(m, "CameraLensModel")
        .def(py::init<>()).def_rw("projection", &raisim::CameraLensModel::projection)
        .def_rw("intrinsics", &raisim::CameraLensModel::intrinsics)
        .def_rw("fisheye", &raisim::CameraLensModel::fisheye)
        .def("isFisheye", &raisim::CameraLensModel::isFisheye)
        .def("projectionName", &raisim::CameraLensModel::projectionName)
        .def("distortionModel", &raisim::CameraLensModel::distortionModel)
        .def("setOpenCvFisheye", &raisim::CameraLensModel::setOpenCvFisheye,
             py::arg("fx"),py::arg("fy"),py::arg("cx"),py::arg("cy"),
             py::arg("k1")=0.,py::arg("k2")=0.,py::arg("k3")=0.,py::arg("k4")=0.);

    py::enum_<raisim::RGBCamera::RGBCameraProperties::NoiseType>(m, "RGBCameraNoiseType",
                                                                 py::is_arithmetic())
        .value("GAUSSIAN", raisim::RGBCamera::RGBCameraProperties::NoiseType::GAUSSIAN)
        .value("UNIFORM", raisim::RGBCamera::RGBCameraProperties::NoiseType::UNIFORM)
        .value("NO_NOISE", raisim::RGBCamera::RGBCameraProperties::NoiseType::NO_NOISE);

    py::class_<raisim::RGBCamera::RGBCameraProperties>(m, "RGBCameraProperties")
        .def(py::init<>())
        .def_rw("name", &raisim::RGBCamera::RGBCameraProperties::name)
        .def_rw("full_name", &raisim::RGBCamera::RGBCameraProperties::full_name)
        .def_rw("width", &raisim::RGBCamera::RGBCameraProperties::width)
        .def_rw("height", &raisim::RGBCamera::RGBCameraProperties::height)
        .def_rw("xOffset", &raisim::RGBCamera::RGBCameraProperties::xOffset)
        .def_rw("yOffset", &raisim::RGBCamera::RGBCameraProperties::yOffset)
        .def_rw("clipNear", &raisim::RGBCamera::RGBCameraProperties::clipNear)
        .def_rw("clipFar", &raisim::RGBCamera::RGBCameraProperties::clipFar)
        .def_rw("hFOV", &raisim::RGBCamera::RGBCameraProperties::hFOV)
        .def_rw("lens", &raisim::RGBCamera::RGBCameraProperties::lens)
        .def_rw("noiseType", &raisim::RGBCamera::RGBCameraProperties::noiseType)
        .def_rw("mean", &raisim::RGBCamera::RGBCameraProperties::mean)
        .def_rw("std", &raisim::RGBCamera::RGBCameraProperties::std)
        .def_rw("shotNoise", &raisim::RGBCamera::RGBCameraProperties::shotNoise)
        .def_rw("rowNoiseStd", &raisim::RGBCamera::RGBCameraProperties::rowNoiseStd)
        .def_rw("seed", &raisim::RGBCamera::RGBCameraProperties::seed)
        .def_rw("format", &raisim::RGBCamera::RGBCameraProperties::format)
        .def_static("stringToNoiseType",
                    &raisim::RGBCamera::RGBCameraProperties::stringToNoiseType);

    py::class_<raisim::RGBCamera, raisim::Sensor>(m, "RGBCamera")
        .def_static("getStaticType", &raisim::RGBCamera::getType)
        .def("applyNoise", py::overload_cast<>(&raisim::RGBCamera::applyNoise))
        .def("resetNoise", &raisim::RGBCamera::resetNoise)
        .def("setProperties", &raisim::RGBCamera::setProperties)
        .def("getProperties", &raisim::RGBCamera::getProperties, py::rv_policy::copy)
        .def("getImageBuffer", [](const raisim::RGBCamera &self) {
            const auto &image = self.getImageBuffer();
            return py::bytes(image.data(), image.size());
        })
        .def("setImageBuffer", &raisim::RGBCamera::setImageBuffer)
        .def("setImageBuffer", [](raisim::RGBCamera &self, const py::bytes &image) {
            self.setImageBuffer(std::vector<char>(image.c_str(), image.c_str() + image.size()));
        });

    py::enum_<raisim::DepthCamera::DepthCameraProperties::NoiseType>(m, "DepthCameraNoiseType",
                                                                     py::is_arithmetic())
        .value("GAUSSIAN", raisim::DepthCamera::DepthCameraProperties::NoiseType::GAUSSIAN)
        .value("UNIFORM", raisim::DepthCamera::DepthCameraProperties::NoiseType::UNIFORM)
        .value("NO_NOISE", raisim::DepthCamera::DepthCameraProperties::NoiseType::NO_NOISE);

    py::enum_<raisim::DepthCamera::Frame>(m, "DepthCameraFrame", py::is_arithmetic())
        .value("SENSOR_FRAME", raisim::DepthCamera::Frame::SENSOR_FRAME)
        .value("ROOT_FRAME", raisim::DepthCamera::Frame::ROOT_FRAME)
        .value("WORLD_FRAME", raisim::DepthCamera::Frame::WORLD_FRAME);

    py::class_<raisim::DepthCamera::DepthCameraProperties>(m, "DepthCameraProperties")
        .def(py::init<>())
        .def_rw("name", &raisim::DepthCamera::DepthCameraProperties::name)
        .def_rw("full_name", &raisim::DepthCamera::DepthCameraProperties::full_name)
        .def_rw("width", &raisim::DepthCamera::DepthCameraProperties::width)
        .def_rw("height", &raisim::DepthCamera::DepthCameraProperties::height)
        .def_rw("xOffset", &raisim::DepthCamera::DepthCameraProperties::xOffset)
        .def_rw("yOffset", &raisim::DepthCamera::DepthCameraProperties::yOffset)
        .def_rw("clipNear", &raisim::DepthCamera::DepthCameraProperties::clipNear)
        .def_rw("clipFar", &raisim::DepthCamera::DepthCameraProperties::clipFar)
        .def_rw("hFOV", &raisim::DepthCamera::DepthCameraProperties::hFOV)
        .def_rw("lens", &raisim::DepthCamera::DepthCameraProperties::lens)
        .def_rw("noiseType", &raisim::DepthCamera::DepthCameraProperties::noiseType)
        .def_rw("mean", &raisim::DepthCamera::DepthCameraProperties::mean)
        .def_rw("std", &raisim::DepthCamera::DepthCameraProperties::std)
        .def_rw("stdDepthSquared", &raisim::DepthCamera::DepthCameraProperties::stdDepthSquared)
        .def_rw("resolution", &raisim::DepthCamera::DepthCameraProperties::resolution)
        .def_rw("dropoutProbability", &raisim::DepthCamera::DepthCameraProperties::dropoutProbability)
        .def_rw("seed", &raisim::DepthCamera::DepthCameraProperties::seed)
        .def_rw("format", &raisim::DepthCamera::DepthCameraProperties::format)
        .def_static("stringToNoiseType",
                    &raisim::DepthCamera::DepthCameraProperties::stringToNoiseType);

    py::class_<raisim::DepthCamera, raisim::Sensor>(m, "DepthCamera")
        .def_static("getStaticType", &raisim::DepthCamera::getType)
        .def("applyNoise", py::overload_cast<>(&raisim::DepthCamera::applyNoise))
        .def("resetNoise", &raisim::DepthCamera::resetNoise)
        .def("updateRayDirections", &raisim::DepthCamera::updateRayDirections)
        .def("getDepthArray", py::overload_cast<>(&raisim::DepthCamera::getDepthArray, py::const_))
        .def("setDepthArray", &raisim::DepthCamera::setDepthArray)
        .def("get3DPoints", [](const raisim::DepthCamera &self) {
            return vec3_list_to_ndarray(self.get3DPoints());
        })
        .def("set3DPoints", [](raisim::DepthCamera &self, const NDArray &points) {
            self.set3DPoints(ndarray_to_vec3_list(points));
        })
        .def("getProperties", py::overload_cast<>(&raisim::DepthCamera::getProperties), py::rv_policy::reference_internal)
        .def("depthToPointCloud", [](const raisim::DepthCamera &self,
                                     const std::vector<float> &depthArray,
                                     bool isInSensorFrame) {
            std::vector<raisim::Vec<3>> points;
            self.depthToPointCloud(depthArray, points, isInSensorFrame);
            return vec3_list_to_ndarray(points);
        }, py::arg("depthArray"), py::arg("isInSensorFrame") = false)
        .def("getPrecomputedRayDir", [](const raisim::DepthCamera &self) {
            return vec3_list_to_ndarray(self.getPrecomputedRayDir());
        });

    using SegmentationCamera = raisim::SegmentationCamera;
    using SegmentationProperties = SegmentationCamera::SegmentationCameraProperties;
    py::enum_<SegmentationCamera::SegmentationType>(m, "SegmentationType", py::is_arithmetic())
        .value("SEMANTIC", SegmentationCamera::SegmentationType::SEMANTIC)
        .value("INSTANCE", SegmentationCamera::SegmentationType::INSTANCE)
        .value("PANOPTIC", SegmentationCamera::SegmentationType::PANOPTIC);

    py::class_<SegmentationProperties>(m, "SegmentationCameraProperties")
        .def(py::init<>())
        .def_rw("name", &SegmentationProperties::name)
        .def_rw("full_name", &SegmentationProperties::full_name)
        .def_rw("width", &SegmentationProperties::width)
        .def_rw("height", &SegmentationProperties::height)
        .def_rw("xOffset", &SegmentationProperties::xOffset)
        .def_rw("yOffset", &SegmentationProperties::yOffset)
        .def_rw("clipNear", &SegmentationProperties::clipNear)
        .def_rw("clipFar", &SegmentationProperties::clipFar)
        .def_rw("hFOV", &SegmentationProperties::hFOV)
        .def_rw("lens", &SegmentationProperties::lens)
        .def_rw("segmentationType", &SegmentationProperties::segmentationType)
        .def_static("stringToSegmentationType", &SegmentationProperties::stringToSegmentationType);

    py::class_<SegmentationCamera, raisim::Sensor>(m, "SegmentationCamera",
        "Object, body and label images: (H, W) int32 arrays, -1 where nothing was hit. "
        "The images the segmentation type does not fill are None.")
        .def_static("getStaticType", &SegmentationCamera::getType)
        .def_ro_static("BACKGROUND", &SegmentationCamera::kBackground)
        .def_ro_static("NOT_WORLD_OBJECT", &SegmentationCamera::kNotWorldObject)
        .def("getProperties", &SegmentationCamera::getProperties, py::rv_policy::copy)
        .def("setProperties", &SegmentationCamera::setProperties)
        .def("getObjectImage", [](const SegmentationCamera &self) {
            return segmentation_image(self, self.getObjectImage());
        }, "World index (World.getObject(index)) of the object seen in each pixel.")
        .def("getBodyImage", [](const SegmentationCamera &self) {
            return segmentation_image(self, self.getBodyImage());
        }, "Body index within that object.")
        .def("getLabelImage", [](const SegmentationCamera &self) {
            return segmentation_image(self, self.getLabelImage());
        }, "Segmentation label (0 for unlabeled geometry).")
        .def("setObjectImage", [](SegmentationCamera &self, const py::ndarray<int32_t, py::c_contig, py::device::cpu> &image) {
            self.setObjectImage(segmentation_ids(image));
        })
        .def("setBodyImage", [](SegmentationCamera &self, const py::ndarray<int32_t, py::c_contig, py::device::cpu> &image) {
            self.setBodyImage(segmentation_ids(image));
        })
        .def("setLabelImage", [](SegmentationCamera &self, const py::ndarray<int32_t, py::c_contig, py::device::cpu> &image) {
            self.setLabelImage(segmentation_ids(image));
        })
        .def("update", &SegmentationCamera::update, "Ray-cast the images (the RAISIM measurement source).",
             py::arg("world"))
        .def_static("colorize", [](const py::ndarray<int32_t, py::c_contig, py::device::cpu> &ids,
                                   std::array<unsigned char, 4> background) {
            std::vector<unsigned char> rgba;
            SegmentationCamera::colorize(segmentation_ids(ids), rgba, background);
            auto *data = new unsigned char[rgba.size()];
            std::copy(rgba.begin(), rgba.end(), data);
            py::capsule owner(data, [](void *p) noexcept { delete[] static_cast<unsigned char *>(p); });
            std::vector<size_t> shape(ids.shape_ptr(), ids.shape_ptr() + ids.ndim());
            shape.push_back(4);
            return py::ndarray<py::numpy, unsigned char, py::c_contig>(data, shape.size(), shape.data(), owner);
        }, "Display colors (..., 4) uint8 for an id image; negative ids get the background color.",
           py::arg("ids"), py::arg("background") = std::array<unsigned char, 4>{0, 0, 0, 0});

    py::enum_<raisim::InertialMeasurementUnit::ImuProperties::NoiseType>(m, "ImuNoiseType",
                                                                         py::is_arithmetic())
        .value("GAUSSIAN", raisim::InertialMeasurementUnit::ImuProperties::NoiseType::GAUSSIAN)
        .value("UNIFORM", raisim::InertialMeasurementUnit::ImuProperties::NoiseType::UNIFORM)
        .value("NO_NOISE", raisim::InertialMeasurementUnit::ImuProperties::NoiseType::NO_NOISE);

    py::class_<raisim::InertialMeasurementUnit::ChannelProperties>(m, "ImuChannelProperties")
        .def(py::init<>())
        .def_rw("bias", &raisim::InertialMeasurementUnit::ChannelProperties::bias)
        .def_rw("noiseDensity", &raisim::InertialMeasurementUnit::ChannelProperties::noiseDensity)
        .def_rw("biasRandomWalk", &raisim::InertialMeasurementUnit::ChannelProperties::biasRandomWalk)
        .def_rw("cutoffFrequency", &raisim::InertialMeasurementUnit::ChannelProperties::cutoffFrequency)
        .def_rw("resolution", &raisim::InertialMeasurementUnit::ChannelProperties::resolution);

    py::class_<raisim::InertialMeasurementUnit::ImuProperties>(m, "ImuProperties")
        .def(py::init<>())
        .def_rw("name", &raisim::InertialMeasurementUnit::ImuProperties::name)
        .def_rw("full_name", &raisim::InertialMeasurementUnit::ImuProperties::full_name)
        .def_rw("maxAcc", &raisim::InertialMeasurementUnit::ImuProperties::maxAcc)
        .def_rw("maxAngVel", &raisim::InertialMeasurementUnit::ImuProperties::maxAngVel)
        .def_rw("noiseType", &raisim::InertialMeasurementUnit::ImuProperties::noiseType)
        .def_rw("mean", &raisim::InertialMeasurementUnit::ImuProperties::mean)
        .def_rw("std", &raisim::InertialMeasurementUnit::ImuProperties::std)
        .def_rw("accelerometer", &raisim::InertialMeasurementUnit::ImuProperties::accelerometer)
        .def_rw("gyroscope", &raisim::InertialMeasurementUnit::ImuProperties::gyroscope)
        .def_rw("latency", &raisim::InertialMeasurementUnit::ImuProperties::latency)
        .def_rw("seed", &raisim::InertialMeasurementUnit::ImuProperties::seed)
        .def_static("stringToNoiseType",
                    &raisim::InertialMeasurementUnit::ImuProperties::stringToNoiseType);

    py::class_<raisim::InertialMeasurementUnit, raisim::Sensor>(m, "InertialMeasurementUnit")
        .def_static("getStaticType", &raisim::InertialMeasurementUnit::getType)
        .def("setProperties", &raisim::InertialMeasurementUnit::setProperties)
        .def("reset", &raisim::InertialMeasurementUnit::reset)
        .def("resetNoise", &raisim::InertialMeasurementUnit::resetNoise)
        .def("getDroppedMeasurementCount", &raisim::InertialMeasurementUnit::getDroppedMeasurementCount)
        .def("getDeliveryTimeStamp", &raisim::InertialMeasurementUnit::getDeliveryTimeStamp)
        .def("getGroundTruthOrientation", [](const raisim::InertialMeasurementUnit &self) {
            return convert_vec_to_np(self.getGroundTruthOrientation());
        })
        .def("getAccelerometerBias", [](const raisim::InertialMeasurementUnit &self) {
            return self.getAccelerometerBias();
        })
        .def("getGyroscopeBias", [](const raisim::InertialMeasurementUnit &self) {
            return self.getGyroscopeBias();
        })
        .def("getLinearAcceleration",
             [](const raisim::InertialMeasurementUnit &self) { return self.getLinearAcceleration(); })
        .def("getAngularVelocity",
             [](const raisim::InertialMeasurementUnit &self) { return self.getAngularVelocity(); })
        .def("getOrientation", [](const raisim::InertialMeasurementUnit &self) {
            return convert_vec_to_np(self.getOrientation());
        })
        .def("setLinearAcceleration", &raisim::InertialMeasurementUnit::setLinearAcceleration)
        .def("setAngularVelocity", &raisim::InertialMeasurementUnit::setAngularVelocity)
        .def("setOrientation", [](raisim::InertialMeasurementUnit &self, const NDArray &orientation) {
            self.setOrientation(convert_np_to_vec<4>(orientation));
        })
        .def("getProperties", py::overload_cast<>(&raisim::InertialMeasurementUnit::getProperties, py::const_),
             py::rv_policy::copy);

    py::class_<raisim::SpinningLidar::SpinningLidarProperties>(m, "SpinningLidarProperties")
        .def(py::init<>())
        .def_rw("name", &raisim::SpinningLidar::SpinningLidarProperties::name)
        .def_rw("full_name", &raisim::SpinningLidar::SpinningLidarProperties::full_name)
        .def_rw("yawSamples", &raisim::SpinningLidar::SpinningLidarProperties::yawSamples)
        .def_rw("pitchSamples", &raisim::SpinningLidar::SpinningLidarProperties::pitchSamples)
        .def_rw("pitchMinAngle", &raisim::SpinningLidar::SpinningLidarProperties::pitchMinAngle)
        .def_rw("pitchMaxAngle", &raisim::SpinningLidar::SpinningLidarProperties::pitchMaxAngle)
        .def_rw("rangeMin", &raisim::SpinningLidar::SpinningLidarProperties::rangeMin)
        .def_rw("rangeMax", &raisim::SpinningLidar::SpinningLidarProperties::rangeMax)
        .def_rw("spinningRate", &raisim::SpinningLidar::SpinningLidarProperties::spinningRate)
        .def_rw("spinDirection", &raisim::SpinningLidar::SpinningLidarProperties::spinDirection)
        .def_rw("rangeBias", &raisim::SpinningLidar::SpinningLidarProperties::rangeBias)
        .def_rw("rangeNoiseStd", &raisim::SpinningLidar::SpinningLidarProperties::rangeNoiseStd)
        .def_rw("angularNoiseStd", &raisim::SpinningLidar::SpinningLidarProperties::angularNoiseStd)
        .def_rw("resolution", &raisim::SpinningLidar::SpinningLidarProperties::resolution)
        .def_rw("dropoutProbability", &raisim::SpinningLidar::SpinningLidarProperties::dropoutProbability)
        .def_rw("seed", &raisim::SpinningLidar::SpinningLidarProperties::seed);

    py::class_<raisim::SpinningLidar, raisim::Sensor>(m, "SpinningLidar")
        .def_static("getStaticType", &raisim::SpinningLidar::getType)
        .def("applyNoise", &raisim::SpinningLidar::applyNoise)
        .def("resetNoise", &raisim::SpinningLidar::resetNoise)
        .def("getProperties", &raisim::SpinningLidar::getProperties, py::rv_policy::copy)
        .def("setProperties", &raisim::SpinningLidar::setProperties)
        .def("getScan", [](const raisim::SpinningLidar &self) {
            return vec3_list_to_ndarray(self.getScan());
        })
        .def("setScan", [](raisim::SpinningLidar &self, const NDArray &scan) {
            self.setScan(ndarray_to_vec3_list(scan));
        })
        .def("getTimeStamp", &raisim::SpinningLidar::getTimeStamp)
        .def("setTimeStamp", &raisim::SpinningLidar::setTimeStamp)
        .def("getCurrentYaw", &raisim::SpinningLidar::getCurrentYaw);
#if RAISIMPY_HAS_WRENCH_SENSORS
    init_wrench_sensors(m);
#endif
#if RAISIMPY_HAS_SEGMENTATION
    init_segmentation(m);
#endif
}
