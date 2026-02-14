/**
 * Python wrappers for raisim.sensors using nanobind.
 */

#include "nanobind_helpers.hpp"

#include <sstream>

#include "raisim/sensors/SensorSet.hpp"
#include "raisim/sensors/InertialMeasurementUnit.hpp"
#include "raisim/sensors/RGBSensor.hpp"
#include "raisim/sensors/DepthSensor.hpp"
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

}  // namespace

void init_sensors(py::module_ &m) {
    py::enum_<raisim::Sensor::Type>(m, "SensorType", py::is_arithmetic())
        .value("UNKNOWN", raisim::Sensor::Type::UNKNOWN)
        .value("RGB", raisim::Sensor::Type::RGB)
        .value("DEPTH", raisim::Sensor::Type::DEPTH)
        .value("IMU", raisim::Sensor::Type::IMU)
        .value("SPINNING_LIDAR", raisim::Sensor::Type::SPINNING_LIDAR);

    py::enum_<raisim::Sensor::MeasurementSource>(m, "SensorMeasurementSource", py::is_arithmetic())
        .value("RAISIM", raisim::Sensor::MeasurementSource::RAISIM)
        .value("VISUALIZER", raisim::Sensor::MeasurementSource::VISUALIZER)
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
        .def_rw("noiseType", &raisim::RGBCamera::RGBCameraProperties::noiseType)
        .def_rw("mean", &raisim::RGBCamera::RGBCameraProperties::mean)
        .def_rw("std", &raisim::RGBCamera::RGBCameraProperties::std)
        .def_rw("format", &raisim::RGBCamera::RGBCameraProperties::format)
        .def_static("stringToNoiseType",
                    &raisim::RGBCamera::RGBCameraProperties::stringToNoiseType);

    py::class_<raisim::RGBCamera, raisim::Sensor>(m, "RGBCamera")
        .def_static("getType", &raisim::RGBCamera::getType)
        .def("getProperties", &raisim::RGBCamera::getProperties, py::rv_policy::reference_internal)
        .def("getImageBuffer", py::overload_cast<>(&raisim::RGBCamera::getImageBuffer, py::const_))
        .def("setImageBuffer", &raisim::RGBCamera::setImageBuffer);

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
        .def_rw("noiseType", &raisim::DepthCamera::DepthCameraProperties::noiseType)
        .def_rw("mean", &raisim::DepthCamera::DepthCameraProperties::mean)
        .def_rw("std", &raisim::DepthCamera::DepthCameraProperties::std)
        .def_rw("format", &raisim::DepthCamera::DepthCameraProperties::format)
        .def_static("stringToNoiseType",
                    &raisim::DepthCamera::DepthCameraProperties::stringToNoiseType);

    py::class_<raisim::DepthCamera, raisim::Sensor>(m, "DepthCamera")
        .def_static("getType", &raisim::DepthCamera::getType)
        .def("updateRayDirections", &raisim::DepthCamera::updateRayDirections)
        .def("getDepthArray", py::overload_cast<>(&raisim::DepthCamera::getDepthArray, py::const_))
        .def("setDepthArray", &raisim::DepthCamera::setDepthArray)
        .def("get3DPoints", [](const raisim::DepthCamera &self) {
            return vec3_list_to_ndarray(self.get3DPoints());
        })
        .def("set3DPoints", [](raisim::DepthCamera &self, const NDArray &points) {
            self.set3DPoints(ndarray_to_vec3_list(points));
        })
        .def("getProperties", &raisim::DepthCamera::getProperties, py::rv_policy::reference_internal)
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

    py::enum_<raisim::InertialMeasurementUnit::ImuProperties::NoiseType>(m, "ImuNoiseType",
                                                                         py::is_arithmetic())
        .value("GAUSSIAN", raisim::InertialMeasurementUnit::ImuProperties::NoiseType::GAUSSIAN)
        .value("UNIFORM", raisim::InertialMeasurementUnit::ImuProperties::NoiseType::UNIFORM)
        .value("NO_NOISE", raisim::InertialMeasurementUnit::ImuProperties::NoiseType::NO_NOISE);

    py::class_<raisim::InertialMeasurementUnit::ImuProperties>(m, "ImuProperties")
        .def(py::init<>())
        .def_rw("name", &raisim::InertialMeasurementUnit::ImuProperties::name)
        .def_rw("full_name", &raisim::InertialMeasurementUnit::ImuProperties::full_name)
        .def_rw("maxAcc", &raisim::InertialMeasurementUnit::ImuProperties::maxAcc)
        .def_rw("maxAngVel", &raisim::InertialMeasurementUnit::ImuProperties::maxAngVel)
        .def_rw("noiseType", &raisim::InertialMeasurementUnit::ImuProperties::noiseType)
        .def_rw("mean", &raisim::InertialMeasurementUnit::ImuProperties::mean)
        .def_rw("std", &raisim::InertialMeasurementUnit::ImuProperties::std)
        .def_static("stringToNoiseType",
                    &raisim::InertialMeasurementUnit::ImuProperties::stringToNoiseType);

    py::class_<raisim::InertialMeasurementUnit, raisim::Sensor>(m, "InertialMeasurementUnit")
        .def_static("getType", &raisim::InertialMeasurementUnit::getType)
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
        .def("getProperties", &raisim::InertialMeasurementUnit::getProperties,
             py::rv_policy::reference_internal);

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
        .def_rw("spinDirection", &raisim::SpinningLidar::SpinningLidarProperties::spinDirection);

    py::class_<raisim::SpinningLidar, raisim::Sensor>(m, "SpinningLidar")
        .def_static("getType", &raisim::SpinningLidar::getType)
        .def("getProperties", &raisim::SpinningLidar::getProperties, py::rv_policy::reference_internal)
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
}
