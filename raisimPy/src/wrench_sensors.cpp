// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#include "nanobind_helpers.hpp"
#include "converter.hpp"
#include "raisim/World.hpp"
#include "raisim/sensors/ForceSensor.hpp"
#include "raisim/sensors/TactileSensor.hpp"

namespace py = nanobind;

void init_wrench_sensors(py::module_& m) {
  using raisim::WrenchSensor;
  using raisim::ForceSensor;
  using raisim::TactileSensor;
  py::class_<WrenchSensor::Model>(m, "WrenchSensorModel")
      .def(py::init<>())
      .def_rw("forceBias", &WrenchSensor::Model::forceBias)
      .def_rw("torqueBias", &WrenchSensor::Model::torqueBias)
      .def_rw("forceNoiseStd", &WrenchSensor::Model::forceNoiseStd)
      .def_rw("torqueNoiseStd", &WrenchSensor::Model::torqueNoiseStd)
      .def_rw("maxForce", &WrenchSensor::Model::maxForce)
      .def_rw("maxTorque", &WrenchSensor::Model::maxTorque)
      .def_rw("cutoffFrequency", &WrenchSensor::Model::cutoffFrequency)
      .def_rw("latency", &WrenchSensor::Model::latency)
      .def_rw("seed", &WrenchSensor::Model::seed);
  py::class_<WrenchSensor, raisim::Sensor>(m, "WrenchSensor")
      .def("getForce", &WrenchSensor::getForce, py::rv_policy::copy)
      .def("getTorque", &WrenchSensor::getTorque, py::rv_policy::copy)
      .def("getAcquisitionStartTime", &WrenchSensor::getAcquisitionStartTime)
      .def("getDeliveryTimeStamp", &WrenchSensor::getDeliveryTimeStamp)
      .def("getDroppedMeasurementCount", &WrenchSensor::getDroppedMeasurementCount)
      .def("reset", &WrenchSensor::reset)
      .def("resetNoise", &WrenchSensor::resetNoise)
      .def("update", &WrenchSensor::update);
  py::class_<ForceSensor::ForceSensorProperties>(m, "ForceSensorProperties")
      .def(py::init<>())
      .def_rw("name", &ForceSensor::ForceSensorProperties::name)
      .def_rw("full_name", &ForceSensor::ForceSensorProperties::full_name)
      .def_rw("joint", &ForceSensor::ForceSensorProperties::joint)
      .def_rw("model", &ForceSensor::ForceSensorProperties::model);
  py::class_<ForceSensor, WrenchSensor>(m, "ForceSensor")
      .def_static("getStaticType", &ForceSensor::getType)
      .def("getProperties", &ForceSensor::getProperties, py::rv_policy::copy)
      .def("setProperties", &ForceSensor::setProperties);
  py::class_<TactileSensor::TactileSensorProperties>(m, "TactileSensorProperties")
      .def(py::init<>())
      .def_rw("name", &TactileSensor::TactileSensorProperties::name)
      .def_rw("full_name", &TactileSensor::TactileSensorProperties::full_name)
      .def_rw("link", &TactileSensor::TactileSensorProperties::link)
      .def_rw("collision", &TactileSensor::TactileSensorProperties::collision)
      .def_rw("width", &TactileSensor::TactileSensorProperties::width)
      .def_rw("height", &TactileSensor::TactileSensorProperties::height)
      .def_rw("depth", &TactileSensor::TactileSensorProperties::depth)
      .def_rw("columns", &TactileSensor::TactileSensorProperties::columns)
      .def_rw("rows", &TactileSensor::TactileSensorProperties::rows)
      .def_rw("model", &TactileSensor::TactileSensorProperties::model);
  py::class_<TactileSensor, WrenchSensor>(m, "TactileSensor")
      .def_static("getStaticType", &TactileSensor::getType)
      .def("getProperties", &TactileSensor::getProperties, py::rv_policy::copy)
      .def("setProperties", &TactileSensor::setProperties)
      .def("getNormalForce", &TactileSensor::getNormalForce)
      .def("hasCenterOfPressure", &TactileSensor::hasCenterOfPressure)
      .def("getCenterOfPressure", &TactileSensor::getCenterOfPressure, py::rv_policy::copy)
      .def("getTaxelForces", [](const TactileSensor& sensor) {
        const auto& values = sensor.getTaxelForces();
        auto result = make_ndarray_2d(values.size(), 3);
        for (size_t i = 0; i < values.size(); ++i)
          for (size_t j = 0; j < 3; ++j) *ndarray_mutable_data(result, i, j) = values[i][j];
        return result;
      })
      .def("getTaxelSlipVelocities", [](const TactileSensor& sensor) {
        const auto& values = sensor.getTaxelSlipVelocities();
        auto result = make_ndarray_2d(values.size(), 2);
        for (size_t i = 0; i < values.size(); ++i)
          for (size_t j = 0; j < 2; ++j) *ndarray_mutable_data(result, i, j) = values[i][j];
        return result;
      });
}
