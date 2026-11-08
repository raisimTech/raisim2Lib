// This file is part of RaiSim. A valid RaiSim license is required.
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "example_resources.hpp"
#include "raisim/World.hpp"
#include "raisim/sensors/InertialMeasurementUnit.hpp"

namespace {
constexpr double kStep = 0.001;
constexpr double kPi = 3.14159265358979323846;

struct IdealSample {
  double time;
  Eigen::Vector3d acc, gyro;
};

void writeVector(const Eigen::Vector3d& value) {
  for (int axis = 0; axis < 3; ++axis) std::cout << ',' << value[axis];
}
}  // namespace

int main(int, char** argv) {
  try {
    raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));
    raisim::World world;
    world.setTimeStep(kStep);
    world.setSleepingEnabled(false);

    // Both sensors share the same mounting point and 400 Hz acquisition clock.
    // An offset exposes centripetal and tangential acceleration when the body rotates.
    const std::string urdf = R"xml(
      <robot name="imu_demo"><link name="base">
        <inertial><mass value="1"/>
          <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
        <sensor name="ideal" type="imu" update_rate="400"><origin xyz="0.1 0 0"/></sensor>
        <sensor name="modeled" type="imu" update_rate="400"><origin xyz="0.1 0 0"/></sensor>
      </link></robot>)xml";
    raisim::ArticulatedSystemOption options;
    options.convexifyCollisionMeshes = true;
    auto* body = world.addArticulatedSystem(urdf, "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
    auto* sensors = body->getSensorSet("base");
    auto* ideal = sensors->getSensor<raisim::InertialMeasurementUnit>("ideal");
    auto* modeled = sensors->getSensor<raisim::InertialMeasurementUnit>("modeled");

    // Illustrative settings, not a calibrated preset for a particular device.
    // #region properties
    auto properties = std::as_const(*modeled).getProperties();
    properties.seed = 43;
    properties.accelerometer.bias = {0.01, -0.02, 0.03};
    properties.accelerometer.noiseDensity = 0.002;
    properties.accelerometer.biasRandomWalk = 0.0001;
    properties.accelerometer.cutoffFrequency = 30.;
    properties.accelerometer.resolution = 0.001;
    properties.gyroscope.bias = {0.001, -0.002, 0.003};
    properties.gyroscope.noiseDensity = 0.0002;
    properties.gyroscope.biasRandomWalk = 0.00001;
    properties.gyroscope.cutoffFrequency = 50.;
    properties.gyroscope.resolution = 0.0001;
    properties.maxAcc = 16. * 9.81;
    properties.maxAngVel = 2000. * kPi / 180.;
    properties.latency = 0.003;
    modeled->setProperties(properties);
    // #endregion properties

    std::cout << std::setprecision(17)
              << "acquisition_s,delivery_s,ideal_ax,ideal_ay,ideal_az,measured_ax,measured_ay,measured_az,"
                 "ideal_gx,ideal_gy,ideal_gz,measured_gx,measured_gy,measured_gz,"
                 "ground_truth_qw,ground_truth_qx,ground_truth_qy,ground_truth_qz\n";
    std::deque<IdealSample> history;
    double lastIdealTime = -1., lastMeasurementTime = -1.;
    for (int step = 0; step < 1000; ++step) {
      const double time = world.getWorldTime();
      // A 1 kg body with unit inertia: compensate gravity and apply varying force/torque.
      body->setGeneralizedForce({0., 0., 9.81 + 2. * std::sin(2. * kPi * time),
                                 0., 0., 0.5 * std::cos(kPi * time)});
      world.integrate();
      if (ideal->getUpdateTimeStamp() != lastIdealTime) {
        lastIdealTime = ideal->getUpdateTimeStamp();
        history.push_back({lastIdealTime, ideal->getLinearAcceleration(), ideal->getAngularVelocity()});
      }
      // Negative timestamps mean no sample has been delivered. Otherwise consume each
      // acquisition once; the getters hold the last delivered value between updates.
      const double acquired = modeled->getUpdateTimeStamp();
      if (acquired < 0. || acquired == lastMeasurementTime) continue;
      lastMeasurementTime = acquired;
      while (!history.empty() && history.front().time < acquired) history.pop_front();
      if (history.empty() || history.front().time != acquired)
        throw std::runtime_error("missing ideal sample at the modeled acquisition time");
      const auto& truth = history.front();
      std::cout << acquired << ',' << modeled->getDeliveryTimeStamp();
      writeVector(truth.acc);
      writeVector(modeled->getLinearAcceleration());
      writeVector(truth.gyro);
      writeVector(modeled->getAngularVelocity());
      // This quaternion is ground truth at acquisition, not an attitude estimate.
      const auto& orientation = modeled->getGroundTruthOrientation();
      for (int i = 0; i < 4; ++i) std::cout << ',' << orientation[i];
      std::cout << '\n';
      history.pop_front();
    }
    if (!std::cout) throw std::runtime_error("failed to write IMU CSV output");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "imu_sensor_model: " << error.what() << '\n';
    return 1;
  }
}
