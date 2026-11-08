# Example: IMU Sensor Model

This headless example compares an ideal IMU and a modeled IMU attached to the
same moving body. It demonstrates separate accelerometer/gyro errors, sensor
bandwidth, acquisition timestamps and delayed delivery. It prints CSV and does
not start a renderer or TCP server.

For the measurement equations and parameter units, see [the IMU guide](../../Sensors.md#imu-sensor-model).
The [Server Example: Sensor Suite](./sensor_suite.md) example covers cameras and LiDAR alongside the IMU.

## Build and run

CMake target: `imu_sensor_model`. Its source is
`examples/src/server/sensors/imu_sensor_model.cpp`.
Build it from an examples build configured as described in [Build, Test, and Benchmark](../../BuildAndTest.md):

```bash
cmake --build build-examples --target imu_sensor_model --config Release --parallel 12
./build-examples/examples/imu_sensor_model > /tmp/imu.csv
```

On Windows, use the build's `bin` directory:

```powershell
.\build-examples\bin\imu_sensor_model.exe > "$env:TEMP\imu.csv"
```

The example takes no arguments and finds `activation.raisim` in the examples'
resource directory via `exampleRscPath(argv[0], ...)`. Install your activation
key in that resource directory before running.

No robot or mesh assets are needed: the source declares a small URDF in memory.
Use matching RaiSim headers and binaries with
`InertialMeasurementUnit::ChannelProperties` support. Older binary packages do
not provide the modeled-sensor API. CMake omits this target with an explanatory
status message when the selected package lacks that API.

## What it simulates

A floating 1 kg body with unit rotational inertia carries two IMUs, both 0.1 m
along its +X axis with the same orientation. Gravity is compensated by an applied
force, with an additional vertical sinusoid and an oscillating torque about +Z.
The ideal IMU therefore exposes vertical specific force, angular velocity, and
both tangential and centripetal acceleration from the offset mount.

Physics runs at 1000 Hz; both IMUs acquire at 400 Hz. Their acquisition intervals
alternate between 2 ms and 3 ms. The modeled IMU adds initial bias, white noise,
bias random walk, first-order filtering, quantization, per-axis clipping and
3 ms of delivery latency. Parameters illustrate the interface rather than a
particular commercial device.

<<< @/../examples/src/server/sensors/imu_sensor_model.cpp#properties{cpp}

The settings are fixed in the source, with seed 43 and 1000 physics steps.
Change the property assignments to try another model. The ideal sensor already
provides the zero-error reference in every CSV row.

## Reading the CSV

Each row represents one **delivered** modeled sample. The columns are:

**CSV fields**

| Columns | Meaning |
| --- | --- |
| `acquisition_s`, `delivery_s` | Acquisition and availability in simulation seconds. |
| `ideal_ax`, `ideal_ay`, `ideal_az` | Ideal specific force at the same acquisition time, in sensor axes, m/s². |
| `measured_ax`, `measured_ay`, `measured_az` | Modeled accelerometer output, m/s². |
| `ideal_gx`, `ideal_gy`, `ideal_gz` | Ideal angular velocity at that acquisition time, rad/s. |
| `measured_gx`, `measured_gy`, `measured_gz` | Modeled gyro output, rad/s. |
| `ground_truth_qw`, `ground_truth_qx`, `ground_truth_qy`, `ground_truth_qz` | Perfect sensor-to-world orientation at acquisition; not an estimated attitude. |

The example retains a short history of ideal samples and matches each delayed
reading by **acquisition time**. Comparing it to the current ideal reading would
mix latency-induced motion differences into the apparent sensor error. The
remaining difference includes the configured bias, filtering and noise.

Plot either measured channel against its ideal counterpart using
`acquisition_s` on the horizontal axis. Plot `delivery_s - acquisition_s` to
inspect availability latency. The first default reading is acquired at 0.001 s
and delivered at 0.004 s; no startup buffers with negative timestamps are emitted.

At the end of the one-second simulation, the newest delayed samples may still be
pending. The example does not advance extra physics solely to flush them. Running
the same motion with the same seed reproduces CSV on the same platform/build;
a different seed changes the stochastic readings while preserving ideal motion.

[Download the source](https://github.com/raisimTech/raisim2Lib/blob/main/examples/src/server/sensors/imu_sensor_model.cpp).
