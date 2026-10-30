/**
 * Python wrappers for the fluids of raisim (raisim/Fluid.hpp) using nanobind.
 */

#include "nanobind_helpers.hpp"

#include <nanobind/stl/vector.h>

#include "converter.hpp"
#include "raisim/Fluid.hpp"

namespace py = nanobind;
using namespace raisim;

void init_fluid(py::module_ &m) {
  py::class_<FluidMedium>(m, "FluidMedium", "A fluid: water, air, oil. A density of zero is no fluid.")
      .def(py::init<>())
      .def_rw("density", &FluidMedium::density, "Density [kg/m^3].")
      .def_rw("viscosity", &FluidMedium::viscosity, "Dynamic viscosity [Pa s].")
      .def_prop_rw("velocity",
          [](const FluidMedium &self) { return convert_vec_to_np(self.velocity); },
          [](FluidMedium &self, const NDArray &array) { self.velocity = convert_np_to_vec<3>(array); },
          "Velocity of the fluid (a current or the wind) [m/s], in the world frame.",
          py::for_getter(py::rv_policy::move))
      .def_static("water", &FluidMedium::water, "Fresh water at 20 C.")
      .def_static("seawater", &FluidMedium::seawater, "Sea water at 15 C.")
      .def_static("air", &FluidMedium::air, "Air at 20 C and sea level.");

  py::class_<FluidSample>(m, "FluidSample", "The liquid of a FluidVolume where a body is.")
      .def(py::init<>())
      .def_rw("surfaceHeight", &FluidSample::surfaceHeight, "Height of the free surface over the point [m].")
      .def_prop_rw("surfaceNormal",
          [](const FluidSample &self) { return convert_vec_to_np(self.surfaceNormal); },
          [](FluidSample &self, const NDArray &array) { self.surfaceNormal = convert_np_to_vec<3>(array); },
          "Normal of the free surface there, in the world frame.",
          py::for_getter(py::rv_policy::move))
      .def_prop_rw("velocity",
          [](const FluidSample &self) { return convert_vec_to_np(self.velocity); },
          [](FluidSample &self, const NDArray &array) { self.velocity = convert_np_to_vec<3>(array); },
          "Velocity of the liquid at the point [m/s], in the world frame.",
          py::for_getter(py::rv_policy::move))
      .def_prop_rw("acceleration",
          [](const FluidSample &self) { return convert_vec_to_np(self.acceleration); },
          [](FluidSample &self, const NDArray &array) { self.acceleration = convert_np_to_vec<3>(array); },
          "Acceleration of the liquid at the point [m/s^2], in the world frame, as in waves.",
          py::for_getter(py::rv_policy::move));

  py::class_<FluidWave>(m, "FluidWave", "A linear wave on the surface of a deep liquid.")
      .def(py::init<>())
      .def("__init__",
          [](FluidWave *self, const NDArray &direction, double wavelength, double amplitude, double phase) {
            new (self) FluidWave{convert_np_to_vec<2>(direction), wavelength, amplitude, phase};
          },
          py::arg("direction"), py::arg("wavelength"), py::arg("amplitude"), py::arg("phase") = 0.0)
      .def_prop_rw("direction",
          [](const FluidWave &self) { return convert_vec_to_np(self.direction); },
          [](FluidWave &self, const NDArray &array) { self.direction = convert_np_to_vec<2>(array); },
          "Direction the wave travels in, (x, y).",
          py::for_getter(py::rv_policy::move))
      .def_rw("wavelength", &FluidWave::wavelength, "Distance from crest to crest [m].")
      .def_rw("amplitude", &FluidWave::amplitude, "Height of a crest above the mean surface [m].")
      .def_rw("phase", &FluidWave::phase, "Phase at the origin at time zero [rad].");

  py::class_<FluidVolume>(m, "FluidVolume", "A body of liquid under a free surface: a pool, a lake, the sea.")
      .def(py::init<>())
      .def_rw("medium", &FluidVolume::medium, "The liquid.")
      .def_rw("surfaceHeight", &FluidVolume::surfaceHeight, "Height of the free surface [m].")
      .def_rw("bottomHeight", &FluidVolume::bottomHeight, "Height of the bottom [m].")
      .def_prop_rw("min",
          [](const FluidVolume &self) { return convert_vec_to_np(self.min); },
          [](FluidVolume &self, const NDArray &array) { self.min = convert_np_to_vec<2>(array); },
          "Lower horizontal corner (x, y) [m].",
          py::for_getter(py::rv_policy::move))
      .def_prop_rw("max",
          [](const FluidVolume &self) { return convert_vec_to_np(self.max); },
          [](FluidVolume &self, const NDArray &array) { self.max = convert_np_to_vec<2>(array); },
          "Upper horizontal corner (x, y) [m].",
          py::for_getter(py::rv_policy::move))
      .def_rw("waveRadiation", &FluidVolume::waveRadiation,
          "Scale of the damping of the waves a floating body makes as it bobs; 0 turns it off.")
      .def_rw("waves", &FluidVolume::waves, "Waves on the surface (list of FluidWave); a sampler takes their place.")
      .def("setSampler",
          [](FluidVolume &self, std::function<FluidSample(NDArray)> sampler) {
            self.sampler = [sampler = std::move(sampler)](const Vec<3> &point) { return sampler(convert_vec_to_np(point)); };
          },
          "Give the volume waves and currents: callable(point: np.array[3]) -> FluidSample, the surface and the "
          "velocity of the liquid around a point.", py::arg("sampler"))
      .def("clearSampler", [](FluidVolume &self) { self.sampler = nullptr; }, "Remove the sampler.")
      .def("hasSampler", [](const FluidVolume &self) { return static_cast<bool>(self.sampler); },
          "Whether the volume has a sampler.");

  py::class_<FluidCoefficients>(m, "FluidCoefficients", "How a collision shape interacts with fluids.")
      .def(py::init<>())
      .def_rw("enabled", &FluidCoefficients::enabled, "Whether fluids act on the shape at all.")
      .def_rw("buoyancy", &FluidCoefficients::buoyancy, "Whether the fluid holds the shape up.")
      .def_rw("drag", &FluidCoefficients::drag, "Pressure drag coefficient; negative takes the shape's own.")
      .def_rw("skinFriction", &FluidCoefficients::skinFriction, "Skin friction coefficient.")
      .def_rw("magnusLift", &FluidCoefficients::magnusLift, "Magnus lift coefficient of a spinning shape.")
      .def_rw("addedMass", &FluidCoefficients::addedMass, "Scale of the added mass.");
}
