#include "modern_bindings.hpp"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace raisim;

void init_force_devices(py::module_& m, py::class_<ArticulatedSystem, Object>& system) {
    auto thruster = py::class_<ThrusterDefinition>(m, "ThrusterDefinition",
        "A thruster fixed to a link: every step it pushes the link with thrust * axis at its position and turns it "
        "with torqueCoefficient * thrust * axis. Position and axis are in the link frame.");
    thruster.def(py::init<>())
        .def_rw("name", &ThrusterDefinition::name)
        .def_rw("link", &ThrusterDefinition::link)
        .def_rw("minThrust", &ThrusterDefinition::minThrust)
        .def_rw("maxThrust", &ThrusterDefinition::maxThrust)
        .def_rw("torqueCoefficient", &ThrusterDefinition::torqueCoefficient);
    bind_vector_property(thruster, "position", &ThrusterDefinition::position);
    bind_vector_property(thruster, "axis", &ThrusterDefinition::axis);

    auto gripper = py::class_<GripperDefinition>(m, "GripperDefinition",
        "A suction cup or an electromagnet on a link. While active, it pulls the first surface within range in "
        "front of its pad (along axis) toward the pad. Position and axis are in the link frame.");
    py::enum_<GripperDefinition::Type>(gripper, "Type")
        .value("SUCTION", GripperDefinition::Type::SUCTION)
        .value("MAGNETIC", GripperDefinition::Type::MAGNETIC);
    gripper.def(py::init<>())
        .def_rw("name", &GripperDefinition::name)
        .def_rw("link", &GripperDefinition::link)
        .def_rw("type", &GripperDefinition::type)
        .def_rw("maxForce", &GripperDefinition::maxForce)
        .def_rw("range", &GripperDefinition::range)
        .def_rw("sealAngle", &GripperDefinition::sealAngle)
        .def_rw("cupRadius", &GripperDefinition::cupRadius)
        .def_rw("leakCoefficient", &GripperDefinition::leakCoefficient);
    bind_vector_property(gripper, "position", &GripperDefinition::position);
    bind_vector_property(gripper, "axis", &GripperDefinition::axis);

    py::class_<GripperState>(m, "GripperState", "What a gripper did in the last step.")
        .def_ro("engaged", &GripperState::engaged)
        .def_prop_ro("object", [](const GripperState& self) { return const_cast<Object*>(self.object); },
                     py::rv_policy::reference)
        .def_ro("localIdx", &GripperState::localIdx)
        .def_ro("force", &GripperState::force)
        .def_ro("gap", &GripperState::gap)
        .def_prop_ro("position", [](const GripperState& self) { return convert_vec_to_np(self.position); },
                     py::rv_policy::move);

    const auto pose = [](const ArticulatedSystem& self, size_t index, bool thrusterPose) {
        Vec<3> position, axis;
        if (thrusterPose) self.getThrusterPose(index, position, axis);
        else self.getGripperPose(index, position, axis);
        return py::make_tuple(convert_vec_to_np(position), convert_vec_to_np(axis));
    };

    system
        .def("setThrusters", &ArticulatedSystem::setThrusters, py::arg("thrusters"))
        .def("getThrusters", &ArticulatedSystem::getThrusters, py::rv_policy::copy)
        .def("getThrusterIndex", &ArticulatedSystem::getThrusterIndex, py::arg("name"))
        .def("setThrust", py::overload_cast<size_t, double>(&ArticulatedSystem::setThrust),
             py::arg("index"), py::arg("thrust"))
        .def("setThrust", py::overload_cast<const std::string&, double>(&ArticulatedSystem::setThrust),
             py::arg("name"), py::arg("thrust"))
        .def("setThrusts", &ArticulatedSystem::setThrusts, py::arg("thrusts"))
        .def("getThrust", &ArticulatedSystem::getThrust, py::arg("index"))
        .def("getThrusts", &ArticulatedSystem::getThrusts, py::rv_policy::copy)
        .def("getThrusterPose", [pose](const ArticulatedSystem& self, size_t index) { return pose(self, index, true); },
             py::arg("index"), "(position, axis) of the thruster in the world frame")
        .def("setGrippers", &ArticulatedSystem::setGrippers, py::arg("grippers"))
        .def("getGrippers", &ArticulatedSystem::getGrippers, py::rv_policy::copy)
        .def("getGripperIndex", &ArticulatedSystem::getGripperIndex, py::arg("name"))
        .def("setGripperActive", py::overload_cast<size_t, bool>(&ArticulatedSystem::setGripperActive),
             py::arg("index"), py::arg("active"))
        .def("setGripperActive", py::overload_cast<const std::string&, bool>(&ArticulatedSystem::setGripperActive),
             py::arg("name"), py::arg("active"))
        .def("isGripperActive", &ArticulatedSystem::isGripperActive, py::arg("index"))
        .def("getGripperState", &ArticulatedSystem::getGripperState, py::arg("index"), py::rv_policy::copy)
        .def("getGripperPose", [pose](const ArticulatedSystem& self, size_t index) { return pose(self, index, false); },
             py::arg("index"), "(position, axis) of the pad in the world frame");
}

void init_force_devices_world(py::class_<World>& world) {
    world
        .def("setMagneticMaterial", &World::setMagneticMaterial, py::arg("material"), py::arg("magnetic") = true,
             "Mark a collision material as magnetic or not; magnetic grippers pull only magnetic materials.")
        .def("isMagneticMaterial", &World::isMagneticMaterial, py::arg("material"));
}
