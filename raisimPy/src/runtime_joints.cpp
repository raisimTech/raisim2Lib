#include "modern_bindings.hpp"

namespace py = nanobind;
using namespace raisim;

void init_runtime_joints(py::module_& m) {
  auto joint = py::class_<RuntimeJoint>(m, "RuntimeJoint");
  py::enum_<RuntimeJoint::Type>(joint, "Type")
      .value("Fixed", RuntimeJoint::Type::Fixed)
      .value("Ball", RuntimeJoint::Type::Ball)
      .value("Hinge", RuntimeJoint::Type::Hinge)
      .value("Slider", RuntimeJoint::Type::Slider);
  py::class_<RuntimeJoint::Attachment>(joint, "Attachment")
      .def("__init__", [](RuntimeJoint::Attachment* self, Object* object, size_t localIndex,
                           const std::array<double, 3>& position, const Eigen::Matrix3d& orientation) {
        new (self) RuntimeJoint::Attachment{object, localIndex, {position[0], position[1], position[2]}};
        self->orientation.e() = orientation;
      }, py::arg("object").none() = nullptr, py::arg("localIndex") = 0,
         py::arg("position") = std::array<double, 3>{0., 0., 0.},
         py::arg("orientation") = Eigen::Matrix3d(Eigen::Matrix3d::Identity()), py::keep_alive<1, 2>())
      .def_prop_rw("object", [](const RuntimeJoint::Attachment& self) { return self.object; },
          [](RuntimeJoint::Attachment& self, Object* object) { self.object = object; },
          py::for_getter(py::rv_policy::reference_internal),
          py::for_setter(py::keep_alive<1, 2>()), py::for_setter(py::arg("object").none()))
      .def_rw("localIndex", &RuntimeJoint::Attachment::localIndex)
      .def_prop_rw("position", [](const RuntimeJoint::Attachment& self) { return convert_vec_to_np(self.position); },
          [](RuntimeJoint::Attachment& self, const std::array<double, 3>& p) { self.position = {p[0], p[1], p[2]}; },
          py::for_getter(py::rv_policy::move))
      .def_prop_rw("orientation", [](const RuntimeJoint::Attachment& self) { return convert_mat_to_np(self.orientation); },
          [](RuntimeJoint::Attachment& self, const NDArray& r) { self.orientation = convert_np_to_mat<3, 3>(r); },
          py::for_getter(py::rv_policy::move));
  py::class_<RuntimeJoint::Properties>(joint, "Properties")
      .def(py::init<>())
      .def_rw("positionCorrection", &RuntimeJoint::Properties::positionCorrection)
      .def_rw("breakForce", &RuntimeJoint::Properties::breakForce)
      .def_rw("breakTorque", &RuntimeJoint::Properties::breakTorque)
      .def_rw("collideConnected", &RuntimeJoint::Properties::collideConnected);
  joint.def("getName", &RuntimeJoint::getName)
      .def("getType", &RuntimeJoint::getType)
      .def("getAttachmentA", &RuntimeJoint::getAttachmentA, py::rv_policy::copy, py::keep_alive<0, 1>())
      .def("getAttachmentB", &RuntimeJoint::getAttachmentB, py::rv_policy::copy, py::keep_alive<0, 1>())
      .def("getProperties", &RuntimeJoint::getProperties, py::rv_policy::copy)
      .def("setProperties", &RuntimeJoint::setProperties, py::arg("properties"))
      .def("setEnabled", &RuntimeJoint::setEnabled, py::arg("enabled"))
      .def("isEnabled", &RuntimeJoint::isEnabled)
      .def("isBroken", &RuntimeJoint::isBroken)
      .def("resetBreakage", &RuntimeJoint::resetBreakage)
      .def("getForce", [](const RuntimeJoint& self) { return convert_vec_to_np(self.getForce()); }, py::rv_policy::move)
      .def("getTorque", [](const RuntimeJoint& self) { return convert_vec_to_np(self.getTorque()); }, py::rv_policy::move);
}

void init_runtime_joint_world(py::class_<World>& world) {
  world.def("addRuntimeJoint", &World::addRuntimeJoint,
      py::arg("name"), py::arg("type"), py::arg("a"), py::arg("b"),
      py::arg("properties") = RuntimeJoint::Properties{}, py::rv_policy::reference_internal)
      .def("getRuntimeJoint", &World::getRuntimeJoint, py::arg("name"), py::rv_policy::reference_internal)
      .def("getRuntimeJoints", [](World& self) {
        std::vector<RuntimeJoint*> joints;
        for (const auto& joint : self.getRuntimeJoints()) joints.push_back(joint.get());
        return joints;
      }, py::rv_policy::reference_internal)
      .def("removeRuntimeJoint", &World::removeRuntimeJoint, py::arg("joint"));
}
