#include "modern_bindings.hpp"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace raisim;

void init_articulated_join(py::module_& m) {
  auto joint = py::class_<Joint>(m, "Joint",
      "A joint of an articulated system, used by World.joinObjects(). Its frame (position, orientation) is "
      "expressed in the parent body frame and its axis in the joint frame.");
  joint.def(py::init<>())
      .def("__init__", [](Joint* self, Joint::Type type, const std::string& name, const std::array<double, 3>& position,
                          const std::array<double, 3>& axis, const std::optional<Eigen::Matrix3d>& orientation) {
        new (self) Joint();
        self->type = type;
        self->name = name;
        self->pos_P = {position[0], position[1], position[2]};
        self->axis = {axis[0], axis[1], axis[2]};
        self->rot = orientation_or_identity(orientation);
      }, py::arg("type"), py::arg("name"), py::arg("position") = std::array<double, 3>{0., 0., 0.},
         py::arg("axis") = std::array<double, 3>{0., 0., 1.},
         py::arg("orientation") = py::none())
      .def_rw("type", &Joint::type)
      .def_rw("name", &Joint::name)
      .def_prop_rw("orientation", [](const Joint& self) { return convert_mat_to_np(self.rot); },
          [](Joint& self, const NDArray& value) { self.rot = convert_np_to_mat<3, 3>(value); },
          py::for_getter(py::rv_policy::move))
      .def_rw("damping", &Joint::damping)
      .def_rw("friction", &Joint::friction)
      .def_rw("stiffness", &Joint::stiffness)
      .def_rw("rotorInertia", &Joint::rotor_inertia)
      .def_rw("effort", &Joint::effort)
      .def_rw("velocityLimit", &Joint::velocity_limit)
      .def_rw("jointRef", &Joint::jointRef);
  bind_vector_property(joint, "position", &Joint::pos_P);
  bind_vector_property(joint, "axis", &Joint::axis);
  bind_vector_property(joint, "limit", &Joint::limit);
  bind_vector_property(joint, "springMount", &Joint::springMount);

  py::class_<JoinOptions>(m, "JoinOptions", "Options of World.joinObjects().")
      .def(py::init<>())
      .def_rw("name", &JoinOptions::name)
      .def_rw("childPrefix", &JoinOptions::childPrefix);
}

void init_articulated_join_world(py::class_<World>& world) {
  world.def("joinObjects", &World::joinObjects, py::arg("parent").none(), py::arg("parentBody"), py::arg("child"),
            py::arg("childBody"), py::arg("joint"), py::arg("options") = JoinOptions{},
            py::rv_policy::reference_internal,
            "Connect two objects (parent None: the world) with a joint and replace them with one articulated "
            "system. The parent and child handles become invalid.")
      .def("detachJoint", &World::detachJoint, py::arg("system"), py::arg("jointName"),
           py::arg("detachedName") = std::string(), py::rv_policy::reference_internal,
           "Remove a joint from an articulated system and return the resulting systems; the input handle "
           "becomes invalid.");
}
