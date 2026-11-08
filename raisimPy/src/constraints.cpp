/**
 * Python wrappers for raisim.constraints using nanobind.
 *
 * Copyright (c) 2019, jhwangbo (C++), Brian Delhaisse <briandelhaisse@gmail.com> (Python wrappers)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "nanobind_helpers.hpp"
#include <nanobind/stl/array.h>
#include "raisim/constraints/Tendon.hpp"
#include "raisim/constraints/PinConstraint.hpp"
#include "raisim/World.hpp"
#include "converter.hpp"

namespace py = nanobind;
using namespace raisim;

void init_constraints(py::module_ &m) {
    auto constraints_module = m.def_submodule("constraints", "Closed-loop pin constraints.");
    auto tendon = py::class_<Tendon>(m, "Tendon");
    py::enum_<Tendon::Type>(tendon, "Type")
        .value("Spatial", Tendon::Type::Spatial).value("Fixed", Tendon::Type::Fixed);
    py::class_<Tendon::Site>(tendon, "Site")
        .def("__init__", [](Tendon::Site* self, Object* object, size_t localIndex,
                           const std::array<double, 3>& position) {
            new (self) Tendon::Site{object, localIndex, {position[0], position[1], position[2]}};
        }, py::arg("object").none() = nullptr, py::arg("localIndex") = 0,
           py::arg("position") = std::array<double, 3>{0., 0., 0.})
        .def_rw("object", &Tendon::Site::object)
        .def_rw("localIndex", &Tendon::Site::localIndex)
        .def_prop_rw("position", [](const Tendon::Site& self) { return convert_vec_to_np(self.position); },
            [](Tendon::Site& self, const std::array<double, 3>& p) { self.position = {p[0], p[1], p[2]}; }, py::for_getter(py::rv_policy::move));
    auto path = py::class_<Tendon::PathElement>(tendon, "PathElement");
    py::enum_<Tendon::PathElement::Kind>(path, "Kind")
        .value("Site", Tendon::PathElement::Kind::Site)
        .value("Sphere", Tendon::PathElement::Kind::Sphere)
        .value("Cylinder", Tendon::PathElement::Kind::Cylinder)
        .value("Pulley", Tendon::PathElement::Kind::Pulley);
    path.def(py::init<>())
        .def_static("via", &Tendon::PathElement::via, py::arg("site"))
        .def_static("sphere", &Tendon::PathElement::sphere, py::arg("center"), py::arg("radius"))
        .def_static("cylinder", [](const Tendon::Site& center, double radius, const std::array<double, 3>& axis) {
            return Tendon::PathElement::cylinder(center, radius, {axis[0], axis[1], axis[2]});
        }, py::arg("center"), py::arg("radius"), py::arg("axis") = std::array<double, 3>{0., 0., 1.})
        .def_static("pulley", &Tendon::PathElement::pulley, py::arg("divisor"))
        .def("withSideSite", &Tendon::PathElement::withSideSite, py::arg("side"), py::rv_policy::reference_internal)
        .def_rw("kind", &Tendon::PathElement::kind)
        .def_rw("site", &Tendon::PathElement::site)
        .def_rw("radius", &Tendon::PathElement::radius)
        .def_rw("hasSideSite", &Tendon::PathElement::hasSideSite)
        .def_rw("sideSite", &Tendon::PathElement::sideSite)
        .def_rw("divisor", &Tendon::PathElement::divisor)
        .def_prop_rw("axis", [](const Tendon::PathElement& self) { return convert_vec_to_np(self.axis); },
            [](Tendon::PathElement& self, const std::array<double, 3>& a) { self.axis = {a[0], a[1], a[2]}; }, py::for_getter(py::rv_policy::move));
    py::class_<Tendon::JointTerm>(tendon, "JointTerm")
        .def(py::init<ArticulatedSystem*, std::string, double>(),
             py::arg("system"), py::arg("joint"), py::arg("coefficient") = 1.)
        .def_rw("system", &Tendon::JointTerm::system)
        .def_rw("joint", &Tendon::JointTerm::joint)
        .def_rw("coefficient", &Tendon::JointTerm::coefficient);
    py::class_<Tendon::Properties>(tendon, "Properties")
        .def(py::init<>())
        .def_rw("stiffness", &Tendon::Properties::stiffness)
        .def_rw("damping", &Tendon::Properties::damping)
        .def_rw("springLower", &Tendon::Properties::springLower)
        .def_rw("springUpper", &Tendon::Properties::springUpper)
        .def_rw("lowerLimit", &Tendon::Properties::lowerLimit)
        .def_rw("upperLimit", &Tendon::Properties::upperLimit)
        .def_rw("limitMargin", &Tendon::Properties::limitMargin)
        .def_rw("limitCompliance", &Tendon::Properties::limitCompliance)
        .def_rw("frictionLoss", &Tendon::Properties::frictionLoss)
        .def_rw("frictionCompliance", &Tendon::Properties::frictionCompliance)
        .def_rw("armature", &Tendon::Properties::armature)
        .def_rw("positionCorrection", &Tendon::Properties::positionCorrection)
        .def_rw("actuationLower", &Tendon::Properties::actuationLower)
        .def_rw("actuationUpper", &Tendon::Properties::actuationUpper)
        .def_rw("width", &Tendon::Properties::width)
        .def_prop_rw("color", [](const Tendon::Properties& self) { return convert_vec_to_np(self.color); },
            [](Tendon::Properties& self, const std::array<double, 4>& c) { self.color = {c[0], c[1], c[2], c[3]}; }, py::for_getter(py::rv_policy::move));
    py::class_<Tendon::Drive>(tendon, "Drive")
        .def(py::init<>())
        .def_rw("force", &Tendon::Drive::force)
        .def_rw("targetLength", &Tendon::Drive::targetLength)
        .def_rw("targetVelocity", &Tendon::Drive::targetVelocity)
        .def_rw("positionGain", &Tendon::Drive::positionGain)
        .def_rw("velocityGain", &Tendon::Drive::velocityGain)
        .def_rw("activationTime", &Tendon::Drive::activationTime);
    py::class_<Tendon::VisualSegment>(tendon, "VisualSegment")
        .def_prop_ro("start", [](const Tendon::VisualSegment& self) { return convert_vec_to_np(self.start); }, py::rv_policy::move)
        .def_prop_ro("end", [](const Tendon::VisualSegment& self) { return convert_vec_to_np(self.end); }, py::rv_policy::move)
        .def_ro("wrapped", &Tendon::VisualSegment::wrapped);
    tendon
        .def("getType", &Tendon::getType)
        .def("getName", &Tendon::getName)
        .def("getLength", &Tendon::getLength)
        .def("getVelocity", &Tendon::getVelocity)
        .def("getReferenceLength", &Tendon::getReferenceLength)
        .def("getForce", &Tendon::getForce)
        .def("getTension", &Tendon::getTension)
        .def("getActuationForce", &Tendon::getActuationForce)
        .def("getLimitForce", &Tendon::getLimitForce)
        .def("getFrictionForce", &Tendon::getFrictionForce)
        .def("getPotentialEnergy", &Tendon::getPotentialEnergy)
        .def("getKineticEnergy", &Tendon::getKineticEnergy)
        .def("isEnabled", &Tendon::isEnabled)
        .def("setName", &Tendon::setName, py::arg("name"))
        .def("setProperties", &Tendon::setProperties, py::arg("properties"))
        .def("setDrive", &Tendon::setDrive, py::arg("drive"))
        .def("setTension", &Tendon::setTension, py::arg("tension"))
        .def("setActuationForce", &Tendon::setActuationForce, py::arg("force"))
        .def("setEnabled", &Tendon::setEnabled, py::arg("enabled"))
        .def("getProperties", &Tendon::getProperties, py::rv_policy::copy)
        .def("getDrive", &Tendon::getDrive, py::rv_policy::copy)
        .def("getPath", &Tendon::getPath, py::rv_policy::copy)
        .def("getJoints", &Tendon::getJoints, py::rv_policy::copy)
        .def("getVisualSegments", &Tendon::getVisualSegments, py::rv_policy::copy)
        .def("updateGeometry", &Tendon::updateGeometry, py::arg("visuals") = false);
    auto coupling = py::class_<TendonCoupling>(m, "TendonCoupling");
    py::class_<TendonCoupling::Properties>(coupling, "Properties")
        .def(py::init<>())
        .def_rw("coefficients", &TendonCoupling::Properties::coefficients)
        .def_rw("compliance", &TendonCoupling::Properties::compliance)
        .def_rw("positionCorrection", &TendonCoupling::Properties::positionCorrection)
        .def_rw("enabled", &TendonCoupling::Properties::enabled);
    coupling.def("getName", &TendonCoupling::getName)
        .def("getFirst", &TendonCoupling::getFirst, py::rv_policy::reference_internal)
        .def("getSecond", &TendonCoupling::getSecond, py::rv_policy::reference_internal)
        .def("getProperties", &TendonCoupling::getProperties, py::rv_policy::copy)
        .def("setProperties", &TendonCoupling::setProperties, py::arg("properties"))
        .def("getForce", &TendonCoupling::getForce);
    /* PinConstraint */
    /*****************/
    py::class_<raisim::PinConstraintDefinition>(constraints_module, "PinConstraintDefinition")
        .def(py::init<>())
        .def_rw("body1", &raisim::PinConstraintDefinition::body1)
        .def_rw("body2", &raisim::PinConstraintDefinition::body2)
        .def_prop_rw("anchor",
                      [](const raisim::PinConstraintDefinition &self) {
                        return convert_vec_to_np(self.anchor);
                      }, [](raisim::PinConstraintDefinition &self, NDArray anchor) {
                        self.anchor = convert_np_to_vec<3>(anchor);
                      }, py::for_getter(py::rv_policy::move));

    py::class_<raisim::PinConstraint>(constraints_module, "PinConstraint")
        .def(py::init<size_t, raisim::Vec<3>, size_t, raisim::Vec<3>>(),
             py::arg("local_idx1"), py::arg("pos1_b"), py::arg("local_idx2"), py::arg("pos2_b"))
        .def_prop_rw("pos1_b",
                      [](const raisim::PinConstraint &self) { return convert_vec_to_np(self.pos1_b); },
                      [](raisim::PinConstraint &self, NDArray pos) { self.pos1_b = convert_np_to_vec<3>(pos); }, py::for_getter(py::rv_policy::move))
        .def_prop_rw("pos2_b",
                      [](const raisim::PinConstraint &self) { return convert_vec_to_np(self.pos2_b); },
                      [](raisim::PinConstraint &self, NDArray pos) { self.pos2_b = convert_np_to_vec<3>(pos); }, py::for_getter(py::rv_policy::move))
        .def_prop_rw("pos1_w",
                      [](const raisim::PinConstraint &self) { return convert_vec_to_np(self.pos1_w); },
                      [](raisim::PinConstraint &self, NDArray pos) { self.pos1_w = convert_np_to_vec<3>(pos); }, py::for_getter(py::rv_policy::move))
        .def_prop_rw("pos2_w",
                      [](const raisim::PinConstraint &self) { return convert_vec_to_np(self.pos2_w); },
                      [](raisim::PinConstraint &self, NDArray pos) { self.pos2_w = convert_np_to_vec<3>(pos); }, py::for_getter(py::rv_policy::move))
        .def_rw("localIdx1", &raisim::PinConstraint::localIdx1)
        .def_rw("localIdx2", &raisim::PinConstraint::localIdx2);

}
