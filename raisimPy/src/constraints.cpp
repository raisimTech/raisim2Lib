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

#include "raisim/math.hpp"   // contains the definitions of Vec, Mat, etc.
#include "raisim/object/Object.hpp"
#include "raisim/constraints/Constraints.hpp"
#include "raisim/constraints/LengthConstraint.hpp"
#include "raisim/constraints/StiffLengthConstraint.hpp"
#include "raisim/constraints/CompliantLengthConstraint.hpp"
#include "raisim/constraints/CustomLengthConstraint.hpp"
#include "raisim/constraints/PinConstraint.hpp"
#include "raisim/contact/BisectionContactSolver.hpp"

#include "converter.hpp"  // contains code that allows to convert between the Vec, Mat to numpy arrays.

namespace py = nanobind;
using namespace raisim;


void init_constraints(py::module_ &m) {


    // create submodule
    py::module_ constraints_module = m.def_submodule("constraints", "RaiSim contact submodule.");

    py::enum_<raisim::LengthConstraint::StretchType>(m, "StretchType", py::is_arithmetic())
        .value("STRETCH_RESISTANT_ONLY", raisim::LengthConstraint::StretchType::STRETCH_RESISTANT_ONLY)
        .value("COMPRESSION_RESISTANT_ONLY", raisim::LengthConstraint::StretchType::COMPRESSION_RESISTANT_ONLY)
        .value("BOTH", raisim::LengthConstraint::StretchType::BOTH);
    py::enum_<raisim::LengthConstraint::WireType>(m, "WireType", py::is_arithmetic())
        .value("STIFF", raisim::LengthConstraint::WireType::STIFF)
        .value("COMPLIANT", raisim::LengthConstraint::WireType::COMPLIANT)
        .value("CUSTOM", raisim::LengthConstraint::WireType::CUSTOM);
    /**************/
    /* Constraint */
    /**************/
    py::class_<raisim::Constraints>(constraints_module, "Constraints", "Raisim Constraints from which all other constraints inherit from.")
        .def("getColor", [](const raisim::Constraints &self) {
            return convert_vec_to_np(self.getColor());
        })
        .def("setColor", [](raisim::Constraints &self, NDArray color) {
            self.setColor(convert_np_to_vec<4>(color));
        }, py::arg("color"))
        .def("lockMutex", &raisim::Constraints::lockMutex)
        .def("unlockMutex", &raisim::Constraints::unlockMutex)
        .def("lock", &raisim::Constraints::lock)
        .def("unlock", &raisim::Constraints::unlock);


    /********/
    /* Wire */
    /********/
    py::class_<raisim::LengthConstraint, raisim::Constraints>(constraints_module, "LengthConstraint", "Raisim LengthConstraint constraint class; it creates a LengthConstraint constraint between 2 bodies.")

        .def("update", [](raisim::LengthConstraint &self,
                          const std::vector<raisim::contact::Single3DContactProblem> &problems) {
            raisim::contact::ContactProblems contact_problems;
            contact_problems.reserve(problems.size());
            for (const auto &problem : problems)
                contact_problems.push_back(problem);
            self.update(contact_problems);
            std::vector<raisim::contact::Single3DContactProblem> out(contact_problems.begin(), contact_problems.end());
            return out;
        }, "update internal variables (called by `integrate1()`).",
             py::arg("contact_problems"))


        .def("getLength", &raisim::LengthConstraint::getLength, R"mydelimiter(
	    Get the length of the LengthConstraint constraint.

	    Returns:
	        float: length of the LengthConstraint constraint.
	    )mydelimiter")
        .def("getVisualizationWidth", &raisim::LengthConstraint::getVisualizationWidth)
        .def("setVisualizationWidth", &raisim::LengthConstraint::setVisualizationWidth, py::arg("width"))
        .def("getDistance", &raisim::LengthConstraint::getDistance)


        .def("getP1", [](raisim::LengthConstraint &self) {
            Vec<3> p1 = self.getP1();
            return convert_vec_to_np(p1);
        }, R"mydelimiter(
	    Return the first attachment point in the World frame.

	    Returns:
	        np.array[float[3]]: first point position expressed in the world frame.
	    )mydelimiter")


	    .def("getP2", [](raisim::LengthConstraint &self) {
            Vec<3> p2 = self.getP2();
            return convert_vec_to_np(p2);
        }, R"mydelimiter(
	    Return the second attachment point in the World frame.

	    Returns:
	        np.array[float[3]]: second point position expressed in the world frame.
	    )mydelimiter")


        .def("getBody1", &raisim::LengthConstraint::getBody1, R"mydelimiter(
	    Return the first object to which the LengthConstraint is attached.

	    Returns:
	        Object: first object.
	    )mydelimiter")


        .def("getBody2", &raisim::LengthConstraint::getBody2, R"mydelimiter(
	    Return the second object to which the LengthConstraint is attached.

	    Returns:
	        Object: second object.
	    )mydelimiter")


        .def("getNorm", [](raisim::LengthConstraint &self) {
            Vec<3> normal = self.getNorm();
            return convert_vec_to_np(normal);
        }, R"mydelimiter(
	    Return the direction of the normal (i.e., p2-p1 normalized)

	    Returns:
	        np.array[float[3]]: direction of the normal.
	    )mydelimiter")


        .def("getLocalIdx1", &raisim::LengthConstraint::getLocalIdx1, R"mydelimiter(
	    Return the local index of object1.

	    Returns:
	        int: local index of object1.
	    )mydelimiter")


        .def("getLocalIdx2", &raisim::LengthConstraint::getLocalIdx2, R"mydelimiter(
	    Return the local index of object2.

	    Returns:
	        int: local index of object2.
	    )mydelimiter")


        .def("getStretch", &raisim::LengthConstraint::getStretch, R"mydelimiter(
	    Return the stretch length (i.e., constraint violation).

	    Returns:
	        float: stretch length.
	    )mydelimiter")

        .def("setStretchType", &raisim::LengthConstraint::setStretchType, R"mydelimiter(
	    Return the stretch type (i.e., constraint violation).

	    Returns:
	        stretch_type: stretch type.
	    )mydelimiter")
        .def("getStretchType", &raisim::LengthConstraint::getStretchType)
        .def("getWireType", &raisim::LengthConstraint::getWireType)
        .def("getOb1MountPos", [](const raisim::LengthConstraint &self) {
            return convert_vec_to_np(self.getOb1MountPos());
        })
        .def("getOb2MountPos", [](const raisim::LengthConstraint &self) {
            return convert_vec_to_np(self.getOb2MountPos());
        })

        .def_prop_rw("name", &raisim::LengthConstraint::getName, &raisim::LengthConstraint::setName)
	    .def("getName", &raisim::LengthConstraint::getName, "Get the LengthConstraint constraint's name.")
	    .def("setName", &raisim::LengthConstraint::setName, "Set the LengthConstraint constraint's name.", py::arg("name"))
	    .def_rw("isActive", &raisim::LengthConstraint::isActive)
    ;


    /*************/
    /* StiffLengthConstraint */
    /*************/

    py::class_<raisim::StiffLengthConstraint, raisim::LengthConstraint>(constraints_module, "StiffLengthConstraint", "Raisim StiffLengthConstraint constraint class; it creates a stiff wire constraint between 2 bodies.");


    /*************/
    /* CustomLengthConstraint */
    /*************/
    py::class_<raisim::CustomLengthConstraint, raisim::LengthConstraint>(constraints_module, "CustomLengthConstraint", "Raisim CustomLengthConstraint class; it creates a stiff wire constraint between 2 bodies.")
        .def("setTension", &raisim::CustomLengthConstraint::setTension, "Set the tension in the wire.\n"
                                                                        "Args:\n"
                                                                        "   tension (float): tension in the wire", py::arg("tension"));

  /*****************/
    /* CompliantLengthConstraint */
    /*****************/

    py::class_<raisim::CompliantLengthConstraint, raisim::LengthConstraint>(constraints_module, "CompliantLengthConstraint", "Raisim Compliant Wire constraint class; it creates a compliant wire constraint between 2 bodies.")
        .def("getStiffness", &raisim::CompliantLengthConstraint::getStiffness)
        .def("getPotentialEnergy", &raisim::CompliantLengthConstraint::getPotentialEnergy)
        .def("getTension", [](const raisim::CompliantLengthConstraint &self) {
            return convert_vec_to_np(self.getTension());
        });

    /*****************/
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
                      });

    py::class_<raisim::PinConstraint>(constraints_module, "PinConstraint")
        .def(py::init<size_t, raisim::Vec<3>, size_t, raisim::Vec<3>>(),
             py::arg("local_idx1"), py::arg("pos1_b"), py::arg("local_idx2"), py::arg("pos2_b"))
        .def_prop_rw("pos1_b",
                      [](const raisim::PinConstraint &self) { return convert_vec_to_np(self.pos1_b); },
                      [](raisim::PinConstraint &self, NDArray pos) { self.pos1_b = convert_np_to_vec<3>(pos); })
        .def_prop_rw("pos2_b",
                      [](const raisim::PinConstraint &self) { return convert_vec_to_np(self.pos2_b); },
                      [](raisim::PinConstraint &self, NDArray pos) { self.pos2_b = convert_np_to_vec<3>(pos); })
        .def_prop_rw("pos1_w",
                      [](const raisim::PinConstraint &self) { return convert_vec_to_np(self.pos1_w); },
                      [](raisim::PinConstraint &self, NDArray pos) { self.pos1_w = convert_np_to_vec<3>(pos); })
        .def_prop_rw("pos2_w",
                      [](const raisim::PinConstraint &self) { return convert_vec_to_np(self.pos2_w); },
                      [](raisim::PinConstraint &self, NDArray pos) { self.pos2_w = convert_np_to_vec<3>(pos); })
        .def_rw("localIdx1", &raisim::PinConstraint::localIdx1)
        .def_rw("localIdx2", &raisim::PinConstraint::localIdx2);

}
