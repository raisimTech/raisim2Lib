/**
 * Python wrappers for raisim.contact using nanobind.
 *
 * Copyright (c) 2019, kangd and jhwangbo (C++), Brian Delhaisse <briandelhaisse@gmail.com> (Python wrappers)
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
#include "raisim/contact/Contact.hpp"
#include "raisim/contact/BisectionContactSolver.hpp"
#include "raisim/contact_engine/math.h"
#include "raisim/object/Object.hpp"

#include "converter.hpp"  // contains code that allows to convert between the Vec, Mat to numpy arrays.

namespace py = nanobind;
using namespace raisim;


void init_contact(py::module_ &m) {


    // create submodule
    py::module_ contact_module = m.def_submodule("contact", "RaiSim contact submodule.");

    /*****************/
    /* Contact class */
    /*****************/
    py::class_<raisim::Contact>(contact_module, "Contact", "Raisim Contact.")

        .def("getPosition", [](raisim::Contact &self) {
            Vec<3> position = self.getPosition();
            return convert_vec_to_np(position);
        }, R"mydelimiter(
	    Get the contact position.

	    Returns:
	        np.array[float[3]]: contact position in the world.
	    )mydelimiter")
        .def("get_position", [](raisim::Contact &self) {
            Vec<3> position = self.getPosition();
            return convert_vec_to_np(position);
        })


        .def("getNormal", [](raisim::Contact &self) {
            Vec<3> normal = self.getNormal();
            return convert_vec_to_np(normal);
        }, R"mydelimiter(
	    Get the contact normal.

	    Returns:
	        np.array[float[3]]: contact normal in the world.
	    )mydelimiter")


        .def("getContactFrame", [](raisim::Contact &self) {
            Mat<3, 3> frame = self.getContactFrame();
            return convert_mat_to_np(frame);
        }, R"mydelimiter(
	    Get the contact frame.

	    Returns:
	        np.array[float[3, 3]]: contact frame.
	    )mydelimiter")


	    .def("getIndexContactProblem", &raisim::Contact::getIndexContactProblem, R"mydelimiter(
	    Get the index contact problem.

	    Returns:
	        int: index.
	    )mydelimiter")


	    .def("getPairObjectIndex", &raisim::Contact::getPairObjectIndex, R"mydelimiter(
	    Get the pair object index.

	    Returns:
	        int: index.
	    )mydelimiter")


	    .def("getPairContactIndexInPairObject", &raisim::Contact::getPairContactIndexInPairObject, R"mydelimiter(
	    Get the pair contact index in pair objects.

	    Returns:
	        int: index.
	    )mydelimiter")

        .def("getIndexInObjectContactList", &raisim::Contact::getIndexInObjectContactList, R"mydelimiter(
	    Get the index of this contact in the object's contact list.

	    Returns:
	        int: index.
	    )mydelimiter")


	    .def("getImpulse", [](raisim::Contact &self) {
            Vec<3> impulse = self.getImpulse();
            return convert_vec_to_np(impulse);
        }, R"mydelimiter(
	    Get the impulse.

	    Returns:
	        np.array[float[3]]: impulse.
	    )mydelimiter")

        .def("getInvInertia", [](raisim::Contact &self) {
            Mat<3, 3> inv_inertia = self.getInvInertia();
            return convert_mat_to_np(inv_inertia);
        }, R"mydelimiter(
	    Get the inverse apparent inertia.

	    Returns:
	        np.array[float[3,3]]: inverse apparent inertia.
	    )mydelimiter")


	    .def("isObjectA", &raisim::Contact::isObjectA, R"mydelimiter(
	    Check if it is object A.

	    Returns:
	        bool: True if object A is in contact.
	    )mydelimiter")

	    .def("getPairObjectBodyType", &raisim::Contact::getPairObjectBodyType, R"mydelimiter(
	    Get the pair object body type.

	    Returns:
	        raisim.BodyType: the body type (STATIC, KINEMATIC, DYNAMIC)
	    )mydelimiter")

	    .def("getlocalBodyIndex", &raisim::Contact::getlocalBodyIndex, R"mydelimiter(
	    Get local body index.

	    Returns:
	        int: local body index.
	    )mydelimiter")


        .def("getDepth", &raisim::Contact::getDepth, R"mydelimiter(
	    Get the depth.

	    Returns:
	        float: depth.
	    )mydelimiter")


        .def("isSelfCollision", &raisim::Contact::isSelfCollision, R"mydelimiter(
	    Return True if self-collision is enabled.

	    Returns:
	        bool: True if self-collision is enabled.
	    )mydelimiter")


	    .def("skip", &raisim::Contact::skip, R"mydelimiter(
	    Return True if we contact is skipped.

	    Returns:
	        bool: True if the contact is skipped.
	    )mydelimiter")

        .def("getCollisionBodyA", &raisim::Contact::getCollisionBodyA, py::rv_policy::reference_internal)
        .def("getCollisionBodyB", &raisim::Contact::getCollisionBodyB, py::rv_policy::reference_internal);

    /*********/
    /* AABB */
    /*********/
    py::class_<::contact::AABB>(contact_module, "AABB", "Contact engine axis-aligned bounding box.")
        .def(py::init<>())
        .def_prop_rw("min",
                      [](const ::contact::AABB &self) { return convert_vec_to_np(self.min); },
                      [](::contact::AABB &self, NDArray value) { self.min = convert_np_to_vec<3>(value); })
        .def_prop_rw("max",
                      [](const ::contact::AABB &self) { return convert_vec_to_np(self.max); },
                      [](::contact::AABB &self, NDArray value) { self.max = convert_np_to_vec<3>(value); })
        .def("overlaps", &::contact::AABB::overlaps, py::arg("other"));

    /****************************/
    /* Single3DContactProblem   */
    /****************************/
    py::class_<raisim::contact::Single3DContactProblem>(contact_module, "Single3DContactProblem")
        .def(py::init<>())
        .def_prop_rw("imp_i",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_vec_to_np(self.imp_i); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.imp_i = convert_np_to_vec<3>(value); })
        .def_prop_rw("velInit",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_vec_to_np(self.velInit); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.velInit = convert_np_to_vec<3>(value); })
        .def_prop_rw("MappInv_i",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_mat_to_np(self.MappInv_i); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.MappInv_i = convert_np_to_mat<3, 3>(value); })
        .def_prop_rw("Mapp_i",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_mat_to_np(self.Mapp_i); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.Mapp_i = convert_np_to_mat<3, 3>(value); })
        .def_prop_rw("Mapp_iInv22",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_mat_to_np(self.Mapp_iInv22); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.Mapp_iInv22 = convert_np_to_mat<2, 2>(value); })
        .def_prop_rw("Mapp_i22",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_mat_to_np(self.Mapp_i22); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.Mapp_i22 = convert_np_to_mat<2, 2>(value); })
        .def_prop_rw("basisMat",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_mat_to_np(self.basisMat); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.basisMat = convert_np_to_mat<3, 2>(value); })
        .def_prop_rw("axis",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_vec_to_np(self.axis); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.axis = convert_np_to_vec<3>(value); })
        .def_prop_rw("MappInv_red",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_mat_to_np(self.MappInv_red); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.MappInv_red = convert_np_to_mat<3, 2>(value); })
        .def_rw("mu", &raisim::contact::Single3DContactProblem::mu)
        .def_rw("n2_mu", &raisim::contact::Single3DContactProblem::n2_mu)
        .def_rw("muinv", &raisim::contact::Single3DContactProblem::muinv)
        .def_rw("coeffRes", &raisim::contact::Single3DContactProblem::coeffRes)
        .def_rw("bounceThres", &raisim::contact::Single3DContactProblem::bounceThres)
        .def_rw("bouceVel", &raisim::contact::Single3DContactProblem::bouceVel)
        .def_rw("Mapp_iInv11", &raisim::contact::Single3DContactProblem::Mapp_iInv11)
        .def_rw("impact_vel", &raisim::contact::Single3DContactProblem::impact_vel)
        .def_rw("depth", &raisim::contact::Single3DContactProblem::depth)
        .def_rw("tempE", &raisim::contact::Single3DContactProblem::tempE)
        .def_rw("rank", &raisim::contact::Single3DContactProblem::rank)
        .def_rw("pointIdA", &raisim::contact::Single3DContactProblem::pointIdA)
        .def_rw("jointId", &raisim::contact::Single3DContactProblem::jointId)
        .def_rw("pointIdB", &raisim::contact::Single3DContactProblem::pointIdB)
        .def_prop_rw("position_W",
                      [](const raisim::contact::Single3DContactProblem &self) { return convert_vec_to_np(self.position_W); },
                      [](raisim::contact::Single3DContactProblem &self, NDArray value) { self.position_W = convert_np_to_vec<3>(value); })
        .def_rw("mu_static", &raisim::contact::Single3DContactProblem::mu_static)
        .def_rw("mu_static_vel_thresh", &raisim::contact::Single3DContactProblem::mu_static_vel_thresh)
        .def_rw("mu_static_vel_thresh_inv", &raisim::contact::Single3DContactProblem::mu_static_vel_thresh_inv)
        .def_prop_ro("obA", [](const raisim::contact::Single3DContactProblem &self) { return self.obA; },
                               py::rv_policy::reference_internal)
        .def_prop_ro("obB", [](const raisim::contact::Single3DContactProblem &self) { return self.obB; },
                               py::rv_policy::reference_internal);

    /****************************/
    /* BisectionContactSolver   */
    /****************************/
    py::class_<raisim::contact::BisectionContactSolver::SolverConfiguration>(contact_module, "SolverConfiguration")
        .def(py::init<>())
        .def_rw("alpha_init", &raisim::contact::BisectionContactSolver::SolverConfiguration::alpha_init)
        .def_rw("alpha_low", &raisim::contact::BisectionContactSolver::SolverConfiguration::alpha_low)
        .def_rw("alpha_decay", &raisim::contact::BisectionContactSolver::SolverConfiguration::alpha_decay)
        .def_rw("error_to_terminate", &raisim::contact::BisectionContactSolver::SolverConfiguration::error_to_terminate)
        .def_rw("erp", &raisim::contact::BisectionContactSolver::SolverConfiguration::erp)
        .def_rw("erp2", &raisim::contact::BisectionContactSolver::SolverConfiguration::erp2)
        .def_rw("maxIteration", &raisim::contact::BisectionContactSolver::SolverConfiguration::maxIteration);

    py::class_<raisim::contact::BisectionContactSolver>(contact_module, "BisectionContactSolver")
        .def(py::init<>())
        .def("solve", [](raisim::contact::BisectionContactSolver &self,
                         const std::vector<raisim::contact::Single3DContactProblem> &problems) {
            raisim::contact::ContactProblems contact_problems;
            contact_problems.reserve(problems.size());
            for (const auto &problem : problems)
                contact_problems.push_back(problem);
            self.solve(contact_problems);
            std::vector<raisim::contact::Single3DContactProblem> out(contact_problems.begin(), contact_problems.end());
            return out;
        }, py::arg("contact_problems"))
        .def("updateConfig", &raisim::contact::BisectionContactSolver::updateConfig, py::arg("config"))
        .def("setTimestep", &raisim::contact::BisectionContactSolver::setTimestep, py::arg("dt"))
        .def("setOrder", &raisim::contact::BisectionContactSolver::setOrder, py::arg("order"))
        .def("getLoopCounter", &raisim::contact::BisectionContactSolver::getLoopCounter)
        .def("getConfig", py::overload_cast<>(&raisim::contact::BisectionContactSolver::getConfig),
             py::rv_policy::reference_internal);
}
