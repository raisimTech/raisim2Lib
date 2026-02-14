/**
 * Python wrappers for raisim.Materials using nanobind.
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

#include <iostream>

#include "raisim/Materials.hpp"


namespace py = nanobind;
using namespace raisim;


void init_materials(py::module_ &m) {


	/**************************/
	/* MaterialPairProperties */
    /**************************/
    py::class_<raisim::MaterialPairProperties>(m, "MaterialPairProperties", "Raisim Material Pair Properties (friction and restitution).")
        .def(py::init<>(), "Initialize the material pair properties.")
        .def(py::init<double, double, double>(),
             "Initialize the material pair properties.\n\n"
             "Args:\n"
             "    friction (float): coefficient of friction.\n"
             "    restitution (float): coefficient of restitution.\n"
             "    threshold (float): restitution threshold.",
             py::arg("friction"), py::arg("restitution"), py::arg("restitution_threshold"))

        .def(py::init<double, double, double, double, double>(),
        "Initialize the material pair properties.\n\n"
        "Args:\n"
        "    friction (float): coefficient of friction.\n"
        "    restitution (float): coefficient of restitution.\n"
        "    threshold (float): restitution threshold.\n"
        "    static_friction (float): coefficient of static friction.\n"
        "    static_friction_velocity_threshold (float): If the relative velocity of two contact points is bigger than this value, then the dynamic coefficient of friction is applied. Otherwise, the coefficient of friction is interpolated between the static and dynamic one proportional to the relative velocity.\n",
        py::arg("friction"), py::arg("restitution"), py::arg("restitution_threshold"), py::arg("static_friction"), py::arg("static_friction_velocity_threshold"))
        .def_rw("c_f", &raisim::MaterialPairProperties::c_f)
        .def_rw("c_r", &raisim::MaterialPairProperties::c_r)
        .def_rw("r_th", &raisim::MaterialPairProperties::r_th)
        .def_rw("c_static_f", &raisim::MaterialPairProperties::c_static_f)
        .def_rw("v_static_speed", &raisim::MaterialPairProperties::v_static_speed)
        .def_rw("v_static_speed_inv", &raisim::MaterialPairProperties::v_static_speed_inv);

    /*******************/
    /* MaterialManager */
    /*******************/
    py::class_<raisim::MaterialManager>(m, "MaterialManager", "Raisim Material Manager.")
        .def(py::init<>(), "Initialize the material pair manager.")
        .def(py::init<const std::string>(),
        "Initialize the material manager by uploading the material data from a file.\n\n"
        "Args:\n"
        "    xml_file (float): xml file.",
        py::arg("xml_file"))

        .def("setMaterialPairProp",
             py::overload_cast<const std::string &, const std::string &, double, double, double, double, double>(&raisim::MaterialManager::setMaterialPairProp), R"mydelimiter(
        Set the material pair properties (friction and restitution).

        Args:
            material1 (str): first material.
            material2 (str): second material.
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            restitution_threshold (float): restitution threshold.
            static friction (float): coefficient of static friction
            static friction velocity threshold (float): If the relative velocity of two contact points is bigger than this value, then the dynamic coefficient of friction is applied. Otherwise, the coefficient of friction is interpolated between the static and dynamic one proportional to the relative velocity.

        )mydelimiter",
        py::arg("material1"), py::arg("material2"), py::arg("friction"), py::arg("restitution"), py::arg("restitution_threshold"), py::arg("static_friction"), py::arg("static_friction_velocity_threshold"))

        .def("setMaterialPairProp",
             py::overload_cast<const std::string &, const std::string &, double, double, double>(&raisim::MaterialManager::setMaterialPairProp), R"mydelimiter(
        Set the material pair properties (friction and restitution).

        Args:
            material1 (str): first material.
            material2 (str): second material.
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            restitution_threshold (float): restitution threshold.

             )mydelimiter",
             py::arg("material1"), py::arg("material2"), py::arg("friction"), py::arg("restitution"), py::arg("restitution_threshold"))


        .def("getMaterialPairProp", py::overload_cast<const std::string &, const std::string &>(&raisim::MaterialManager::getMaterialPairProp, py::const_), R"mydelimiter(
        Get the material pair properties (friction and restitution).

        Args:
            material1 (str): first material.
            material2 (str): second material.

        Returns:
            MaterialPairProperties: material pair properties (friction, restitution, and restitution threshold).
        )mydelimiter",
        py::arg("material1"), py::arg("material2"))
        .def("setDefaultMaterialProperties", &raisim::MaterialManager::setDefaultMaterialProperties, R"mydelimiter(
        Set the default material properties.

        Args:
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            threshold (float): restitution threshold.
            static friction (float): coefficient of static friction.
            static friction velocity threshold (float): If the relative velocity of two contact points is bigger than this value, then the dynamic coefficient of friction is applied. Otherwise, the coefficient of friction is interpolated between the static and dynamic one proportional to the relative velocity.
        )mydelimiter",
        py::arg("friction"), py::arg("restitution"), py::arg("restitution_threshold"), py::arg("static_friction"), py::arg("static_friction_velocity_threshold"))
        .def("getMaterialPairProp",
             py::overload_cast<unsigned int, unsigned int>(&raisim::MaterialManager::getMaterialPairProp, py::const_),
             py::arg("material1"), py::arg("material2"))
        .def("getMaterialIdOrDefault", &raisim::MaterialManager::getMaterialIdOrDefault, py::arg("material_name"))
        .def("getDefaultMaterialId", &raisim::MaterialManager::getDefaultMaterialId);

}
