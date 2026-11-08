/**
 * Python wrappers for RaiSim using nanobind.
 *
 * Copyright (c) 2019, Brian Delhaisse <briandelhaisse@gmail.com>
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

namespace py = nanobind;


void init_math(py::module_ &);
void init_materials(py::module_ &);
void init_fluid(py::module_ &);
void init_explosion(py::module_ &);
void init_sensors(py::module_ &);
void init_object(py::module_ &);
void init_constraints(py::module_ &);
void init_contact(py::module_ &);
void init_world(py::module_ &);


// The NB_MODULE() macro creates a function that will be called when an import statement is issued from within
// Python. In the following, "raisim" is the module name, "m" is a variable of type py::module which is the main
// interface for creating bindings. The method module::def() generates binding code that exposes the C++ function
// to Python.
NB_MODULE(raisimpy, m) {

    // add meta information
	m.doc() = "Python wrappers for the RaiSim library."; // docstring for the module
	m.attr("__license__") = "PROPRIETARY";
    m.attr("__version__") = RAISIMPY_VERSION;
    m.attr("hasWrenchSensors") = bool(RAISIMPY_HAS_WRENCH_SENSORS);
    m.attr("hasSegmentation") = bool(RAISIMPY_HAS_SEGMENTATION);
	py::list authors;
	authors.append("Jemin Hwangbo (raisim)");
	authors.append("Donho Kang (raisim)");
	authors.append("Joonho Lee (raisim)");
	authors.append("Brian Delhaisse (raisimpy)");
	m.attr("__authors__") = authors;

    /********/
    /* Math */
    /********/
    init_math(m);

	/*************/
	/* Materials */
    /*************/
    init_materials(m);

    /*********/
    /* Fluid */
    /*********/
    init_fluid(m);

    /*************************/
    /* Explosions, projectiles */
    /*************************/
    init_explosion(m);

    /***********/
    /* Sensors */
    /***********/
    init_sensors(m);

    /******************/
	/* raisim.contact */
	/******************/
	init_contact(m);

    /*****************/
	/* raisim.object */
	/*****************/
	init_object(m);  // define primitive shapes and articulated systems)

    /*********************/
	/* raisim.constraint */
	/*********************/
  init_constraints(m);

    /*********/
	/* World */
	/*********/
	init_world(m);
    m.attr("hasRayrai") = bool(RAISIMPY_HAS_RAYRAI);
    auto rayrai = m.def_submodule("rayrai", "In-process Rayrai rendering (loaded on first use).");
    rayrai.attr("available") = bool(RAISIMPY_HAS_RAYRAI);
    rayrai.def("__dir__", []() -> nb::object {
        try {
            return nb::module_::import_("builtins").attr("dir")(nb::module_::import_("raisimpy_rayrai"));
        } catch(nb::python_error& error) {
            if(!error.matches(PyExc_ImportError)) throw;
            nb::list names; names.append("available"); return names;
        }
    });
    rayrai.def("__getattr__", [](const std::string& name) -> nb::object {
        try {
            return nb::module_::import_("raisimpy_rayrai").attr(name.c_str());
        } catch(nb::python_error& error) {
            if(!error.matches(PyExc_ImportError)) throw;
            nb::raise_from(error,PyExc_AttributeError,"Rayrai backend is unavailable: %s",name.c_str());
        }
    });
    m.def("__getattr__", [](const std::string& name) -> nb::object {
        if (name == "RayraiWindow" || name == "RayraiViewer") {
            try {
                return nb::module_::import_("raisimpy_rayrai").attr(name.c_str());
            } catch(nb::python_error& error) {
                if(!error.matches(PyExc_ImportError)) throw;
                nb::raise_from(error,PyExc_AttributeError,"Rayrai backend is unavailable: %s",name.c_str());
            }
        }
        throw nb::attribute_error(("module raisimpy has no attribute " + name).c_str());
    });
}
