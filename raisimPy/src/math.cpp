/**
 * Python wrappers for raisim.math using nanobind. Have also a look at
 * `converter.hpp` and `converter.cpp` which contains the code to convert between
 * np.array to raisim::Vec, raisim::Mat, raisim::VecDyn, raisim::MatDyn, and
 * raisim::Transformation.
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


#include "raisim/math.hpp"   // contains the definitions of Vec, Mat, Transformation, etc.
#include "raisim/helper.hpp"

#include "converter.hpp"  // contains code that allows to convert between the Vec, Mat to numpy arrays.

namespace py = nanobind;
using namespace raisim;


void init_math(py::module_ &m) {
    /******************/
    /* Transformation */
    /******************/
    py::class_<raisim::Transformation>(m, "Transformation", "Raisim homogeneous transformation.")
        .def(py::init<>())  // default constructor
        .def_prop_rw("rot",
            [](raisim::Transformation &self) { // getter
                // convert from Mat<3,3> to np.array[3,3]
                return convert_mat_to_np(self.rot);
            }, [](raisim::Transformation &self, const NDArray &array) { // setter
                // convert from np.array[3,3] to Mat<3,3>
                Mat<3,3> rot = convert_np_to_mat<3,3>(array);
                self.rot = rot;
            })
        .def_prop_rw("pos",
            [](raisim::Transformation &self) { // getter
                // convert from Vec<3> to np.array[3]
                return convert_vec_to_np(self.pos);
            }, [](raisim::Transformation &self, const NDArray &array) { // setter
                // convert from np.array[3] to Vec<3>
                Vec<3> pos = convert_np_to_vec<3>(array);
                self.pos = pos;
            });


    /******************/
    /* SparseJacobian */
    /******************/

    py::class_<raisim::SparseJacobian>(m, "SparseJacobian", "Raisim sparse Jacobian (3 x n).")
        .def(py::init<>(), "Initialize an empty sparse Jacobian.")
        .def("resize", &raisim::SparseJacobian::resize, "Resize the Jacobian.", py::arg("cols"))
        .def_prop_ro("size", [](const raisim::SparseJacobian &self) { return self.size; })
        .def_prop_ro("capacity", [](const raisim::SparseJacobian &self) { return self.capacity; })
        .def_rw("idx", &raisim::SparseJacobian::idx)
        .def("toDense", [](const raisim::SparseJacobian &self) {
            NDArray array = make_ndarray_2d(3, self.size);
            for (size_t j = 0; j < self.size; ++j)
                for (size_t i = 0; i < 3; ++i)
                    *ndarray_mutable_data(array, i, j) = self(i, j);
            return array;
        }, "Return a dense (3 x n) numpy array.");

    /************/
    /* ColorRGB */
    /************/
    py::class_<raisim::ColorRGB>(m, "ColorRGB", "RGB color with 8-bit channels.")
        .def(py::init<>())
        .def_rw("r", &raisim::ColorRGB::r)
        .def_rw("g", &raisim::ColorRGB::g)
        .def_rw("b", &raisim::ColorRGB::b);

    /*************/
    /* ColorRGBA */
    /*************/
    py::class_<raisim::ColorRGBA>(m, "ColorRGBA", "RGBA color with 8-bit channels.")
        .def(py::init<>())
        .def_rw("r", &raisim::ColorRGBA::r)
        .def_rw("g", &raisim::ColorRGBA::g)
        .def_rw("b", &raisim::ColorRGBA::b)
        .def_rw("a", &raisim::ColorRGBA::a);

}
