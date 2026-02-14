#ifndef RAISIMPY_NANOBIND_HELPERS_HPP
#define RAISIMPY_NANOBIND_HELPERS_HPP

#include <cstddef>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>

#if defined(__has_include)
#  if __has_include(<nanobind/stl/functional.h>)
#    include <nanobind/stl/functional.h>
#  elif __has_include(<nanobind/stl/function.h>)
#    include <nanobind/stl/function.h>
#  endif
#else
#  include <nanobind/stl/function.h>
#endif
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using NDArray = nb::ndarray<nb::numpy, double, nb::c_contig>;

inline const double *ndarray_data(const NDArray &array, size_t i) {
    return array.data() + static_cast<size_t>(array.stride(0)) * i;
}

inline const double *ndarray_data(const NDArray &array, size_t i, size_t j) {
    return array.data() + static_cast<size_t>(array.stride(0)) * i
                        + static_cast<size_t>(array.stride(1)) * j;
}

inline double *ndarray_mutable_data(NDArray &array, size_t i) {
    return array.data() + static_cast<size_t>(array.stride(0)) * i;
}

inline double *ndarray_mutable_data(NDArray &array, size_t i, size_t j) {
    return array.data() + static_cast<size_t>(array.stride(0)) * i
                        + static_cast<size_t>(array.stride(1)) * j;
}

inline NDArray make_ndarray_1d(size_t size) {
    auto *data = new double[size];
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] static_cast<double *>(p);
    });
    return NDArray(data, {size}, owner);
}

inline NDArray make_ndarray_2d(size_t rows, size_t cols) {
    auto *data = new double[rows * cols];
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] static_cast<double *>(p);
    });
    return NDArray(data, {rows, cols}, owner);
}

#endif
