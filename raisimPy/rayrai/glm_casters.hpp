#pragma once

#include "nanobind_helpers.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <nanobind/stl/array.h>

namespace nanobind::detail {

template <typename T>
handle owned_array(T *data, std::initializer_list<size_t> shape) {
  std::unique_ptr<T[]> allocation(data);
  capsule owner(
      data, [](void *pointer) noexcept { delete[] static_cast<T *>(pointer); });
  allocation.release();
  return ndarray<numpy, T>(data, shape, owner).cast(rv_policy::move).release();
}

template <glm::length_t N, typename T, glm::qualifier Q>
struct type_caster<glm::vec<N, T, Q>> {
  using Native = glm::vec<N, T, Q>;
  NB_TYPE_CASTER(Native, const_name("numpy.ndarray"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    make_caster<std::array<T, N>> caster;
    if (!caster.from_python(src, flags, cleanup))
      return false;
    const auto &values = caster.operator cast_t<std::array<T, N>>();
    for (glm::length_t i = 0; i < N; ++i)
      value[i] = values[i];
    return true;
  }
  static handle from_cpp(const Value &vector, rv_policy, cleanup_list *) {
    auto *data = new T[N];
    for (glm::length_t i = 0; i < N; ++i)
      data[i] = vector[i];
    return owned_array(data, {N});
  }
};

template <glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
struct type_caster<glm::mat<C, R, T, Q>> {
  using Native = glm::mat<C, R, T, Q>;
  NB_TYPE_CASTER(Native, const_name("numpy.ndarray"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    make_caster<std::array<std::array<T, C>, R>> caster;
    if (!caster.from_python(src, flags, cleanup))
      return false;
    const auto &values =
        caster.operator cast_t<std::array<std::array<T, C>, R>>();
    for (glm::length_t row = 0; row < R; ++row)
      for (glm::length_t col = 0; col < C; ++col)
        value[col][row] = values[row][col];
    return true;
  }
  static handle from_cpp(const Value &matrix, rv_policy, cleanup_list *) {
    auto *data = new T[C * R];
    for (glm::length_t row = 0; row < R; ++row)
      for (glm::length_t col = 0; col < C; ++col)
        data[row * C + col] = matrix[col][row];
    return owned_array(data, {R, C});
  }
};

template <typename T, glm::qualifier Q> struct type_caster<glm::qua<T, Q>> {
  using Native = glm::qua<T, Q>;
  NB_TYPE_CASTER(Native, const_name("numpy.ndarray"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    make_caster<std::array<T, 4>> caster;
    if (!caster.from_python(src, flags, cleanup))
      return false;
    const auto &v = caster.operator cast_t<std::array<T, 4>>();
    value = Value(v[0], v[1], v[2], v[3]);
    return true;
  }
  static handle from_cpp(const Value &q, rv_policy, cleanup_list *) {
    return owned_array(new T[4]{q.w, q.x, q.y, q.z}, {4});
  }
};

} // namespace nanobind::detail
