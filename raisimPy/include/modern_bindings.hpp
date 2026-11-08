#pragma once

#include "converter.hpp"
#include <nanobind/stl/array.h>
#include "raisim/World.hpp"

void init_articulated_extensions(nanobind::module_& m,
    nanobind::class_<raisim::ArticulatedSystem, raisim::Object>& system);
void init_particle_objects(nanobind::module_& m);
void init_particle_world(nanobind::class_<raisim::World>& world);
void init_world_extensions(nanobind::module_& m, nanobind::class_<raisim::World>& world);

template<size_t n, typename T, typename... Bases>
void bind_vector_property(nanobind::class_<T, Bases...>& cls, const char* name,
                          raisim::Vec<n> T::*member) {
    cls.def_prop_rw(name,
        [member](const T& self) { return convert_vec_to_np(self.*member); },
        [member](T& self, const NDArray& value) { self.*member = convert_np_to_vec<n>(value); },
        nanobind::for_getter(nanobind::rv_policy::move));
}
