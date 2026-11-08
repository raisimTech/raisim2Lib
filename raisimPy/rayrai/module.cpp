#include "runtime.hpp"
NB_MODULE(raisimpy_rayrai, m) {
  nanobind::module_::import_("raisimpy");
  // Generated GLM defaults require NumPy. Report a normal dependency error
  // before type conversion starts; missing NumPy must never terminate Python.
  nanobind::module_::import_("numpy");
  rayrai_py::bind_runtime(m);
  rayrai_py::declare_generated(m);
  rayrai_py::define_generated(m);
  rayrai_py::bind_manual(m);
  nanobind::module_::import_("atexit").attr("register")(
      nanobind::cpp_function(&rayrai_py::close_all));
}
