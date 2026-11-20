#include "nanobind_helpers.hpp"
#if RAISIMPY_HAS_VOXEL_MAP
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include "converter.hpp"
#include "raisim/World.hpp"

namespace py = nanobind;
using raisim::CollisionGroup;

namespace {

using KeyArray = py::ndarray<py::numpy, int32_t, py::c_contig>;
using ArrayInput = py::ndarray<py::ro, py::c_contig, py::device::cpu>;

template <typename T>
T elementAt(const ArrayInput& array, size_t index) {
  T value;
  std::memcpy(&value, static_cast<const unsigned char*>(array.data()) + index * sizeof(T), sizeof(T));
  return value;
}

bool isInteger(const py::dlpack::dtype& type) {
  return type.lanes == 1 &&
         (type.code == uint8_t(py::dlpack::dtype_code::Int) || type.code == uint8_t(py::dlpack::dtype_code::UInt)) &&
         (type.bits == 8 || type.bits == 16 || type.bits == 32 || type.bits == 64);
}

bool isOccupancy(const py::dlpack::dtype& type) {
  return isInteger(type) || (type.lanes == 1 && type.code == uint8_t(py::dlpack::dtype_code::Bool) && type.bits == 8) ||
         (type.lanes == 1 && type.code == uint8_t(py::dlpack::dtype_code::Float) && (type.bits == 32 || type.bits == 64));
}

// Every element is read in its own type, without numpy's narrowing conversions.
std::int64_t integerAt(const ArrayInput& array, size_t index) {
  const auto type = array.dtype();
  if (type.lanes == 1 && type.code == uint8_t(py::dlpack::dtype_code::Int)) {
    switch (type.bits) {
      case 8: return elementAt<std::int8_t>(array, index);
      case 16: return elementAt<std::int16_t>(array, index);
      case 32: return elementAt<std::int32_t>(array, index);
      case 64: return elementAt<std::int64_t>(array, index);
      default: break;
    }
  }
  if (type.lanes == 1 && type.code == uint8_t(py::dlpack::dtype_code::UInt)) {
    switch (type.bits) {
      case 8: return elementAt<std::uint8_t>(array, index);
      case 16: return elementAt<std::uint16_t>(array, index);
      case 32: return elementAt<std::uint32_t>(array, index);
      case 64: {
        const std::uint64_t value = elementAt<std::uint64_t>(array, index);
        return value > std::uint64_t(std::numeric_limits<std::int64_t>::max()) ? std::numeric_limits<std::int64_t>::max()
                                                                                : std::int64_t(value);
      }
      default: break;
    }
  }
  throw std::invalid_argument("voxel keys must be an integer array");
}

bool nonzeroAt(const ArrayInput& array, size_t index) {
  const auto type = array.dtype();
  if (type.lanes == 1 && type.code == uint8_t(py::dlpack::dtype_code::Bool) && type.bits == 8)
    return elementAt<std::uint8_t>(array, index) != 0;
  if (type.lanes == 1 && type.code == uint8_t(py::dlpack::dtype_code::Float)) {
    if (type.bits == 32) return elementAt<float>(array, index) != 0.0f;
    if (type.bits == 64) return elementAt<double>(array, index) != 0.0;
  }
  if (type.lanes == 1 && (type.code == uint8_t(py::dlpack::dtype_code::Int) ||
                          type.code == uint8_t(py::dlpack::dtype_code::UInt))) {
    switch (type.bits) {
      case 8: return elementAt<std::uint8_t>(array, index) != 0;
      case 16: return elementAt<std::uint16_t>(array, index) != 0;
      case 32: return elementAt<std::uint32_t>(array, index) != 0;
      case 64: return elementAt<std::uint64_t>(array, index) != 0;
      default: break;
    }
  }
  throw std::invalid_argument("the occupancy must be a boolean, integer or floating-point array");
}

// Keys beyond the int range become int limits, which lie outside every voxel-map domain.
std::vector<std::array<int, 3>> keysFromArray(const ArrayInput& keys) {
  if (keys.ndim() != 2 || keys.shape(1) != 3) throw std::invalid_argument("keys must have shape (n, 3)");
  if (!isInteger(keys.dtype())) throw std::invalid_argument("voxel keys must be an integer array");
  std::vector<std::array<int, 3>> out(keys.shape(0));
  for (size_t i = 0; i < out.size(); ++i) {
    for (size_t a = 0; a < 3; ++a) {
      const std::int64_t value = integerAt(keys, 3 * i + a);
      out[i][a] = int(std::clamp<std::int64_t>(value, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()));
    }
  }
  return out;
}

KeyArray keysToArray(const std::vector<std::array<int, 3>>& keys) {
  auto* data = new int32_t[keys.size() * 3];
  for (size_t i = 0; i < keys.size(); ++i) {
    data[3 * i] = keys[i][0];
    data[3 * i + 1] = keys[i][1];
    data[3 * i + 2] = keys[i][2];
  }
  py::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
  const size_t shape[2] = {keys.size(), 3};
  return KeyArray(data, 2, shape, owner);
}

std::vector<std::uint8_t> occupancyFromArray(const ArrayInput& occupancy, const std::array<int, 3>& size) {
  if (occupancy.ndim() != 3 || occupancy.shape(0) != size_t(size[2]) || occupancy.shape(1) != size_t(size[1]) ||
      occupancy.shape(2) != size_t(size[0])) {
    throw std::invalid_argument("the occupancy must have shape (size_z, size_y, size_x)");
  }
  if (!isOccupancy(occupancy.dtype())) {
    throw std::invalid_argument("the occupancy must be a boolean, integer or floating-point array");
  }
  std::vector<std::uint8_t> out(occupancy.size());
  for (size_t i = 0; i < out.size(); ++i) out[i] = nonzeroAt(occupancy, i) ? 1 : 0;
  return out;
}

}  // namespace

void init_voxel_map(py::module_& m) {
  py::class_<raisim::VoxelMapResource>(m, "VoxelMapResource",
                                       "Immutable voxel occupancy that several voxel maps can share.")
      .def_static("fromKeys", [](double voxelSize, const std::array<int, 3>& minKey, const std::array<int, 3>& size,
                                 const ArrayInput& keys) {
        return raisim::VoxelMapResource::fromKeys(voxelSize, minKey, size, keysFromArray(keys));
      }, py::arg("voxel_size"), py::arg("min_key"), py::arg("size"), py::arg("keys"),
         "Occupancy from an (n, 3) integer array of occupied voxel keys.")
      .def_static("fromOccupancy", [](double voxelSize, const std::array<int, 3>& minKey, const std::array<int, 3>& size,
                                      const ArrayInput& occupancy) {
        return raisim::VoxelMapResource::fromOccupancy(voxelSize, minKey, size, occupancyFromArray(occupancy, size));
      }, py::arg("voxel_size"), py::arg("min_key"), py::arg("size"), py::arg("occupancy"),
         "Occupancy from an array of shape (size_z, size_y, size_x); nonzero entries are occupied.");

  py::class_<raisim::VoxelMap, raisim::SingleBodyObject>(m, "VoxelMap",
      "A static, editable 3D occupancy map of cubic voxels. Voxel key (i, j, k) occupies the cube "
      "[i s, (i + 1) s) x [j s, (j + 1) s) x [k s, (k + 1) s) of the map frame.")
      .def("getVoxelSize", &raisim::VoxelMap::getVoxelSize)
      .def("getMinKey", &raisim::VoxelMap::getMinKey)
      .def("getDomainSize", &raisim::VoxelMap::getDomainSize)
      .def("getOccupiedCount", &raisim::VoxelMap::getOccupiedCount)
      .def("getRevision", &raisim::VoxelMap::getRevision)
      .def("isOccupied", py::overload_cast<int, int, int>(&raisim::VoxelMap::isOccupied, py::const_),
           py::arg("x"), py::arg("y"), py::arg("z"))
      .def("getKey", [](const raisim::VoxelMap& self, const Eigen::Vector3d& point) {
        raisim::Vec<3> p;
        p = point;
        return self.getKey(p);
      }, py::arg("point_in_map_frame"))
      .def("getKeyAtWorldPosition", [](const raisim::VoxelMap& self, const Eigen::Vector3d& point) {
        raisim::Vec<3> p;
        p = point;
        return self.getKeyAtWorldPosition(p);
      }, py::arg("world_point"))
      .def("getVoxelCenter", [](const raisim::VoxelMap& self, const std::array<int, 3>& key) {
        return Eigen::Vector3d(self.getVoxelCenter(key).e());
      }, py::arg("key"))
      .def("setVoxel", &raisim::VoxelMap::setVoxel, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("occupied"))
      .def("setVoxels", [](raisim::VoxelMap& self, const ArrayInput& keys, bool occupied) {
        return self.setVoxels(keysFromArray(keys), occupied);
      }, py::arg("keys"), py::arg("occupied"), "Set or clear the voxels of an (n, 3) integer key array.")
      .def("fillBox", &raisim::VoxelMap::fillBox, py::arg("lo"), py::arg("hi"), py::arg("occupied"))
      .def("fillSphere", [](raisim::VoxelMap& self, const Eigen::Vector3d& center, double radius, bool occupied) {
        raisim::Vec<3> c;
        c = center;
        return self.fillSphere(c, radius, occupied);
      }, py::arg("center_in_map_frame"), py::arg("radius"), py::arg("occupied"))
      .def("setOccupancy", [](raisim::VoxelMap& self, const ArrayInput& occupancy) {
        self.setOccupancy(occupancyFromArray(occupancy, self.getDomainSize()));
      }, py::arg("occupancy"), "Replace the occupancy with an array of shape (size_z, size_y, size_x); nonzero "
         "entries are occupied.")
      .def("clear", &raisim::VoxelMap::clear)
      .def("getOccupiedKeys", [](const raisim::VoxelMap& self) { return keysToArray(self.getOccupiedKeys()); },
           "Keys of every occupied voxel as an (n, 3) int32 array.")
      .def("makeResource", &raisim::VoxelMap::makeResource);
}

void init_voxel_map_world(py::class_<raisim::World>& world) {
  world
      .def("addVoxelMap",
           py::overload_cast<double, const std::array<int, 3>&, const std::array<int, 3>&, const std::string&,
                             raisim::CollisionGroup, raisim::CollisionGroup>(&raisim::World::addVoxelMap),
           py::arg("voxel_size"), py::arg("min_key"), py::arg("size"), py::arg("material") = "default",
           py::arg("collision_group") = RAISIM_STATIC_COLLISION_GROUP,
           py::arg("collision_mask") = raisim::CollisionGroup(-1), py::rv_policy::reference_internal)
      .def("addVoxelMap",
           py::overload_cast<const raisim::VoxelMapResource&, const std::string&, raisim::CollisionGroup,
                             raisim::CollisionGroup>(&raisim::World::addVoxelMap),
           py::arg("resource"), py::arg("material") = "default",
           py::arg("collision_group") = RAISIM_STATIC_COLLISION_GROUP,
           py::arg("collision_mask") = raisim::CollisionGroup(-1), py::rv_policy::reference_internal);
}
#endif
