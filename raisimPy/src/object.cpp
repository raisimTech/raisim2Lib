/**
 * Python wrappers for raisim.object using nanobind.
 *
 * Copyright (c) 2019, kangd (original C++), Brian Delhaisse <briandelhaisse@gmail.com> (Python wrappers)
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

#include "raisim/object/Object.hpp"
#include "raisim/contact_engine/math.h"

#include "converter.hpp"  // contains code that allows to convert between the Vec, Mat to numpy arrays.

namespace py = nanobind;
using namespace raisim;


void init_single_bodies(py::module_ &);
void init_articulated_system(py::module_ &);  // , py::module &);
void init_terrain(py::module_ &);


void init_object(py::module_ &m) {
    auto vec3_list_to_np = [](const std::vector<raisim::Vec<3>> &vecs) {
        NDArray array = make_ndarray_2d(vecs.size(), 3);
        for (size_t i = 0; i < vecs.size(); ++i) {
            for (size_t j = 0; j < 3; ++j) {
                *ndarray_mutable_data(array, i, j) = vecs[i][j];
            }
        }
        return array;
    };

    // create submodule
//    py::module object_module = m.def_submodule("object", "RaiSim object submodule.");


    /**************/
    /* ObjectType */
    /**************/
	// object type enum (from include/raisim/configure.hpp)
	py::enum_<raisim::ObjectType>(m, "ObjectType", py::is_arithmetic())
	    .value("SPHERE", raisim::ObjectType::SPHERE)
	    .value("BOX", raisim::ObjectType::BOX)
	    .value("CYLINDER", raisim::ObjectType::CYLINDER)
	    .value("CONE", raisim::ObjectType::CONE)
	    .value("CAPSULE", raisim::ObjectType::CAPSULE)
	    .value("MESH", raisim::ObjectType::MESH)
	    .value("HALFSPACE", raisim::ObjectType::HALFSPACE)
	    .value("COMPOUND", raisim::ObjectType::COMPOUND)
	    .value("HEIGHTMAP", raisim::ObjectType::HEIGHTMAP)
	    .value("ARTICULATED_SYSTEM", raisim::ObjectType::ARTICULATED_SYSTEM);

    /*********************/
    /* MeshCollisionMode */
    /*********************/
    // mesh collision mode enum (from include/raisim/configure.hpp)
    py::enum_<raisim::MeshCollisionMode>(m, "MeshCollisionMode", py::is_arithmetic())
        .value("ORIGINAL_MESH", raisim::MeshCollisionMode::ORIGINAL_MESH)
        .value("CONVEX_HULL", raisim::MeshCollisionMode::CONVEX_HULL);


    /************/
    /* BodyType */
    /************/
	// body type enum (from include/raisim/configure.hpp)
	py::enum_<raisim::BodyType>(m, "BodyType", py::is_arithmetic())
	    .value("STATIC", raisim::BodyType::STATIC)
	    .value("KINEMATIC", raisim::BodyType::KINEMATIC)
	    .value("DYNAMIC", raisim::BodyType::DYNAMIC);


	/**********/
	/* Object */
	/**********/
	// Object class (from include/raisim/object/Object.hpp)
	py::class_<raisim::Object>(m, "Object", "Raisim Object from which all other objects/bodies inherit from.")
	    .def_prop_rw("name", &raisim::Object::getName, &raisim::Object::setName)
	    .def("getName", &raisim::Object::getName, R"mydelimiter(
	    Get the object's name.

	    Returns:
	        str: object's name.
	    )mydelimiter")
	    .def("setName", &raisim::Object::setName, R"mydelimiter(
	    Set the object's name.

	    Args:
	        name (str): object's name.
	    )mydelimiter",
	    py::arg("name"))

	    .def("setIndexInWorld", &raisim::Object::setIndexInWorld, R"mydelimiter(
	    Set the object index in the world.

	    Args:
	        index (int): object index.
	    )mydelimiter",
	    py::arg("index"))

	    .def("getIndexInWorld", &raisim::Object::getIndexInWorld, R"mydelimiter(
	    Get the object index in the world.

	    Returns:
	        int: object index in the world.
	    )mydelimiter")

	    .def("getContacts", py::overload_cast<>(&raisim::Object::getContacts), R"mydelimiter(
	    Get the list of contact points.

	    Returns:
	        list[Contact]: list of contact points.
	    )mydelimiter")
	    .def("getContacts", py::overload_cast<>(&raisim::Object::getContacts, py::const_), R"mydelimiter(
	    Get the list of contact points.

	    Returns:
	        list[Contact]: list of contact points.
	    )mydelimiter")

	    .def("setExternalForce", [](raisim::Object &self, size_t local_idx, const NDArray &force) {
	        // convert np.array[3] to Vec<3>
	        Vec<3> f = convert_np_to_vec<3>(force);
	        self.setExternalForce(local_idx, f);
	    }, R"mydelimiter(
	    Set the external force on the body.

	    Args:
	        local_idx (int): local index.
	        force (np.array[float[3]]): force vector.
	    )mydelimiter",
	    py::arg("local_idx"), py::arg("force"))
	    .def("setExternalForce", [](raisim::Object &self, size_t local_idx, const NDArray &position, const NDArray &force) {
	        Vec<3> p = convert_np_to_vec<3>(position);
	        Vec<3> f = convert_np_to_vec<3>(force);
	        self.setExternalForce(local_idx, p, f);
	    }, R"mydelimiter(
	    Set the external force at a position on the body.

	    Args:
	        local_idx (int): local index.
	        position (np.array[float[3]]): position in the body frame.
	        force (np.array[float[3]]): force vector in world frame.
	    )mydelimiter",
	    py::arg("local_idx"), py::arg("position"), py::arg("force"))


	    .def("setExternalTorque", [](raisim::Object &self, size_t local_idx, const NDArray &torque) {
	        // convert np.array[3] to Vec<3>
	        Vec<3> t = convert_np_to_vec<3>(torque);
	        self.setExternalTorque(local_idx, t);
	    }, R"mydelimiter(
	    Set the external force on the body.

	    Args:
	        local_idx (int): local index.
	        force (np.array[float[3]]): force vector.
	    )mydelimiter",
	    py::arg("local_idx"), py::arg("torque"))
	    .def("setConstraintForce", [](raisim::Object &self, size_t local_idx, const NDArray &position, const NDArray &force) {
	        Vec<3> p = convert_np_to_vec<3>(position);
	        Vec<3> f = convert_np_to_vec<3>(force);
	        self.setConstraintForce(local_idx, p, f);
	    }, R"mydelimiter(
	    Set the constraint force at a position on the body.

	    Args:
	        local_idx (int): local index.
	        position (np.array[float[3]]): position in the body frame.
	        force (np.array[float[3]]): force vector in world frame.
	    )mydelimiter",
	    py::arg("local_idx"), py::arg("position"), py::arg("force"))

	    .def("getMass", &raisim::Object::getMass, py::arg("local_idx"))
	    .def("getObjectType", &raisim::Object::getObjectType)


	    .def("getPosition", [](raisim::Object &self, size_t local_idx) {
	        Vec<3> pos;
	        self.getPosition(local_idx, pos);
	        // convert vec<3> to np.array[3]
	        return convert_vec_to_np(pos);
	    }, R"mydelimiter(
	    Get the world position.

	    Args:
	        local_idx (int): local index.

	    Returns:
	        np.array[float[3]]: position expressed in the Cartesian world frame.
	    )mydelimiter",
	    py::arg("local_idx"))


	    .def("getVelocity", [](raisim::Object &self, size_t local_idx) {
	        Vec<3> vel;
	        self.getVelocity(local_idx, vel);
	        return convert_vec_to_np(vel);
	    }, R"mydelimiter(
	    Get the world linear velocity.

	    Args:
	        local_idx (int): local index.

	    Returns:
	        np.array[float[3]]: linear velocity expressed in the Cartesian world frame.
	    )mydelimiter",
	    py::arg("local_idx"))
	    .def("getVelocity", [](raisim::Object &self, size_t local_idx, const NDArray &body_pos) {
	        Vec<3> pos_b = convert_np_to_vec<3>(body_pos);
	        Vec<3> vel;
	        self.getVelocity(local_idx, pos_b, vel);
	        return convert_vec_to_np(vel);
	    }, py::arg("local_idx"), py::arg("body_pos"))


	    .def("getOrientation", [](raisim::Object &self, size_t local_idx) {
	        Mat<3,3> rot;
	        self.getOrientation(local_idx, rot);
	        return convert_mat_to_np(rot);
	    }, R"mydelimiter(
	    Get the world orientation as a rotation matrix.

	    Args:
	        local_idx (int): local index.

	    Returns:
	        np.array[float[3,3]]: rotation matrix.
	    )mydelimiter",
	    py::arg("local_idx"))


	    .def("getPosition", [](raisim::Object &self, size_t local_idx, const NDArray &body_pos) {
	        Vec<3> pos_b =convert_np_to_vec<3>(body_pos);
	        Vec<3> pos;
	        self.getPosition(local_idx, pos_b, pos);
	        return convert_vec_to_np(pos);
	     })
	    .def("getBodyType", py::overload_cast<size_t>(&raisim::Object::getBodyType, py::const_))
	    .def("getBodyType", py::overload_cast<>(&raisim::Object::getBodyType, py::const_))
	    .def("getContactPointVel", [](raisim::Object &self, size_t point_id) {
	        Vec<3> vel;
	        self.getContactPointVel(point_id, vel);
	        return convert_vec_to_np(vel);
	    }, R"mydelimiter(
	    Get the contact point velocity.

	    Args:
	        point_id (int): point id.

	    Returns:
	        np.array[float[3]]: contact point velocity.
	    )mydelimiter",
	    py::arg("point_id"))
	    .def("getContactAabbCount", &raisim::Object::getContactAabbCount)
	    .def("getContactAabb", [](raisim::Object &self, size_t idx) -> const ::contact::AABB& {
	        return self.getContactAabb(idx);
	    }, py::rv_policy::reference_internal, py::arg("idx"))
	    .def("getExternalForce", [vec3_list_to_np](const raisim::Object &self) {
	        return vec3_list_to_np(self.getExternalForce());
	    })
	    .def("getExternalForcePosition", [vec3_list_to_np](const raisim::Object &self) {
	        return vec3_list_to_np(self.getExternalForcePosition());
	    })
	    .def("getExternalTorque", [vec3_list_to_np](const raisim::Object &self) {
	        return vec3_list_to_np(self.getExternalTorque());
	    })
	    .def("getExternalTorquePosition", [vec3_list_to_np](const raisim::Object &self) {
	        return vec3_list_to_np(self.getExternalTorquePosition());
	    })
	    .def("clearExternalForcesAndTorques", &raisim::Object::clearExternalForcesAndTorques)
	    .def("lockMutex", &raisim::Object::lockMutex)
	    .def("unlockMutex", &raisim::Object::unlockMutex)
	    .def("lock", &raisim::Object::lock)
	    .def("unlock", &raisim::Object::unlock);


	// raisim.object.singleBodies
	init_single_bodies(m);  // object_module

	// raisim.object.ArticulatedSystem
	init_articulated_system(m);  // object_module

	// raisim.object.terrain
	init_terrain(m);  // object_module

}
