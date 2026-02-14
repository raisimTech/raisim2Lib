/**
 * Python wrappers for raisim.object.ArticulatedSystem using nanobind.
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

#include "raisim/math.hpp"   // contains the definitions of Vec, Mat, and others, as well as transformations.
#include "raisim/World.hpp"
#include "raisim/object/ArticulatedSystem/loaders.hpp"
#include "raisim/object/ArticulatedSystem/JointAndBodies.hpp"
#include "raisim/object/ArticulatedSystem/ArticulatedSystem.hpp"

#include "converter.hpp"  // contains code that allows to convert between the Vec, Mat to numpy arrays.

// Important note: for the above include ("ode/src/collision_kernel.h"), you have to add a `extras` folder in the
// `$LOCAL_BUILD/include/ode/` which should contain the following header files:
// array.h, collision_kernel.h, common.h, error.h, objects.h, odeou.h, odetls.h, threading_base.h, and typedefs.h.
// These header files can be found in the `ode/src` folder (like here: https://github.com/thomasmarsh/ODE/tree/master/ode/src)
//
// Why do we need to do that? The reason is that for `raisim::Mesh`, the authors of RaiSim use the `dSpaceID` variable
// type which has been forward declared in `ode/common.h` (but not defined there) as such:
//
// struct dxSpace;
// typedef struct dxSpace *dSpaceID;
//
// Thus for `dSpaceID` we need the definition of `dxSpace`, and this one is defined in `ode/src/collision_kernel.h` (in
// the `src` folder and not in the `include` folder!!). Pybind11 is looking for that definition, if you don't include
// it, nanobind will complain and raise errors.


namespace py = nanobind;
using namespace raisim;


void init_articulated_system(py::module_ &m) { // py::module &main_module) {
    auto vec3_list_to_pylist = [](const std::vector<raisim::Vec<3>> &vecs) {
        py::list out;
        for (const auto &v : vecs)
            out.append(convert_vec_to_np(v));
        return out;
    };
    auto mat3_list_to_pylist = [](const std::vector<raisim::Mat<3, 3>> &mats) {
        py::list out;
        for (const auto &mat : mats)
            out.append(convert_mat_to_np(mat));
        return out;
    };
    auto matdyn_list_to_pylist = [](const std::vector<raisim::MatDyn> &mats) {
        py::list out;
        for (const auto &mat : mats)
            out.append(convert_matdyn_to_np(mat));
        return out;
    };
    auto vecdyn_list_to_pylist = [](const std::vector<raisim::VecDyn> &vecs) {
        py::list out;
        for (const auto &vec : vecs)
            out.append(convert_vecdyn_to_np(vec));
        return out;
    };

    /***************/
    /* ControlMode */
    /***************/
    py::enum_<raisim::ControlMode::Type>(m, "ControlMode", py::is_arithmetic())
        .value("FORCE_AND_TORQUE", raisim::ControlMode::Type::FORCE_AND_TORQUE)
        .value("PD_PLUS_FEEDFORWARD_TORQUE", raisim::ControlMode::Type::PD_PLUS_FEEDFORWARD_TORQUE);

    /***********************/
    /* CollisionBodyHandle */
    /***********************/
    py::class_<raisim::CollisionBodyHandle>(m, "CollisionBodyHandle", "Raisim collision body handle.")
        .def(py::init<>())
        .def_rw("objectType", &raisim::CollisionBodyHandle::objectType)
        .def_rw("bodyType", &raisim::CollisionBodyHandle::bodyType)
        .def_rw("objectIndex", &raisim::CollisionBodyHandle::objectIndex)
        .def_rw("localIdx", &raisim::CollisionBodyHandle::localIdx)
        .def_rw("name", &raisim::CollisionBodyHandle::name)
        .def_rw("material", &raisim::CollisionBodyHandle::material)
        .def_rw("materialVersion", &raisim::CollisionBodyHandle::materialVersion)
        .def_rw("materialId", &raisim::CollisionBodyHandle::materialId);

    /***********************/
    /* CollisionDefinition */
    /***********************/
    py::class_<raisim::CollisionDefinition>(m, "CollisionDefinition", "Raisim CollisionDefinition struct.")

        .def_prop_rw("rotOffset",
            [](raisim::CollisionDefinition &self) {  // getter
                return convert_mat_to_np(self.rotOffset);
            }, [](raisim::CollisionDefinition &self, NDArray array) {  // setter
                Mat<3,3> rot = convert_np_to_mat<3,3>(array);
                self.rotOffset = rot;
            })
        .def_prop_rw("posOffset",
            [](raisim::CollisionDefinition &self) {  // getter
                return convert_vec_to_np(self.posOffset);
            }, [](raisim::CollisionDefinition &self, NDArray array) {  // setter
                Vec<3> pos = convert_np_to_vec<3>(array);
                self.posOffset = pos;
            })
        .def_rw("localIdx", &raisim::CollisionDefinition::localIdx)
        .def_rw("shape", &raisim::CollisionDefinition::shape)
        .def_prop_rw("shapeParam",
            [](raisim::CollisionDefinition &self) {  // getter
                return convert_vec_to_np(self.shapeParam);
            }, [](raisim::CollisionDefinition &self, NDArray array) {  // setter
                Vec<4> param = convert_np_to_vec<4>(array);
                self.shapeParam = param;
            })
        .def_prop_rw("scale",
            [](raisim::CollisionDefinition &self) {  // getter
                return convert_vec_to_np(self.scale);
            }, [](raisim::CollisionDefinition &self, NDArray array) {  // setter
                Vec<3> scale = convert_np_to_vec<3>(array);
                self.scale = scale;
            })
        .def_rw("meshIndex", &raisim::CollisionDefinition::meshIndex)
        .def_prop_rw("name",
                      [](raisim::CollisionDefinition &self) {  // getter
                        return self.colBody.name;
                      }, [](raisim::CollisionDefinition &self, py::str name) {  // setter
              self.colBody.name = py::cast<std::string>(name);
            })
        .def("setMaterial", &raisim::CollisionDefinition::setMaterial, py::arg("material"))
        .def("getMaterial", &raisim::CollisionDefinition::getMaterial)
        .def("setCollisionGroup", &raisim::CollisionDefinition::setCollisionGroup, py::arg("group"))
        .def("setCollisionMask", &raisim::CollisionDefinition::setCollisionMask, py::arg("mask"))
        .def("getCollisionGroup", &raisim::CollisionDefinition::getCollisionGroup)
        .def("getCollisionMask", &raisim::CollisionDefinition::getCollisionMask)
        .def("getCollisionBody", py::overload_cast<>(&raisim::CollisionDefinition::getCollisionBody),
             py::rv_policy::reference_internal);


    /*********/
    /* Shape */
    /*********/
    py::enum_<raisim::Shape::Type>(m, "ShapeType", py::is_arithmetic())
        .value("Box", raisim::Shape::Type::Box)
        .value("Cylinder", raisim::Shape::Type::Cylinder)
        .value("Sphere", raisim::Shape::Type::Sphere)
        .value("Mesh", raisim::Shape::Type::Mesh)
        .value("Capsule", raisim::Shape::Type::Capsule)
        .value("Cone", raisim::Shape::Type::Cone);

    /************/
    /* JointType */
    /************/
    py::enum_<raisim::Joint::Type>(m, "JointType", py::is_arithmetic())
        .value("FIXED", raisim::Joint::Type::FIXED)
        .value("REVOLUTE", raisim::Joint::Type::REVOLUTE)
        .value("PRISMATIC", raisim::Joint::Type::PRISMATIC)
        .value("SPHERICAL", raisim::Joint::Type::SPHERICAL)
        .value("FLOATING", raisim::Joint::Type::FLOATING);

    /************/
    /* VisObject */
    /************/
    py::class_<raisim::VisObject>(m, "VisObject", "Raisim visual object definition.")
        .def_prop_rw("visShapeParam",
            [](raisim::VisObject &self) {  // getter
                return convert_vec_to_np(self.visShapeParam);
            }, [](raisim::VisObject &self, NDArray array) {  // setter
                Vec<4> param = convert_np_to_vec<4>(array);
                self.visShapeParam = param;
            })
        .def_prop_rw("offset",
            [](raisim::VisObject &self) {  // getter
                return convert_vec_to_np(self.offset);
            }, [](raisim::VisObject &self, NDArray array) {  // setter
                Vec<3> offset = convert_np_to_vec<3>(array);
                self.offset = offset;
            })
        .def_prop_rw("rot",
            [](raisim::VisObject &self) {  // getter
                return convert_mat_to_np(self.rot);
            }, [](raisim::VisObject &self, NDArray array) {  // setter
                Mat<3,3> rot = convert_np_to_mat<3,3>(array);
                self.rot = rot;
            })
        .def_prop_rw("color",
            [](raisim::VisObject &self) {  // getter
                return convert_vec_to_np(self.color);
            }, [](raisim::VisObject &self, NDArray array) {  // setter
                Vec<4> color = convert_np_to_vec<4>(array);
                self.color = color;
            })
        .def_prop_rw("scale",
            [](raisim::VisObject &self) {  // getter
                return convert_vec_to_np(self.scale);
            }, [](raisim::VisObject &self, NDArray array) {  // setter
                Vec<3> scale = convert_np_to_vec<3>(array);
                self.scale = scale;
            })
        .def_rw("shape", &raisim::VisObject::shape)
        .def_rw("fileName", &raisim::VisObject::fileName)
        .def_rw("name", &raisim::VisObject::name)
        .def_rw("localIdx", &raisim::VisObject::localIdx)
        .def_rw("material", &raisim::VisObject::material);

    /*******************/
    /* Spring          */
    /*******************/
    py::class_<raisim::ArticulatedSystem::SpringElement>(m, "SpringElement", "Spring element in an articulated system")
        .def_prop_rw("q_ref", &raisim::ArticulatedSystem::SpringElement::getSpringMount, &raisim::ArticulatedSystem::SpringElement::setSpringMount)
        .def_rw("childBodyId", &raisim::ArticulatedSystem::SpringElement::childBodyId)
        .def_rw("stiffness", &raisim::ArticulatedSystem::SpringElement::stiffness);

    /*******************/
    /* CoordinateFrame */
    /*******************/
    py::class_<raisim::CoordinateFrame>(m, "CoordinateFrame", "Raisim Coordinate Frame class.")
        .def_prop_rw("position",
            [](raisim::CoordinateFrame &self) {  // getter
                return convert_vec_to_np(self.position);
            }, [](raisim::CoordinateFrame &self, NDArray array) {  // setter
                auto pos = convert_np_to_vec<3>(array);
                self.position = pos;
            })
        .def_prop_rw("orientation",
            [](raisim::CoordinateFrame &self) {  // getter
                return convert_mat_to_np(self.orientation);
            }, [](raisim::CoordinateFrame &self, NDArray array) {  // setter
                Mat<3,3> rot = convert_np_to_mat<3,3>(array);
                self.orientation = rot;
            })
        .def_rw("parentId", &raisim::CoordinateFrame::parentId)
        .def_rw("parentName", &raisim::CoordinateFrame::parentName)
        .def_rw("name", &raisim::CoordinateFrame::name)
        .def_rw("isChild", &raisim::CoordinateFrame::isChild);  // child is the first body after movable joint. All fixed bodies attached to a child is not a child


    /***********/
    /* LinkRef */
    /***********/
    py::class_<raisim::ArticulatedSystem::LinkRef>(m, "LinkRef")
        .def("getPosition", [](const raisim::ArticulatedSystem::LinkRef &self) {
            Vec<3> pos;
            self.getPosition(pos);
            return convert_vec_to_np(pos);
        })
        .def("getOrientation", [](const raisim::ArticulatedSystem::LinkRef &self) {
            Mat<3, 3> rot;
            self.getOrientation(rot);
            return convert_mat_to_np(rot);
        })
        .def("getPose", [](const raisim::ArticulatedSystem::LinkRef &self) {
            Vec<3> pos;
            Mat<3, 3> rot;
            self.getPose(pos, rot);
            return py::make_tuple(convert_vec_to_np(pos), convert_mat_to_np(rot));
        })
        .def("getCollisionDefinition", &raisim::ArticulatedSystem::LinkRef::getCollisionDefinition,
             py::arg("name"), py::rv_policy::reference_internal)
        .def("getVisualObject", &raisim::ArticulatedSystem::LinkRef::getVisualObject,
             py::arg("name"), py::rv_policy::reference_internal)
        .def("setWeight", &raisim::ArticulatedSystem::LinkRef::setWeight, py::arg("weight"))
        .def("getWeight", &raisim::ArticulatedSystem::LinkRef::getWeight)
        .def("setInertia", [](raisim::ArticulatedSystem::LinkRef &self, NDArray inertia) {
            self.setInertia(convert_np_to_mat<3, 3>(inertia));
        }, py::arg("inertia"))
        .def("getInertia", [](const raisim::ArticulatedSystem::LinkRef &self) {
            return convert_mat_to_np(self.getInertia());
        })
        .def("setComPositionInParentFrame", [](raisim::ArticulatedSystem::LinkRef &self, NDArray com) {
            self.setComPositionInParentFrame(convert_np_to_vec<3>(com));
        }, py::arg("com"))
        .def("getComPositionInParentFrame", [](const raisim::ArticulatedSystem::LinkRef &self) {
            return convert_vec_to_np(self.getComPositionInParentFrame());
        })
        .def("getCollisionSet", &raisim::ArticulatedSystem::LinkRef::getCollisionSet,
             py::rv_policy::reference_internal)
        .def("getVisualSet", &raisim::ArticulatedSystem::LinkRef::getVisualSet,
             py::rv_policy::reference_internal);

    /************/
    /* JointRef */
    /************/
    py::class_<raisim::ArticulatedSystem::JointRef>(m, "JointRef")
        .def("getPosition", [](const raisim::ArticulatedSystem::JointRef &self) {
            Vec<3> pos;
            self.getPosition(pos);
            return convert_vec_to_np(pos);
        })
        .def("getOrientation", [](const raisim::ArticulatedSystem::JointRef &self) {
            return convert_mat_to_np(self.getOrientation());
        })
        .def("getPose", [](const raisim::ArticulatedSystem::JointRef &self) {
            Vec<3> pos;
            Mat<3, 3> rot;
            self.getPose(pos, rot);
            return py::make_tuple(convert_vec_to_np(pos), convert_mat_to_np(rot));
        })
        .def("getJointCoordinate", [](const raisim::ArticulatedSystem::JointRef &self) {
            VecDyn coord;
            self.getJointCoordinate(coord);
            return convert_vecdyn_to_np(coord);
        })
        .def("getJointAngle", &raisim::ArticulatedSystem::JointRef::getJointAngle)
        .def("getPositionInParentFrame", [](const raisim::ArticulatedSystem::JointRef &self) {
            return convert_vec_to_np(self.getPositionInParentFrame());
        })
        .def("getJointAxis", [](const raisim::ArticulatedSystem::JointRef &self) {
            return convert_vec_to_np(self.getJointAxis());
        })
        .def("getType", &raisim::ArticulatedSystem::JointRef::getType)
        .def("getIdxInGeneralizedCoordinate", &raisim::ArticulatedSystem::JointRef::getIdxInGeneralizedCoordinate)
        .def("getLinearVelocity", [](const raisim::ArticulatedSystem::JointRef &self) {
            return convert_vec_to_np(self.getLinearVelocity());
        });


    /***************************/
    /* ArticulatedSystemOption */
    /***************************/
    py::class_<raisim::ArticulatedSystemOption>(m, "ArticulatedSystemOption", "Raisim Articulated System Option.")
        .def_rw("do_not_collide_with_parent", &raisim::ArticulatedSystemOption::doNotCollideWithParent);


    /*********************/
    /* ArticulatedSystem */
    /*********************/

    // From the `ArticulatedSystem.h` file:
    /* list of vocabs
     1. body: body here refers to only rotating bodies. Fixed bodies are optimized out.
              Position of a body refers to the position of the joint connecting the body and its parent.
     2. Coordinate frame: coordinate frames are defined on every joint (even at the fixed joints). If you want
                          to define a custom frame, define a fixed zero-mass object and a joint in the URDF */

    py::class_<raisim::ArticulatedSystem, raisim::Object> system(m, "ArticulatedSystem", "Raisim Articulated System.");

    py::enum_<raisim::ArticulatedSystem::Frame>(system, "Frame")
        .value("WORLD_FRAME", raisim::ArticulatedSystem::Frame::WORLD_FRAME)
        .value("PARENT_FRAME", raisim::ArticulatedSystem::Frame::PARENT_FRAME)
        .value("BODY_FRAME", raisim::ArticulatedSystem::Frame::BODY_FRAME);
    py::enum_<raisim::ArticulatedSystem::IntegrationScheme>(system, "IntegrationScheme")
        .value("TRAPEZOID", raisim::ArticulatedSystem::IntegrationScheme::TRAPEZOID)
        .value("SEMI_IMPLICIT", raisim::ArticulatedSystem::IntegrationScheme::SEMI_IMPLICIT)
        .value("EULER", raisim::ArticulatedSystem::IntegrationScheme::EULER)
        .value("RUNGE_KUTTA_4", raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);

    system.def(py::init<>(), "Initialize the Articulated System.")

        .def(py::init<const std::string &, const std::string &, std::vector<std::string>, raisim::ArticulatedSystemOption>(),
        "Initialize the Articulated System.\n\n"
        "Do not call this method yourself. use World class to create an Articulated system.\n\n"
        "Args:\n"
        "    filename (str): path to the robot description file (URDF, etc).\n"
        "    resource_directory (str): path the resource directory. If empty, it will use the robot description folder.\n"
        "    joint_order (list[str]): specify the joint order, if we want it to be different from the URDF file.\n"
        "    options (ArticulatedSystemOption): options.",
        py::arg("filename"), py::arg("resource_directory"), py::arg("joint_order"), py::arg("options"))


        .def("getGeneralizedCoordinate", [](raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getGeneralizedCoordinate());
        }, R"mydelimiter(
        Get the generalized coordinates of the system.

        The dimension of the returned vector is equal to the number of degrees of freedom that each joint provides.
        A floating joint provides 6 DoFs (3 linear + 3 revolute), a prismatic/revolute joint 1 DoF, a fixed joint 0
        DoF, etc.

        Returns:
            np.array[float[n]]: generalized coordinates.
        )mydelimiter")


        .def("getBaseOrientation", [](raisim::ArticulatedSystem &self) {
            Vec<4> quaternion;
            self.getBaseOrientation(quaternion);
            return convert_vec_to_np(quaternion);
        }, R"mydelimiter(
        Get the base orientation (expressed as a quaternion [w,x,y,z]).

        Returns:
            np.array[float[4]]: base orientation (expressed as a quaternion [w,x,y,z])
        )mydelimiter")

        .def("getBaseOrientation", [](raisim::ArticulatedSystem &self) {
            Mat<3,3> rot;
            self.getBaseOrientation(rot);
            return convert_mat_to_np(rot);
        }, R"mydelimiter(
        Get the base orientation (expressed as a rotation matrix).

        Returns:
            np.array[float[3,3]]: rotation matrix
        )mydelimiter")

        .def("getGeneralizedVelocity", [](raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getGeneralizedVelocity());
        }, R"mydelimiter(
        Get the generalized velocities of the system.

        The dimension of the returned vector is equal to the number of degrees of freedom that each joint provides.
        A floating joint provides 6 DoFs (3 linear + 3 revolute), a prismatic/revolute joint 1 DoF, a fixed joint 0
        DoF, etc.

        Returns:
            np.array[float[n]]: generalized velocities.
        )mydelimiter")

        .def("getGeneralizedAcceleration", [](raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getGeneralizedAcceleration());
        }, R"mydelimiter(
        Get the generalized accelerations of the system.

        Returns:
            np.array[float[n]]: generalized accelerations.
        )mydelimiter")

        .def("updateKinematics", &raisim::ArticulatedSystem::updateKinematics,  R"mydelimiter(
        Update the kinematics.

        It is unnecessary to call this function if you are simulating your system. `integrate1` calls this function.
        Call this function if you want to get kinematic properties but you don't want to integrate.
        )mydelimiter")

        .def("setGeneralizedCoordinate", py::overload_cast<std::initializer_list<double>>(&raisim::ArticulatedSystem::setGeneralizedCoordinate), R"mydelimiter(
        Set the generalized coordinates.

        Args:
            coordinates (list[float]): generalized coordinates to set.
        )mydelimiter",
        py::arg("coordinates"))
        .def("setGeneralizedCoordinate", py::overload_cast<const Eigen::VectorXd &>(&raisim::ArticulatedSystem::setGeneralizedCoordinate), R"mydelimiter(
        Set the generalized coordinates.

        Args:
            coordinates (np.array[float[n]]): generalized coordinates to set.
        )mydelimiter",
        py::arg("coordinates"))


        .def("setGeneralizedVelocity", py::overload_cast<std::initializer_list<double>>(&raisim::ArticulatedSystem::setGeneralizedVelocity), R"mydelimiter(
        Set the generalized velocities.

        Args:
            velocities (list[float]): generalized velocities to set.
        )mydelimiter",
        py::arg("velocities"))
        .def("setGeneralizedVelocity", py::overload_cast<const Eigen::VectorXd &>(&raisim::ArticulatedSystem::setGeneralizedVelocity), R"mydelimiter(
        Set the generalized velocities.

        Args:
            velocities (np.array[float[n]]): generalized velocities to set.
        )mydelimiter",
        py::arg("velocities"))


        .def("setGeneralizedForce", py::overload_cast<std::initializer_list<double>>(&raisim::ArticulatedSystem::setGeneralizedForce), R"mydelimiter(
        Set the generalized forces.

        These are the feedforward generalized forces. In the PD control mode, this differs from the actual
        generalizedForces. The dimension should be the same as the number of DoFs.

        Args:
            forces (list[float]): generalized forces to set.
        )mydelimiter",
        py::arg("forces"))


        .def("setGeneralizedForce", py::overload_cast<const Eigen::VectorXd &>(&raisim::ArticulatedSystem::setGeneralizedForce), R"mydelimiter(
        Set the generalized forces.

        These are the feedforward generalized forces. In the PD control mode, this differs from the actual
        generalizedForces. The dimension should be the same as the number of DoFs.

        Args:
            forces (np.array[float[n]]): generalized forces to set.
        )mydelimiter",
        py::arg("forces"))

        .def("getState", [](raisim::ArticulatedSystem &self) {
            Eigen::VectorXd gc;
            Eigen::VectorXd gv;

            self.getState(gc, gv);

            return py::make_tuple(gc, gv);
        }, R"mydelimiter(
        Get the joint states.

        Returns:
            np.array[float[n]]: generalized coordinates.
            np.array[float[n]]: generalized velocities.
        )mydelimiter")


        .def("setState", &raisim::ArticulatedSystem::setState, R"mydelimiter(
        Set the joint states.

        Args:
            coordinates (np.array[float[n]]): generalized coordinates to set.
            velocities (np.array[float[n]]): generalized velocities to set.
        )mydelimiter",
        py::arg("coordinates"), py::arg("velocities"))


        /* get dynamics properties. Make sure that after integration you call "integrate1()" of the world object
        before using this method. Generalized force is the actual force */
        .def("getGeneralizedForce", [](raisim::ArticulatedSystem &self) {
            VecDyn vec = self.getGeneralizedForce();
            return convert_vecdyn_to_np(vec);
        }, R"mydelimiter(
        Get the generalized forces.

        Returns:
            np.array[float[n]]: generalized forces.
        )mydelimiter")


        .def("getFeedForwardGeneralizedForce", [](raisim::ArticulatedSystem &self) {
            VecDyn vec = self.getFeedForwardGeneralizedForce();
            return convert_vecdyn_to_np(vec);
        }, R"mydelimiter(
        Get the feedforward generalized forces.

        Returns:
            np.array[float[n]]: feedforward generalized forces.
        )mydelimiter")


        .def("getMassMatrix", [](raisim::ArticulatedSystem &self) {
            MatDyn mat = self.getMassMatrix();
            return convert_matdyn_to_np(mat);
        }, R"mydelimiter(
        Get the mass (inertia) matrix which is present in the dynamic equation of motion:

        .. math:: H(q) \ddot{q} + N(q, \dot{q}) = \tau

        where :math:`H(q)` is the mass inertia matrix, :math:`N(q, \dot{q})` are the non-linear terms, and
        :math:`\tau` are the forces/torques.

        Returns:
            np.array[float[n,n]]: mass inertia matrix.
        )mydelimiter")


        .def("getNonlinearities", [](raisim::ArticulatedSystem &self, NDArray gravity) {
          Vec<3> g = convert_np_to_vec<3>(gravity);
          VecDyn vec = self.getNonlinearities(g);
          return convert_vecdyn_to_np(vec);
        }, R"mydelimiter(
        Get the non linearity terms that are present in the dynamic equation of motion:

        .. math:: H(q) \ddot{q} + N(q, \dot{q}) = \tau

        where :math:`N(q, \dot{q})` are the non-linear terms, :math:`H(q)` is the inertia matrix, and :math:`\tau` are
        the forces/torques.

        Returns:
            np.array[float[n]]: non-linearity forces.
        )mydelimiter")


        .def("getInverseMassMatrix", [](raisim::ArticulatedSystem &self) {
            MatDyn mat = self.getInverseMassMatrix();
            return convert_matdyn_to_np(mat);
        }, R"mydelimiter(
        Get the inverse of the mass (inertia) matrix, where the inertia matrix appears in the dynamic equation of motion:

        .. math:: H(q) \ddot{q} + N(q, \dot{q}) = \tau

        where :math:`H(q)` is the mass inertia matrix, :math:`N(q, \dot{q})` are the non-linear terms, and
        :math:`\tau` are the forces/torques.

        Returns:
            np.array[float[n,n]]: inverse of the mass inertia matrix.
        )mydelimiter")

        .def("getCompositeCOM", [](raisim::ArticulatedSystem &self) {
          py::list list;
          for (auto pos : self.getCompositeCOM())
            list.append(convert_vec_to_np(pos));

          return list;
        }, R"mydelimiter(
        Get the center of mass position of the composite body (i.e. the byproduct of the composite rigid body algorithm).

        Returns:
            list of np.array[float[3]]: center of mass position of the composite bodies.
        )mydelimiter")

        .def("getCompositeInertia", [](raisim::ArticulatedSystem &self) {
          py::list list;
          for (auto pos : self.getCompositeInertia())
            list.append(convert_mat_to_np(pos));

          return list;
        }, R"mydelimiter(
        Get the composite moment of inertia of the composite body (i.e. the byproduct of the composite rigid body algorithm).

        Returns:
            list of np.array[float[3, 3]]: moment inertia of the composite bodies
        )mydelimiter")

        .def("getCompositeMass", [](raisim::ArticulatedSystem &self) {
          return self.getCompositeMass();
        }, R"mydelimiter(
        Get the composite mass of the composite body (i.e. the byproduct of the composite rigid body algorithm)

        Returns:
            list of float64: mass of the composite bodies
        )mydelimiter")

        .def("getLinearMomentum", [](raisim::ArticulatedSystem &self) {
            Vec<3> vec = self.getLinearMomentum();
            return convert_vec_to_np(vec);
        }, R"mydelimiter(
        Get the linear momentum of the whole system in Cartesian space.

        Returns:
            np.array[float[3]]: total linear momentum.
        )mydelimiter")

        .def("getGeneralizedMomentum", [](raisim::ArticulatedSystem &self) {
            VecDyn vec = self.getGeneralizedMomentum();
            return convert_vecdyn_to_np(vec);
        }, R"mydelimiter(
        Get the generalized momentum which is simply the mass matrix multiplied by the generalized velocities:

        .. math:: H(q) \dot{q}

        where :math:`H(q)` is the mass/inertia matrix, and :math:`\dot{q}` are the generalized velocities.

        Returns:
            np.array[float[n]]: generalized momentum.
        )mydelimiter")

        .def("get_kinetic_energy", &raisim::ArticulatedSystem::getKineticEnergy, "Return the total kinetic energy of the whole system.")

        .def("getPotentialEnergy", [](raisim::ArticulatedSystem &self, NDArray gravity) {
            Vec<3> g = convert_np_to_vec<3>(gravity);
            return self.getPotentialEnergy(g);
        }, R"mydelimiter(
        Get the system's potential energy due to gravity.

        Args:
            gravity (np.array[float[3]]): gravity vector.

        Returns:
            float: potential energy.
        )mydelimiter",
        py::arg("gravity"))

        .def("getEnergy", [](raisim::ArticulatedSystem &self, NDArray gravity) {
            Vec<3> g = convert_np_to_vec<3>(gravity);
            return self.getEnergy(g);
        }, R"mydelimiter(
        Get the system's total energy.

        Args:
            gravity (np.array[float[3]]): gravity vector.

        Returns:
            float: total energy.
        )mydelimiter",
        py::arg("gravity"))

        .def("printOutBodyNamesInOrder", &raisim::ArticulatedSystem::printOutBodyNamesInOrder,
            "Print the moving bodies in the order. Fixed bodies are optimized out.")

        .def("printOutFrameNamesInOrder", &raisim::ArticulatedSystem::printOutFrameNamesInOrder,
            "Print the frames (that are attached to every joint coordinate) in the order.")

        .def("getPosition", [](raisim::ArticulatedSystem &self, size_t body_idx, NDArray body_point) {
            Vec<3> pos;
            Vec<3> body_pos = convert_np_to_vec<3>(body_point);
            self.getPosition(body_idx, body_pos, pos);
            return convert_vec_to_np(pos);
        }, R"mydelimiter(
        Get the body's position with respect to the world frame.

        Args:
            body_idx (int): body/link index.
            body_point (np.array[float[3]]): position of point on the body expressed in the body frame.

        Returns:
            np.array[float[3]]: position of the body point in the world frame.
        )mydelimiter",
        py::arg("body_idx"), py::arg("body_point"))

        /* CoordinateFrame contains the pose relative to its parent expressed in the parent frame.
         If you want the position expressed in the world frame,
         you have to use this with "void getPosition_W(size_t bodyIdx, const Vec<3> &point_B, Vec<3> &point_W)".
         If you want the orientation expressed in the world frame,
         you have to get the parent body orientation and pre-multiply it by the relative orientation*/
        .def("getFrameByName", py::overload_cast<const std::string &>(&raisim::ArticulatedSystem::getFrameByName), py::rv_policy::reference, R"mydelimiter(
        Get the coordinate frame from its name.

        Args:
            name (str): name of the frame.

        Returns:
            CoordinateFrame: the coordinate frame.
        )mydelimiter",
        py::arg("name"))


        .def("getFrameByIdx", py::overload_cast<size_t>(&raisim::ArticulatedSystem::getFrameByIdx), py::rv_policy::reference, R"mydelimiter(
        Get the coordinate frame from its index.

        Args:
            idx (int): index of the frame.

        Returns:
            CoordinateFrame: the coordinate frame.
        )mydelimiter",
        py::arg("idx"))

        .def("getSprings", py::overload_cast<>(&raisim::ArticulatedSystem::getSprings), py::rv_policy::reference, R"mydelimiter(
        Get the spring elements.

        Returns:
            springs: list of spring elements.
        )mydelimiter")

        .def("getParentVector", &raisim::ArticulatedSystem::getParentVector, py::rv_policy::reference_internal, R"mydelimiter(
        Get the parent vector. parent[i] is the parent body id of body i.

        Returns:
            list[int]: parent body indices.
        )mydelimiter")

        .def("getFrameIdxByName", &raisim::ArticulatedSystem::getFrameIdxByName, py::rv_policy::reference, R"mydelimiter(
        Get the coordinate frame index from its name.

        Args:
            name (str): name of the frame.

        Returns:
            int: the corresponding index.
        )mydelimiter",
        py::arg("name"))

        .def("getFrames", py::overload_cast<>(&raisim::ArticulatedSystem::getFrames), py::rv_policy::reference, R"mydelimiter(
        Get all the coordinate frames.

        Returns:
            list[CoordinateFrame]: the coordinate frames.
        )mydelimiter")

        .def("getSensorSet", &raisim::ArticulatedSystem::getSensorSet, py::rv_policy::reference_internal, R"mydelimiter(
        Get the sensor set by name.

        Args:
            name (str): sensor set name.

        Returns:
            SensorSet: sensor set.
        )mydelimiter",
        py::arg("name"))

        .def("getSensorSets", [](raisim::ArticulatedSystem &self) { return self.getSensorSets(); },
             py::rv_policy::reference_internal, R"mydelimiter(
        Get all sensor sets.

        Returns:
            list[SensorSet]: sensor sets.
        )mydelimiter")

        /* returns position and orientation in the world frame of a frame defined in the robot description
          Frames are attached to the joint position */
        .def("getFramePosition", [](raisim::ArticulatedSystem &self, size_t frame_id) {
            Vec<3> vec;
            self.getFramePosition(frame_id, vec);
            return convert_vec_to_np(vec);
        }, R"mydelimiter(
        Get the frame position expressed in the Cartesian world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: the coordinate frame position in the world space.
        )mydelimiter",
        py::arg("frame_id"))

        /* returns position and orientation in the world frame of a frame defined in the robot description
         * Frames are attached to the joint position */
        .def("getFramePosition", [](raisim::ArticulatedSystem &self, const std::string& name) {
               Vec<3> vec;
               self.getFramePosition(name, vec);
               return convert_vec_to_np(vec);
             }, R"mydelimiter(
        Get the frame position expressed in the Cartesian world frame.

        Args:
            frame_name (string): frame name.

        Returns:
            np.array[float[3]]: the coordinate frame position in the world space.
        )mydelimiter",
             py::arg("frame_name"))

        .def("getFrameOrientation", [](raisim::ArticulatedSystem &self, size_t frame_id) {
            Mat<3, 3> mat;
            self.getFrameOrientation(frame_id, mat);
            return convert_mat_to_np(mat);
        }, R"mydelimiter(
        Get the frame orientation as a rotation matrix expressed in the Cartesian world frame

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3,3]]: the coordinate frame orientation in the world space.
        )mydelimiter",
        py::arg("frame_id"))

        .def("getFrameOrientation", [](raisim::ArticulatedSystem &self, const std::string& frame_name) {
            Mat<3, 3> mat;
            self.getFrameOrientation(frame_name, mat);
            return convert_mat_to_np(mat);
           }, R"mydelimiter(
        Get the frame orientation as a rotation matrix expressed in the Cartesian world frame

        Args:
            frame_name (string): frame name.

        Returns:
            np.array[float[3,3]]: the coordinate frame orientation (quaternion) in the world space.
        )mydelimiter",
        py::arg("frame_name"))

        .def("getFrameVelocity", [](raisim::ArticulatedSystem &self, size_t frame_id) {
            Vec<3> vec;
            self.getFrameVelocity(frame_id, vec);
            return convert_vec_to_np(vec);
        }, R"mydelimiter(
        Get the frame linear velocity expressed in the Cartesian world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: the coordinate frame linear velocity in the world space.
        )mydelimiter",
        py::arg("frame_id"))

        .def("getFrameAngularVelocity", [](raisim::ArticulatedSystem &self, size_t frame_id) {
            Vec<3> vec;
            self.getFrameAngularVelocity(frame_id, vec);
            return convert_vec_to_np(vec);
        }, R"mydelimiter(
        Get the frame angular velocity expressed in the Cartesian world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: the coordinate frame angular velocity in the world space.
        )mydelimiter",
        py::arg("frame_id"))

        .def("getFrameAcceleration", [](raisim::ArticulatedSystem &self, const std::string &frame_name) {
            Vec<3> acc;
            self.getFrameAcceleration(frame_name, acc);
            return convert_vec_to_np(acc);
        }, R"mydelimiter(
        Get the frame linear acceleration expressed in the Cartesian world frame.

        Args:
            frame_name (string): frame name.

        Returns:
            np.array[float[3]]: the coordinate frame linear acceleration in the world space.
        )mydelimiter",
        py::arg("frame_name"))

        .def("getFrameAcceleration", [](raisim::ArticulatedSystem &self, const raisim::CoordinateFrame &frame) {
            Vec<3> acc;
            self.getFrameAcceleration(frame, acc);
            return convert_vec_to_np(acc);
        }, R"mydelimiter(
        Get the frame linear acceleration expressed in the Cartesian world frame.

        Args:
            frame (CoordinateFrame): coordinate frame.

        Returns:
            np.array[float[3]]: the coordinate frame linear acceleration in the world space.
        )mydelimiter",
        py::arg("frame"))


        .def("getPosition", [](raisim::ArticulatedSystem &self, size_t frame_id) {
            Vec<3> pos;
            self.getPosition(frame_id, pos);
            return convert_vec_to_np(pos);
        }, R"mydelimiter(
        Get the joint frame's position with respect to the world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: position of the joint frame expressed in the world frame.
        )mydelimiter",
        py::arg("frame_id"))


        .def("getFrameOrientation", [](raisim::ArticulatedSystem &self, size_t frame_id) {
            Mat<3, 3> rot;
            self.getFrameOrientation(frame_id, rot);
            return convert_mat_to_np(rot);
        }, R"mydelimiter(
        Get the joint frame's orientation (expressed as a rotation matrix) with respect to the world frame.

        Args:
            frame_id (int): frame index.

        Returns:
            np.array[float[3]]: orientation (rotation matrix) of the joint frame expressed in the world frame.
        )mydelimiter",
        py::arg("frame_id"))


        .def("getFrameOrientation", [](raisim::ArticulatedSystem &self, size_t frame_id) {
            Mat<3, 3> rot;
            self.getFrameOrientation(frame_id, rot);
            Vec<4> quat;
            rotMatToQuat(rot, quat);
            return convert_vec_to_np(quat);
        }, R"mydelimiter(
        Get the joint frame's orientation (expressed as a quaternion [w,x,y,z]) with respect to the world frame.

        Args:
            frame_id (int): frame index.

        Returns:
            np.array[float[3]]: orientation (quaternion) of the joint frame expressed in the world frame.
        )mydelimiter",
        py::arg("frame_id"))


        .def("getVelocity", [](raisim::ArticulatedSystem &self, size_t frame_id) {
            Vec<3> vel;
            self.getVelocity(frame_id, vel);
            return convert_vec_to_np(vel);
        }, R"mydelimiter(
        Get the joint frame's linear velocity with respect to the world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: velocity of the joint frame expressed in the world frame.
        )mydelimiter",
        py::arg("frame_id"))


        .def("getVelocity", [](raisim::ArticulatedSystem &self, NDArray jacobian) {
            SparseJacobian jac;
            MatDyn mat = convert_np_to_matdyn(jacobian);
            jac.v = mat;
            Vec<3> vel;
            self.getVelocity(jac, vel);
            return convert_vec_to_np(vel);
        }, R"mydelimiter(
        Return the velocity of the point of the sparse linear jacobian.

        Args:
            sparse_linear_jacobian (np.array[float[3,n]]): sparse linear jacobian.

        Returns:
            np.array[float[3]]: velocity of the point expressed in the world frame.
        )mydelimiter",
        py::arg("sparse_linear_jacobian"))


        .def("getVelocity", [](raisim::ArticulatedSystem &self, size_t body_id,
                NDArray body_pos) {
            Vec<3> pos = convert_np_to_vec<3>(body_pos);
            Vec<3> vel;
            self.getVelocity(body_id, pos, vel);
            return convert_vec_to_np(vel);
        }, R"mydelimiter(
        Return the velocity of a point (expressed in the body frame) in the world frame.

        Args:
            body_id (int): body id.
            body_pos (np.array[float[3]]): position of a point on the body frame.

        Returns:
            np.array[float[3]]: velocity of the body expressed in the world frame.
        )mydelimiter",
        py::arg("body_id"), py::arg("body_pos"))


        .def("getAngularVelocity", [](raisim::ArticulatedSystem &self, size_t body_id) {
            Vec<3> vel;
            self.getAngularVelocity(body_id, vel);
            return convert_vec_to_np(vel);
        }, R"mydelimiter(
        Get the angular velocity of the body with respect to the world frame.

        Args:
            body_id (int): body id.

        Returns:
            np.array[float[3]]: angular velocity of the body expressed in the world frame.
        )mydelimiter",
        py::arg("body_id"))


        .def("getDenseFrameJacobian", [](raisim::ArticulatedSystem &self, std::string frameName) {
            size_t n = self.getDOF();
            Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(3, n);
            self.getDenseFrameJacobian(frameName, jac);
            return jac;
        }, R"mydelimiter(
        Get the dense frame linear jacobian.

        .. math:: v_{lin} = J(q) \dot{q}

        Args:
            frame_name (string): frame name defined in the URDF. A frame is attached to every joint by default.

        Returns:
            np.array[float[3, n]]: dense linear jacobian of the frame.
        )mydelimiter",
        py::arg("frame_name"))


        .def("getDenseFrameRotationalJacobian", [](raisim::ArticulatedSystem &self, std::string frameName) {
            size_t n = self.getDOF();
            Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(3, n);
            self.getDenseFrameRotationalJacobian(frameName, jac);
            return jac;
        }, R"mydelimiter(
        Get the dense rotational jacobian.

        .. math:: \omega = J(q) \dot{q}

        Args:
            body_idx (int): body index.

        Returns:
            np.array[float[3, n]]: dense rotational jacobian of the frame.
        )mydelimiter",
        py::arg("body_idx"))


        .def("getBodyIdx", &raisim::ArticulatedSystem::getBodyIdx, R"mydelimiter(
        Return the body index associated with the given name.

        Args:
            name (str): body name.

        Returns:
            int: body index.
        )mydelimiter",
        py::arg("name"))


        .def("getDOF", &raisim::ArticulatedSystem::getDOF, R"mydelimiter(
        Return the number of degrees of freedom.

        Returns:
            int: the number of degrees of freedom.
        )mydelimiter")

        .def("getGeneralizedCoordinateDim", &raisim::ArticulatedSystem::getGeneralizedCoordinateDim, R"mydelimiter(
        Return the dimension/size of the generalized coordinates vector.

        Returns:
            int: the number of generalized coordinates.
        )mydelimiter")


        .def("getBodyPosition", [](raisim::ArticulatedSystem &self, size_t body_id) {
               Vec<3> pos;
               self.getBodyPosition(body_id, pos);
               auto pos_np = convert_vec_to_np(pos);
               return pos_np;
               }, R"mydelimiter(
        Return the body position.

        Args:
            body_id (int): body id.
        Returns:
            np.array[float[3]]: body position.

        )mydelimiter",
        py::arg("body_id"))

        .def("getBodyOrientation", [](raisim::ArticulatedSystem &self, size_t body_id) {
               Mat<3,3> ori;
               self.getBodyOrientation(body_id, ori);
               auto ori_np = convert_mat_to_np(ori);
               return ori_np;
             }, R"mydelimiter(
        Return the body orientation.

        Args:
            body_id (int): body id.
        Returns:
            np.array[float[3,3]]: body orientation.

        )mydelimiter",
             py::arg("body_id"))

        /* The following 5 methods can be used to directly modify dynamic/kinematic properties of the robot.
         They are made for dynamic randomization. Use them with caution since they will change the
         the model permanently. After you change the dynamic properties, call "void updateMassInfo()" to update
         some precomputed dynamic properties.
         returns the reference to joint position relative to its parent, expressed in the parent frame. */

        .def("getJointPos_P", [](raisim::ArticulatedSystem &self) {
            std::vector<raisim::Vec<3>>& positions = self.getJointPos_P();

            py::list list;
            for (auto pos : positions)
                list.append(convert_vec_to_np(pos));

            return list;
        }, R"mydelimiter(
        Return the joint Cartesian positions relative to their parent frame.

        Returns:
            list[np.array[float[3]]]: joint cartesian positions (relative to their parent frame).
        )mydelimiter")

        .def("setJointPos_P", [](raisim::ArticulatedSystem &self, py::list &list) {
            // get references to joint cartesian positions
            std::vector<raisim::Vec<3>>& positions = self.getJointPos_P();

            // check list and vector sizes
            if (list.size() != positions.size()) {
                std::ostringstream s;
                s << "error: expecting the given list to have the same size as the number of joint positions: "
                    << list.size() << " != " << positions.size() << ".";
                throw std::domain_error(s.str());
            }

            // copy each element from the list
            for (size_t i=0; i<list.size(); i++) {
                NDArray e = py::cast<NDArray>(list[i]);
                positions[i] = convert_np_to_vec<3>(e);
            }

            // update mass info (this is more pythonic)
            self.updateMassInfo();
        }, R"mydelimiter(
        Set the joint Cartesian positions relative to their parent frame.

        Args:
            positions (list[np.array[float[3]]]): joint cartesian positions (relative to their parent frame).
        )mydelimiter",
        py::arg("positions"))

        .def_prop_rw("getJointPos_P",
            [](raisim::ArticulatedSystem &self) {  // getter
                std::vector<raisim::Vec<3>> positions = self.getJointPos_P();

                py::list list;
                for (auto pos : positions)
                    list.append(convert_vec_to_np(pos));

                return list;
            },
            [](raisim::ArticulatedSystem &self, py::list &list) { // setter
                // get references to joint cartesian positions
                std::vector<raisim::Vec<3>>& positions = self.getJointPos_P();

                // check list and vector sizes
                if (list.size() != positions.size()) {
                    std::ostringstream s;
                    s << "error: expecting the given list to have the same size as the number of joint positions: "
                        << list.size() << " != " << positions.size() << ".";
                    throw std::domain_error(s.str());
                }

                // copy each element from the list
                for (size_t i=0; i<list.size(); i++) {
                    NDArray e = py::cast<NDArray>(list[i]);
                    positions[i] = convert_np_to_vec<3>(e);
                }

                // update mass info (this is more pythonic)
                self.updateMassInfo();
            })



        .def("getMass", py::overload_cast<>(&raisim::ArticulatedSystem::getMass), py::rv_policy::reference, R"mydelimiter(
        Return the body/link masses.

        Returns:
            list[double]: body masses.
        )mydelimiter")

        .def("setMass", [](raisim::ArticulatedSystem &self, std::vector<double> &list) {
            // get references to masses
            std::vector<double>& masses = self.getMass();

            // check list and vector sizes
            if (list.size() != masses.size()) {
                std::ostringstream s;
                s << "error: expecting the given list to have the same size as the number of masses: "
                    << list.size() << " != " << masses.size() << ".";
                throw std::domain_error(s.str());
            }

            // copy each element
            for (size_t i=0; i<list.size(); i++) {
                masses[i] = list[i];
            }

            // update mass info (this is more pythonic)
            self.updateMassInfo();
        }, R"mydelimiter(
        Set the body/link masses.

        Args:
            masses (list[double]): body masses.
        )mydelimiter",
        py::arg("masses"))

        .def_prop_rw("masses",
            py::overload_cast<>(&raisim::ArticulatedSystem::getMass), // getter
            [](raisim::ArticulatedSystem &self, std::vector<double> &list) { // setter
                // get references to masses
                std::vector<double>& masses = self.getMass();

                // check list and vector sizes
                if (list.size() != masses.size()) {
                    std::ostringstream s;
                    s << "error: expecting the given list to have the same size as the number of masses: "
                        << list.size() << " != " << masses.size() << ".";
                    throw std::domain_error(s.str());
                }

                // copy each element
                for (size_t i=0; i<list.size(); i++) {
                    masses[i] = list[i];
                }

                // update mass info (this is more pythonic)
                self.updateMassInfo();
            })


        .def("getInertia", [](raisim::ArticulatedSystem &self) {
            std::vector<raisim::Mat<3, 3> >& inertia = self.getInertia();

            py::list list;
            for (auto I : inertia)
                list.append(convert_mat_to_np(I));

            return list;
        }, R"mydelimiter(
        Return the inertias (one for each body).

        Returns:
            list[np.array[float[3,3]]]: inertias.
        )mydelimiter")

        .def("setInertia", [](raisim::ArticulatedSystem &self, py::list &list) {
            // get references to inertias
            std::vector<raisim::Mat<3, 3> >& inertia = self.getInertia();

            // check list and vector sizes
            if (list.size() != inertia.size()) {
                std::ostringstream s;
                s << "error: expecting the given list to have the same size as the number of inertia: "
                    << list.size() << " != " << inertia.size() << ".";
                throw std::domain_error(s.str());
            }

            // copy each element from the list
            for (size_t i=0; i<list.size(); i++) {
                NDArray e = py::cast<NDArray>(list[i]);
                inertia[i] = convert_np_to_mat<3, 3>(e);
            }

            // update mass info (this is more pythonic)
            self.updateMassInfo();
        }, R"mydelimiter(
        Set the inertias (one for each body).

        Args:
            inertias (list[np.array[float[3,3]]]): body inertias.
        )mydelimiter",
        py::arg("inertias"))

        .def_prop_rw("inertias",
            [](raisim::ArticulatedSystem &self) {  // getter
                std::vector<raisim::Mat<3, 3> >& inertia = self.getInertia();

                py::list list;
                for (auto I : inertia)
                    list.append(convert_mat_to_np(I));

                return list;
            },
            [](raisim::ArticulatedSystem &self, py::list &list) { // setter
                // get references to inertias
                std::vector<raisim::Mat<3, 3> >& inertia = self.getInertia();

                // check list and vector sizes
                if (list.size() != inertia.size()) {
                    std::ostringstream s;
                    s << "error: expecting the given list to have the same size as the number of inertia: "
                        << list.size() << " != " << inertia.size() << ".";
                    throw std::domain_error(s.str());
                }

                // copy each element from the list
                for (size_t i=0; i<list.size(); i++) {
                    NDArray e = py::cast<NDArray>(list[i]);
                    inertia[i] = convert_np_to_mat<3, 3>(e);
                }

                // update mass info (this is more pythonic)
                self.updateMassInfo();
            })



        .def("getBodyCOM_B", [](raisim::ArticulatedSystem &self) {
            std::vector<raisim::Vec<3>> positions = self.getBodyCOM_B();

            py::list list;
            for (auto pos : positions)
                list.append(convert_vec_to_np(pos));

            return list;
        }, R"mydelimiter(
        Return the center of mass of each link (expressed in their body frame).

        Returns:
            list[np.array[float[3]]]: center of mass of each link (expressed in the body frame).
        )mydelimiter")

        .def("setLinkCOM", [](raisim::ArticulatedSystem &self, py::list &list) {
            // get references to joint cartesian positions
            std::vector<raisim::Vec<3>>& positions = self.getBodyCOM_B();

            // check list and vector sizes
            if (list.size() != positions.size()) {
                std::ostringstream s;
                s << "error: expecting the given list to have the same size as the number of CoMs: "
                    << list.size() << " != " << positions.size() << ".";
                throw std::domain_error(s.str());
             }

            // copy each element from the list
            for (size_t i=0; i<list.size(); i++) {
                NDArray e = py::cast<NDArray>(list[i]);
                positions[i] = convert_np_to_vec<3>(e);
            }

            // update mass info (this is more pythonic)
            self.updateMassInfo();
        }, R"mydelimiter(
        Set the center of mass of each link (expressed in their body frame).

        Args:
            coms (list[np.array[float[3]]]): center of mass of each link (expressed in the body frame).
        )mydelimiter",
        py::arg("coms"))

        .def_prop_rw("linkComs",
            [](raisim::ArticulatedSystem &self) {  // getter
                std::vector<raisim::Vec<3>>& positions = self.getBodyCOM_B();

                py::list list;
                for (auto pos : positions)
                    list.append(convert_vec_to_np(pos));

                return list;
            },
            [](raisim::ArticulatedSystem &self, py::list &list) { // setter
                // get references to joint cartesian positions
                std::vector<raisim::Vec<3>>& positions = self.getBodyCOM_B();

                // check list and vector sizes
                if (list.size() != positions.size()) {
                    std::ostringstream s;
                    s << "error: expecting the given list to have the same size as the number of CoMs: "
                        << list.size() << " != " << positions.size() << ".";
                    throw std::domain_error(s.str());
                 }

                // copy each element from the list
                for (size_t i=0; i<list.size(); i++) {
                    NDArray e = py::cast<NDArray>(list[i]);
                    positions[i] = convert_np_to_vec<3>(e);
                }

                // update mass info (this is more pythonic)
                self.updateMassInfo();
            })



//        .def("get_collision_bodies", [](raisim::ArticulatedSystem &self) {
//            return self.getCollisionBodies();
//        }, R"mydelimiter(
//        Return the collision bodies.
//
//        Returns:
//            list[CollisionDefinition]: collision bodies.
//        )mydelimiter")
//        .def("set_collision_bodies", [](raisim::ArticulatedSystem &self, raisim::CollisionSet &list) {
//            // get references to collision bodies
//            std::vector<CollisionDefinition> collisions = self.getCollisionBodies();
//
//            // check list and vector sizes
//            if (list.size() != collisions.size()) {
//                std::ostringstream s;
//                s << "error: expecting the given list to have the same size as the number of collision bodies: "
//                     << list.size() << " != " << collisions.size() << ".";
//                throw std::domain_error(s.str());
//            }
//
//            // copy each element from the list
//            for (size_t i=0; i<list.size(); i++) {
//                collisions[i] = list[i];
//            }
//
//            // update mass info (this is more pythonic)
//            self.updateMassInfo();
//        }, R"mydelimiter(
//        Set the collision bodies.
//
//        Args:
//            collisions (list[CollisionDefinition]): collision bodies.
//        )mydelimiter",
//        py::arg("collisions"))
//
//        .def_prop_rw("collision_bodies",
//            [](raisim::ArticulatedSystem &self) {  // getter
//                return self.getCollisionBodies();
//            },
//            [](raisim::ArticulatedSystem &self, py::list list) { // setter
//                // get references to collision bodies
//                std::vector<CollisionDefinition> collisions = self.getCollisionBodies();
//
//                // check list and vector sizes
//                if (list.size() != collisions.size()) {
//                    std::ostringstream s;
//                    s << "error: expecting the given list to have the same size as the number of collision bodies: "
//                         << list.size() << " != " << collisions.size() << ".";
//                    throw std::domain_error(s.str());
//                }
//
//                // copy each element from the list
//                for (size_t i=0; i<list.size(); i++) {
//                    collisions[i] = list[i].cast<CollisionDefinition>();
//                }
//
//                // update mass info (this is more pythonic)
//                self.updateMassInfo();
//            })
//
//
//        .def("get_collision_body", &raisim::ArticulatedSystem::getCollisionBody, R"mydelimiter(
//        Return the collision body associated with the given name.
//
//        Args:
//            name (str): collision body name.
//
//        Returns:
//            CollisionDefinition: collision body.
//        )mydelimiter",
//        py::arg("name"))


        // this is automatically done when you use properties so you don't have to call it (more pythonic)
        .def("updateMassInfo", &raisim::ArticulatedSystem::updateMassInfo,
            "Update the mass information. This function must be called after we change the dynamic parameters.")


        .def("getMass", py::overload_cast<size_t>(&raisim::ArticulatedSystem::getMass, py::const_), R"mydelimiter(
        Get the mass of the link.

        Args:
            link_idx (int): link index.

        Returns:
            float: mass value.
        )mydelimiter",
        py::arg("localIdx"))

        .def("setMass", &raisim::ArticulatedSystem::setMass, R"mydelimiter(
        Set the mass of the link.

        Args:
            link_idx (int): link index.
            value (float): mass value.
        )mydelimiter",
        py::arg("link_idx"), py::arg("value"))


        .def("getTotalMass", &raisim::ArticulatedSystem::getTotalMass, R"mydelimiter(
        Get the total mass of the system.

        Returns:
            float: total mass value.
        )mydelimiter")


        .def("setExternalForce", [](raisim::ArticulatedSystem &self, size_t local_idx, NDArray pos, NDArray force) {
	        Vec<3> f = convert_np_to_vec<3>(force);
            Vec<3> p = convert_np_to_vec<3>(pos);
	        self.setExternalForce(local_idx, p, f);
	    }, R"mydelimiter(
	    Set the external force on the body.

	    Args:
	        local_idx (int): local/link index.
	        force (np.array[float[3]]): force vector.
	    )mydelimiter",
	    py::arg("local_idx"), py::arg("pos"), py::arg("force"))


	    .def("setExternalTorque", [](raisim::ArticulatedSystem &self, size_t local_idx, NDArray torque) {
	        Vec<3> t = convert_np_to_vec<3>(torque);
	        self.setExternalTorque(local_idx, t);
	    }, R"mydelimiter(
	    Set the external force on the body.

	    Args:
	        local_idx (int): local/link index.
	        force (np.array[float[3]]): force vector.
	    )mydelimiter",
	    py::arg("local_idx"), py::arg("torque"))


	    .def("setExternalForce", [](raisim::ArticulatedSystem &self, size_t local_idx,
	            raisim::ArticulatedSystem::Frame force_frame, NDArray force,
	            raisim::ArticulatedSystem::Frame pos_frame, NDArray position) {
	        Vec<3> f = convert_np_to_vec<3>(force);
	        self.setExternalForce(local_idx, f);
	    }, R"mydelimiter(
	    Set the external force on the specified point on the body.

	    Args:
	        local_idx (int): local/link index.
	        force_frame (Frame): frame in which the force vector is expressed in, select between {WORLD_FRAME,
	            BODY_FRAME, PARENT_FRAME}.
	        force (np.array[float[3]]): force vector.
	        pos_frame (Frame): frame in which the position vector is expressed in, select between {WORLD_FRAME,
	            BODY_FRAME, PARENT_FRAME}.
	        position (np.array[float[3]]): position vector.

	    )mydelimiter",
	    py::arg("local_idx"), py::arg("force_frame"), py::arg("force"), py::arg("pos_frame"), py::arg("position"))


        .def("setControlMode", &raisim::ArticulatedSystem::setControlMode, R"mydelimiter(
	    Set the control mode.

	    Args:
	        mode (ControlMode.Type): control mode type, select between {FORCE_AND_TORQUE, PD_PLUS_FEEDFORWARD_TORQUE,
	            VELOCITY_PLUS_FEEDFORWARD_TORQUE}
	    )mydelimiter",
	    py::arg("mode"))

        .def("getControlMode", &raisim::ArticulatedSystem::getControlMode, R"mydelimiter(
	    Get the control mode.

	    Returns:
	        ControlMode.Type: control mode type which is one of {FORCE_AND_TORQUE, PD_PLUS_FEEDFORWARD_TORQUE,
	            VELOCITY_PLUS_FEEDFORWARD_TORQUE}
	    )mydelimiter")


        /* set PD targets. It is effective only in the control mode "PD_PLUS_FEEDFORWARD_TORQUE". set any arbitrary
        number for unactuated degrees of freedom */

        .def("setPdTarget", py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &>(&raisim::ArticulatedSystem::setPdTarget), R"mydelimiter(
	    Set the PD targets. It is effective only in the control mode 'PD_PLUS_FEEDFORWARD_TORQUE'. Set any arbitrary
        number for unactuated degrees of freedom.

	    Args:
	        pos_targets (np.array[float[n]]): position targets.
	        vel_targets (np.array[float[n]]): velocity targets.
	    )mydelimiter",
	    py::arg("pos_targets"), py::arg("vel_targets"))


        .def("setPdGains", py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &>(&raisim::ArticulatedSystem::setPdGains), R"mydelimiter(
	    Set the PD gains. It is effective only in the control mode 'PD_PLUS_FEEDFORWARD_TORQUE'. Set any arbitrary
        number for unactuated degrees of freedom.

	    Args:
	        p_gains (np.array[float[n]]): P gains.
	        d_gains (np.array[float[n]]): D gains.
	    )mydelimiter",
	    py::arg("p_gains"), py::arg("d_gains"))


	    .def("setJointDamping", py::overload_cast<const Eigen::VectorXd &>(&raisim::ArticulatedSystem::setJointDamping), R"mydelimiter(
	    Set the joint dampings (passive elements at the joints).

	    Args:
	        dampings (np.array[float[n]]): joint damping coefficients.
	    )mydelimiter",
	    py::arg("dampings"))


        .def("computeSparseInverse", [](raisim::ArticulatedSystem &self, NDArray mass) {
            MatDyn M = convert_np_to_matdyn(mass);
            MatDyn Minv;
            self.computeSparseInverse(M, Minv);
            return convert_matdyn_to_np(Minv);
        }, R"mydelimiter(
	    This computes the inverse mass matrix given the mass matrix. The return type is dense. It exploits the
	    sparsity of the mass matrix to efficiently perform the computation.

	    Args:
	        mass_matrix (np.array[float[n,n]]): mass matrix.

	    Returns:
	        np.array(float[n,n]): dense inverse matrix.
	    )mydelimiter",
	    py::arg("mass_matrix"))
        .def("massMatrixVecMul", [](raisim::ArticulatedSystem &self, NDArray vector) {
            VecDyn vec1 = convert_np_to_vecdyn(vector);
            VecDyn vec;
            self.massMatrixVecMul(vec1, vec);
            return convert_vecdyn_to_np(vec);
        }, py::arg("vector"))


//	    .def("mass_matrix_vector_multiplication", [](raisim::ArticulatedSystem &self, NDArray vector) {
//            VecDyn vec1 = convert_np_to_vecdyn(vector);
//            VecDyn vec;
//            self.massMatrixVecMul(vec1, vec);
//            return convert_vecdyn_to_np(vec);
//        }, R"mydelimiter(
//	    This method exploits the sparsity of the mass matrix. If the mass matrix is nearly dense, it will be slower
//	    than your ordinary matrix multiplication which is probably vectorized.
//
//	    Args:
//	        vector (np.array[float[n]]): vector to be multiplied by the mass matrix.
//
//	    Returns:
//	        np.array(float[n]): resulting vector.
//	    )mydelimiter",
//	    py::arg("vector"))


	    .def("ignoreCollisionBetween", &raisim::ArticulatedSystem::ignoreCollisionBetween, R"mydelimiter(
	    Ignore collision between the 2 specified bodies.

	    Args:
	        body_idx1 (int): first body index.
	        body_idx2 (int): second body index.
	    )mydelimiter",
	    py::arg("body_idx1"), py::arg("body_idx2"))


	    .def("getOptions", &raisim::ArticulatedSystem::getOptions, R"mydelimiter(
	    Return the options associated with the articulated system.

	    Returns:
	        ArticulatedSystemOption: options for the articulated system.
	    )mydelimiter")


	    .def("getBodyNames", &raisim::ArticulatedSystem::getBodyNames, R"mydelimiter(
	    Return the body names.

	    Returns:
	        list[str]: list of body names.
	    )mydelimiter")


        .def("getVisOb", py::overload_cast<>(&raisim::ArticulatedSystem::getVisOb), R"mydelimiter(
	    Get the visual objects.

	    Returns:
	        list[VisObject]: list of visual objects.
	    )mydelimiter")


	    .def("getVisColOb", py::overload_cast<>(&raisim::ArticulatedSystem::getVisColOb), R"mydelimiter(
	    Get the visual collision objects.

	    Returns:
	        list[VisObject]: list of visual collision objects.
	    )mydelimiter")


	    .def("getVisualObjectPosition", [](raisim::ArticulatedSystem &self, size_t body_idx) {
	        Vec<3> pos;
            Mat<3,3> rot;
            self.getVisObPose(body_idx, rot, pos);

            Vec<4> quat;
            rotMatToQuat(rot, quat);

            auto position = convert_vec_to_np(pos);
            return position;
	    }, R"mydelimiter(
	    Get the visual object position.

	    Args:
	        body_idx (int): body index.

	    Returns:
	        np.array[float[3]]: visual object position.
	    )mydelimiter")

	    .def("getVisualObjectOrientation", [](raisim::ArticulatedSystem &self, size_t body_idx) {
	      Vec<3> pos;
	      Mat<3,3> rot;
	      self.getVisObPose(body_idx, rot, pos);
	      Vec<4> quat;
	      rotMatToQuat(rot, quat);

	      auto orientation = convert_vec_to_np(quat);
	      return orientation;
	      }, R"mydelimiter(
	    Get the visual object orientation in quaternion.

	    Args:
	        body_idx (int): body index.

	    Returns:
	        np.array[float[4]]: visual object orientation (expressed as a quaternion [w,x,y,z]).
	      )mydelimiter")


	    .def("getVisColObPosition", [](raisim::ArticulatedSystem &self, size_t body_idx) {
	        Vec<3> pos;
            Mat<3,3> rot;
            self.getVisColObPose(body_idx, rot, pos);

            Vec<4> quat;
            rotMatToQuat(rot, quat);

            auto position = convert_vec_to_np(pos);
            return position;
	    }, R"mydelimiter(
	    Get the visual collision object position.

	    Args:
	        body_idx (int): body index.

	    Returns:
	        np.array[float[3]]: visual object position.
	    )mydelimiter")

	    .def("getVisColObOrientation", [](raisim::ArticulatedSystem &self, size_t body_idx) {
	      Vec<3> pos;
	      Mat<3,3> rot;
	      self.getVisColObPose(body_idx, rot, pos);

	      Vec<4> quat;
	      rotMatToQuat(rot, quat);

	      auto orientation = convert_vec_to_np(quat);
	      return orientation;
	      }, R"mydelimiter(
	    Get the visual collision object orientation in a quaternion.

	    Args:
	        body_idx (int): body index.

	    Returns:
	        np.array[float[4]]: visual object orientation (expressed as a quaternion [w,x,y,z]).
	      )mydelimiter")


        .def("getResourceDir", &raisim::ArticulatedSystem::getResourceDir, R"mydelimiter(
	    Get the robot resource directory.

	    Returns:
	        str: robot resource directory.
	    )mydelimiter")


	    .def("getRobotDescriptionfFileName", &raisim::ArticulatedSystem::getRobotDescriptionfFileName, R"mydelimiter(
	    Get the robot description filename (e.g. path to the URDF).

	    Returns:
	        str: robot description filename.
	    )mydelimiter")


	    .def("getRobotDescriptionfTopDirName", &raisim::ArticulatedSystem::getRobotDescriptionfTopDirName, R"mydelimiter(
	    Get the robot description top directory name.

	    Returns:
	        str: robot description top directory name.
	    )mydelimiter")


        /* change the base position and orientation of the base. */
        .def("setBasePos", [](raisim::ArticulatedSystem &self, NDArray position) {
	        Vec<3> pos = convert_np_to_vec<3>(position);
            self.setBasePos(pos);
	    }, R"mydelimiter(
	    Set the base position.

	    Args:
	        position (np.array[float[3]]): new base position.
	    )mydelimiter",
	    py::arg("position"))

	    .def("setBaseOrientation", [](raisim::ArticulatedSystem &self, NDArray orientation) {
	        Mat<3,3> rot;
	        if (orientation.size() == 3) { // rpy angles
	            Vec<3> rpy = convert_np_to_vec<3>(orientation);
                rpyToRotMat_intrinsic(rpy, rot);
	        } else if (orientation.size() == 4) {
	            Vec<4> quat = convert_np_to_vec<4>(orientation);
                quatToRotMat(quat, rot);
	        } else if (orientation.size() == 9) {
	            rot = convert_np_to_mat<3,3>(orientation);
	        } else {
                std::ostringstream s;
                s << "error: expecting the given orientation to have a size of 3 (RPY), 4 (quaternion), or 9 " <<
                    "(rotation matrix), but got instead a size of " << orientation.size() << ".";
                throw std::domain_error(s.str());
	        }
	        self.setBaseOrientation(rot);
	    }, R"mydelimiter(
	    Set the base orientation.

	    Args:
	        orientation (np.array[float[3]], np.array[float[4]], np.array[float[3,3]]): new base orientation.
	    )mydelimiter",
	    py::arg("orientation"))


	    .def("setActuationLimits", &raisim::ArticulatedSystem::setActuationLimits, R"mydelimiter(
	    Set the upper and lower limits in actuation forces.

	    Args:
	        upper (np.array[float[n]]): upper limits.
	        lower (np.array[float[n]]): lower limits.
	    )mydelimiter",
	    py::arg("upper"), py::arg("lower"))

	    .def("setComputeInverseDynamics", &raisim::ArticulatedSystem::setComputeInverseDynamics, R"mydelimiter(
	    Enable or disable inverse dynamics computation.

	    Args:
	        flag (bool): true to enable inverse dynamics.
	    )mydelimiter",
	    py::arg("flag"))

	    .def("getComputeInverseDynamics", &raisim::ArticulatedSystem::getComputeInverseDynamics, R"mydelimiter(
	    Check if inverse dynamics computation is enabled.

	    Returns:
	        bool: true if inverse dynamics is enabled.
	    )mydelimiter")

	    .def("getForceAtJointInWorldFrame", [](const raisim::ArticulatedSystem &self, size_t joint_id) {
            return convert_vec_to_np(self.getForceAtJointInWorldFrame(joint_id));
	    }, R"mydelimiter(
	    Get the force at the specified joint in the world frame.

	    Args:
	        joint_id (int): joint id.

	    Returns:
	        np.array[float[3]]: joint force in world frame.
	    )mydelimiter",
	    py::arg("joint_id"))

	    .def("getTorqueAtJointInWorldFrame", [](const raisim::ArticulatedSystem &self, size_t joint_id) {
            return convert_vec_to_np(self.getTorqueAtJointInWorldFrame(joint_id));
	    }, R"mydelimiter(
	    Get the torque at the specified joint in the world frame.

	    Args:
	        joint_id (int): joint id.

	    Returns:
	        np.array[float[3]]: joint torque in world frame.
	    )mydelimiter",
	    py::arg("joint_id"))

	    .def("getAllowedNumberOfInternalContactsBetweenTwoBodies",
	         &raisim::ArticulatedSystem::getAllowedNumberOfInternalContactsBetweenTwoBodies, R"mydelimiter(
	    Get the allowed number of internal contacts between two bodies.

	    Returns:
	        int: maximum number of internal contacts.
	    )mydelimiter")

	    .def("setAllowedNumberOfInternalContactsBetweenTwoBodies",
	         &raisim::ArticulatedSystem::setAllowedNumberOfInternalContactsBetweenTwoBodies, R"mydelimiter(
	    Set the allowed number of internal contacts between two bodies.

	    Args:
	        count (int): maximum number of internal contacts.
	    )mydelimiter",
	    py::arg("count"))

        .def("isSecondOrderOrHigher", &raisim::ArticulatedSystem::isSecondOrderOrHigher, py::arg("scheme"))
        .def("getBasePosition", [](const raisim::ArticulatedSystem &self) {
            return convert_vec_to_np(self.getBasePosition());
        })
        .def("setBasePos_e", [](raisim::ArticulatedSystem &self, NDArray position) {
            Vec<3> pos = convert_np_to_vec<3>(position);
            Eigen::Vector3d eig = pos.e();
            self.setBasePos_e(eig);
        }, py::arg("position"))
        .def("setBaseOrientation_e", [](raisim::ArticulatedSystem &self, NDArray orientation) {
            Mat<3, 3> rot = convert_np_to_mat<3, 3>(orientation);
            Eigen::Matrix3d eig = rot.e();
            self.setBaseOrientation_e(eig);
        }, py::arg("orientation"))
        .def("setBaseVelocity", [](raisim::ArticulatedSystem &self, NDArray velocity) {
            self.setBaseVelocity(convert_np_to_vec<3>(velocity));
        }, py::arg("velocity"))
        .def("setBaseAngularVelocity", [](raisim::ArticulatedSystem &self, NDArray velocity) {
            self.setBaseAngularVelocity(convert_np_to_vec<3>(velocity));
        }, py::arg("velocity"))
        .def("getBodyPose", [](const raisim::ArticulatedSystem &self, size_t body_idx) {
            Mat<3, 3> rot;
            Vec<3> pos;
            self.getBodyPose(body_idx, rot, pos);
            return py::make_tuple(convert_vec_to_np(pos), convert_mat_to_np(rot));
        }, py::arg("body_idx"))
        .def("getBodyCOM_W", [vec3_list_to_pylist](raisim::ArticulatedSystem &self) {
            return vec3_list_to_pylist(self.getBodyCOM_W());
        })
        .def("getCOM", [](const raisim::ArticulatedSystem &self) {
            return convert_vec_to_np(self.getCOM());
        })
        .def("getKineticEnergy", &raisim::ArticulatedSystem::getKineticEnergy)
        .def("getAngularMomentum", [](const raisim::ArticulatedSystem &self, NDArray reference_point) {
            Vec<3> ref = convert_np_to_vec<3>(reference_point);
            Vec<3> momentum;
            self.getAngularMomentum(ref, momentum);
            return convert_vec_to_np(momentum);
        }, py::arg("reference_point"))
        .def("getGeneralizedVelocityDim", &raisim::ArticulatedSystem::getGeneralizedVelocityDim)
        .def("getGeneralizedVelocityIndex", &raisim::ArticulatedSystem::getGeneralizedVelocityIndex, py::arg("name"))
        .def("getActuationUpperLimits", [](const raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getActuationUpperLimits());
        })
        .def("getActuationLowerLimits", [](const raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getActuationLowerLimits());
        })
        .def("getJointLimits", [](const raisim::ArticulatedSystem &self) {
            py::list out;
            for (const auto &lim : self.getJointLimits())
                out.append(convert_vec_to_np(lim));
            return out;
        })
        .def("setJointLimits", [](raisim::ArticulatedSystem &self, py::list limits) {
            std::vector<raisim::Vec<2>> lims;
            lims.reserve(limits.size());
            for (auto item : limits) {
                NDArray arr = py::cast<NDArray>(item);
                lims.push_back(convert_np_to_vec<2>(arr));
            }
            self.setJointLimits(lims);
        }, py::arg("joint_limits"))
        .def("getJointVelocityLimits", [](const raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getJointVelocityLimits());
        })
        .def("setJointVelocityLimits", [](raisim::ArticulatedSystem &self, NDArray limits) {
            self.setJointVelocityLimits(convert_np_to_vecdyn(limits));
        }, py::arg("velocity_limits"))
        .def("getRotorInertia", [](const raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getRotorInertia());
        })
        .def("setRotorInertia", [](raisim::ArticulatedSystem &self, NDArray inertia) {
            self.setRotorInertia(convert_np_to_vecdyn(inertia));
        }, py::arg("rotor_inertia"))
        .def("getNumberOfJoints", &raisim::ArticulatedSystem::getNumberOfJoints)
        .def("getJointType", py::overload_cast<size_t>(&raisim::ArticulatedSystem::getJointType), py::arg("joint_index"))
        .def("getJointAxis", [](const raisim::ArticulatedSystem &self, size_t joint_index) {
            return convert_vec_to_np(self.getJointAxis(joint_index));
        }, py::arg("joint_index"))
        .def("getJointAxis_P", [vec3_list_to_pylist](raisim::ArticulatedSystem &self) {
            return vec3_list_to_pylist(self.getJointAxis_P());
        })
        .def("getMappingFromBodyIndexToGeneralizedCoordinateIndex",
             &raisim::ArticulatedSystem::getMappingFromBodyIndexToGeneralizedCoordinateIndex)
        .def("getMappingFromBodyIndexToGeneralizedVelocityIndex",
             &raisim::ArticulatedSystem::getMappingFromBodyIndexToGeneralizedVelocityIndex)
        .def("getJoint", &raisim::ArticulatedSystem::getJoint, py::arg("name"))
        .def("getLink", &raisim::ArticulatedSystem::getLink, py::arg("name"))
        .def("getJointLimitViolations", [](raisim::ArticulatedSystem &self,
                                           const std::vector<raisim::contact::Single3DContactProblem> &problems) {
            raisim::contact::ContactProblems contact_problems;
            contact_problems.reserve(problems.size());
            for (const auto &problem : problems)
                contact_problems.push_back(problem);
            auto violations = self.getJointLimitViolations(contact_problems);
            std::vector<raisim::contact::Single3DContactProblem> out;
            out.reserve(violations.size());
            for (const auto *problem : violations)
                out.push_back(*problem);
            return out;
        }, py::arg("contact_problems"))
        .def("getFrameByLinkName", py::overload_cast<const std::string &>(&raisim::ArticulatedSystem::getFrameByLinkName),
             py::rv_policy::reference_internal, py::arg("name"))
        .def("getFrameIdxByLinkName", &raisim::ArticulatedSystem::getFrameIdxByLinkName, py::arg("name"))
        .def("getPositionInBodyCoordinate", [](raisim::ArticulatedSystem &self, size_t body_idx, NDArray pos_w) {
            Vec<3> pos_W = convert_np_to_vec<3>(pos_w);
            Vec<3> pos_B;
            self.getPositionInBodyCoordinate(body_idx, pos_W, pos_B);
            return convert_vec_to_np(pos_B);
        }, py::arg("body_idx"), py::arg("pos_w"))
        .def("getPositionInFrame", [](raisim::ArticulatedSystem &self, size_t frame_id, NDArray local_pos) {
            Vec<3> local = convert_np_to_vec<3>(local_pos);
            Vec<3> pos_W;
            self.getPositionInFrame(frame_id, local, pos_W);
            return convert_vec_to_np(pos_W);
        }, py::arg("frame_id"), py::arg("local_pos"))
        .def("getSparseJacobian", [](raisim::ArticulatedSystem &self, size_t body_idx, NDArray point_w) {
            Vec<3> point = convert_np_to_vec<3>(point_w);
            raisim::SparseJacobian jaco;
            jaco.resize(self.getDOF());
            self.getSparseJacobian(body_idx, point, jaco);
            return jaco;
        }, py::arg("body_idx"), py::arg("point_w"))
        .def("getSparseJacobian", [](raisim::ArticulatedSystem &self, size_t body_idx, raisim::ArticulatedSystem::Frame frame, NDArray point) {
            Vec<3> p = convert_np_to_vec<3>(point);
            raisim::SparseJacobian jaco;
            jaco.resize(self.getDOF());
            self.getSparseJacobian(body_idx, frame, p, jaco);
            return jaco;
        }, py::arg("body_idx"), py::arg("frame"), py::arg("point"))
        .def("getSparseRotationalJacobian", [](raisim::ArticulatedSystem &self, size_t body_idx) {
            raisim::SparseJacobian jaco;
            jaco.resize(self.getDOF());
            self.getSparseRotationalJacobian(body_idx, jaco);
            return jaco;
        }, py::arg("body_idx"))
        .def("getTimeDerivativeOfSparseJacobian", [](raisim::ArticulatedSystem &self, size_t body_idx, raisim::ArticulatedSystem::Frame frame, NDArray point) {
            Vec<3> p = convert_np_to_vec<3>(point);
            raisim::SparseJacobian jaco;
            jaco.resize(self.getDOF());
            self.getTimeDerivativeOfSparseJacobian(body_idx, frame, p, jaco);
            return jaco;
        }, py::arg("body_idx"), py::arg("frame"), py::arg("point"))
        .def("getTimeDerivativeOfSparseRotationalJacobian", [](raisim::ArticulatedSystem &self, size_t body_idx) {
            raisim::SparseJacobian jaco;
            jaco.resize(self.getDOF());
            self.getTimeDerivativeOfSparseRotationalJacobian(body_idx, jaco);
            return jaco;
        }, py::arg("body_idx"))
        .def_static("convertSparseJacobianToDense", [](const raisim::SparseJacobian &jaco, size_t cols) {
            if (cols == 0) {
                size_t max_idx = 0;
                for (size_t i = 0; i < jaco.size; ++i)
                    max_idx = std::max(max_idx, jaco.idx[i]);
                cols = max_idx + 1;
            }
            Eigen::MatrixXd dense(3, cols);
            dense.setZero();
            raisim::ArticulatedSystem::convertSparseJacobianToDense(jaco, dense);
            return dense;
        }, py::arg("sparse_jacobian"), py::arg("cols") = 0)
        .def("getDenseJacobian", [](raisim::ArticulatedSystem &self, size_t body_idx, NDArray point_w) {
            Vec<3> point = convert_np_to_vec<3>(point_w);
            Eigen::MatrixXd jaco(3, self.getDOF());
            jaco.setZero();
            self.getDenseJacobian(body_idx, point, jaco);
            return jaco;
        }, py::arg("body_idx"), py::arg("point_w"))
        .def("getDenseRotationalJacobian", [](raisim::ArticulatedSystem &self, size_t body_idx) {
            Eigen::MatrixXd jaco(3, self.getDOF());
            jaco.setZero();
            self.getDenseRotationalJacobian(body_idx, jaco);
            return jaco;
        }, py::arg("body_idx"))
        .def("getUdot", [](raisim::ArticulatedSystem &self) {
            return convert_vecdyn_to_np(self.getUdot());
        })
        .def("getMinvJT", [matdyn_list_to_pylist](raisim::ArticulatedSystem &self) {
            return matdyn_list_to_pylist(self.getMinvJT());
        })
        .def("getj_MinvJT_T1D", [vecdyn_list_to_pylist](raisim::ArticulatedSystem &self) {
            return vecdyn_list_to_pylist(self.getj_MinvJT_T1D());
        })
        .def("getFullDelassusAndTauStar", &raisim::ArticulatedSystem::getFullDelassusAndTauStar, py::arg("dt"))
        .def("getVisObPose", [](raisim::ArticulatedSystem &self, size_t idx) {
            Mat<3, 3> rot;
            Vec<3> pos;
            self.getVisObPose(idx, rot, pos);
            return py::make_tuple(convert_vec_to_np(pos), convert_mat_to_np(rot));
        }, py::arg("idx"))
        .def("getVisColObPose", [](raisim::ArticulatedSystem &self, size_t idx) {
            Mat<3, 3> rot;
            Vec<3> pos;
            self.getVisColObPose(idx, rot, pos);
            return py::make_tuple(convert_vec_to_np(pos), convert_mat_to_np(rot));
        }, py::arg("idx"))
        .def("exportRobotDescriptionToURDF", &raisim::ArticulatedSystem::exportRobotDescriptionToURDF, py::arg("file_path"))
        .def("getRobotDescriptionFullPath", &raisim::ArticulatedSystem::getRobotDescriptionFullPath)
        .def("getRobotDescription", &raisim::ArticulatedSystem::getRobotDescription)
        .def("setIntegrationScheme", &raisim::ArticulatedSystem::setIntegrationScheme, py::arg("scheme"))
        .def("setExternalTorqueInBodyFrame", [](raisim::ArticulatedSystem &self, size_t body_idx, NDArray torque) {
            self.setExternalTorqueInBodyFrame(body_idx, convert_np_to_vec<3>(torque));
        }, py::arg("body_idx"), py::arg("torque"))
        .def("addSpring", &raisim::ArticulatedSystem::addSpring, py::arg("spring"))
        .def("addConstraints", [](raisim::ArticulatedSystem &self,
                                  const std::vector<raisim::PinConstraintDefinition> &pin_def,
                                  NDArray nominal_config) {
            self.addConstraints(pin_def, convert_np_to_vecdyn(nominal_config));
        }, py::arg("pin_def"), py::arg("pin_constraint_nominal_config"))
        .def("initializeConstraints", &raisim::ArticulatedSystem::initializeConstraints)
        .def("dumpCollisionBodyPairInfo", &raisim::ArticulatedSystem::dumpCollisionBodyPairInfo,
             py::arg("link_a"), py::arg("link_b"))
        .def("getIgnoredCollisionPairs", [](const raisim::ArticulatedSystem &self) {
            py::list out;
            for (const auto &pair : self.getIgnoredCollisionPairs())
                out.append(py::make_tuple(pair.first, pair.second));
            return out;
        })
        .def("updateSelfCollisionCache", &raisim::ArticulatedSystem::updateSelfCollisionCache, py::arg("world"))
        .def("printOutMovableJointNamesInOrder", &raisim::ArticulatedSystem::printOutMovableJointNamesInOrder)
        .def("getMovableJointNames", &raisim::ArticulatedSystem::getMovableJointNames)
        .def("getCollisionBodies", py::overload_cast<>(&raisim::ArticulatedSystem::getCollisionBodies),
             py::rv_policy::reference_internal)
        .def("getCollisionBody", py::overload_cast<const std::string &>(&raisim::ArticulatedSystem::getCollisionBody),
             py::rv_policy::reference_internal, py::arg("name"))
        .def("setCollisionBodyShapeParameters", [](raisim::ArticulatedSystem &self, size_t id_, NDArray params) {
            self.setCollisionBodyShapeParameters(id_, convert_np_to_vec<4>(params));
        }, py::arg("id_"), py::arg("parameters"))
        .def("setCollisionBodyPositionOffset", [](raisim::ArticulatedSystem &self, size_t id_, NDArray position) {
            self.setCollisionBodyPositionOffset(id_, convert_np_to_vec<3>(position));
        }, py::arg("id_"), py::arg("position"))
        .def("setCollisionBodyOrientationOffset", [](raisim::ArticulatedSystem &self, size_t id_, NDArray orientation) {
            Mat<3,3> rot;
            if (orientation.size() == 3) { // rpy angles
                Vec<3> rpy = convert_np_to_vec<3>(orientation);
                rpyToRotMat_intrinsic(rpy, rot);
            } else if (orientation.size() == 4) {
                Vec<4> quat = convert_np_to_vec<4>(orientation);
                quatToRotMat(quat, rot);
            } else if (orientation.size() == 9) {
                rot = convert_np_to_mat<3,3>(orientation);
            } else {
                std::ostringstream s;
                s << "error: expecting the given orientation to have a size of 3 (RPY), 4 (quaternion), or 9 "
                  << "(rotation matrix), but got instead a size of " << orientation.size() << ".";
                throw std::domain_error(s.str());
            }
            self.setCollisionBodyOrientationOffset(id_, rot);
        }, py::arg("id_"), py::arg("orientation"))

	    .def("setCollisionObjectShapeParameters", [](raisim::ArticulatedSystem &self, size_t id_, NDArray params) {
            self.setCollisionBodyShapeParameters(id_, convert_np_to_vec<4>(params));
        }, R"mydelimiter(
	    Set the collision object shape parameters.

	    Args:
	        id_ (int): collision object id.
	        parameters (list[float]): parameters.
	    )mydelimiter",
	    py::arg("id_"), py::arg("parameters"))


        .def("setCollisionObjectPositionOffset", [](raisim::ArticulatedSystem &self, size_t id_, NDArray &position) {
            Vec<3> pos = convert_np_to_vec<3>(position);
            self.setCollisionBodyPositionOffset(id_, pos);
        }, R"mydelimiter(
	    Set the collision object position offset.

	    Args:
	        id_ (int): collision object id.
	        position (np.array[float[3]]): position offset.
	    )mydelimiter",
	    py::arg("id_"), py::arg("position"))


	    .def("setCollisionObjectOrientationOffset", [](raisim::ArticulatedSystem &self, size_t id_, NDArray &orientation) {
            Mat<3,3> rot;
	        if (orientation.size() == 3) { // rpy angles
	            Vec<3> rpy = convert_np_to_vec<3>(orientation);
                rpyToRotMat_intrinsic(rpy, rot);
	        } else if (orientation.size() == 4) {
	            Vec<4> quat = convert_np_to_vec<4>(orientation);
                quatToRotMat(quat, rot);
	        } else if (orientation.size() == 9) {
	            rot = convert_np_to_mat<3,3>(orientation);
	        } else {
                std::ostringstream s;
                s << "error: expecting the given orientation to have a size of 3 (RPY), 4 (quaternion), or 9 " <<
                    "(rotation matrix), but got instead a size of " << orientation.size() << ".";
                throw std::domain_error(s.str());
	        }
	        self.setCollisionBodyOrientationOffset(id_, rot);
        }, R"mydelimiter(
	    Set the collision object orientation offset.

	    Args:
	        id_ (int): collision object id.
	        orientation (np.array[float[3]], np.array[float[4]], np.array[float[3,3]]): orientation offset.
	    )mydelimiter",
	    py::arg("id_"), py::arg("orientation"));

}
