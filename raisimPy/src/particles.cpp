#include "modern_bindings.hpp"
#include "raisim/object/deformable/DeformableObject.hpp"
#include "raisim/object/granular/GranularSystem.hpp"
#include "raisim/object/terrain/DeformableTerrain.hpp"

namespace py = nanobind;
using namespace raisim;

namespace {
using Vector = std::array<double, 3>;
Vec<3> vec(const Vector& value) { return {value[0], value[1], value[2]}; }

void require_compatible_world(World& world, bool addingSoil) {
    for (size_t i = 0; i < world.getObjectCount(); ++i) {
        const auto* object = world.getObject(i);
        if ((addingSoil && (object->getObjectType() == DEFORMABLE || object->getObjectType() == GRANULAR)) ||
            (!addingSoil && dynamic_cast<const DeformableTerrain*>(object)))
            throw std::invalid_argument("Deformable terrain cannot share a world with deformable or granular particles.");
    }
}

template<bool AddingSoil, typename Result, typename... Args>
auto compatible_factory(Result (World::*factory)(Args...)) {
    return [factory](World& self, Args... args) {
        require_compatible_world(self, AddingSoil);
        return (self.*factory)(std::forward<Args>(args)...);
    };
}
}

void init_particle_objects(py::module_& m) {
    using D = DeformableObject;
    auto deformable = py::class_<D, Object>(m, "DeformableObject");
    py::enum_<D::SolverMode>(deformable, "SolverMode")
        .value("XPBD", D::SolverMode::XPBD).value("PBD", D::SolverMode::PBD);
    py::class_<D::Material>(deformable, "Material")
        .def(py::init<>())
        .def_rw("totalMass", &D::Material::totalMass)
        .def_rw("distanceCompliance", &D::Material::distanceCompliance)
        .def_rw("distanceStiffness", &D::Material::distanceStiffness)
        .def_rw("bendCompliance", &D::Material::bendCompliance)
        .def_rw("bendStiffness", &D::Material::bendStiffness)
        .def_rw("youngsModulus", &D::Material::youngsModulus)
        .def_rw("poissonRatio", &D::Material::poissonRatio)
        .def_rw("thickness", &D::Material::thickness)
        .def_rw("crossSectionArea", &D::Material::crossSectionArea)
        .def_rw("damping", &D::Material::damping)
        .def_rw("airDrag", &D::Material::airDrag)
        .def_rw("collisionRadius", &D::Material::collisionRadius)
        .def_rw("iterations", &D::Material::iterations)
        .def_rw("substeps", &D::Material::substeps)
        .def_rw("solverMode", &D::Material::solverMode)
        .def_rw("selfCollision", &D::Material::selfCollision);
    auto struts = py::class_<D::InternalStrutOptions>(deformable, "InternalStrutOptions");
    py::enum_<D::InternalStrutOptions::Mode>(struts, "Mode")
        .value("None_", D::InternalStrutOptions::Mode::None)
        .value("PairsWithinRadius", D::InternalStrutOptions::Mode::PairsWithinRadius);
    struts.def(py::init<>()).def_rw("mode", &D::InternalStrutOptions::mode)
        .def_rw("radius", &D::InternalStrutOptions::radius);
    auto particles = py::class_<D::MeshParticleOptions>(deformable, "MeshParticleOptions");
    py::enum_<D::MeshParticleOptions::Mode>(particles, "Mode")
        .value("Surface", D::MeshParticleOptions::Mode::Surface)
        .value("Filled", D::MeshParticleOptions::Mode::Filled);
    particles.def(py::init<>()).def_rw("mode", &D::MeshParticleOptions::mode)
        .def_rw("scale", &D::MeshParticleOptions::scale)
        .def_rw("spacing", &D::MeshParticleOptions::spacing)
        .def_rw("maxFillParticles", &D::MeshParticleOptions::maxFillParticles);
    py::class_<D::MeshBuildOptions>(deformable, "MeshBuildOptions")
        .def(py::init<>()).def_rw("particles", &D::MeshBuildOptions::particles)
        .def_rw("internalStruts", &D::MeshBuildOptions::internalStruts);
    deformable
        .def("getNumParticles", &D::getNumParticles)
        .def("getPositions", [](const D& self) { return convert_vec_list_to_np(self.getPositions()); })
        .def("getVelocities", [](const D& self) { return convert_vec_list_to_np(self.getVelocities()); })
        .def("getTriangles", &D::getTriangles, py::rv_policy::copy)
        .def("getTetrahedra", &D::getTetrahedra, py::rv_policy::copy)
        .def("getVisualPosition", [](const D& self, size_t index) {
            return convert_vec_to_np(self.getVisualPosition(index));
        }, py::arg("index"))
        .def("getDistanceConstraintCount", &D::getDistanceConstraintCount)
        .def("getBendingConstraintCount", &D::getBendingConstraintCount)
        .def("getMaterial", &D::getMaterial, py::rv_policy::copy)
        .def("getCollisionBodyAt", &D::getCollisionBodyAt, py::arg("index"), py::rv_policy::reference_internal)
        .def("getCollisionBodyCount", &D::getCollisionBodyCount)
        .def("setPositionOffset", [](D& self, const NDArray& offset) {
            self.setPositionOffset(convert_np_to_vec<3>(offset));
        }, py::arg("offset"))
        .def("applyRigidTransform", [](D& self, const NDArray& translation, const NDArray& rotation) {
            self.applyRigidTransform(convert_np_to_vec<3>(translation), convert_np_to_mat<3, 3>(rotation));
        }, py::arg("translation"), py::arg("rotation"))
        .def("pinVertex", &D::pinVertex, py::arg("index"))
        .def("unpinVertex", &D::unpinVertex, py::arg("index"), py::arg("mass"))
        .def("addDistanceConstraint", &D::addDistanceConstraint, py::arg("i"), py::arg("j"))
        .def("addDistanceConstraints", &D::addDistanceConstraints, py::arg("edges"))
        .def("addInternalStruts", &D::addInternalStruts, py::arg("options"))
        .def("setGravity", [](D& self, const NDArray& gravity) {
            self.setGravity(convert_np_to_vec<3>(gravity));
        }, py::arg("gravity"));

    using G = GranularSystem;
    auto granular = py::class_<G, Object>(m, "GranularSystem");
    py::enum_<G::NormalContactModel>(granular, "NormalContactModel")
        .value("LINEAR", G::NormalContactModel::LINEAR).value("HERTZ", G::NormalContactModel::HERTZ);
    py::class_<G::Material>(granular, "Material")
        .def(py::init<>())
        .def_rw("density", &G::Material::density)
        .def_rw("normalStiffness", &G::Material::normalStiffness)
        .def_rw("normalDamping", &G::Material::normalDamping)
        .def_rw("tangentialStiffness", &G::Material::tangentialStiffness)
        .def_rw("tangentialDamping", &G::Material::tangentialDamping)
        .def_rw("friction", &G::Material::friction)
        .def_rw("rollingFriction", &G::Material::rollingFriction)
        .def_rw("cohesionStiffness", &G::Material::cohesionStiffness)
        .def_rw("cohesionMaxDistance", &G::Material::cohesionMaxDistance)
        .def_rw("normalContactModel", &G::Material::normalContactModel)
        .def_rw("substeps", &G::Material::substeps)
        .def_rw("maxSpeed", &G::Material::maxSpeed)
        .def_rw("maxAngularSpeed", &G::Material::maxAngularSpeed);
    py::class_<G::ParticleMaterial>(granular, "ParticleMaterial")
        .def(py::init<>()).def_rw("friction", &G::ParticleMaterial::friction)
        .def_rw("rollingFriction", &G::ParticleMaterial::rollingFriction);
    py::class_<G::StepStats>(granular, "StepStats")
        .def_ro("candidatePairs", &G::StepStats::candidatePairs)
        .def_ro("rigidCandidatePairs", &G::StepStats::rigidCandidatePairs)
        .def_ro("meshTriangleCandidatePairs", &G::StepStats::meshTriangleCandidatePairs)
        .def_ro("meshInsideTriangleTests", &G::StepStats::meshInsideTriangleTests)
        .def_ro("particleContacts", &G::StepStats::particleContacts)
        .def_ro("boundaryContacts", &G::StepStats::boundaryContacts)
        .def_ro("rigidContacts", &G::StepStats::rigidContacts)
        .def_ro("cohesiveContacts", &G::StepStats::cohesiveContacts)
        .def_ro("activeContactHistories", &G::StepStats::activeContactHistories)
        .def_ro("staleContactHistoriesRemoved", &G::StepStats::staleContactHistoriesRemoved)
        .def_ro("emittedParticles", &G::StepStats::emittedParticles)
        .def_ro("removedParticles", &G::StepStats::removedParticles)
        .def_ro("wrappedParticles", &G::StepStats::wrappedParticles)
        .def_ro("velocityClamps", &G::StepStats::velocityClamps)
        .def_ro("maxPenetration", &G::StepStats::maxPenetration)
        .def_ro("maxSpeed", &G::StepStats::maxSpeed);
    auto box = py::class_<G::BoxOptions>(granular, "BoxOptions");
    box.def(py::init<>()).def_rw("radius", &G::BoxOptions::radius)
        .def_rw("spacing", &G::BoxOptions::spacing).def_rw("maxParticles", &G::BoxOptions::maxParticles)
        .def_rw("materialId", &G::BoxOptions::materialId).def_rw("fixed", &G::BoxOptions::fixed);
    bind_vector_property(box, "minCorner", &G::BoxOptions::minCorner);
    bind_vector_property(box, "maxCorner", &G::BoxOptions::maxCorner);
    bind_vector_property(box, "velocity", &G::BoxOptions::velocity);
    bind_vector_property(box, "angularVelocity", &G::BoxOptions::angularVelocity);
    granular
        .def("getNumParticles", &G::getNumParticles)
        .def("getPositions", [](const G& self) { return convert_vec_list_to_np(self.getPositions()); })
        .def("getVelocities", [](const G& self) { return convert_vec_list_to_np(self.getVelocities()); })
        .def("getAngularVelocities", [](const G& self) { return convert_vec_list_to_np(self.getAngularVelocities()); })
        .def("getRadii", &G::getRadii, py::rv_policy::copy)
        .def("getParticleMaterialIds", &G::getParticleMaterialIds, py::rv_policy::copy)
        .def("getFixedParticleFlags", &G::getFixedParticleFlags, py::rv_policy::copy)
        .def("getMaterial", &G::getMaterial, py::rv_policy::copy)
        .def("getBoundaryMaterial", &G::getBoundaryMaterial, py::rv_policy::copy)
        .def("getLastStepStats", &G::getLastStepStats, py::rv_policy::copy)
        .def("setParticleMaterial", &G::setParticleMaterial, py::arg("materialId"), py::arg("material"))
        .def("setParticleMaterialId", &G::setParticleMaterialId, py::arg("index"), py::arg("materialId"))
        .def("setBoundaryMaterial", &G::setBoundaryMaterial, py::arg("material"))
        .def("reserveParticles", &G::reserveParticles, py::arg("capacity"))
        .def("addParticle", [](G& self, const Vector& position, double radius, const Vector& velocity,
                               const Vector& angularVelocity, uint16_t materialId, bool fixed) {
            return self.addParticle(vec(position), radius, vec(velocity), vec(angularVelocity), materialId, fixed);
        }, py::arg("position"), py::arg("radius"), py::arg("velocity") = Vector{},
           py::arg("angularVelocity") = Vector{}, py::arg("materialId") = 0, py::arg("fixed") = false)
        .def("emitBox", [](G& self, const Vector& min, const Vector& max, double radius,
                           double spacing, size_t maxParticles, const Vector& velocity,
                           const Vector& angularVelocity, uint16_t materialId, bool fixed) {
            return self.emitBox(vec(min), vec(max), radius, spacing, maxParticles,
                                vec(velocity), vec(angularVelocity), materialId, fixed);
        }, py::arg("minCorner"), py::arg("maxCorner"), py::arg("radius"), py::arg("spacing"),
           py::arg("maxParticles"), py::arg("velocity") = Vector{}, py::arg("angularVelocity") = Vector{},
           py::arg("materialId") = 0, py::arg("fixed") = false)
        .def("emitBoxRandom", [](G& self, const Vector& min, const Vector& max, double minRadius,
                           double maxRadius, double spacing, size_t maxParticles, uint64_t seed,
                           const Vector& velocity, const Vector& angularVelocity, uint16_t materialId, bool fixed) {
            return self.emitBoxRandom(vec(min), vec(max), minRadius, maxRadius, spacing, maxParticles,
                                      seed, vec(velocity), vec(angularVelocity), materialId, fixed);
        }, py::arg("minCorner"), py::arg("maxCorner"), py::arg("minRadius"), py::arg("maxRadius"),
           py::arg("spacing"), py::arg("maxParticles"), py::arg("seed"), py::arg("velocity") = Vector{},
           py::arg("angularVelocity") = Vector{}, py::arg("materialId") = 0, py::arg("fixed") = false)
        .def("removeParticle", &G::removeParticle, py::arg("index"))
        .def("removeParticlesBelowZ", &G::removeParticlesBelowZ, py::arg("zThreshold"))
        .def("removeParticlesInBox", [](G& self, const Vector& min, const Vector& max) {
            return self.removeParticlesInBox(vec(min), vec(max));
        }, py::arg("minCorner"), py::arg("maxCorner"))
        .def("saveGranularState", &G::saveGranularState, py::arg("path"), py::arg("includeContactHistories") = true)
        .def("loadGranularState", &G::loadGranularState, py::arg("path"))
        .def("setVelocity", [](G& self, size_t index, const Vector& velocity) {
            self.setVelocity(index, vec(velocity));
        }, py::arg("index"), py::arg("velocity"))
        .def("setAngularVelocity", [](G& self, size_t index, const Vector& velocity) {
            self.setAngularVelocity(index, vec(velocity));
        }, py::arg("index"), py::arg("angularVelocity"))
        .def("setParticleFixed", &G::setParticleFixed, py::arg("index"), py::arg("fixed"))
        .def("setGroundPlane", &G::setGroundPlane, py::arg("height"), py::arg("enabled") = true)
        .def("disableGroundPlane", &G::disableGroundPlane)
        .def("setContainerBox", [](G& self, const Vector& min, const Vector& max, bool enabled) {
            self.setContainerBox(vec(min), vec(max), enabled);
        }, py::arg("minCorner"), py::arg("maxCorner"), py::arg("enabled") = true)
        .def("disableContainerBox", &G::disableContainerBox)
        .def("setPeriodicBoundary", [](G& self, const Vector& min, const Vector& max, bool x, bool y, bool z) {
            self.setPeriodicBoundary(vec(min), vec(max), x, y, z);
        }, py::arg("minCorner"), py::arg("maxCorner"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("disablePeriodicBoundary", &G::disablePeriodicBoundary);

    using T = DeformableTerrain;
    auto terrain = py::class_<T, HeightMap>(m, "DeformableTerrain");
    py::class_<T::Material>(terrain, "Material")
        .def(py::init<>())
        .def_rw("bekkerKc", &T::Material::bekkerKc)
        .def_rw("bekkerKphi", &T::Material::bekkerKphi)
        .def_rw("sinkageExponent", &T::Material::sinkageExponent)
        .def_rw("elasticStiffness", &T::Material::elasticStiffness)
        .def_rw("damping", &T::Material::damping)
        .def_rw("cohesion", &T::Material::cohesion)
        .def_rw("frictionCoefficient", &T::Material::frictionCoefficient)
        .def_rw("shearDisplacement", &T::Material::shearDisplacement);
    py::class_<T::State>(terrain, "State")
        .def(py::init<>()).def_rw("sinkage", &T::State::sinkage)
        .def_rw("plasticSinkage", &T::State::plasticSinkage).def_rw("shear", &T::State::shear);
    py::class_<T::StepStats>(terrain, "StepStats")
        .def_ro("candidateCells", &T::StepStats::candidateCells)
        .def_ro("rayQueries", &T::StepStats::rayQueries)
        .def_ro("contacts", &T::StepStats::contacts)
        .def_ro("updatedVertices", &T::StepStats::updatedVertices)
        .def_ro("contactArea", &T::StepStats::contactArea)
        .def_ro("normalForce", &T::StepStats::normalForce);
    terrain.def("getSoilMaterial", &T::getSoilMaterial, py::rv_policy::copy)
        .def("getUndeformedHeights", &T::getUndeformedHeights, py::rv_policy::copy)
        .def("getSoilState", &T::getSoilState, py::rv_policy::copy)
        .def("getSoilStepStats", &T::getSoilStepStats, py::rv_policy::copy)
        .def("setSoilState", &T::setSoilState, py::arg("state"))
        .def("resetSoil", &T::resetSoil)
        .def("pressure", &T::pressure, py::arg("sinkage"), py::arg("plasticSinkage"), py::arg("contactWidth"));
}

void init_particle_world(py::class_<World>& world) {
    using D = DeformableObject;
    using G = GranularSystem;
    world
        .def("addDeformableCloth", [](World& self, const NDArray& vertices,
                const std::vector<D::Triangle>& triangles, const std::vector<unsigned int>& pinned,
                const D::Material& material, const std::string& contactMaterial, CollisionGroup group, CollisionGroup mask) {
            require_compatible_world(self, false);
            return self.addDeformableCloth(convert_np_to_vec_list<3>(vertices), triangles, pinned,
                                           material, contactMaterial, group, mask);
        }, py::arg("vertices"), py::arg("triangles"), py::arg("pinnedVertices") = std::vector<unsigned int>{},
           py::arg("material") = D::Material{}, py::arg("contactMaterial") = "default",
           py::arg("collision_group") = 1, py::arg("collision_mask") = CollisionGroup(-1), py::rv_policy::reference_internal)
        .def("addDeformableCloth",
             compatible_factory<false>(py::overload_cast<const std::string&, double, const std::vector<unsigned int>&,
                 const D::Material&, const std::string&, CollisionGroup, CollisionGroup>(&World::addDeformableCloth)),
             py::arg("fileName"), py::arg("scale") = 1., py::arg("pinnedVertices") = std::vector<unsigned int>{},
             py::arg("material") = D::Material{}, py::arg("contactMaterial") = "default",
             py::arg("collision_group") = 1, py::arg("collision_mask") = CollisionGroup(-1), py::rv_policy::reference_internal)
        .def("addDeformableVolume", [](World& self, const NDArray& vertices,
                const std::vector<D::Tetrahedron>& tetrahedra, const std::vector<unsigned int>& pinned,
                const D::Material& material, const std::string& contactMaterial, CollisionGroup group, CollisionGroup mask) {
            require_compatible_world(self, false);
            return self.addDeformableVolume(convert_np_to_vec_list<3>(vertices), tetrahedra, pinned,
                                            material, contactMaterial, group, mask);
        }, py::arg("vertices"), py::arg("tetrahedra"), py::arg("pinnedVertices") = std::vector<unsigned int>{},
           py::arg("material") = D::Material{}, py::arg("contactMaterial") = "default",
           py::arg("collision_group") = 1, py::arg("collision_mask") = CollisionGroup(-1), py::rv_policy::reference_internal)
        .def("addDeformableObject", compatible_factory<false>(&World::addDeformableObject), py::arg("fileName"), py::arg("material"),
             py::arg("options"), py::arg("pinnedVertices") = std::vector<unsigned int>{},
             py::arg("contactMaterial") = "default", py::arg("collision_group") = 1,
             py::arg("collision_mask") = CollisionGroup(-1), py::rv_policy::reference_internal)
        .def("addGranularParticles", [](World& self, const NDArray& positions,
                                          const std::vector<double>& radii, const G::Material& material) {
            require_compatible_world(self, false);
            return self.addGranularParticles(convert_np_to_vec_list<3>(positions), radii, material);
        }, py::arg("positions"), py::arg("radii"), py::arg("material") = G::Material{}, py::rv_policy::reference_internal)
        .def("addGranularBox", compatible_factory<false>(&World::addGranularBox), py::arg("options"),
             py::arg("material") = G::Material{}, py::rv_policy::reference_internal)
        .def("addDeformableTerrain", compatible_factory<true>(&World::addDeformableTerrain),
             py::arg("xSamples"), py::arg("ySamples"), py::arg("xSize"), py::arg("ySize"),
             py::arg("centerX"), py::arg("centerY"), py::arg("heights"),
             py::arg("soil") = DeformableTerrain::Material{}, py::arg("material") = "default",
             py::arg("collision_group") = RAISIM_STATIC_COLLISION_GROUP,
             py::arg("collision_mask") = CollisionGroup(-1), py::rv_policy::reference_internal);
}
