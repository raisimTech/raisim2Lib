#include "modern_bindings.hpp"

namespace py = nanobind;
using namespace raisim;

void init_world_extensions(py::module_& m, py::class_<World>& world) {
    py::class_<World::Checkpoint>(world, "Checkpoint")
        .def(py::init<>())
        .def("__bool__", [](const World::Checkpoint& self) { return bool(self); })
        .def("getWorldTime", &World::Checkpoint::getWorldTime);
    auto snapshot = py::class_<World::SingleBodySnapshot>(world, "SingleBodySnapshot");
    snapshot.def(py::init<>())
        .def_ro("objectId", &World::SingleBodySnapshot::objectId)
        .def_ro("objectType", &World::SingleBodySnapshot::objectType)
        .def_rw("bodyType", &World::SingleBodySnapshot::bodyType)
        .def_rw("material", &World::SingleBodySnapshot::material)
        .def_rw("collisionGroup", &World::SingleBodySnapshot::collisionGroup)
        .def_rw("collisionMask", &World::SingleBodySnapshot::collisionMask);
    bind_vector_property(snapshot, "position", &World::SingleBodySnapshot::position);
    bind_vector_property(snapshot, "quaternion", &World::SingleBodySnapshot::quaternion);
    bind_vector_property(snapshot, "linearVelocity", &World::SingleBodySnapshot::linearVelocity);
    bind_vector_property(snapshot, "angularVelocity", &World::SingleBodySnapshot::angularVelocity);
    py::class_<World::SleepSettings>(world, "SleepSettings")
        .def_ro("enabled", &World::SleepSettings::enabled)
        .def_ro("linearVelThreshold", &World::SleepSettings::linearVelThreshold)
        .def_ro("angularVelThreshold", &World::SleepSettings::angularVelThreshold)
        .def_ro("quietSteps", &World::SleepSettings::quietSteps);
    py::class_<World::CollisionCounters>(world, "CollisionCounters")
        .def_ro("deformableParticleCandidates", &World::CollisionCounters::deformableParticleCandidates)
        .def_ro("deformableParticleAabbTests", &World::CollisionCounters::deformableParticleAabbTests)
        .def_ro("deformableParticleContacts", &World::CollisionCounters::deformableParticleContacts)
        .def_ro("deformableParticleGridCells", &World::CollisionCounters::deformableParticleGridCells)
        .def_ro("sweptCcdCandidates", &World::CollisionCounters::sweptCcdCandidates)
        .def_ro("sweptCcdContacts", &World::CollisionCounters::sweptCcdContacts);

    py::enum_<::contact::BroadphaseType>(m, "BroadphaseType")
        .value("None_", ::contact::BroadphaseType::None)
        .value("Sap3Axis", ::contact::BroadphaseType::Sap3Axis)
        .value("MultiBoxPrune", ::contact::BroadphaseType::MultiBoxPrune);
    auto broadphase = py::class_<::contact::BroadphaseSettings>(m, "BroadphaseSettings");
    broadphase.def(py::init<>())
        .def_rw("type", &::contact::BroadphaseSettings::type)
        .def_rw("mbpUseWorldBounds", &::contact::BroadphaseSettings::mbpUseWorldBounds)
        .def_rw("mbpPadding", &::contact::BroadphaseSettings::mbpPadding)
        .def_rw("mbpMaxCellsPerAxis", &::contact::BroadphaseSettings::mbpMaxCellsPerAxis)
        .def_rw("mbpMaxCellsPerObject", &::contact::BroadphaseSettings::mbpMaxCellsPerObject);
    bind_vector_property(broadphase, "mbpWorldMin", &::contact::BroadphaseSettings::mbpWorldMin);
    bind_vector_property(broadphase, "mbpWorldMax", &::contact::BroadphaseSettings::mbpWorldMax);
    bind_vector_property(broadphase, "mbpCellSize", &::contact::BroadphaseSettings::mbpCellSize);
    py::class_<::contact::ContactSettings>(m, "ContactSettings")
        .def(py::init<>())
        .def_rw("gjkMaxIterations", &::contact::ContactSettings::gjkMaxIterations)
        .def_rw("gjkTolerance", &::contact::ContactSettings::gjkTolerance)
        .def_rw("epaMaxIterations", &::contact::ContactSettings::epaMaxIterations)
        .def_rw("epaTolerance", &::contact::ContactSettings::epaTolerance)
        .def_rw("maxContactsPerPair", &::contact::ContactSettings::maxContactsPerPair)
        .def_rw("sweptCcdEnabled", &::contact::ContactSettings::sweptCcdEnabled)
        .def_rw("sweptCcdMinSpeed", &::contact::ContactSettings::sweptCcdMinSpeed)
        .def_rw("sweptCcdSpeculativeMargin", &::contact::ContactSettings::sweptCcdSpeculativeMargin)
        .def_rw("broadphase", &::contact::ContactSettings::broadphase);

    world
        .def("captureCheckpoint", &World::captureCheckpoint)
        .def("restoreCheckpoint", [](World& self, const World::Checkpoint& checkpoint, bool raiseOnError) {
            std::string error;
            const bool restored = self.restoreCheckpoint(checkpoint, &error);
            if (!restored && raiseOnError) throw std::invalid_argument(error);
            return restored;
        }, py::arg("checkpoint"), py::arg("raiseOnError") = false,
           "Restore a checkpoint. Return false on rejection, or raise ValueError with raiseOnError=True.")
        .def("captureSingleBodySnapshot", [](const World& self, const SingleBodyObject* object) {
            World::SingleBodySnapshot result;
            if (!self.captureSingleBodySnapshot(object, result))
                throw std::invalid_argument("The object does not belong to this world.");
            return result;
        }, py::arg("object"))
        .def("restoreSingleBodySnapshot",
             py::overload_cast<SingleBodyObject*, const World::SingleBodySnapshot&, bool>(&World::restoreSingleBodySnapshot),
             py::arg("object"), py::arg("snapshot"), py::arg("restoreCollisionProperties") = true)
        .def("restoreSingleBodySnapshot",
             py::overload_cast<World::ObjectId, const World::SingleBodySnapshot&, bool>(&World::restoreSingleBodySnapshot),
             py::arg("id"), py::arg("snapshot"), py::arg("restoreCollisionProperties") = true)
        .def("cloneSingleBodyObject", &World::cloneSingleBodyObject, py::arg("source"), py::arg("name") = "",
             py::rv_policy::reference_internal)
        .def("getObjectCount", &World::getObjectCount)
        .def("getObjectById", py::overload_cast<World::ObjectId>(&World::getObjectById),
             py::arg("id"), py::rv_policy::reference_internal)
        .def("isObjectAlive", &World::isObjectAlive, py::arg("id"))
        .def("setObjectCollisionFilter", &World::setObjectCollisionFilter,
             py::arg("object"), py::arg("collision_group"), py::arg("collision_mask"), py::arg("localIdx") = size_t(-1))
        .def("setObjectMaterial", &World::setObjectMaterial,
             py::arg("object"), py::arg("material"), py::arg("localIdx") = size_t(-1))
        .def("setContactSolverParam", &World::setContactSolverParam,
             py::arg("alpha_init"), py::arg("alpha_min"), py::arg("alpha_decay"),
             py::arg("maxIter"), py::arg("threshold"))
        .def("setMonolithicDeformables", &World::setMonolithicDeformables, py::arg("enabled"))
        .def("isMonolithicDeformables", &World::isMonolithicDeformables)
        .def("setSleepingEnabled", &World::setSleepingEnabled, py::arg("enabled"))
        .def("isSleepingEnabled", &World::isSleepingEnabled)
        .def("setSleepingParameters", &World::setSleepingParameters,
             py::arg("linearVelThreshold"), py::arg("angularVelThreshold"), py::arg("quietSteps"))
        .def("getSleepingParameters", &World::getSleepingParameters, py::rv_policy::copy)
        .def("setSleepingVelocityThresholds", &World::setSleepingVelocityThresholds,
             py::arg("linearVelThreshold"), py::arg("angularVelThreshold"))
        .def("wakeObject", &World::wakeObject, py::arg("object"))
        .def("wakeAll", &World::wakeAll)
        .def("isObjectSleeping", py::overload_cast<const Object*>(&World::isObjectSleeping, py::const_), py::arg("object"))
        .def("isObjectSleeping", py::overload_cast<size_t>(&World::isObjectSleeping, py::const_), py::arg("index"))
        .def("setContactSettings", &World::setContactSettings, py::arg("settings"))
        .def("getContactSettings", &World::getContactSettings, py::rv_policy::copy)
        .def("setBroadphaseSettings", &World::setBroadphaseSettings, py::arg("settings"))
        .def("getBroadphaseSettings", &World::getBroadphaseSettings, py::rv_policy::copy)
        .def("setCollisionCountersEnabled", &World::setCollisionCountersEnabled, py::arg("enabled"))
        .def("getCollisionCountersEnabled", &World::getCollisionCountersEnabled)
        .def("getLastCollisionCounters", &World::getLastCollisionCounters, py::rv_policy::copy)
        .def("addUsdArticulatedSystem", &World::addUsdArticulatedSystem, py::arg("usdPath"),
             py::arg("collision_group") = 1, py::arg("collision_mask") = CollisionGroup(-1),
             py::arg("options") = ArticulatedSystemOption{}, py::rv_policy::reference_internal)
        .def("addArticulatedSystem",
             py::overload_cast<const std::string&, const std::vector<std::string>&, const std::string&,
                 const std::vector<std::string>&, CollisionGroup, CollisionGroup, ArticulatedSystemOption>(&World::addArticulatedSystem),
             py::arg("filePathOrURDFScript"), py::arg("modules"), py::arg("resPath") = "",
             py::arg("jointOrder") = std::vector<std::string>{}, py::arg("collision_group") = 1,
             py::arg("collision_mask") = CollisionGroup(-1), py::arg("options") = ArticulatedSystemOption{},
             py::rv_policy::reference_internal)
        .def("exportMeshAssetsToObj", &World::exportMeshAssetsToObj, py::arg("directory"), py::arg("prefix") = "mesh")
        .def("loadMjcfFile", &World::loadMjcfFile, py::arg("configFile"));
}
