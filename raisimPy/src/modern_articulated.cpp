#include "modern_bindings.hpp"
#include "raisim/object/ArticulatedSystem/MotorOperatingRegion.hpp"
#include "raisim/object/ArticulatedSystem/Track.hpp"

namespace py = nanobind;
using namespace raisim;

#if RAISIMPY_HAS_SEGMENTATION
void init_segmentation_articulated(py::class_<ArticulatedSystem, Object>& system);
#endif

void init_articulated_extensions(py::module_& m, py::class_<ArticulatedSystem, Object>& system) {
#if RAISIMPY_HAS_SEGMENTATION
    init_segmentation_articulated(system);
#endif
    using Options = ArticulatedSystem::IKOptions;
    using Result = ArticulatedSystem::IKResult;
    py::class_<Options>(system, "IKOptions")
        .def(py::init<>())
        .def_rw("maxIterations", &Options::maxIterations)
        .def_rw("positionTolerance", &Options::positionTolerance)
        .def_rw("orientationTolerance", &Options::orientationTolerance)
        .def_rw("orientationWeight", &Options::orientationWeight)
        .def_rw("damping", &Options::damping)
        .def_rw("stepSize", &Options::stepSize)
        .def_rw("maxStep", &Options::maxStep)
        .def_rw("svdTolerance", &Options::svdTolerance)
        .def_rw("finiteDifferenceStep", &Options::finiteDifferenceStep)
        .def_rw("useOrientation", &Options::useOrientation)
        .def_rw("applySolution", &Options::applySolution)
        .def_rw("applyBestOnFailure", &Options::applyBestOnFailure)
        .def_rw("enforceJointLimits", &Options::enforceJointLimits)
        .def_rw("inverseMethod", &Options::inverseMethod)
        .def_rw("transposeGain", &Options::transposeGain)
        .def_rw("seed", &Options::seed);
    py::class_<Result>(system, "IKResult")
        .def_ro("success", &Result::success)
        .def_ro("iterations", &Result::iterations)
        .def_ro("bestIteration", &Result::bestIteration)
        .def_ro("residual", &Result::residual)
        .def_ro("positionResidual", &Result::positionResidual)
        .def_ro("orientationResidual", &Result::orientationResidual)
        .def_ro("bestResidual", &Result::bestResidual)
        .def_ro("bestPositionResidual", &Result::bestPositionResidual)
        .def_ro("bestOrientationResidual", &Result::bestOrientationResidual)
        .def_ro("solution", &Result::solution)
        .def_ro("bestSolution", &Result::bestSolution)
        .def_prop_ro("bestFramePosition", [](const Result& result) {
            return convert_vec_to_np(result.bestFramePosition);
        }, py::rv_policy::move)
        .def_prop_ro("bestFrameOrientation", [](const Result& result) {
            return convert_mat_to_np(result.bestFrameOrientation);
        }, py::rv_policy::move)
        .def_ro("message", &Result::message);
    m.attr("IKOptions") = system.attr("IKOptions");
    m.attr("IKResult") = system.attr("IKResult");

    py::class_<DcMotorParameters>(m, "DcMotorParameters")
        .def(py::init<>())
        .def_rw("resistance", &DcMotorParameters::resistance)
        .def_rw("torqueConstant", &DcMotorParameters::torqueConstant)
        .def_rw("velocityConstant", &DcMotorParameters::velocityConstant)
        .def_rw("busVoltage", &DcMotorParameters::busVoltage)
        .def_rw("peakTorque", &DcMotorParameters::peakTorque)
        .def("isValid", &DcMotorParameters::isValid)
        .def("torquePerVolt", &DcMotorParameters::torquePerVolt)
        .def("backEmfDamping", &DcMotorParameters::backEmfDamping)
        .def("stallTorque", &DcMotorParameters::stallTorque)
        .def("noLoadSpeed", &DcMotorParameters::noLoadSpeed);
    py::enum_<MotorSaturation>(m, "MotorSaturation")
        .value("NONE", MotorSaturation::NONE)
        .value("PEAK_TORQUE", MotorSaturation::PEAK_TORQUE)
        .value("VOLTAGE", MotorSaturation::VOLTAGE);
    py::class_<MotorTorqueBounds>(m, "MotorTorqueBounds")
        .def_ro("lower", &MotorTorqueBounds::lower)
        .def_ro("upper", &MotorTorqueBounds::upper)
        .def_ro("lowerSaturation", &MotorTorqueBounds::lowerSaturation)
        .def_ro("upperSaturation", &MotorTorqueBounds::upperSaturation);
    py::class_<ActuatorCoupling>(m, "ActuatorCoupling")
        .def(py::init<>())
        .def_rw("joint", &ActuatorCoupling::joint)
        .def_rw("ratio", &ActuatorCoupling::ratio);
    py::class_<ActuatorDefinition>(m, "ActuatorDefinition")
        .def(py::init<>())
        .def_rw("name", &ActuatorDefinition::name)
        .def_rw("model", &ActuatorDefinition::model)
        .def_rw("joint", &ActuatorDefinition::joint)
        .def_rw("gearRatio", &ActuatorDefinition::gearRatio)
        .def_rw("couplings", &ActuatorDefinition::couplings)
        .def_rw("motor", &ActuatorDefinition::motor)
        .def_rw("outputDamping", &ActuatorDefinition::outputDamping)
        .def_rw("outputFriction", &ActuatorDefinition::outputFriction)
        .def("isValid", &ActuatorDefinition::isValid);
    py::class_<ActuatorState>(m, "ActuatorState")
        .def_ro("motorSpeed", &ActuatorState::motorSpeed)
        .def_ro("averageMotorSpeed", &ActuatorState::averageMotorSpeed)
        .def_ro("commandedMotorTorque", &ActuatorState::commandedMotorTorque)
        .def_ro("motorTorque", &ActuatorState::motorTorque)
        .def_ro("jointLossTorque", &ActuatorState::jointLossTorque)
        .def_ro("saturation", &ActuatorState::saturation);
    m.def("motorTorqueBounds", py::overload_cast<const DcMotorParameters&, double>(&motorTorqueBounds),
          py::arg("motor"), py::arg("speed"));
    m.def("clipMotorTorque", [](const DcMotorParameters& motor, double speed, double torque) {
        return clipMotorTorque(motor, speed, torque);
    }, py::arg("motor"), py::arg("speed"), py::arg("torque"));
    m.def("motorOperatingRegionViolation", &motorOperatingRegionViolation,
          py::arg("motor"), py::arg("speed"), py::arg("torque"));

    py::class_<TrackDrive>(m, "TrackDrive")
        .def_ro("joint", &TrackDrive::joint)
        .def_ro("gain", &TrackDrive::gain)
        .def_ro("gvIndex", &TrackDrive::gvIndex)
        .def_ro("bodyIdx", &TrackDrive::bodyIdx);
    py::class_<Track>(m, "Track")
        .def_ro("name", &Track::name)
        .def_ro("link", &Track::link)
        .def_ro("bodyIdx", &Track::bodyIdx)
        .def_prop_ro("axis", [](const Track& track) { return convert_vec_to_np(track.axis); }, py::rv_policy::move)
        .def_ro("width", &Track::width)
        .def_ro("thickness", &Track::thickness)
        .def_ro("drives", &Track::drives)
        .def_ro("collisionBodies", &Track::collisionBodies);

    system
        .def("solveIK", [](ArticulatedSystem& self, const std::string& frame,
                           const NDArray& position, const Options& options) {
            return self.solveIK(frame, convert_np_to_vec<3>(position), options);
        }, py::arg("frame"), py::arg("targetPosition"), py::arg("options") = Options{})
        .def("solveIK", [](ArticulatedSystem& self, size_t frame,
                           const NDArray& position, const Options& options) {
            return self.solveIK(frame, convert_np_to_vec<3>(position), options);
        }, py::arg("frame"), py::arg("targetPosition"), py::arg("options") = Options{})
        .def("solveIK", [](ArticulatedSystem& self, const std::string& frame,
                           const NDArray& position, const NDArray& orientation, Options options) {
            return self.solveIK(frame, convert_np_to_vec<3>(position),
                                convert_np_to_mat<3, 3>(orientation), options);
        }, py::arg("frame"), py::arg("targetPosition"), py::arg("targetOrientation"),
           py::arg("options") = Options{})
        .def("solveIK", [](ArticulatedSystem& self, size_t frame,
                           const NDArray& position, const NDArray& orientation, Options options) {
            options.useOrientation = true;
            return self.solveIK(frame, convert_np_to_vec<3>(position),
                                convert_np_to_mat<3, 3>(orientation), options);
        }, py::arg("frame"), py::arg("targetPosition"), py::arg("targetOrientation"),
           py::arg("options") = Options{})
        .def("setActuators", &ArticulatedSystem::setActuators, py::arg("actuators"))
        .def("getNumberOfActuators", &ArticulatedSystem::getNumberOfActuators)
        .def("getActuatorNames", &ArticulatedSystem::getActuatorNames, py::rv_policy::copy)
        .def("getActuators", &ArticulatedSystem::getActuators, py::rv_policy::copy)
        .def("getActuatorIndex", &ArticulatedSystem::getActuatorIndex, py::arg("name"))
        .def("setActuatorTorques", &ArticulatedSystem::setActuatorTorques, py::arg("torques"))
        .def("setActuatorTorque", py::overload_cast<const std::string&, double>(&ArticulatedSystem::setActuatorTorque),
             py::arg("name"), py::arg("torque"))
        .def("setActuatorTorque", py::overload_cast<size_t, double>(&ArticulatedSystem::setActuatorTorque),
             py::arg("index"), py::arg("torque"))
        .def("getActuatorTorques", &ArticulatedSystem::getActuatorTorques, py::rv_policy::copy)
        .def("getActuatorTorqueBounds", [](const ArticulatedSystem& self) {
            VecDyn lower, upper;
            self.getActuatorTorqueBounds(lower, upper);
            return py::make_tuple(convert_vecdyn_to_np(lower), convert_vecdyn_to_np(upper));
        })
        .def("setBusVoltage", py::overload_cast<double>(&ArticulatedSystem::setBusVoltage), py::arg("voltage"))
        .def("setBusVoltage", py::overload_cast<const std::string&, double>(&ArticulatedSystem::setBusVoltage),
             py::arg("actuatorName"), py::arg("voltage"))
        .def("getActuatorStates", &ArticulatedSystem::getActuatorStates, py::rv_policy::copy)
        .def("setMotorOperatingRegionEnforced", &ArticulatedSystem::setMotorOperatingRegionEnforced, py::arg("enforce"))
        .def("isMotorOperatingRegionEnforced", &ArticulatedSystem::isMotorOperatingRegionEnforced)
        .def("getTracks", &ArticulatedSystem::getTracks, py::rv_policy::copy)
        .def("getTrackIndex", &ArticulatedSystem::getTrackIndex, py::arg("name"))
        .def("getTrackSpeed", &ArticulatedSystem::getTrackSpeed, py::arg("index"));
}
