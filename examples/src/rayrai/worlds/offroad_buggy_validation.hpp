#pragma once
// Physics checks of the off-road buggy (rayrai_offroad_buggy --validate) and the autopilot that
// drives their lap.
//
// Each check compares the simulation with an independent prediction:
//   1. Static equilibrium on flat ground: the wheel loads carry the weight, and their front/rear
//      split is the lever rule about the centre of mass.
//   2. Heave: the natural frequency and damping ratio of the body bouncing on its four coilovers,
//      against the linear model from the spring rates, damper rates and motion ratios.
//   3. Steering geometry: the outer wheel angle against Ackermann's condition for the inner one.
//   4. Steady cornering: the lateral load transfer against m a_y h, with the wheels' gyroscopic
//      moment.
//   5. A lap of the circuit with its jump, whoops, washboard and rock garden, past the venue's
//      barriers, hay bales and cones: loop-closure error, suspension travel used, air time and
//      simulation speed.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>

#include "offroad_buggy_scene.hpp"
#include "offroad_buggy_vehicle.hpp"
#include "offroad_buggy_venue.hpp"

namespace offroad {

inline double headingOf(const raisim::ArticulatedSystem& system) {
  const auto& q = system.getGeneralizedCoordinate();
  return std::atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
}

/// Follows the track centre line (pure pursuit) at the speed that the corners ahead and the track
/// features allow.
class Autopilot {
 public:
  static constexpr double kTopSpeed = 17.0, kLateral = 5.0, kBraking = 4.0;  // m/s, m/s^2, m/s^2

  Autopilot(const TerrainInfo& info, const TrackLocator& track) : info_(info), track_(track) {}

  /// Speed limit at a station: the feature's speed, or the cornering speed at kLateral.
  double limit(double station) const {
    for (const auto& f : info_.features)
      if (station > f.start - 10 && station < f.start + f.length) {
        if (f.name == "whoops") return 9.0;
        if (f.name == "tabletop jump") return 12.0;
        if (f.name == "rock garden") return 3.5;
        if (f.name == "washboard") return 11.0;
      }
    return std::min(kTopSpeed, std::sqrt(kLateral / std::max(std::abs(track_.curvature(station)), 1e-4)));
  }
  /// The speed from which the car can still brake to every limit within the next 60 m.
  double targetSpeed(double station) const {
    double target = kTopSpeed;
    for (double d = 0; d <= 60; d += 2) target = std::min(target, std::sqrt(limit(station + d) * limit(station + d) + 2 * kBraking * d));
    return target;
  }

  DriverInput update(const Buggy& buggy) {
    const auto* system = buggy.system();
    const auto& q = system->getGeneralizedCoordinate();
    const double speed = buggy.telemetry().speed, heading = headingOf(*system);
    station_ = track_.locate(q[0], q[1])[1];
    Eigen::Vector3d ahead;
    Eigen::Vector2d direction;
    const double lookahead = 5.0 + 0.45 * std::max(speed, 0.0);
    track_.at(station_ + lookahead, ahead, direction);
    const double alpha = std::remainder(std::atan2(ahead.y() - q[1], ahead.x() - q[0]) - heading, 2 * M_PI);
    // Pure pursuit: the wheel angle that puts the car on the arc through the look-ahead point.
    const double wheelAngle = std::atan(2 * 2.714 * std::sin(alpha) / lookahead);
    const double maxWheelAngle = 0.53 / (1.0 + speed * speed / 500.0);
    DriverInput input;
    input.steer = std::clamp(wheelAngle / maxWheelAngle, -1.0, 1.0);
    const double error = targetSpeed(station_) - speed;
    input.throttle = std::clamp(0.35 * error + 0.15, 0.0, 1.0);
    input.brake = error < -1.5 ? std::clamp(-0.25 * error, 0.0, 1.0) : 0.0;
    if (input.brake > 0) input.throttle = 0;
    return input;
  }
  double station() const { return station_; }

 private:
  const TerrainInfo& info_;
  const TrackLocator& track_;
  double station_ = 0;
};

namespace validation {

struct Flat {
  raisim::World world;
  std::unique_ptr<Buggy> buggy;
  explicit Flat(const std::filesystem::path& root) {
    world.setTimeStep(0.001);
    world.addGround(0, "terrain");
    world.setMaterialPairProp("tire", "terrain", 0.85, 0.0, 0.0);
    buggy = std::make_unique<Buggy>(world, (root / "buggy.urdf").string());
    buggy->place(0, 0, 0, [](double, double) { return 0.0; });
  }
  /// Step for a duration, calling observe(time) after each frame of 16 steps.
  template <class Input, class Observe>
  void run(double duration, Input&& input, Observe&& observe) {
    const int frames = int(duration / 0.016);
    for (int f = 0; f < frames; ++f) {
      for (int i = 0; i < 16; ++i) {
        buggy->applyControls(input(), 0.001);
        world.integrate();
        buggy->accumulate(0.001);
      }
      buggy->updateTelemetry(0.016);
      observe(world.getWorldTime());
    }
  }
};

inline void heading(const std::string& title) { std::cout << "\n" << title << "\n"; }
inline void row(const std::string& what, double sim, double reference, const std::string& unit, int digits = 2) {
  const double error = reference != 0 ? 100 * (sim - reference) / std::abs(reference) : 0.0;
  std::cout << "  " << std::left << std::setw(46) << what << std::right << std::fixed << std::setprecision(digits) << std::setw(10) << sim << std::setw(10) << reference << " " << std::setw(6) << unit
            << std::setprecision(2) << std::setw(8) << error << " %\n";
}

/// Mean contact x of each corner and the system centre of mass, in the chassis frame.
struct Statics {
  std::array<double, 4> load{}, contactX{};
  double comX = 0, comZ = 0, groundZ = 0;
};

inline Statics measureStatics(Buggy& buggy, raisim::World& world, double duration) {
  Statics s;
  int samples = 0;
  const auto* system = buggy.system();
  for (int step = 0; step < int(duration / 0.001); ++step) {
    buggy.applyControls(DriverInput{}, 0.001);
    world.integrate();
    raisim::Mat<3, 3> R;
    raisim::Vec<3> p;
    system->getBodyPose(0, R, p);
    for (auto& contact : system->getContacts()) {
      if (contact.skip() || contact.getPairObjectIndex() == system->getIndexInWorld() || !contact.getImpulsePtr()) continue;
      for (int i = 0; i < 4; ++i)
        if (contact.getlocalBodyIndex() == buggy.wheelBodyIndex(i)) {
          Eigen::Vector3d f = contact.getContactFrame().e().transpose() * contact.getImpulse().e() / 0.001;
          if (!contact.isObjectA()) f = -f;
          const double fz = f.z();
          s.load[i] += fz;
          s.contactX[i] += fz * (R.e().transpose() * (contact.getPosition().e() - p.e())).x();
          s.groundZ += fz * contact.getPosition()[2];
        }
    }
    const Eigen::Vector3d com = system->getCOM().e();
    s.comX += (R.e().transpose() * (com - p.e())).x();
    s.comZ += com.z();
    ++samples;
  }
  double total = 0;
  for (int i = 0; i < 4; ++i) {
    s.contactX[i] /= s.load[i];
    total += s.load[i];
    s.load[i] /= samples;
  }
  s.groundZ /= total;
  s.comX /= samples;
  s.comZ /= samples;
  return s;
}

}  // namespace validation

inline int validate(const std::filesystem::path& root) {
  using namespace validation;
  std::cout << "rayrai_offroad_buggy physics checks (time step 1 ms)\n";
  std::cout << "  " << std::left << std::setw(46) << "quantity" << std::right << std::setw(10) << "sim" << std::setw(10) << "reference" << " " << std::setw(6) << "unit" << std::setw(10) << "error\n";
  int failures = 0;
  auto check = [&](bool ok) { failures += ok ? 0 : 1; };

  // 1. Static equilibrium.
  {
    Flat flat(root);
    auto& buggy = *flat.buggy;
    flat.run(3.0, [] { return DriverInput{}; }, [](double) {});
    const auto s = measureStatics(buggy, flat.world, 1.0);
    const double weight = buggy.mass() * 9.81;
    const double sum = std::accumulate(s.load.begin(), s.load.end(), 0.0);
    const double xf = 0.5 * (s.contactX[0] + s.contactX[1]), xr = 0.5 * (s.contactX[2] + s.contactX[3]);
    heading("1. Static equilibrium on flat ground");
    row("sum of wheel loads vs m g", sum, weight, "N");
    row("front axle load vs lever rule about the COM", s.load[0] + s.load[1], weight * (s.comX - xr) / (xf - xr), "N");
    const auto& t = buggy.telemetry();
    // The radial tyre spring carries the wheel load less the weight of the wheel below it.
    const double wheelWeight = buggy.system()->getMass()[buggy.wheelBodyIndex(0)] * 9.81;
    for (int i = 0; i < 4; i += 2)
      row(std::string("tyre spring force vs load - wheel weight, ") + Buggy::kCornerNames[i], buggy.tyreStiffness(i) * t.corners[i].tyreDeflection, s.load[i] - wheelWeight, "N");
    double sag = 0;
    for (int i = 0; i < 4; ++i) sag = std::max(sag, std::abs(t.corners[i].shockExtension));
    std::cout << "  ride height: largest damper offset from the design pose " << std::setprecision(2) << sag * 1000 << " mm\n";
    check(std::abs(sum / weight - 1) < 0.002);
  }

  // 2. Heave: push the body down at its centre of mass, release, and compare the free vibration
  // with a linear half-car model (body heave and pitch, front and rear unsprung masses).
  {
    Flat flat(root);
    auto& buggy = *flat.buggy;
    auto* system = buggy.system();
    flat.run(3.0, [] { return DriverInput{}; }, [](double) {});
    // Per axle: wheel rate k MR^2 + k_lat s^2 (the lateral tyre spring loaded by the scrub s of
    // the contact patch), damping likewise, and the radial tyre spring under the unsprung mass.
    // Motion ratios and scrub rates at the contact point from offroad_buggy_design.wheel_kinematics().
    const std::array<double, 2> mr = {0.6318, 0.6648}, scrub = {0.1071, 0.1131};
    double unsprung = 0;
    const std::string c0 = Buggy::kCornerNames[0];
    for (const char* body : {"_wheel", "_upright", "_tyre_radial", "_tyre_lateral"}) unsprung += system->getMass()[system->getBodyIdx(c0 + body)];
    for (const char* body : {"_lower_arm", "_upper_arm", "_tie_rod", "_shock_body", "_shock_rod"}) unsprung += 0.5 * system->getMass()[system->getBodyIdx(c0 + body)];
    const double kLat = 110e3, cLat = 600.0;  // offroad_buggy_design.TYRE
    std::array<double, 2> kw, cw, kt, ct, x;
    for (int axle = 0; axle < 2; ++axle) {
      const int i = 2 * axle;
      kw[axle] = 2 * (buggy.stiffness(i) * mr[axle] * mr[axle] + kLat * scrub[axle] * scrub[axle]);
      cw[axle] = 2 * (buggy.damperCoefficient(i) * mr[axle] * mr[axle] + cLat * scrub[axle] * scrub[axle]);
      kt[axle] = 2 * buggy.tyreStiffness(i);
      ct[axle] = 2 * buggy.tyreDamping(i);
    }
    const double ms = buggy.mass() - 4 * unsprung, inertia = 800.0, xcg = -0.14;  // chassis link of the URDF
    x = {1.3571 - xcg, -1.3571 - xcg};
    // q = (body heave, body pitch, front and rear unsprung heave); suspension deflection z + x th - zu.
    Eigen::Matrix4d M = Eigen::Vector4d(ms, inertia, 2 * unsprung, 2 * unsprung).asDiagonal(), K, C;
    K.setZero();
    C.setZero();
    for (int axle = 0; axle < 2; ++axle) {
      Eigen::Vector4d g(1, x[axle], 0, 0), t = Eigen::Vector4d::Zero();
      g[2 + axle] = -1;
      t[2 + axle] = 1;
      K += kw[axle] * g * g.transpose() + kt[axle] * t * t.transpose();
      C += cw[axle] * g * g.transpose() + ct[axle] * t * t.transpose();
    }
    // A 1.5 kN push for 0.1 s at the centre of mass of both, then free motion.
    const double rest = system->getGeneralizedCoordinate()[2];
    std::vector<double> time, simulated, linear;
    Eigen::Vector4d q = Eigen::Vector4d::Zero(), u = Eigen::Vector4d::Zero();
    const Eigen::Matrix4d Minv = M.inverse();
    DriverInput rolling;
    rolling.parkingHold = false;  // the wheels roll freely, so that the body can pitch as in the model
    for (int step = 0; step < 2500; ++step) {
      buggy.applyControls(rolling, 0.001);
      const double push = step < 100 ? -1500.0 : 0.0;
      if (push != 0) {
        raisim::Mat<3, 3> R;
        raisim::Vec<3> p;
        system->getBodyPose(0, R, p);
        const Eigen::Vector3d local = R.e().transpose() * (system->getCOM().e() - p.e());
        system->setExternalForce(0, raisim::Vec<3>{local.x(), local.y(), local.z()}, raisim::Vec<3>{0, 0, push});
      }
      flat.world.integrate();
      for (int sub = 0; sub < 10; ++sub) {  // the linear model, semi-implicit Euler at 0.1 ms
        u += 1e-4 * Minv * (Eigen::Vector4d(push, 0, 0, 0) - K * q - C * u);
        q += 1e-4 * u;
      }
      time.push_back(step * 0.001);
      simulated.push_back(system->getGeneralizedCoordinate()[2] - rest);
      linear.push_back(q[0] + q[1] * (0.0 - xcg));  // the chassis origin, xcg ahead of the body's centre of mass
    }
    // The first half cycle: the deepest compression and its time, and the time the body passes its
    // rest height again. Later on, tyre scrub at standstill adds friction the linear model lacks.
    auto firstSwing = [&](const std::vector<double>& z) {
      size_t trough = 0;
      while (trough + 1 < z.size() && z[trough + 1] <= z[trough]) ++trough;
      size_t k = trough;
      while (k + 1 < z.size() && z[k + 1] < 0) ++k;
      const double crossing = time[k] + 0.001 * z[k] / (z[k] - z[k + 1]);
      return std::array<double, 3>{-z[trough], time[trough], crossing};
    };
    const auto sim = firstSwing(simulated), lin = firstSwing(linear);
    double rms = 0;
    size_t count = 0;
    for (size_t k = 0; k < time.size() && time[k] <= lin[2]; ++k, ++count) rms += (simulated[k] - linear[k]) * (simulated[k] - linear[k]);
    rms = std::sqrt(rms / std::max<size_t>(count, 1));
    heading("2. Heave: body pushed down and released vs a linear half-car model (heave, pitch, unsprung masses)");
    row("deepest compression", sim[0] * 1000, lin[0] * 1000, "mm", 2);
    row("time of the deepest compression", sim[1], lin[1], "s", 3);
    row("time back at rest height (quarter period after)", sim[2], lin[2], "s", 3);
    std::cout << "  RMS difference over that half cycle " << std::setprecision(2) << rms * 1000 << " mm (" << std::setprecision(1) << 100 * rms / lin[0] << " % of the peak)\n";
    check(std::abs(sim[0] / lin[0] - 1) < 0.06 && std::abs(sim[2] / lin[2] - 1) < 0.06);
  }

  // 3. Steering: rack and pinion (mimic constraint) and the steering geometry.
  {
    Flat flat(root);
    auto& buggy = *flat.buggy;
    heading("3. Steering: rack travel = pinion radius x steering-wheel angle; outer wheel vs Ackermann");
    double worstRack = 0;
    for (double steer : {0.25, 0.5, 0.75, 1.0}) {
      DriverInput input;
      input.steer = steer;
      flat.run(
          1.0, [&] { return input; },
          [&](double) {
            const auto& t = buggy.telemetry();
            worstRack = std::max(worstRack, std::abs(t.rackPosition + 0.0095 * t.steeringWheelAngle));
          });
      const auto& t = buggy.telemetry();
      const double inner = t.corners[0].steerAngle, outer = t.corners[1].steerAngle;
      // Ackermann: cot(outer) - cot(inner) = kingpin track / wheelbase (offroad_buggy_design.py).
      const double ideal = std::atan(1.0 / (1.0 / std::tan(inner) + 2.0258 / 2.7142));
      row("outer angle at inner " + std::to_string(int(std::round(inner * 180 / M_PI))) + " deg (design)", outer * 180 / M_PI, ideal * 180 / M_PI, "deg", 2);
    }
    std::cout << "  largest rack-and-pinion error " << std::setprecision(3) << worstRack * 1e6 << " um\n";
    check(worstRack < 1e-5);
  }

  // 4. Steady cornering: the vertical wheel loads balance the roll moment m a_y h.
  {
    Flat flat(root);
    auto& buggy = *flat.buggy;
    auto* system = buggy.system();
    DriverInput input;
    input.steer = 0.35;
    double integral = 0;
    auto control = [&] {
      const double error = 8.0 - buggy.telemetry().speed;
      integral += error * 0.001;
      input.throttle = std::clamp(0.2 * error + 0.05 * integral + 0.1, 0.0, 1.0);
      return input;
    };
    flat.run(10.0, control, [](double) {});
    double rollMoment = 0, pitchMoment = 0, ay = 0, ax = 0, h = 0, gyro = 0;
    int n = 0;
    flat.run(2.0, control, [&](double) {
      const auto& t = buggy.telemetry();
      const auto& gv = system->getGeneralizedVelocity();
      // Moments of the vertical loads about the centre of mass, from where they act.
      for (int i = 0; i < 4; ++i) {
        rollMoment += t.corners[i].wheelLoad * (t.corners[i].contact2d.y() - t.com2d.y());
        pitchMoment += t.corners[i].wheelLoad * (t.corners[i].contact2d.x() - t.com2d.x());
      }
      ay += t.acceleration.y();
      ax += t.acceleration.x();
      h += t.comHeight;
      // The spinning wheels turning with the car change their angular momentum by -I w Omega about x.
      double spin = 0;
      for (int i = 0; i < 4; ++i) spin += t.corners[i].wheelSpeed;
      gyro += 1.25 * spin * gv[5];
      ++n;
    });
    rollMoment /= n, pitchMoment /= n, ay /= n, ax /= n, h /= n, gyro /= n;
    heading("4. Steady cornering at 8 m/s: moments of the vertical wheel loads about the centre of mass");
    std::cout << "  lateral acceleration " << std::setprecision(2) << ay << " m/s^2, centre of mass " << h << " m above the ground\n";
    // The tyres' lateral forces act at the ground, h below the centre of mass.
    row("roll moment of the loads vs m a_y h + I w Omega", -rollMoment, buggy.mass() * ay * h + gyro, "Nm", 1);
    check(std::abs(-rollMoment / (buggy.mass() * ay * h + gyro) - 1) < 0.02);
  }

  // 5. A lap of the track.
  {
    const auto info = readTerrainInfo(root / "terrain" / "terrain.txt");
    const auto surface = readSurface(root / "terrain", info.extent);
    raisim::World world;
    world.setTimeStep(0.001);
    setMaterials(world);
    const auto heights = readHeights(root / "terrain", info);
    auto* terrain = addTerrain(world, info, heights);
    PlacementGrid occupied(4.0);
    auto venue = buildVenue(world, *terrain, info, root.parent_path(), occupied);
    placeScenery(world, *terrain, info, surface, root.parent_path(), 0.0, occupied);
    Buggy buggy(world, (root / "buggy.urdf").string());
    const TrackLocator track(info);
    buggy.place(info.start.x(), info.start.y(), info.start.z(), [&](double x, double y) { return terrain->getHeight(x, y); });
    Autopilot pilot(info, track);
    double maxError = 0, airTime = 0, maxLoad = 0, maxSpeed = 0, travelled = 0, lastStation = -1;
    std::array<double, 4> minExt, maxExt;
    minExt.fill(1);
    maxExt.fill(-1);
    double bumpStop = 0;
    bool upset = false;
    const auto start = std::chrono::steady_clock::now();
    int steps = 0;
    DriverInput input;
    std::vector<Eigen::Vector3d> placed;
    for (const auto& prop : venue.loose) placed.push_back(prop.body->getPosition());
    while (travelled < info.length && world.getWorldTime() < 150) {
      input = pilot.update(buggy);
      for (int i = 0; i < 16; ++i) {
        buggy.applyControls(input, 0.001);
        world.integrate();
        buggy.accumulate(0.001);
        ++steps;
      }
      buggy.updateTelemetry(0.016);
      const auto& q = buggy.system()->getGeneralizedCoordinate();
      updateLooseProps(venue, Eigen::Vector3d(q[0], q[1], q[2]), 0.016);
      const auto& t = buggy.telemetry();
      maxError = std::max(maxError, t.maxLoopError);
      maxSpeed = std::max(maxSpeed, t.speed);
      bool airborne = true;
      for (int i = 0; i < 4; ++i) {
        const auto& c = t.corners[i];
        minExt[i] = std::min(minExt[i], c.shockExtension);
        maxExt[i] = std::max(maxExt[i], c.shockExtension);
        maxLoad = std::max(maxLoad, c.wheelLoad);
        airborne &= !c.contact;
        if (c.shockExtension < buggy.shockLimits(i)[0] + 0.002) bumpStop += 0.016 / 4;
      }
      if (airborne) airTime += 0.016;
      if (std::abs(t.rollAngle) > 1.2 || std::abs(t.pitchAngle) > 1.2) upset = true;
      const double station = pilot.station();
      if (lastStation >= 0) travelled += std::clamp(std::remainder(station - lastStation, info.length), -5.0, 5.0);
      lastStation = station;
      if (upset) break;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    heading("5. A lap of the circuit (tabletop jump, whoops, washboard, rock garden), autopilot");
    std::cout << std::setprecision(1) << "  distance " << travelled << " m in " << world.getWorldTime() << " s, top speed " << maxSpeed * 3.6 << " km/h, " << (upset ? "rolled over" : "stayed upright")
              << "\n";
    std::cout << "  air time with all four wheels off the ground " << airTime << " s, peak wheel load " << maxLoad / 1000 << " kN, " << std::setprecision(2) << bumpStop << " s on the bump stops\n";
    for (int i = 0; i < 4; ++i)
      std::cout << "  damper travel used " << Buggy::kCornerNames[i] << ": " << std::setprecision(0) << minExt[i] * 1000 << " .. " << maxExt[i] * 1000 << " mm (stroke "
                << buggy.shockLimits(i)[0] * 1000 << " .. " << buggy.shockLimits(i)[1] * 1000 << ")\n";
    std::cout << std::setprecision(2) << "  largest loop-closure error " << maxError * 1e6 << " um\n";
    int moved = 0;
    for (size_t i = 0; i < placed.size(); ++i) moved += (venue.loose[i].body->getPosition() - placed[i]).norm() > 0.1;
    std::cout << "  " << moved << " of " << placed.size() << " hay bales and cones moved more than 10 cm\n";
    std::cout << std::setprecision(1) << "  " << seconds / steps * 1e6 << " us per step with " << world.getObjList().size() << " objects, " << world.getWorldTime() / seconds << "x real time\n";
    check(!upset && travelled > 0.95 * info.length && maxError < 1e-3);
  }
  std::cout << "\n" << (failures ? "FAILED " + std::to_string(failures) + " check(s)" : std::string("all checks passed")) << "\n";
  return failures ? 1 : 0;
}

}  // namespace offroad
