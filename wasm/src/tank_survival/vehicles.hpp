// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#pragma once

#include "weapons.hpp"

namespace tank {

enum class Role { Player, Scout, Medium, Heavy, Destroyer, Artillery, Boss };

// Armor is in millimetres of rolled steel; gunPower scales the penetration and damage of the
// vehicle's gun. model names the folder of the vehicle's meshes (vehicles/<model>).
struct VehicleSpec {
  const char* name;
  const char* description;
  Role role;
  const char* model;
  double mass, health;
  double frontArmor, sideArmor, rearArmor, topArmor;
  double speed, turnRate;
  bool fixedTurret;
  WeaponId gun;
  double gunPower, reload, spread, range;
  int score;
  glm::vec4 marker;
};

inline const std::array<VehicleSpec, 7> kVehicles{{
    {"Leopard 2A4", "your tank", Role::Player, "leopard_2a4", 55000, 240, 140, 70, 40, 25, 10.0, 3.8, false,
     WeaponId::AP, 1.0, 3.0, 0.0, 0, 0, {1, 1, 1, 1}},
    {"BMD-2", "airborne fighting vehicle: fast, thin armor, autocannon", Role::Scout, "bmd2", 8000, 45, 18, 12, 10, 8,
     15.0, 5.0, false, WeaponId::MG, 1.6, 0.09, 0.03, 22, 60, {1.0f, 0.85f, 0.3f, 1}},
    {"T-72B3", "main battle tank: balanced", Role::Medium, "t72b3", 45000, 120, 110, 60, 35, 20, 9.0, 3.4, false,
     WeaponId::AP, 0.85, 3.6, 0.03, 28, 120, {1.0f, 0.3f, 0.2f, 1}},
    {"T-90AM", "heavy main battle tank: thick front, weak rear: flank it", Role::Heavy, "t90am", 48000, 260, 260, 120,
     60, 35, 6.5, 2.6, false, WeaponId::AP, 1.25, 5.0, 0.035, 32, 260, {0.75f, 0.5f, 1.0f, 1}},
    {"2S25 Sprut-SD", "tank destroyer: long-range gun, light armor", Role::Destroyer, "sprut_sd", 18000, 150, 190, 50,
     30, 20, 8.0, 3.0, false, WeaponId::AP, 1.45, 6.5, 0.012, 62, 220, {1.0f, 0.6f, 0.1f, 1}},
    {"2S19 Msta-S", "self-propelled howitzer: shells land where you will be", Role::Artillery, "msta_s", 42000, 90, 40,
     25, 15, 10, 6.0, 3.0, false, WeaponId::HE, 1.0, 7.0, 4.0, 85, 180, {0.3f, 1.0f, 0.5f, 1}},
    {"T-14 Armata", "boss: salvos of AP and HE, armor everywhere but the rear", Role::Boss, "t14_armata", 55000, 1100,
     320, 180, 80, 50, 5.5, 2.2, false, WeaponId::AP, 1.6, 2.8, 0.03, 40, 1500, {1.0f, 0.15f, 0.4f, 1}},
}};

inline const VehicleSpec& vehicle(Role role) { return kVehicles[size_t(role)]; }

// Where the parts of a vehicle model meet (prepare_tank_assets.py writes vehicle.json): in metres,
// x forward, z up from the ground under the hull.
struct VehicleModel {
  Vec3 hullMin, hullMax, turretMin, turretMax, gunMin, gunMax, turretPivot, gunPivot;
  double gunRadius = 0.1, muzzle = 1.0;
  double wheelRadius = 0.3, axleHeight = 0.36;
};

inline VehicleModel loadVehicleModel(const std::string& folder) {
  const std::string text = readText(folder + "/vehicle.json");
  const auto numbers = [&](const std::string& key, size_t after = 0) {
    const size_t at = text.find("\"" + key + "\"", after);
    if (at == std::string::npos) throw std::runtime_error(folder + "/vehicle.json has no " + key);
    std::vector<double> values;
    const std::regex number(R"re(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)re");
    const size_t end = text.find_first_of("]}", text.find_first_of("[0123456789-", at));
    const std::string span = text.substr(at, end - at);
    for (auto it = std::sregex_iterator(span.begin(), span.end(), number); it != std::sregex_iterator(); ++it)
      values.push_back(std::stod(it->str()));
    return values;
  };
  const auto point = [&](const std::string& key, size_t after = 0) {
    const auto v = numbers(key, after);
    if (v.size() < 3) throw std::runtime_error(folder + "/vehicle.json: " + key + " is not a point");
    return Vec3(v[0], v[1], v[2]);
  };
  VehicleModel model;
  for (const auto& [part, low, high] : {std::tuple<const char*, Vec3*, Vec3*>{"hull", &model.hullMin, &model.hullMax},
                                        {"turret", &model.turretMin, &model.turretMax},
                                        {"gun", &model.gunMin, &model.gunMax}}) {
    const size_t at = text.find(std::string("\"") + part + "\"");
    *low = point("min", at);
    *high = point("max", at);
  }
  model.turretPivot = point("turretPivot");
  model.gunPivot = point("gunPivot");
  model.gunRadius = numbers("gunRadius").at(0);
  model.muzzle = numbers("muzzle").at(0);
  const double width = model.hullMax.y() - model.hullMin.y();
  model.wheelRadius = 0.33 * width / 3.6;
  model.axleHeight = model.wheelRadius + 0.06;
  return model;
}

// The vehicle's URDF: the hull riding on two driven tracks, the turret turning on its ring and the gun
// elevating at its trunnions, each drawn with its part of the model and colliding as a box (the gun as
// a cylinder). The tracks' belts are the contact surface; the model draws its own tracks.
inline std::string vehicleUrdf(const VehicleSpec& spec, const VehicleModel& m, const std::string& folder) {
  std::ostringstream u;
  u.precision(6);
  const auto v3 = [](const Vec3& v) {
    std::ostringstream s;
    s.precision(6);
    s << v.x() << ' ' << v.y() << ' ' << v.z();
    return s.str();
  };
  const auto box = [&](double mass, const Vec3& size, const Vec3& center) {
    std::ostringstream s;
    s << "<inertial><mass value=\"" << mass << "\"/><origin xyz=\"" << v3(center) << "\"/><inertia ixx=\""
      << mass * (size.y() * size.y() + size.z() * size.z()) / 12 << "\" iyy=\""
      << mass * (size.x() * size.x() + size.z() * size.z()) / 12 << "\" izz=\""
      << mass * (size.x() * size.x() + size.y() * size.y()) / 12 << "\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial>";
    return s.str();
  };
  const double width = m.hullMax.y() - m.hullMin.y(), length = m.hullMax.x() - m.hullMin.x();
  const double axle = m.axleHeight, r = m.wheelRadius;
  const double deck = std::min(m.hullMax.z(), m.turretPivot.z());
  const Vec3 hullSize(length - 0.3, width - 0.1, std::max(0.3, deck - axle - 0.05));
  const Vec3 hullCenter(0.5 * (m.hullMin.x() + m.hullMax.x()), 0.0, axle + 0.05 + 0.5 * hullSize.z());
  const Vec3 turretSize(m.turretMax.x() - m.turretMin.x(), m.turretMax.y() - m.turretMin.y(),
                        std::min(m.turretMax.z(), m.turretPivot.z() + 1.1) - m.turretPivot.z());
  const Vec3 turretCenter(0.5 * (m.turretMin.x() + m.turretMax.x()), 0.5 * (m.turretMin.y() + m.turretMax.y()),
                          m.turretPivot.z() + 0.5 * turretSize.z());
  const double barrel = m.muzzle - m.gunPivot.x();
  const double hullMass = 0.7 * spec.mass, turretMass = 0.24 * spec.mass, gunMass = 0.05 * spec.mass;
  const double sprocketMass = 0.005 * spec.mass;
  const Vec3 origin(0, 0, axle);
  u << "<?xml version=\"1.0\"?><robot name=\"" << spec.model << "\">";
  u << "<link name=\"hull\">" << box(hullMass, hullSize, hullCenter - origin)
    << "<visual><origin xyz=\"" << v3(-origin) << "\"/><geometry><mesh filename=\"" << folder
    << "/hull.gltf\"/></geometry></visual><collision><origin xyz=\"" << v3(hullCenter - origin)
    << "\"/><geometry><box size=\"" << v3(hullSize) << "\"/></geometry></collision></link>";
  const double trackY = 0.5 * width - 0.3 * width / 3.6;
  const double rear = m.hullMin.x() + r + 0.25, front = m.hullMax.x() - r - 0.3;
  for (const char* side : {"left", "right"}) {
    const double y = std::string(side) == "left" ? trackY : -trackY;
    u << "<link name=\"" << side << "_sprocket_link\">" << box(sprocketMass, Vec3(2 * r, 0.5, 2 * r), Vec3::Zero())
      << "</link><joint name=\"" << side << "_sprocket\" type=\"continuous\"><parent link=\"hull\"/><child link=\""
      << side << "_sprocket_link\"/><origin xyz=\"" << rear << ' ' << y << " 0\"/><axis xyz=\"0 1 0\"/></joint>";
    u << "<link name=\"" << side << "_idler_link\"/><joint name=\"" << side << "_idler\" type=\"fixed\"><parent "
      << "link=\"hull\"/><child link=\"" << side << "_idler_link\"/><origin xyz=\"" << front << ' ' << y
      << " 0.05\"/></joint>";
  }
  const Vec3 turretOrigin = m.turretPivot - origin;
  u << "<link name=\"turret_link\">" << box(turretMass, turretSize, turretCenter - m.turretPivot)
    << "<visual><origin xyz=\"" << v3(-m.turretPivot) << "\"/><geometry><mesh filename=\"" << folder
    << "/turret.gltf\"/></geometry></visual><collision><origin xyz=\"" << v3(turretCenter - m.turretPivot)
    << "\"/><geometry><box size=\"" << v3(turretSize) << "\"/></geometry></collision></link>"
    << "<joint name=\"turret\" type=\"continuous\"><parent link=\"hull\"/><child link=\"turret_link\"/><origin xyz=\""
    << v3(turretOrigin) << "\"/><axis xyz=\"0 0 1\"/></joint>";
  u << "<link name=\"gun_link\">" << box(gunMass, Vec3(barrel, 2 * m.gunRadius, 2 * m.gunRadius), Vec3(0.5 * barrel, 0, 0))
    << "<visual><origin xyz=\"" << v3(-m.gunPivot) << "\"/><geometry><mesh filename=\"" << folder
    << "/gun.gltf\"/></geometry></visual><collision><origin xyz=\"" << 0.5 * barrel
    << " 0 0\" rpy=\"0 1.5708 0\"/><geometry><cylinder length=\"" << barrel << "\" radius=\"" << m.gunRadius
    << "\"/></geometry></collision></link><joint name=\"gun\" type=\"revolute\"><parent link=\"turret_link\"/>"
    << "<child link=\"gun_link\"/><origin xyz=\"" << v3(m.gunPivot - m.turretPivot) << "\"/><axis xyz=\"0 1 0\"/>"
    << "<limit lower=\"-0.35\" upper=\"0.15\" effort=\"1e7\" velocity=\"2\"/></joint>";
  for (const char* side : {"left", "right"}) {
    const double y = std::string(side) == "left" ? trackY : -trackY;
    u << "<track name=\"" << side << "\" link=\"hull\" width=\"" << 0.55 * width / 3.6
      << "\" thickness=\"0.06\" visual=\"false\"><axis xyz=\"0 1 0\"/><wheel joint=\"" << side << "_sprocket\" radius=\""
      << r << "\"/><wheel joint=\"" << side << "_idler\" radius=\"" << r << "\"/>";
    for (int k = 1; k <= 3; ++k)
      u << "<wheel xyz=\"" << rear + (front - rear) * k / 4.0 << ' ' << y << " -0.05\" radius=\"" << r - 0.03 << "\"/>";
    u << "<drive joint=\"" << side << "_sprocket\" gain=\"" << r << "\"/></track>";
  }
  u << "</robot>";
  return u.str();
}

struct Tank {
  raisim::ArticulatedSystem* system = nullptr;
  const VehicleSpec* spec = nullptr;
  size_t left = 0, right = 0, turret = 0, gun = 0, turretCoordinate = 0, gunCoordinate = 0;
  size_t hullBody = 0, turretBody = 0, gunBody = 0;
  Eigen::VectorXd pTarget, dTarget;
  double turretAim = 0.0, gunAim = -0.03, health = 0.0, lastShot = -100.0, born = 0.0;
  double strafe = 1.0, nextStrafeSwitch = 0.0, hurtTime = -100.0, upsideDown = 0.0;
  double tracksBrokenUntil = -1.0, gunDamagedUntil = -1.0, stuckTime = 0.0, reverseUntil = -1.0;
  int salvo = 0;
  // Whether water (and air) act on the tank; only near the river (Game::update()).
  bool wading = true;

  VehicleModel model;

  Tank(raisim::World& world, const std::string& urdf, const VehicleModel& shape, const VehicleSpec& kind,
       const Vec3& position, double yaw, double now)
      : spec(&kind), health(kind.health), born(now), model(shape) {
    system = world.addArticulatedSystem(urdf);
    system->setName(kind.name);
    left = system->getGeneralizedVelocityIndex("left_sprocket");
    right = system->getGeneralizedVelocityIndex("right_sprocket");
    turret = system->getGeneralizedVelocityIndex("turret");
    gun = system->getGeneralizedVelocityIndex("gun");
    turretCoordinate = system->getJoint("turret").getIdxInGeneralizedCoordinate();
    gunCoordinate = system->getJoint("gun").getIdxInGeneralizedCoordinate();
    hullBody = system->getBodyIdx("hull");
    turretBody = system->getBodyIdx("turret_link");
    gunBody = system->getBodyIdx("gun_link");
    // The turret and gun swing through trees instead of jamming against them.
    for (auto& body : system->getCollisionBodies())
      if (body.localIdx == turretBody || body.localIdx == gunBody) body.setCollisionMask(~kTreeCollisionGroup);
    Eigen::VectorXd gc = Eigen::VectorXd::Zero(system->getGeneralizedCoordinateDim());
    gc.head<3>() = position;
    gc[3] = std::cos(0.5 * yaw);
    gc[6] = std::sin(0.5 * yaw);
    system->setState(gc, Eigen::VectorXd::Zero(system->getDOF()));
    Eigen::VectorXd pGain = Eigen::VectorXd::Zero(system->getDOF()), dGain = pGain;
    dTarget = pGain;
    pTarget = Eigen::VectorXd::Zero(system->getGeneralizedCoordinateDim());
    const double mass = kind.mass / 12500.0;
    const double turretInertia = 0.24 * kind.mass * std::pow(shape.turretMax.x() - shape.turretMin.x(), 2) / 12.0 / 1800.0;
    const double gunInertia = 0.05 * kind.mass * std::pow(shape.muzzle - shape.gunPivot.x(), 2) / 3.0 / 600.0;
    dGain[left] = dGain[right] = 4000.0 * mass;
    pGain[turret] = 4e5 * turretInertia;
    dGain[turret] = 4e4 * turretInertia;
    pGain[gun] = 1e6 * gunInertia;
    dGain[gun] = 6e4 * gunInertia;
    system->setPdGains(pGain, dGain);
    command(0.0, 0.0, 1.0, now);
  }

  bool alive() const { return system && health > 0.0; }
  Vec3 position() const { return system->getGeneralizedCoordinate().e().head<3>(); }
  Vec3 velocity() const { return system->getGeneralizedVelocity().e().head<3>(); }
  double muzzleLength() const { return model.muzzle - model.gunPivot.x(); }
  double scale() const { return 0.5 * (model.hullMax.y() - model.hullMin.y()) / 1.8; }

  double heading() const {
    const auto& q = system->getGeneralizedCoordinate();
    return std::atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
  }

  Eigen::Matrix3d hullRotation() const {
    const auto& q = system->getGeneralizedCoordinate();
    return Eigen::Quaterniond(q[3], q[4], q[5], q[6]).toRotationMatrix();
  }

  void command(double throttle, double turn, double speedScale, double now) {
    const bool immobile = now < tracksBrokenUntil;
    const double speed = immobile ? 0.0 : spec->speed * speedScale * std::clamp(throttle, -1.0, 1.0);
    const double spin = immobile ? 0.0 : spec->turnRate * std::clamp(turn, -1.0, 1.0);
    const double radius = model.wheelRadius;
    dTarget[left] = (speed - spin) / radius;
    dTarget[right] = (speed + spin) / radius;
    pTarget[turretCoordinate] = spec->fixedTurret ? 0.0 : turretAim;
    pTarget[gunCoordinate] = gunAim;
    system->setPdTarget(pTarget, dTarget);
  }

  void muzzle(Vec3& position, Vec3& direction) const {
    raisim::Vec<3> p;
    raisim::Mat<3, 3> R;
    system->getLink("gun_link").getPose(p, R);
    direction = R.e().col(0);
    position = p.e() + direction * muzzleLength();
  }

  Vec3 turretPivot() const {
    raisim::Vec<3> p;
    raisim::Mat<3, 3> R;
    system->getLink("turret_link").getPose(p, R);
    return p.e() + Vec3(0, 0, 0.5 * (model.turretMax.z() - model.turretPivot.z()));
  }

  Vec3 gunPivot() const {
    raisim::Vec<3> p;
    raisim::Mat<3, 3> R;
    system->getLink("gun_link").getPose(p, R);
    return p.e();
  }

  double turretAngle() const { return system->getGeneralizedCoordinate()[turretCoordinate]; }

  // Points the gun at target for a shell of the given speed (on the flat arc) and returns the
  // remaining aiming error (rad) of the turret, or of the hull for a fixed casemate. A stabilized gun
  // (the player's) sets the direction in the hull's frame, where the turret turns about the hull's up
  // axis and the gun about the turret's side axis: on a slope, or while the hull pitches, it still
  // points where the shell must go. The enemies' guns take bearing and elevation as if the hull were
  // level.
  double aimAt(const Vec3& target, double shellSpeed, bool stabilized = false) {
    const Vec3 offset = target - (stabilized ? gunPivot() : turretPivot());
    const double distance = std::max(1.0, offset.head<2>().norm());
    const double v2 = shellSpeed * shellSpeed;
    const double root = v2 * v2 - kGravity * (kGravity * distance * distance + 2.0 * offset.z() * v2);
    const double elevation = root > 0.0 ? std::atan((v2 - std::sqrt(root)) / (kGravity * distance)) : 0.25 * kPi;
    const double bearing = std::atan2(offset.y(), offset.x());
    const Vec3 direction(std::cos(elevation) * std::cos(bearing), std::cos(elevation) * std::sin(bearing), std::sin(elevation));
    const Vec3 local = stabilized ? Vec3(hullRotation().transpose() * direction)
                                  : Vec3(Eigen::AngleAxisd(-heading(), Vec3::UnitZ()) * direction);
    const double azimuth = std::atan2(local.y(), local.x());
    turretAim = spec->fixedTurret ? 0.0 : azimuth;
    gunAim = std::clamp(-std::atan2(local.z(), local.head<2>().norm()), -0.35, 0.15);
    return std::abs(wrapAngle(azimuth - (spec->fixedTurret ? 0.0 : turretAngle())));
  }

  // A tank that a blast left on its side or roof for a while is set back on its tracks, facing the way
  // it faced.
  void selfRight(double dt) {
    const auto& q = system->getGeneralizedCoordinate();
    const double up = 1.0 - 2.0 * (q[4] * q[4] + q[5] * q[5]);
    upsideDown = up < 0.35 ? upsideDown + dt : 0.0;
    if (upsideDown < 1.5) return;
    Eigen::VectorXd gc = system->getGeneralizedCoordinate().e();
    const double yaw = heading();
    gc[2] += 1.5 * scale();
    gc.segment<4>(3) << std::cos(0.5 * yaw), 0.0, 0.0, std::sin(0.5 * yaw);
    system->setState(gc, Eigen::VectorXd::Zero(system->getDOF()));
    upsideDown = 0.0;
  }
};

// What a round did to the vehicle it hit.
struct HitResult {
  double damage = 0.0;
  const char* label = "";
  ImU32 color = IM_COL32(255, 255, 255, 255);
  bool penetrated = false, ricochet = false, tracks = false, gun = false, critical = false;
};

// Resolves a hit by the armor zone it strikes (front, side, rear or roof of the hull or turret), the
// angle at which it meets the plate and the round's remaining penetration.
inline HitResult resolveHit(const Tank& target, const raisim::ProjectileHit& hit, WeaponId weaponId, double power,
                            std::mt19937& random) {
  const WeaponSpec& round = weapon(weaponId);
  const VehicleSpec& armor = *target.spec;
  HitResult result;
  const Vec3 velocity = hit.velocity.e();
  const double speed = std::max(1.0, velocity.norm());
  const Vec3 direction = velocity / speed;
  Eigen::Matrix3d R = target.hullRotation();
  const bool turretHit = hit.localIdx == target.turretBody || hit.localIdx == target.gunBody;
  if (turretHit) R = R * Eigen::AngleAxisd(target.turretAngle(), Vec3::UnitZ()).toRotationMatrix();
  const Vec3 incoming = R.transpose() * -direction;
  double plate;
  const char* zone;
  const double azimuth = std::abs(std::atan2(incoming.y(), incoming.x()));
  if (incoming.z() > 0.65) {
    plate = armor.topArmor;
    zone = "roof";
  } else if (azimuth < 0.25 * kPi) {
    plate = armor.frontArmor * (turretHit ? 0.9 : 1.0);
    zone = "front";
  } else if (azimuth > 0.75 * kPi) {
    plate = armor.rearArmor;
    zone = "rear";
  } else {
    plate = armor.sideArmor;
    zone = "side";
  }
  const double cosine = std::clamp(std::abs(direction.dot(hit.normal.e())), 0.0, 1.0);
  if (round.kinetic && cosine < 0.34 && uniform(random, 0.0, 1.0) < 0.85) {
    result.ricochet = true;
    result.label = "RICOCHET";
    result.color = IM_COL32(170, 200, 255, 255);
    return result;
  }
  const double effective = plate / std::max(cosine, 0.2);
  double penetration = round.penetration * power;
  if (round.kinetic) penetration *= std::pow(std::min(1.0, speed / round.speed), 1.4);
  if (penetration < effective) {
    result.damage = round.damage * power * (round.kinetic ? 0.06 : 0.3);
    result.label = "NO PENETRATION";
    result.color = IM_COL32(200, 200, 200, 255);
    return result;
  }
  result.penetrated = true;
  result.damage = round.damage * power * uniform(random, 0.85, 1.15);
  result.label = "PENETRATION";
  result.color = IM_COL32(255, 170, 60, 255);
  const Vec3 local = target.hullRotation().transpose() * (hit.position.e() - target.position());
  if (!turretHit && std::string(zone) == "side" && local.z() < 0.3 * target.scale()) {
    result.tracks = true;
    result.label = "TRACKS HIT";
  }
  if (hit.localIdx == target.gunBody) {
    result.gun = true;
    result.label = "GUN DAMAGED";
  }
  if (std::string(zone) == "rear" && uniform(random, 0.0, 1.0) < 0.2) {
    result.critical = true;
    result.damage *= armor.role == Role::Boss ? 1.6 : 4.0;
    result.label = "AMMO RACK HIT";
    result.color = IM_COL32(255, 70, 50, 255);
  }
  return result;
}

}  // namespace tank
