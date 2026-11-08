// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// Checks of Tank Survival's game logic and vehicles that need no window.

#include "hud.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* what) {
  if (condition) return;
  std::fprintf(stderr, "FAILED: %s\n", what);
  ++failures;
}

void arrowsPointAtEnemiesOutOfView() {
  // A camera at the origin looking along +x with z up, so that its right is -y, on a 1600 x 900 screen.
  const ImVec2 size(1600.0f, 900.0f);
  const glm::mat4 viewProjection = glm::perspective(glm::radians(50.0f), size.x / size.y, 0.1f, 1000.0f) *
                                   glm::lookAt(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  const auto toward = [&](const glm::vec3& p) { return tank::offscreenDirection(viewProjection * glm::vec4(p, 1.0f), size); };
  check(toward({-10.0f, -10.0f, 0.0f}).x > 0.99f, "an enemy behind and to the right gets an arrow on the right");
  check(toward({-10.0f, 10.0f, 0.0f}).x < -0.99f, "an enemy behind and to the left gets an arrow on the left");
  check(toward({-10.0f, 0.0f, -5.0f}).y > 0.99f, "an enemy behind and below gets an arrow at the bottom");
  check(toward({10.0f, -40.0f, 0.0f}).x > 0.99f, "an enemy ahead, far to the right, gets an arrow on the right");
  // As far out of view to the right as up in clip space: the arrow follows its bearing on the wide screen.
  const glm::vec2 diagonal = tank::offscreenDirection(glm::vec4(2.0f, 2.0f, 0.5f, 1.0f), size);
  check(std::abs(std::atan2(-diagonal.y, diagonal.x) - std::atan2(size.y, size.x)) < 1e-4f,
        "a diagonal arrow keeps the screen's aspect ratio");
  check(tank::offscreenDirection(glm::vec4(0.0f, 0.0f, 1.0f, -1.0f), size) == glm::vec2(0.0f),
        "an enemy straight behind gets no direction");
}

void burstsOfHitsGatherPerVehicle() {
  std::vector<tank::Notice> notes;
  const int first = 0, second = 0;
  tank::gatherHit(notes, &first, "MG", 0, tank::Vec3(0, 0, 0), 1.0, 4.0);
  tank::gatherHit(notes, &second, "MG", 0, tank::Vec3(2, 0, 0), 1.1, 4.0);
  check(notes.size() == 2 && notes[0].damage == 4.0 && notes[1].damage == 4.0,
        "hits on two vehicles side by side keep notes of their own");
  tank::gatherHit(notes, &first, "MG", 0, tank::Vec3(0.5, 0, 0), 1.3, 4.0);
  check(notes.size() == 2 && notes[0].damage == 8.0 && notes[0].text == "MG  -8", "a burst on one vehicle gathers into one note");
  tank::gatherHit(notes, &first, "PENETRATION", 0, tank::Vec3(0.5, 0, 0), 1.4, 30.0);
  check(notes.size() == 3, "a hit of another kind gets a note of its own");
  tank::gatherHit(notes, &first, "MG", 0, tank::Vec3(0.5, 0, 0), 2.0, 4.0);
  check(notes.size() == 4, "a hit more than half a second after the last starts a new note");
  tank::gatherHit(notes, &first, "MG", 0, tank::Vec3(8.0, 0, 0), 2.1, 4.0);
  check(notes.size() == 5, "a hit far from the last starts a new note");
  tank::forgetTarget(notes, &first);
  tank::gatherHit(notes, &first, "MG", 0, tank::Vec3(8.0, 0, 0), 2.2, 4.0);
  check(notes.size() == 6 && notes[4].damage == 4.0, "a vehicle made at the address of one that is gone starts a new note");
}

void qualityStepsDownWhileFramesAreSlow() {
  tank::QualityTuner tuner;
  int quality = 2, frames = 0;
  while (quality == 2 && frames < 1000) {
    quality = tuner.update(quality, 0.05);
    ++frames;
  }
  check(quality == 1 && frames == 90, "slow frames lower the quality once the load has settled and 30 are timed");
  while (quality > 0 && frames < 1000) {
    quality = tuner.update(quality, 0.05);
    ++frames;
  }
  check(quality == 0 && tuner.update(quality, 0.05) == 0, "the quality stops at Fast");

  tank::QualityTuner fast;
  quality = 2;
  for (int i = 0; i < 200; ++i) quality = fast.update(quality, i % 4 == 0 ? 0.2 : 0.016);
  check(quality == 2 && !fast.active, "a few slow frames among fast ones keep the quality and end the tuning");

  tank::QualityTuner interrupted;
  quality = 2;
  for (int i = 0; i < 89; ++i) quality = interrupted.update(quality, 0.05);
  interrupted.restart();
  quality = interrupted.update(quality, 0.05);
  check(quality == 2 && interrupted.frames.empty(), "frames timed before other frames are not counted after them");

  tank::QualityTuner chosen;
  chosen.active = false;
  quality = 3;
  for (int i = 0; i < 200; ++i) quality = chosen.update(quality, 0.05);
  check(quality == 3, "a quality the player chose is kept");
}

void windArrowTurnsWithTheCamera() {
  // Looking along +x (heading 0), a wind blowing along +x points up on the radar and one along +y to the left.
  const glm::vec2 ahead = tank::radarDirection(0.0, tank::Vec3(5, 0, 0)), left = tank::radarDirection(0.0, tank::Vec3(0, 5, 0));
  check(std::abs(ahead.x) < 1e-5f && ahead.y < -4.99f, "a wind along the view points up");
  check(left.x < -4.99f && std::abs(left.y) < 1e-5f, "a wind from the right points left");
  const glm::vec2 turned = tank::radarDirection(0.5 * tank::kPi, tank::Vec3(0, 5, 0));
  check(std::abs(turned.x) < 1e-5f && turned.y < -4.99f, "turning the camera turns the arrow with it");
}

void restartNeedsAHeldKey() {
  tank::HoldToConfirm key;
  bool acted = false;
  for (int i = 0; i < 6; ++i) acted = key.update(true, 0.1) || acted;
  acted = key.update(false, 0.1) || acted;
  check(!acted, "a tap on the key does nothing");
  int times = 0;
  for (int i = 0; i < 30; ++i) times += key.update(true, 0.1);
  check(times == 1, "holding the key acts once, however long it is held");
  key.update(false, 0.1);
  for (int i = 0; i < 9; ++i) times += key.update(true, 0.1);
  check(times == 2, "after the key is let go, holding it acts again");
}

void riverBanksCanBeClimbed() {
  // Across each river, from its middle to beyond the banks, the ground rises no steeper than the banks of
  // the rivers tanks are seen to drive out of (a tank in the town's steep canal could not).
  for (const tank::Terrain terrain : {tank::Terrain::Hills, tank::Terrain::Town, tank::Terrain::Ridges, tank::Terrain::Fortress}) {
    const tank::River river = tank::riverOf(terrain);
    double steepest = 0.0;
    for (double x = -60.0; x <= 60.0; x += 7.5)
      for (const double side : {-1.0, 1.0})
        for (double d = 0.0; d < tank::riverHalfWidth(terrain) + 3.0; d += 0.25) {
          const double y = tank::riverY(river, x) + side * d, step = side * 0.25;
          steepest = std::max(steepest, std::abs(tank::groundShape(terrain, x, y + step) - tank::groundShape(terrain, x, y)) / 0.25);
        }
    check(steepest < 0.6, "a tank can climb out of every river");
  }
}

void cameraStaysWhereItIsTurned() {
  tank::OrbitHeading orbit;
  orbit.place(0.3);
  orbit.drag(100.0);
  const double turned = orbit.heading;
  check(std::abs(turned - (0.3 - 0.5)) < 1e-12, "a drag turns the camera at once, as far as the mouse moves");
  for (int frame = 0; frame < 120; ++frame) orbit.update(0.02 * frame, 1.0 / 60.0, false);
  check(orbit.heading == turned, "the camera stays where the drag left it while the hull turns");
  orbit.behindHull = true;
  for (int frame = 0; frame < 40; ++frame) orbit.update(2.0, 1.0 / 60.0, false);
  check(std::abs(orbit.heading - 2.0) < 0.002 && !orbit.behindHull, "asked to, the camera swings behind the hull within a second");
  orbit.update(2.5, 1.0 / 60.0, false);
  check(std::abs(orbit.heading - 2.0) < 0.002, "once behind the hull, the camera stays there when the hull turns");
  orbit.update(2.5, 1.0 / 60.0, true);
  check(orbit.heading > 2.01, "the autopilot's camera follows the hull");
}

void mouseLooksAround() {
  tank::OrbitHeading orbit;
  orbit.place(0.0);
  orbit.look(100.0, 0.0, 1.0);
  check(std::abs(orbit.heading + 100.0 * tank::OrbitHeading::kSensitivity) < 1e-12, "the mouse moved right turns the view right");
  orbit.look(100.0, 0.0, 0.25);
  check(std::abs(orbit.heading + 125.0 * tank::OrbitHeading::kSensitivity) < 1e-12, "the sniper sight turns slower");
  orbit.look(0.0, -1e5, 1.0);
  check(orbit.pitch == tank::OrbitHeading::kHighest, "the view looks up no higher than its limit");
  orbit.look(0.0, 1e5, 1.0);
  check(orbit.pitch == tank::OrbitHeading::kLowest, "the view looks down no lower than its limit");
  orbit.behindHull = true;
  orbit.look(0.0, 10.0, 1.0);
  check(orbit.behindHull, "looking up and down keeps the camera swinging behind the hull");
  orbit.look(5.0, 0.0, 1.0);
  check(!orbit.behindHull, "turning the view stops the swing behind the hull");
  for (int turn = 0; turn < 1000; ++turn) orbit.look(500.0, 0.0, 1.0);
  check(std::abs(orbit.heading) <= tank::kPi, "the heading stays within a turn");
  orbit.place(1.0);
  check(orbit.heading == 1.0 && orbit.pitch == tank::OrbitHeading::kPitch, "a new mission looks from behind the hull");
}

void aimingDownTheSightZooms() {
  tank::AimDownSights sights;
  sights.update(false, 3.0);
  check(!sights.held && sights.magnification() == 1.0, "without the right button the view is not magnified");
  sights.update(true, 0.0);
  check(sights.magnification() == 3.0, "holding the right button magnifies three times");
  sights.update(true, 0.5);
  sights.update(true, 0.4);
  check(sights.magnification() == 3.0, "part of a notch does not change the magnification");
  sights.update(true, 0.2);
  check(sights.magnification() == 5.0, "a notch in steps it up");
  sights.update(true, 5.0);
  check(sights.magnification() == 8.0, "it stops at eight times");
  sights.update(true, -0.5);
  sights.update(true, -0.6);
  check(sights.magnification() == 5.0, "turned out it steps down");
  sights.update(false, 0.0);
  sights.update(true, 0.0);
  check(sights.magnification() == 5.0, "aiming again keeps the last magnification");
  sights.update(true, -9.0);
  check(sights.magnification() == 2.0, "it stops at two times");
}

// A round fired from the muzzle along launchDirection() passes through the point under the sight, as the
// world flies it: gravity, the air's drag and the wind.
void roundsReachThePointUnderTheSight() {
  for (const tank::WeaponId id : {tank::WeaponId::AP, tank::WeaponId::HE, tank::WeaponId::MG}) {
    const tank::WeaponSpec& round = tank::weapon(id);
    for (const tank::Vec3 target : {tank::Vec3(30.0, 4.0, 1.0), tank::Vec3(150.0, -40.0, 6.0), tank::Vec3(-280.0, 260.0, -3.0)}) {
      raisim::World world;
      world.setTimeStep(tank::kTimeStep);
      raisim::FluidMedium air = raisim::FluidMedium::air();
      const tank::Vec3 wind(-6.0, 5.0, 0.0);
      air.velocity = tank::vec(wind);
      world.setAmbientFluid(air);
      const tank::Vec3 muzzle(0.0, 0.0, 2.0);
      const tank::Vec3 direction = tank::launchDirection(muzzle, target, round, wind);
      raisim::ProjectileProperties properties;
      properties.mass = round.mass;
      properties.radius = round.radius;
      properties.dragCoefficient = round.dragCoefficient;
      world.addProjectile(tank::vec(muzzle), tank::vec(direction * round.speed), properties);
      const tank::Vec3 axis = (target - muzzle).normalized();
      const double range = (target - muzzle).norm();
      double miss = 1e9;
      tank::Vec3 last = muzzle;
      for (int step = 0; step < 5000 && !world.getProjectiles().empty(); ++step) {
        world.integrate();
        const tank::Vec3 now = world.getProjectiles().front().position.e();
        const double before = (last - muzzle).dot(axis), after = (now - muzzle).dot(axis);
        if (after >= range) {
          const tank::Vec3 passed = last + (range - before) / (after - before) * (now - last);
          miss = (passed - target).norm();
          break;
        }
        last = now;
      }
      check(miss < 0.02, "a round passes within 2 cm of the point under the sight");
    }
  }
}

void sightPicksTheNearestEnemy() {
  const std::vector<ImVec2> points{{100.0f, 100.0f}, {130.0f, 100.0f}, {300.0f, 300.0f}};
  check(tank::nearestWithin(points, ImVec2(120.0f, 100.0f), 40.0f) == 1, "the sight picks the enemy nearest it");
  check(tank::nearestWithin(points, ImVec2(200.0f, 200.0f), 40.0f) == -1, "no enemy within reach, none is picked");
  check(tank::nearestWithin({}, ImVec2(0.0f, 0.0f), 40.0f) == -1, "no enemies, none is picked");
}

// A shot kicks the view up and it settles back where it was. At the top of the view's range the kick turns
// nothing, and settling pulled the view down by the whole kick, shot after shot.
void recoilSettlesWhereTheViewWas() {
  for (const double start : {0.0, tank::OrbitHeading::kHighest}) {
    tank::OrbitHeading orbit;
    orbit.pitch = start;
    tank::Recoil recoil;
    recoil.kick(orbit, 0.03);
    check(start == tank::OrbitHeading::kHighest || std::abs(orbit.pitch - (start + 0.03)) < 1e-9, "a shot kicks the view up");
    for (int shot = 0; shot < 5; ++shot) {
      for (int frame = 0; frame < 6; ++frame) recoil.settle(orbit, 1.0 / 60.0);
      recoil.kick(orbit, 0.03);
    }
    for (int frame = 0; frame < 240; ++frame) recoil.settle(orbit, 1.0 / 60.0);
    check(std::abs(orbit.pitch - start) < 1e-6, "the view settles back where it was after a burst of shots");
  }
}

// The player's gun fires where the crosshair is. Just after the view turns away, a round would leave along
// the barrel, far off the crosshair, so the gun is not ready; it is once the turret has swung round. A point
// below the gun's depression is never on a round's path, and the gun is ready once it has turned as far as
// it can.
void gunFiresOnceItCanReachTheSight() {
  const tank::VehicleSpec& spec = tank::vehicle(tank::Role::Player);
  const std::string folder = std::string(TANK_SURVIVAL_ASSETS) + "/tank_survival/vehicles/" + spec.model;
  const tank::VehicleModel model = tank::loadVehicleModel(folder);
  raisim::World world;
  world.setTimeStep(tank::kTimeStep);
  world.addGround();
  tank::Tank vehicle(world, tank::vehicleUrdf(spec, model, folder), model, spec,
                     tank::Vec3(0.0, 0.0, model.axleHeight + 0.3), 0.0, 0.0);
  const tank::Vec3 wind(3.0, -2.0, 0.0);
  const double speed = tank::weapon(tank::WeaponId::AP).speed;
  // Aimed every frame, as the game does: the turret's swing turns the hull a little.
  const auto settle = [&](const tank::Vec3& target, double seconds) {
    for (int i = 0; i < int(seconds / tank::kTimeStep); ++i) {
      if (i % 8 == 0) {
        vehicle.aimAt(target, speed, true);
        vehicle.command(0.0, 0.0, 1.0, world.getWorldTime());
      }
      world.integrate();
    }
  };
  settle(tank::Vec3(300.0, 0.0, 2.0), 2.0);
  check(tank::Game::gunReadyFor(vehicle, tank::WeaponId::AP, tank::Vec3(300.0, 0.0, 2.0), wind),
        "the gun is ready on the point it has turned to");
  const tank::Vec3 side = vehicle.position() + tank::Vec3(0.0, 300.0, 2.0);
  vehicle.aimAt(side, speed, true);
  check(!tank::Game::gunReadyFor(vehicle, tank::WeaponId::AP, side, wind) &&
            !tank::Game::gunReadyFor(vehicle, tank::WeaponId::MG, side, wind),
        "the gun is not ready on a point it has yet to turn to");
  check(tank::Game::gunReadyFor(vehicle, tank::WeaponId::Missile, side, wind), "missiles do not wait for the gun");
  settle(side, 1.0);
  check(tank::Game::gunReadyFor(vehicle, tank::WeaponId::AP, side, wind), "the gun is ready once the turret swung round");
  tank::Vec3 muzzle, barrel;
  vehicle.muzzle(muzzle, barrel);
  const tank::Vec3 below = vehicle.gunPivot() + 4.0 * tank::Vec3(barrel.x(), barrel.y(), 0.0).normalized() -
                           tank::Vec3(0.0, 0.0, 2.0);
  vehicle.aimAt(below, speed, true);
  bool onTarget = true;
  tank::Vec3 origin;
  tank::Game::roundPath(vehicle, tank::WeaponId::AP, &below, wind, origin, &onTarget);
  check(!onTarget, "a point below the gun's depression is off the barrel");
  settle(below, 1.0);
  check(tank::Game::gunReadyFor(vehicle, tank::WeaponId::AP, below, wind),
        "the gun is ready below its depression once it has turned as far as it can");
}

// The turret swings to a new angle and stops there. Its gains came from a nominal inertia, far below the
// turret's and gun's: it overshot a quarter turn by 0.14 rad (the player's tank) to 0.41 rad (the BMD-2) and
// swung back and forth for most of a second after each turn of the view.
void turretSettlesWithoutRinging() {
  for (const tank::Role role : {tank::Role::Player, tank::Role::Medium, tank::Role::Scout}) {
    const tank::VehicleSpec& spec = tank::vehicle(role);
    const std::string folder = std::string(TANK_SURVIVAL_ASSETS) + "/tank_survival/vehicles/" + spec.model;
    const tank::VehicleModel model = tank::loadVehicleModel(folder);
    raisim::World world;
    world.setTimeStep(tank::kTimeStep);
    world.addGround();
    tank::Tank vehicle(world, tank::vehicleUrdf(spec, model, folder), model, spec,
                       tank::Vec3(0.0, 0.0, model.axleHeight + 0.3), 0.0, 0.0);
    // The turret's angle on the hull, commanded `to` and held for `seconds`.
    const auto swing = [&](double to, double seconds, double& overshoot, double& last) {
      const double from = vehicle.turretAngle();
      vehicle.turretAim = to;
      vehicle.command(0.0, 0.0, 1.0, world.getWorldTime());
      overshoot = 0.0;
      for (int i = 0; i < int(seconds / tank::kTimeStep); ++i) {
        world.integrate();
        const double error = tank::wrapAngle(vehicle.turretAngle() - to);
        if (error * (from - to) < 0.0) overshoot = std::max(overshoot, std::abs(error));
        last = std::abs(error);
      }
    };
    double overshoot = 0.0, last = 0.0;
    swing(0.0, 1.0, overshoot, last);
    swing(0.5 * tank::kPi, 1.5, overshoot, last);
    check(overshoot < 0.02 && last < 0.01, "the turret turns a quarter turn without overshooting and stops there");
    swing(0.5 * tank::kPi + 0.15, 1.0, overshoot, last);
    check(overshoot < 0.005 && last < 0.005, "the turret follows a small turn of the view without overshooting");
  }
}

// Each enemy, driven as the game drives it (its share of top speed, steering held under the sideways
// acceleration of a turn at speed), over rough ground and steered hard from side to side, stays on its
// tracks. At full speed and full steer, the BMD-2 rolled over in five runs of six.
void enemiesStayOnTheirTracks() {
  for (const tank::Role role : {tank::Role::Scout, tank::Role::Medium, tank::Role::Heavy, tank::Role::Destroyer,
                                tank::Role::Artillery, tank::Role::Boss}) {
    const tank::VehicleSpec& spec = tank::vehicle(role);
    const std::string folder = std::string(TANK_SURVIVAL_ASSETS) + "/tank_survival/vehicles/" + spec.model;
    const tank::VehicleModel model = tank::loadVehicleModel(folder);
    const std::string urdf = tank::vehicleUrdf(spec, model, folder);
    int rolls = 0;
    for (const tank::Terrain terrain : {tank::Terrain::Hills, tank::Terrain::Ridges}) {
      constexpr size_t kSamples = 161;
      std::vector<double> heights(kSamples * kSamples);
      for (size_t j = 0; j < kSamples; ++j)
        for (size_t i = 0; i < kSamples; ++i)
          heights[j * kSamples + i] = tank::roughness(terrain, -80.0 + double(i), -80.0 + double(j));
      for (int run = 0; run < 6; ++run) {
        raisim::World world;
        world.setTimeStep(tank::kTimeStep);
        world.addHeightMap(kSamples, kSamples, 160.0, 160.0, 0.0, 0.0, heights);
        const tank::Vec3 start(-40.0 + 12.0 * run, -30.0 + 9.0 * run, 0.0);
        tank::Tank vehicle(world, urdf, model, spec,
                           start + tank::Vec3(0, 0, tank::roughness(terrain, start.x(), start.y()) + model.axleHeight + 0.4),
                           0.7 * run, 0.0);
        bool rolled = false;
        // Until it has driven 12 s or reached the edge of the ground.
        for (int i = 0; i < int(12.0 / tank::kTimeStep) && !rolled &&
                        vehicle.position().head<2>().cwiseAbs().maxCoeff() < 70.0; ++i) {
          if (i % 8 == 0) {
            const double steer = vehicle.steerLimit(tank::kEnemyTurnAcceleration);
            vehicle.command(1.0, std::clamp(std::sin(0.9 * i * tank::kTimeStep), -steer, steer), tank::kEnemySpeedScale,
                            world.getWorldTime());
          }
          world.integrate();
          rolled = vehicle.hullRotation()(2, 2) < 0.35;
        }
        rolls += rolled;
      }
    }
    check(rolls == 0, "an enemy steered hard over rough ground stays on its tracks");
  }
}

// Each vehicle on rough ground at full throttle, then steered: a rigid hull bounced from bump to bump with
// its tracks mostly in the air and went on straight (the player's tank turned 5 degrees in 3 s); on sprung
// track carriers it turns.
void tanksTurnWhileDriving() {
  for (const auto& spec : tank::kVehicles) {
    const std::string folder = std::string(TANK_SURVIVAL_ASSETS) + "/tank_survival/vehicles/" + spec.model;
    const tank::VehicleModel model = tank::loadVehicleModel(folder);
    const std::string urdf = tank::vehicleUrdf(spec, model, folder);
    for (const double turn : {1.0, -1.0}) {
      raisim::World world;
      world.setTimeStep(tank::kTimeStep);
      constexpr size_t kSamples = 161;
      std::vector<double> heights(kSamples * kSamples);
      for (size_t j = 0; j < kSamples; ++j)
        for (size_t i = 0; i < kSamples; ++i)
          heights[j * kSamples + i] = tank::roughness(tank::Terrain::Hills, -80.0 + double(i), -80.0 + double(j));
      world.addHeightMap(kSamples, kSamples, 160.0, 160.0, 0.0, 0.0, heights);
      const tank::Vec3 start(-30.0, 0.0, tank::roughness(tank::Terrain::Hills, -30.0, 0.0) + model.axleHeight + 0.3);
      tank::Tank vehicle(world, urdf, model, spec, start, 0.0, 0.0);
      const auto drive = [&](double throttle, double steer, double seconds) {
        double turned = 0.0, last = vehicle.heading();
        for (int i = 0; i < int(seconds / tank::kTimeStep); ++i) {
          if (i % 8 == 0) vehicle.command(throttle, steer, 1.0, world.getWorldTime());
          world.integrate();
          turned += tank::wrapAngle(vehicle.heading() - last);
          last = vehicle.heading();
        }
        return turned;
      };
      drive(1.0, 0.0, 2.0);
      check(turn * drive(1.0, turn, 3.0) > 0.5, "a vehicle driving over rough ground turns when steered");
    }
  }
}

}  // namespace

int main() {
  arrowsPointAtEnemiesOutOfView();
  burstsOfHitsGatherPerVehicle();
  qualityStepsDownWhileFramesAreSlow();
  riverBanksCanBeClimbed();
  windArrowTurnsWithTheCamera();
  restartNeedsAHeldKey();
  cameraStaysWhereItIsTurned();
  mouseLooksAround();
  aimingDownTheSightZooms();
  roundsReachThePointUnderTheSight();
  sightPicksTheNearestEnemy();
  recoilSettlesWhereTheViewWas();
  gunFiresOnceItCanReachTheSight();
  turretSettlesWithoutRinging();
  enemiesStayOnTheirTracks();
  tanksTurnWhileDriving();
  if (failures == 0) std::printf("tank_survival_test: all checks passed\n");
  return failures == 0 ? 0 : 1;
}
