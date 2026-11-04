// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#pragma once

#include "buildings.hpp"
#include "scene.hpp"

namespace tank {

enum class Bonus { Ammo, Repair, Reactive, Overdrive, Missiles, Radio };

// A supply: its label, the color of its beacon, how often it drops, and the prop it is drawn with.
struct BonusStyle {
  const char* label;
  glm::vec4 color;
  double weight;
  const char* model;
  double scale;
};

inline const std::array<BonusStyle, 6> kBonusStyles{{
    {"AMMO", {1.0f, 0.78f, 0.18f, 1.0f}, 0.33, "ammo_box", 3.6},
    {"REPAIR", {0.25f, 0.95f, 0.35f, 1.0f}, 0.24, "old_military_crate", 0.9},
    {"REACTIVE ARMOR", {0.75f, 0.45f, 1.0f, 1.0f}, 0.12, "old_military_crate", 0.9},
    {"OVERDRIVE", {0.25f, 0.65f, 1.0f, 1.0f}, 0.11, "metal_jerrycan_green", 1.7},
    {"MISSILES", {0.4f, 0.9f, 1.0f, 1.0f}, 0.12, "wooden_military_crate", 1.1},
    {"ARTILLERY RADIO", {0.55f, 1.0f, 0.45f, 1.0f}, 0.08, "ammo_box", 3.6},
}};

enum class Phase { Menu, Playing, Victory, Defeat };

struct Pickup {
  Bonus type;
  Vec3 position;
  double expires = 0.0;
  std::string name;
  std::shared_ptr<raisin::Visuals> crate, beam;
};

struct Barrel {
  raisim::Cylinder* body = nullptr;
  double fuse = -1.0;
};

// A destroyed vehicle: its hull and turret as loose bodies, drawn burnt with the vehicle's meshes.
struct Wreck {
  std::vector<raisim::SingleBodyObject*> parts;
  std::vector<std::string> visuals;
  std::vector<Vec3> offsets;
  double removeAt = 0.0;
};

struct Shell {
  size_t id;
  bool fromPlayer;
  WeaponId weapon;
  double power;
};

struct Missile {
  raisim::Cylinder* body = nullptr;
  size_t shooter = size_t(-1);
  double launched = 0.0, nextPuff = 0.0;
};

struct Strike {
  double time;
  Vec3 origin, velocity;
};

struct Marker {
  std::string name;
  double until;
};

struct PendingBlast {
  double time;
  Vec3 position;
  double charge;
};

struct SmokeZone {
  Vec3 center;
  double until;
};

struct Notice {
  std::string text;
  ImU32 color;
  double time;
  Vec3 position = Vec3::Zero();
};

struct Building {
  Vec3 center, half;
};

struct Stats {
  int shots = 0, hits = 0, penetrations = 0, kills = 0;
  double damageTaken = 0.0;
};

class Game {
 public:
  Game(std::shared_ptr<raisim::World> world, Viewer& viewer, const std::string& assets, unsigned seed)
      : world_(std::move(world)), viewer_(viewer), random_(seed), assets_(assets) {
    for (const auto& spec : kVehicles) {
      const std::string folder = assets + "/tank_survival/vehicles/" + spec.model;
      models_.push_back(loadVehicleModel(folder));
      urdfs_.push_back(vehicleUrdf(spec, models_.back(), folder));
    }
  }

  // Reports how far loading has come (0 to 1) and what it is doing.
  using Progress = std::function<void(double, const char*)>;

  // Builds the arena of a level. Playing starts the waves; otherwise the arena is a quiet backdrop for
  // the menu. progress, if given, hears of each stage.
  void load(int level, bool playing, int quality, const Progress& progress = {}) {
    const auto report = [&](double fraction, const char* what) {
      if (progress) progress(fraction, what);
    };
    report(0.0, "Clearing the battlefield");
    clear();
    level_ = level;
    const LevelSpec& spec = levels()[size_t(level)];
    applyTheme(spec, quality);
    buildArena(spec, report);
    phase_ = playing ? Phase::Playing : Phase::Menu;
    player_ = std::make_unique<Tank>(*world_, urdfs_[0], models_[0], vehicle(Role::Player),
                                     Vec3(spec.playerStart.x(), spec.playerStart.y(),
                                          height(spec.playerStart.x(), spec.playerStart.y()) + models_[0].axleHeight + 0.3),
                                     spec.playerStart.y() < -1.0 ? 0.5 * kPi : 0.0, now());
    ammo_ = {36, 18, 600, 4, 1};
    smokeCharges_ = 3;
    selected_ = WeaponId::AP;
    reactive_ = 0;
    overdriveUntil_ = -1.0;
    for (auto& t : lastFire_) t = -100.0;
    artilleryRecharge_ = now() + 60.0;
    score_ = 0;
    stats_ = Stats();
    wave_ = 0;
    waveCleared_ = false;
    queue_.clear();
    reinforced_ = false;
    nextSpawn_ = now() + 2.0;
    nextWave_ = now() + 3.0;
    nextPickup_ = now() + 4.0;
    started_ = now();
    endedAt_ = -1.0;
    if (playing) {
      notify(std::string(spec.name).append(" - ").append(spec.subtitle), IM_COL32(255, 230, 160, 255));
      for (int i = 0; i < 3; ++i) spawnPickup(i == 0 ? Bonus::Ammo : randomBonus());
    }
  }

  double now() const { return world_->getWorldTime(); }
  Phase phase() const { return phase_; }
  int level() const { return level_; }
  const LevelSpec& levelSpec() const { return levels()[size_t(level_)]; }
  Tank* player() { return phase_ == Phase::Playing && player_ && player_->alive() ? player_.get() : nullptr; }
  const Tank* playerTank() const { return player_.get(); }

  void updateScenery(const glm::vec3& camera) { scenery_.updateGrass(camera); }
  size_t treesFelled() const { return scenery_.treesFelled(); }

  // Where a round of the weapon leaves the tank, and its direction: the machine gun sits beside the cannon.
  static void roundOrigin(const Tank& shooter, WeaponId id, Vec3& position, Vec3& direction) {
    shooter.muzzle(position, direction);
    if (id == WeaponId::MG) {
      const Vec3 side = direction.cross(Vec3::UnitZ()).normalized();
      position += -direction * (0.7 * shooter.muzzleLength()) + side * (0.35 * shooter.scale());
    }
  }

  // The first thing a round flying straight from origin along direction meets within length, as the
  // world's projectiles do (the player's own tank aside): the point, and whether it is a vehicle.
  bool sweep(const Vec3& origin, const Vec3& direction, double length, Vec3& point, bool& vehicle) {
    Vec3 from = origin;
    for (int attempt = 0; attempt < 4 && length > 0.0; ++attempt) {
      const auto& hits = world_->rayTest(from, direction, length);
      if (hits.size() == 0) return false;
      const Vec3 at = hits[0].getPosition();
      if (player_ && player_->system && hits[0].getObject() == player_->system) {
        const double skip = (at - from).norm() + 0.05;
        from += skip * direction;
        length -= skip;
        continue;
      }
      point = at;
      vehicle = hits[0].getObject() && tankAt(hits[0].getObject()->getIndexInWorld()) != nullptr;
      return true;
    }
    return false;
  }

  double height(double x, double y) const {
    if (terrain_ && std::abs(x) < 0.5 * kExtent && std::abs(y) < 0.5 * kExtent) return terrain_->getHeight(x, y);
    return groundShape(levelSpec().terrain, x, y);
  }

  // Intersection of a ray with the ground (or a building), or the point maxRange along it.
  Vec3 groundHit(const Vec3& origin, const Vec3& direction, double maxRange = 260.0) const {
    const auto solid = [&](const Vec3& p) {
      if (p.z() < height(p.x(), p.y())) return true;
      for (const auto& b : buildings_)
        if (((p - b.center).cwiseAbs().array() < b.half.array()).all()) return true;
      return false;
    };
    double previous = 0.0;
    for (double t = 0.5; t < maxRange; t += 0.5) {
      if (solid(origin + t * direction)) {
        double a = previous, b = t;
        for (int i = 0; i < 20; ++i) {
          const double m = 0.5 * (a + b);
          (solid(origin + m * direction) ? b : a) = m;
        }
        return origin + b * direction;
      }
      previous = t;
    }
    return origin + maxRange * direction;
  }

  void setAim(const Vec3& point) { aim_ = point; }
  const Vec3& aim() const { return aim_; }
  WeaponId selected() const { return selected_; }
  void select(WeaponId id) {
    if (selected_ != id) notify(std::string(weapon(id).name), rgba(weapon(id).color));
    selected_ = id;
  }
  void cycle(int step) { select(WeaponId((int(selected_) + step + kWeaponCount) % kWeaponCount)); }
  int ammo(WeaponId id) const { return ammo_[size_t(id)]; }
  int smokeCharges() const { return smokeCharges_; }

  // Fraction of the weapon's reload or cooldown that has passed.
  double ready(WeaponId id) const {
    const double reload = reloadTime(id);
    return std::clamp((now() - lastFire_[slot(id)]) / reload, 0.0, 1.0);
  }

  bool fire() {
    Tank* hero = player();
    if (!hero) return false;
    const WeaponId id = selected_;
    if (ready(id) < 1.0) return false;
    if (ammo_[size_t(id)] <= 0) {
      if (now() - lastEmpty_ > 1.0) notify(std::string("NO ") + weapon(id).shortName + " LEFT", IM_COL32(255, 90, 70, 255));
      lastEmpty_ = now();
      return false;
    }
    switch (id) {
      case WeaponId::AP:
      case WeaponId::HE:
        launchShell(*hero, id, true, 1.0, 0.0006);
        shake_ = std::max(shake_, 0.35);
        break;
      case WeaponId::MG:
        launchShell(*hero, id, true, 1.0, 0.004);
        break;
      case WeaponId::Missile:
        launchMissile(*hero);
        break;
      case WeaponId::Artillery:
        callArtillery(aim_);
        break;
    }
    --ammo_[size_t(id)];
    lastFire_[slot(id)] = now();
    if (id != WeaponId::Artillery) ++stats_.shots;
    return true;
  }

  void smoke() {
    Tank* hero = player();
    if (!hero || smokeCharges_ <= 0 || now() - lastSmoke_ < 2.0) return;
    --smokeCharges_;
    lastSmoke_ = now();
    const double yaw = hero->heading() + hero->turretAngle();
    for (int k = -1; k <= 1; ++k) {
      const double a = yaw + 0.45 * k;
      const Vec3 p = hero->position() + 9.0 * Vec3(std::cos(a), std::sin(a), 0.0);
      const Vec3 center(p.x(), p.y(), height(p.x(), p.y()) + 1.5);
      viewer_.addExplosionEffect({glm::vec3(center.x(), center.y(), center.z()), 1.2f, true, {0.82f, 0.83f, 0.85f}, {0.78f, 0.79f, 0.8f}});
      smoke_.push_back({center, now() + 14.0});
    }
    notify("SMOKE SCREEN", IM_COL32(220, 220, 220, 255));
  }

  void step() {
    world_->integrate();
    const double t = now();
    steerMissiles();
    for (const auto& hit : world_->getProjectileHits()) {
      if (hit.id < nextHit_) continue;
      nextHit_ = hit.id + 1;
      const Shell* shell = findShell(hit.projectileId);
      if (Tank* target = tankAt(hit.objectIndex); target && shell)
        applyHit(*target, resolveHit(*target, hit, shell->weapon, shell->power, random_), shell->fromPlayer,
                 hit.position.e(), -hit.velocity.e());
      for (auto& barrel : barrels_)
        if (barrel.body && barrel.fuse < 0 && barrel.body->getIndexInWorld() == hit.objectIndex) barrel.fuse = hit.time + 0.05;
      if (shell && shell->weapon != WeaponId::MG) scenery_.shootTree(hit.objectIndex, hit.velocity.e());
    }
    for (const auto& explosion : world_->getExplosions()) {
      if (explosion.id < nextExplosion_) continue;
      nextExplosion_ = explosion.id + 1;
      const double charge = explosion.properties.chargeMass * (explosion.properties.surfaceBurst ? 1.8 : 1.0);
      if (explosion.properties.craterRadius > 0.0)
        terrainVisuals_.dig(explosion.position[0], explosion.position[1], explosion.properties.craterRadius);
      const Shell* shell = findShell(explosion.projectileId);
      const bool fromPlayer = !shell || shell->fromPlayer;
      for (Tank* tank : tanks()) {
        const Vec3 center = tank->position() + Vec3(0, 0, 0.6 * tank->scale());
        const double pressure = raisim::blast::wave((center - explosion.position.e()).norm(), charge).overpressure;
        const double damage = std::clamp((pressure - 15e3) / 5e3, 0.0, 50.0) *
                              std::clamp(70.0 / tank->spec->sideArmor, 0.2, 2.0);
        if (damage > 0.0) applyDamage(*tank, damage, fromPlayer, explosion.position.e() - center, false);
        if (tank == player_.get()) shake_ = std::max(shake_, std::clamp(pressure / 6e4, 0.0, 1.0));
      }
      for (auto& barrel : barrels_) {
        if (!barrel.body || barrel.fuse >= 0) continue;
        const double distance = (barrel.body->getPosition() - explosion.position.e()).norm();
        if (raisim::blast::wave(distance, charge).overpressure > 1.4e5) barrel.fuse = t + uniform(random_, 0.08, 0.35);
      }
      scenery_.blastTrees(explosion.position.e(), 2.5 * std::cbrt(charge));
    }
    scenery_.fellMarkedTrees(*world_, t);
    for (Tank* tank : tanks())
      if (tank->alive())
        scenery_.crushTrees(*world_, tank->position(), tank->hullRotation(), tank->model.hullMin, tank->model.hullMax,
                            tank->velocity(), t);
    for (auto& barrel : barrels_) {
      if (!barrel.body || barrel.fuse < 0 || t < barrel.fuse) continue;
      const Vec3 where = barrel.body->getPosition();
      undress(barrel.body);
      world_->removeObject(barrel.body);
      barrel.body = nullptr;
      raisim::ExplosionProperties blast;
      blast.chargeMass = 4.0;
      blast.surfaceBurst = true;
      blast.craterRadius = 1.6;
      blast.craterDepth = 0.6;
      world_->addExplosion(vec(where), blast);
    }
    for (size_t i = 0; i < pending_.size();) {
      if (pending_[i].time > t) {
        ++i;
        continue;
      }
      raisim::ExplosionProperties blast;
      blast.chargeMass = pending_[i].charge;
      blast.craterRadius = 0.8;
      blast.craterDepth = 0.25;
      world_->addExplosion(vec(pending_[i].position), blast);
      pending_.erase(pending_.begin() + long(i));
    }
    for (size_t i = 0; i < strikes_.size();) {
      if (strikes_[i].time > t) {
        ++i;
        continue;
      }
      raisim::ProjectileProperties shell;
      shell.mass = weapon(WeaponId::Artillery).mass;
      shell.radius = weapon(WeaponId::Artillery).radius;
      shell.dragCoefficient = 0.0;
      shell.maxLifetime = 10.0;
      shell.explosion.chargeMass = weapon(WeaponId::Artillery).charge;
      shell.explosion.craterRadius = 2.4;
      shell.explosion.craterDepth = 0.8;
      shells_.push_back({world_->addProjectile(vec(strikes_[i].origin), vec(strikes_[i].velocity), shell), true,
                         WeaponId::Artillery, 1.0});
      strikes_.erase(strikes_.begin() + long(i));
    }
    for (auto& enemy : enemies_)
      if (enemy->system && enemy->health <= 0.0) destroy(*enemy);
    if (player_ && player_->system && player_->health <= 0.0) {
      destroy(*player_);
      phase_ = Phase::Defeat;
      endedAt_ = now();
    }
  }

  // Drives the enemies, runs the waves, supplies and timers, and collects what the player drives over.
  void update(double frameTime) {
    const double t = now();
    viewer_.advanceWaterTime(frameTime);
    scenery_.animateTrees(t);
    std::erase_if(enemies_, [](const std::unique_ptr<Tank>& tank) { return !tank->system; });
    for (auto& wreck : wrecks_) {
      for (size_t i = 0; i < wreck.parts.size(); ++i) {
        if (!wreck.parts[i]) continue;
        const Vec3 p = wreck.parts[i]->getPosition();
        const Eigen::Quaterniond q(wreck.parts[i]->getRotationMatrix());
        const Vec3 at = p + q * wreck.offsets[i];
        if (auto visual = viewer_.getVisualObject(wreck.visuals[i])) {
          visual->setPosition(at.x(), at.y(), at.z());
          visual->setOrientation(q.w(), q.x(), q.y(), q.z());
        }
      }
      if (t > wreck.removeAt) removeWreck(wreck);
    }
    std::erase_if(wrecks_, [](const Wreck& wreck) { return wreck.parts.empty(); });
    std::erase_if(barrels_, [](const Barrel& barrel) { return !barrel.body; });
    for (const auto& dressing : dressings_) {
      const Eigen::Matrix3d R = dressing.body->getRotationMatrix();
      const Vec3 at = dressing.body->getPosition() + R * dressing.offset;
      const Eigen::Quaterniond q(R);
      dressing.mesh->setPosition(at.x(), at.y(), at.z());
      dressing.mesh->setOrientation(q.w(), q.x(), q.y(), q.z());
    }
    if (terrain_) terrainVisuals_.flush(viewer_, *terrain_);
    std::erase_if(missiles_, [](const Missile& missile) { return !missile.body; });
    std::erase_if(smoke_, [&](const SmokeZone& zone) { return zone.until < t; });
    std::erase_if(notices_, [&](const Notice& notice) { return t - notice.time > 4.0; });
    std::erase_if(floating_, [&](const Notice& notice) { return t - notice.time > 1.6; });
    if (!shells_.empty() && shells_.size() > 600) shells_.erase(shells_.begin(), shells_.begin() + 200);
    for (auto& marker : markers_)
      if (marker.until < t) viewer_.removeVisualObject(marker.name);
    std::erase_if(markers_, [&](const Marker& marker) { return marker.until < t; });
    shake_ *= std::exp(-6.0 * frameTime);

    if (phase_ == Phase::Playing) runWaves(t);
    if (phase_ == Phase::Playing && ammo_[size_t(WeaponId::Artillery)] < 2 && t > artilleryRecharge_) {
      ++ammo_[size_t(WeaponId::Artillery)];
      artilleryRecharge_ = t + 60.0;
      notify("ARTILLERY READY", rgba(weapon(WeaponId::Artillery).color));
    }

    Tank* hero = player();
    for (auto& enemy : enemies_) drive(*enemy, hero, frameTime);
    for (Tank* tank : tanks()) tank->selfRight(frameTime);
    // The air barely moves a tank and costs a fifth of a step, so fluids act on a tank only near the
    // river (the wind still drifts the rounds in flight).
    const Terrain terrain = levelSpec().terrain;
    for (Tank* tank : tanks()) {
      const Vec3 p = tank->position();
      const bool wading = riverDistance(terrain, p.x(), p.y()) < riverHalfWidth(terrain) + 6.0;
      if (wading == tank->wading) continue;
      raisim::FluidCoefficients coefficients;
      coefficients.enabled = wading;
      world_->setFluidCoefficients(tank->system, coefficients);
      tank->wading = wading;
    }
    for (auto& missile : missiles_)
      if (missile.body && t > missile.nextPuff) {
        const Vec3 p = missile.body->getPosition();
        viewer_.addExplosionEffect({glm::vec3(p.x(), p.y(), p.z()), 0.004f, false, {0.75f, 0.75f, 0.75f}});
        missile.nextPuff = t + 0.06;
      }

    for (auto& pickup : pickups_) {
      const bool blink = pickup.expires - t < 5.0 && std::fmod(t, 0.4) < 0.2;
      const glm::vec4 color = kBonusStyles[size_t(pickup.type)].color;
      const double pulse = 0.16 + 0.06 * std::sin(4.0 * t + pickup.position.x());
      pickup.beam->setColor(color.r, color.g, color.b, float(blink ? 0.04 : pulse));
      if (hero && (hero->position() - pickup.position).head<2>().norm() < 3.4) {
        collect(pickup.type);
        pickup.expires = -1.0;
      }
    }
    for (auto& pickup : pickups_)
      if (pickup.expires < t) removeVisuals(pickup);
    std::erase_if(pickups_, [&](const Pickup& pickup) { return pickup.expires < t; });
  }

  double overdrive() const { return now() < overdriveUntil_ ? 1.5 : 1.0; }
  double overdriveLeft() const { return std::max(0.0, overdriveUntil_ - now()); }
  int reactive() const { return reactive_; }
  double shake() const { return shake_; }
  int score() const { return score_; }
  const Stats& stats() const { return stats_; }
  int wave() const { return wave_; }
  int waves() const { return levelSpec().endless ? 0 : int(levelSpec().waves.size()); }
  int enemiesLeft() const { return int(enemies_.size() + queue_.size()); }
  double elapsed() const { return (endedAt_ >= 0.0 ? endedAt_ : now()) - started_; }
  double playerHealth() const { return player_ ? std::max(0.0, player_->health) : 0.0; }
  double hurtTime() const { return player_ ? player_->hurtTime : -100.0; }
  const Vec3& hurtFrom() const { return hurtFrom_; }
  double hitMarkerTime() const { return hitMarker_; }
  ImU32 hitMarkerColor() const { return hitMarkerColor_; }
  const std::vector<std::unique_ptr<Tank>>& enemies() const { return enemies_; }
  const std::vector<Pickup>& pickups() const { return pickups_; }
  const std::vector<Barrel>& barrels() const { return barrels_; }
  const std::vector<Notice>& notices() const { return notices_; }
  const std::vector<Notice>& floating() const { return floating_; }
  const std::vector<Notice>& kills() const { return killFeed_; }
  bool tracksBroken() const { return player_ && now() < player_->tracksBrokenUntil; }
  bool gunDamaged() const { return player_ && now() < player_->gunDamagedUntil; }

  void notify(const std::string& text, ImU32 color) { notices_.push_back({text, color, now()}); }

 private:
  static size_t slot(WeaponId id) { return id == WeaponId::HE ? size_t(WeaponId::AP) : size_t(id); }

  double reloadTime(WeaponId id) const {
    return weapon(id).reload * (gunDamaged() && (id == WeaponId::AP || id == WeaponId::HE) ? 2.0 : 1.0);
  }

  void clear() {
    for (const auto& projectile : std::vector<raisim::Projectile>(world_->getProjectiles()))
      world_->removeProjectile(projectile.id);
    for (auto& pickup : pickups_) removeVisuals(pickup);
    pickups_.clear();
    for (auto& marker : markers_) viewer_.removeVisualObject(marker.name);
    markers_.clear();
    for (const auto& dressing : dressings_) viewer_.removeVisualObject(dressing.visual);
    dressings_.clear();
    for (const auto& name : visuals_) viewer_.removeVisualObject(name);
    visuals_.clear();
    terrainVisuals_.clear(viewer_);
    scenery_.removeTreeColliders(*world_);
    scenery_.clear(viewer_);
    viewer_.removeWaterBody("river");
    world_->clearFluids();
    for (auto* object : props_) world_->removeObject(object);
    props_.clear();
    terrain_ = nullptr;
    buildings_.clear();
    town_.clear(viewer_);
    for (auto& wreck : wrecks_) removeWreck(wreck);
    wrecks_.clear();
    for (auto& enemy : enemies_)
      if (enemy->system) world_->removeObject(enemy->system);
    enemies_.clear();
    if (player_ && player_->system) world_->removeObject(player_->system);
    player_.reset();
    for (auto& barrel : barrels_)
      if (barrel.body) world_->removeObject(barrel.body);
    barrels_.clear();
    for (auto& missile : missiles_)
      if (missile.body) world_->removeObject(missile.body);
    missiles_.clear();
    shells_.clear();
    pending_.clear();
    strikes_.clear();
    smoke_.clear();
    notices_.clear();
    floating_.clear();
    killFeed_.clear();
    // Blasts still spreading would push the new tanks; they are spent well within half a second.
    for (int i = 0; i < 250; ++i) world_->integrate();
    if (!world_->getExplosions().empty()) nextExplosion_ = world_->getExplosions().back().id + 1;
    if (!world_->getProjectileHits().empty()) nextHit_ = world_->getProjectileHits().back().id + 1;
    viewer_.clearExplosionEffects();
  }

  void applyTheme(const LevelSpec& spec, int quality) {
    configureRendering(viewer_, spec, quality);
    quality_ = quality;
    raisim::FluidMedium air = raisim::FluidMedium::air();
    air.velocity = vec(spec.wind);
    world_->setAmbientFluid(air);
  }

  // A solid box drawn with the concrete texture (or not at all, for the colliders of the arena walls).
  raisim::Box* staticBox(const Vec3& size, const Vec3& center, double yaw, bool drawn) {
    auto* box = world_->addBox(size.x(), size.y(), size.z(), 1.0);
    box->setPosition(center.x(), center.y(), center.z());
    box->setOrientation(std::cos(0.5 * yaw), 0.0, 0.0, std::sin(0.5 * yaw));
    box->setBodyType(raisim::BodyType::STATIC);
    box->setAppearance("hidden");
    props_.push_back(box);
    if (drawn) {
      const std::string name = "building_" + std::to_string(nextName_++);
      auto visual = texturedBox(viewer_, name, size, concrete(), 3.0);
      visual->setPosition(center.x(), center.y(), center.z());
      visual->setOrientation(std::cos(0.5 * yaw), 0.0, 0.0, std::sin(0.5 * yaw));
      visuals_.push_back(name);
    }
    return box;
  }

  const raisin::Material& concrete() {
    if (!concrete_.albedoMap) {
      concrete_ = raisin::Material::pbr("concrete", glm::vec4(0.74f, 0.72f, 0.69f, 1.f), 0.f, 0.9f);
      concrete_.albedoMap = Viewer::loadTextureWithTiling((assets_ + "/tank_survival/ground/concrete_wall_006_diffuse_1k.jpg").c_str(), true);
      concrete_.normalMap = Viewer::loadTextureWithTiling((assets_ + "/tank_survival/ground/concrete_wall_006_nor_gl_1k.jpg").c_str(), false);
    }
    return concrete_;
  }

  // Draws body with a model: the body's primitive is hidden and the model follows it (update()).
  void dress(raisim::SingleBodyObject* body, const std::string& model, double scale, const Vec3& offset) {
    body->setAppearance("hidden");
    const std::string name = "dressed_" + std::to_string(nextName_++);
    auto mesh = viewer_.addVisualMesh(name, assets_ + "/tank_survival/props/" + model + "/model.gltf", glm::dvec3(scale));
    dressings_.push_back({body, name, offset, mesh});
  }

  void buildArena(const LevelSpec& spec, const std::function<void(double, const char*)>& report) {
    report(0.1, "Shaping the ground");
    constexpr size_t kSamples = 221;
    std::vector<double> heights(kSamples * kSamples);
    for (size_t j = 0; j < kSamples; ++j)
      for (size_t i = 0; i < kSamples; ++i)
        heights[j * kSamples + i] = groundShape(spec.terrain, -0.5 * kExtent + kExtent * double(i) / (kSamples - 1),
                                                -0.5 * kExtent + kExtent * double(j) / (kSamples - 1));
    terrain_ = world_->addHeightMap(kSamples, kSamples, kExtent, kExtent, 0, 0, heights);
    terrain_->setAppearance("hidden");
    props_.push_back(terrain_);
    for (int side = 0; side < 4; ++side) {
      const double along = 2.0 * kArena + 1.0, offset = side % 2 ? kArena : -kArena;
      staticBox(side < 2 ? Vec3(1.0, along, 9.0) : Vec3(along, 1.0, 9.0), side < 2 ? Vec3(offset, 0.0, -2.5) : Vec3(0.0, offset, -2.5),
                0.0, false);
    }
    report(0.25, "Raising buildings");
    switch (spec.terrain) {
      case Terrain::Town: buildTown(heights, kSamples); break;
      case Terrain::Fortress: buildFortress(); break;
      default: break;
    }
    // Supply dumps of stacked crates and of fuel drums, each on ground levelled for it, so that they
    // stand still (and sleep) from the start.
    std::vector<Vec3> stacks, drums;
    for (int stack = 0; stack < 6; ++stack) stacks.push_back(freeGroundPoint(16.0, kArena - 8.0, 3.0));
    for (int cluster = 0; cluster < 7; ++cluster) drums.push_back(freeGroundPoint(14.0, kArena - 6.0, 3.0) + Vec3(0.375, 0.375, 0.0));
    for (const auto& p : stacks) levelGround(heights, kSamples, p, 2.0);
    for (const auto& p : drums) levelGround(heights, kSamples, p, 1.4);
    terrain_->update(0.0, 0.0, kExtent, kExtent, heights);
    report(0.35, "Laying the terrain");
    terrainVisuals_.build(viewer_, *terrain_, spec.terrain, assets_);
    report(0.6, "Filling the river");
    addRiver(spec.terrain);
    std::vector<std::pair<Vec3, Vec3>> footprints;
    for (const auto& b : buildings_) footprints.emplace_back(b.center, b.half);
    for (const auto& p : stacks) footprints.emplace_back(p, Vec3(1.6, 1.6, 1.0));
    for (const auto& p : drums) footprints.emplace_back(p, Vec3(1.2, 1.2, 1.0));
    report(0.65, "Planting trees and grass");
    scenery_.build(*world_, viewer_, *terrain_, spec.terrain, assets_, quality_, random_, footprints, spec.playerStart, props_);
    report(0.9, "Stacking supplies");
    raisim::FluidCoefficients still;
    still.enabled = false;
    for (const auto& p : stacks) {
      const double ground = height(p.x(), p.y());
      for (int layer = 0; layer < 3; ++layer)
        for (int k = 0; k < 3 - layer; ++k) {
          auto* crate = world_->addBox(1.55, 0.65, 0.58, 90.0);
          crate->setPosition(p.x(), p.y() - 0.35 * (2 - layer) + 0.7 * k, ground + 0.2905 + layer * 0.5805);
          world_->setFluidCoefficients(crate, still);
          dress(crate, "wooden_military_crate", 1.25, Vec3(0, 0, -0.29));
          props_.push_back(crate);
        }
    }
    for (const auto& p : drums) {
      const double ground = height(p.x(), p.y());
      for (int k = 0; k < 4; ++k) {
        Barrel barrel;
        barrel.body = world_->addCylinder(0.34, 0.98, 80.0);
        barrel.body->setPosition(p.x() + 0.75 * (k % 2) - 0.375, p.y() + 0.75 * (k / 2) - 0.375, ground + 0.4905);
        world_->setFluidCoefficients(barrel.body, still);
        dress(barrel.body, "barrel_03", 1.05, Vec3(0, 0, -0.49));
        barrels_.push_back(barrel);
      }
    }
  }

  // Levels the heightmap samples within radius of p to their mean, blending back over two metres.
  static void levelGround(std::vector<double>& heights, size_t samples, const Vec3& p, double radius) {
    const double cell = kExtent / double(samples - 1), half = 0.5 * kExtent, reach = radius + 2.0;
    const int i0 = std::max(0, int((p.x() - reach + half) / cell)), i1 = std::min(int(samples) - 1, int((p.x() + reach + half) / cell) + 1);
    const int j0 = std::max(0, int((p.y() - reach + half) / cell)), j1 = std::min(int(samples) - 1, int((p.y() + reach + half) / cell) + 1);
    double sum = 0.0;
    int count = 0;
    for (int j = j0; j <= j1; ++j)
      for (int i = i0; i <= i1; ++i)
        if (std::hypot(-half + i * cell - p.x(), -half + j * cell - p.y()) <= radius) {
          sum += heights[size_t(j) * samples + size_t(i)];
          ++count;
        }
    if (count == 0) return;
    const double mean = sum / count;
    for (int j = j0; j <= j1; ++j)
      for (int i = i0; i <= i1; ++i) {
        const double d = std::hypot(-half + i * cell - p.x(), -half + j * cell - p.y());
        const double w = std::clamp((reach - d) / (reach - radius), 0.0, 1.0);
        double& h = heights[size_t(j) * samples + size_t(i)];
        h += w * w * (3.0 - 2.0 * w) * (mean - h);
      }
  }

  // The river: rayrai simulates its water over the drawn bed (tanks push it aside), and RaiSim a
  // volume of water that follows the same profile, so that it holds up and drags what enters it and
  // slows rounds fired into it.
  void addRiver(Terrain terrain) {
    const River river = riverOf(terrain);
    if (!river.present) return;
    constexpr double kHalfLength = 0.5 * kExtent - 2.0;
    const double halfWidth = river.amplitude + riverHalfWidth(terrain) + 5.0;
    raisin::WaterBody water = raisin::defaultWaterBody(raisin::WaterPreset::CalmLake);
    water.shape = raisin::WaterBodyShape::Rectangle;
    water.center = {0.0, river.offset, riverSurface(0.0)};
    water.halfExtents = {float(kHalfLength), float(halfWidth)};
    water.depth = 2.4f;
    water.edgeFadeDistance = 0.0f;
    water.waves.clear();
    water.absorption = terrain == Terrain::Town ? glm::vec3(0.5f, 0.3f, 0.32f) : glm::vec3(0.42f, 0.15f, 0.2f);
    water.scattering = terrain == Terrain::Town ? glm::vec3(0.06f, 0.07f, 0.05f) : glm::vec3(0.03f, 0.05f, 0.04f);
    water.roughness = 0.03f;
    water.rippleStrength = 0.5f;
    water.rippleScale = 1.2f;
    water.shorelineFoamDepth = 0.0f;
    water.crestFoamStrength = 0.0f;
    water.causticStrength = 0.5f;
    water.causticScale = 0.9f;
    water.flow.enabled = true;
    water.flow.cellSize = 0.3f;
    water.flow.bedCeiling = 3.0f;
    water.flow.manningRoughness = 0.03f;
    water.flow.foamStrength = 0.6f;
    raisin::WaterFlowSource spring;
    const double x0 = -kHalfLength + 4.0;
    spring.position = {x0, riverY(river, x0)};
    spring.radius = 2.5f;
    spring.flowRate = float(kRiverFlow);
    spring.velocity = {float(kRiverSpeed), 0.0f};
    water.flow.sources.push_back(spring);
    viewer_.setWaterBody("river", water);

    raisim::FluidVolume volume;
    volume.surfaceHeight = riverSurface(0.0);
    volume.bottomHeight = riverSurface(kHalfLength) - kRiverDepth - 1.0;
    volume.min = {-kHalfLength, river.offset - halfWidth};
    volume.max = {kHalfLength, river.offset + halfWidth};
    volume.sampler = [terrain, river](const raisim::Vec<3>& point) {
      raisim::FluidSample sample;
      // Only the channel holds water: a crater beside it stays dry, as the drawn river leaves it.
      const double x = point[0], d = std::abs(point[1] - riverY(river, x)) / riverHalfWidth(terrain);
      sample.surfaceHeight = d < 1.1 ? riverSurface(x) : -1e3;
      sample.surfaceNormal = {kRiverSlope, 0.0, 1.0};
      const double slope = river.amplitude * 0.031 * std::cos(0.031 * x + 0.6), speed = kRiverSpeed * std::max(0.0, 1.0 - d * d);
      sample.velocity = {speed / std::sqrt(1.0 + slope * slope), speed * slope / std::sqrt(1.0 + slope * slope), 0.0};
      return sample;
    };
    world_->addFluidVolume(volume);
  }

  void buildTown(std::vector<double>& heights, size_t samples) {
    std::vector<Lot> lots;
    for (double cx = -60.0; cx <= 60.0; cx += 24.0)
      for (double cy = -60.0; cy <= 60.0; cy += 24.0) {
        if (std::abs(cx) < 20.0 && std::abs(cy) < 20.0) continue;
        const bool ruined = uniform(random_, 0.0, 1.0) < 0.3;
        const double lx = 3.0 * std::floor(uniform(random_, 3.0, 5.999)), ly = 3.0 * std::floor(uniform(random_, 3.0, 4.999));
        const double x = cx + uniform(random_, -2.0, 2.0), y = cy + uniform(random_, -2.0, 2.0);
        if (riverDistance(Terrain::Town, x, y) < 0.5 * ly + riverHalfWidth(Terrain::Town) + 3.0) continue;
        Lot lot{x - 0.5 * lx, y - 0.5 * ly, x + 0.5 * lx, y + 0.5 * ly, 0.0,
                ruined ? int(uniform(random_, 1.0, 2.999)) : int(uniform(random_, 2.0, 4.999)), ruined,
                int(uniform(random_, 0.0, 3.999)), float(ruined ? uniform(random_, 0.8, 0.95) : uniform(random_, 0.0, 0.55))};
        lot.ground = levelRectangle(heights, samples, lot.x0, lot.y0, lot.x1, lot.y1);
        lots.push_back(lot);
        const double top = Town::height(lot);
        const Vec3 center(x, y, lot.ground + 0.5 * top - 0.5), size(lx, ly, top + 1.0);
        staticBox(size, center, 0.0, false);
        buildings_.push_back({center, 0.5 * size});
        if (!ruined) continue;
        for (int k = 0; k < 6; ++k) {
          auto* rubble = world_->addBox(uniform(random_, 0.6, 1.4), uniform(random_, 0.6, 1.4), 0.6, 250.0);
          rubble->setPosition(x + uniform(random_, -0.35, 0.35) * lx, y + uniform(random_, -0.35, 0.35) * ly, top + lot.ground + 0.5 + 0.7 * k);
          props_.push_back(rubble);
          raisim::FluidCoefficients still;
          still.enabled = false;
          world_->setFluidCoefficients(rubble, still);
          const std::string name = "rubble_" + std::to_string(nextName_++);
          rubble->setAppearance("hidden");
          const auto dim = rubble->getDim();
          dressings_.push_back({rubble, name, Vec3::Zero(), texturedBox(viewer_, name, Vec3(dim[0], dim[1], dim[2]), concrete(), 1.5)});
        }
      }
    town_.build(viewer_, assets_, lots);
  }

  // Levels the ground under a footprint (and a margin around it) to its mean height, blending into
  // the surrounding ground over a few metres; returns that height.
  static double levelRectangle(std::vector<double>& heights, size_t samples, double x0, double y0, double x1, double y1) {
    constexpr double kMargin = 1.5, kBlend = 4.0;
    const double cell = kExtent / double(samples - 1), half = 0.5 * kExtent;
    const auto outside = [&](double x, double y) {
      return std::hypot(std::max({x0 - kMargin - x, 0.0, x - x1 - kMargin}), std::max({y0 - kMargin - y, 0.0, y - y1 - kMargin}));
    };
    double sum = 0.0;
    int count = 0;
    const int i0 = std::max(0, int((x0 - kMargin - kBlend + half) / cell)), i1 = std::min(int(samples) - 1, int((x1 + kMargin + kBlend + half) / cell) + 1);
    const int j0 = std::max(0, int((y0 - kMargin - kBlend + half) / cell)), j1 = std::min(int(samples) - 1, int((y1 + kMargin + kBlend + half) / cell) + 1);
    for (int j = j0; j <= j1; ++j)
      for (int i = i0; i <= i1; ++i)
        if (outside(-half + i * cell, -half + j * cell) == 0.0) {
          sum += heights[size_t(j) * samples + size_t(i)];
          ++count;
        }
    if (count == 0) return 0.0;
    const double mean = sum / count;
    for (int j = j0; j <= j1; ++j)
      for (int i = i0; i <= i1; ++i) {
        const double d = outside(-half + i * cell, -half + j * cell);
        if (d >= kBlend) continue;
        const double w = 1.0 - d / kBlend, blend = w * w * (3.0 - 2.0 * w);
        double& h = heights[size_t(j) * samples + size_t(i)];
        h += blend * (mean - h);
      }
    return mean;
  }

  void buildFortress() {
    const Vec3 center(0, 12, 0);
    constexpr double kRadius = 30.0;
    std::vector<::Vertex> vertices;
    std::vector<unsigned int> indices;
    // Merlons along the top of a wall or tower, in the wall's own frame.
    const auto merlon = [&](const Vec3& at, double yaw, const Vec3& local, const Vec3& size) {
      const double c = std::cos(yaw), s = std::sin(yaw);
      appendBox(vertices, indices, size, at + Vec3(c * local.x() - s * local.y(), s * local.x() + c * local.y(), local.z()), yaw, 1.5);
    };
    for (int k = 0; k < 16; ++k) {
      if (k % 4 == 0) continue;
      const double a = 2.0 * kPi * (k + 0.5) / 16.0;
      const Vec3 p = center + kRadius * Vec3(std::cos(a), std::sin(a), 0.0);
      const double z = lowest(p.x(), p.y(), 5.5, 5.5) + 1.5;
      staticBox(Vec3(1.6, 11.0, 3.6), Vec3(p.x(), p.y(), z), a, true);
      buildings_.push_back({Vec3(p.x(), p.y(), z), Vec3(5.5, 5.5, 1.8)});
      for (int m = 0; m < 6; ++m) merlon(Vec3(p.x(), p.y(), z), a, Vec3(0.0, -4.5 + 1.8 * m, 2.2), Vec3(1.6, 0.9, 0.8));
    }
    for (int k = 0; k < 4; ++k) {
      const double a = 0.25 * kPi + 0.5 * kPi * k;
      const Vec3 p = center + (kRadius + 1.5) * Vec3(std::cos(a), std::sin(a), 0.0);
      const double z = lowest(p.x(), p.y(), 2.6, 2.6) + 3.3;
      staticBox(Vec3(5.0, 5.0, 7.0), Vec3(p.x(), p.y(), z), a, true);
      buildings_.push_back({Vec3(p.x(), p.y(), z), Vec3(2.6, 2.6, 3.5)});
      for (int side = 0; side < 4; ++side)
        for (int m = 0; m < 4; ++m) {
          const double along = -1.95 + 1.3 * m, edge = side < 2 ? 2.25 : -2.25;
          merlon(Vec3(p.x(), p.y(), z), a, side % 2 ? Vec3(along, edge, 3.95) : Vec3(edge, along, 3.95),
                 side % 2 ? Vec3(0.7, 0.5, 0.9) : Vec3(0.5, 0.7, 0.9));
        }
    }
    for (const auto& [x, y] : {std::array<double, 2>{-12, 22}, std::array<double, 2>{12, 22}, std::array<double, 2>{0, 0}}) {
      const double z = lowest(x, y, 3.0, 2.0) + 0.9;
      staticBox(Vec3(6, 4, 2.2), Vec3(x, y, z), 0.0, true);
      buildings_.push_back({Vec3(x, y, z), Vec3(3, 2, 1.1)});
    }
    visuals_.push_back("fortress merlons");
    meshVisual(viewer_, visuals_.back(), std::move(vertices), std::move(indices), concrete());
  }

  // The lowest ground under a rectangle, so that buildings leave no gap under their walls.
  double lowest(double x, double y, double halfX, double halfY) const {
    double z = height(x, y);
    for (double u = -1.0; u <= 1.0; u += 0.25)
      for (double v = -1.0; v <= 1.0; v += 0.25) z = std::min(z, height(x + u * halfX * 1.42, y + v * halfY * 1.42));
    return z;
  }

  bool blocked(const Vec3& p, double margin) const {
    if (riverDistance(levelSpec().terrain, p.x(), p.y()) < riverHalfWidth(levelSpec().terrain) + margin) return true;
    for (const auto& b : buildings_)
      if (std::abs(p.x() - b.center.x()) < b.half.x() + margin && std::abs(p.y() - b.center.y()) < b.half.y() + margin)
        return true;
    return false;
  }

  Vec3 freeGroundPoint(double minRadius, double maxRadius, double margin) {
    const Vec3 start = levelSpec().playerStart;
    for (int attempt = 0; attempt < 200; ++attempt) {
      const double angle = uniform(random_, 0.0, 2 * kPi), radius = minRadius + (maxRadius - minRadius) * std::sqrt(uniform(random_, 0.0, 1.0));
      const Vec3 p(radius * std::cos(angle), radius * std::sin(angle), 0.0);
      if (std::max(std::abs(p.x()), std::abs(p.y())) < kArena - 3.0 && !blocked(p, margin) &&
          (p - start).head<2>().norm() > 12.0)
        return p;
    }
    return Vec3(minRadius, 0, 0);
  }

  void runWaves(double t) {
    const LevelSpec& spec = levelSpec();
    const bool waveOver = queue_.empty() && enemies_.empty();
    if (waveOver && wave_ > 0 && !waveCleared_) {
      waveCleared_ = true;
      nextWave_ = t + 5.0;
      score_ += 100 * wave_;
      notify("WAVE " + std::to_string(wave_) + " CLEARED  +" + std::to_string(100 * wave_), IM_COL32(150, 255, 150, 255));
      if (!spec.endless && wave_ >= int(spec.waves.size())) {
        phase_ = Phase::Victory;
        endedAt_ = t;
        score_ += int(playerHealth() * 5.0);
        return;
      }
    }
    if (waveOver && (wave_ == 0 || waveCleared_) && t > nextWave_) {
      ++wave_;
      waveCleared_ = false;
      const auto squads = spec.endless ? endlessWave(wave_) : spec.waves[size_t(wave_ - 1)];
      for (const auto& squad : squads)
        for (int i = 0; i < squad.count; ++i) queue_.push_back(squad.role);
      std::shuffle(queue_.begin(), queue_.end(), random_);
      std::stable_partition(queue_.begin(), queue_.end(), [](Role role) { return role != Role::Boss; });
      nextSpawn_ = t + 1.0;
      notify("WAVE " + std::to_string(wave_) + (spec.endless ? "" : " / " + std::to_string(spec.waves.size())),
             IM_COL32(255, 220, 120, 255));
    }
    if (!queue_.empty() && enemies_.size() < 8 && t > nextSpawn_) {
      spawnEnemy(queue_.front());
      queue_.erase(queue_.begin());
      nextSpawn_ = t + std::max(1.0, 2.6 - 0.15 * wave_);
    }
    for (const auto& enemy : enemies_)
      if (enemy->spec->role == Role::Boss && !reinforced_ && enemy->health < 0.5 * enemy->spec->health) {
        reinforced_ = true;
        for (int i = 0; i < 3; ++i) queue_.push_back(Role::Medium);
        notify("REINFORCEMENTS INBOUND", IM_COL32(255, 120, 90, 255));
      }
    if (t > nextPickup_ && pickups_.size() < 6) {
      spawnPickup(randomBonus());
      nextPickup_ = t + uniform(random_, 6.0, 9.0);
    }
  }

  void spawnEnemy(Role role) {
    const LevelSpec& spec = levelSpec();
    const Vec3 hero = player_ ? player_->position() : spec.playerStart;
    Vec3 p = hero;
    for (int attempt = 0; attempt < 60; ++attempt) {
      if (spec.enemyRadius > 0.0) {
        const double a = uniform(random_, 0.0, 2 * kPi), r = spec.enemyRadius * std::sqrt(uniform(random_, 0.0, 1.0));
        p = spec.enemyCenter + Vec3(r * std::cos(a), r * std::sin(a), 0.0);
      } else {
        const double a = uniform(random_, 0.0, 2 * kPi), r = uniform(random_, 50.0, 75.0);
        p = hero + Vec3(r * std::cos(a), r * std::sin(a), 0.0);
      }
      p.x() = std::clamp(p.x(), -kArena + 8.0, kArena - 8.0);
      p.y() = std::clamp(p.y(), -kArena + 8.0, kArena - 8.0);
      if ((p - hero).head<2>().norm() > 35.0 && !blocked(p, 6.0)) break;
    }
    const double yaw = std::atan2(hero.y() - p.y(), hero.x() - p.x());
    const auto& kind = vehicle(role);
    enemies_.push_back(std::make_unique<Tank>(*world_, urdfs_[size_t(role)], models_[size_t(role)], kind,
                                              Vec3(p.x(), p.y(), height(p.x(), p.y()) + 3.0), yaw, now()));
    enemies_.back()->strafe = uniform(random_, 0.0, 1.0) < 0.5 ? -1.0 : 1.0;
    enemies_.back()->lastShot = now() - kind.reload + 2.5;
    if (role == Role::Boss) notify("THE BEHEMOTH HAS ARRIVED", IM_COL32(255, 80, 120, 255));
  }

  Bonus randomBonus() {
    std::vector<double> weights;
    for (const auto& style : kBonusStyles) weights.push_back(style.weight);
    std::discrete_distribution<int> pick(weights.begin(), weights.end());
    Bonus type = Bonus(pick(random_));
    if (ammo_[0] + ammo_[1] < 10 && type != Bonus::Ammo && uniform(random_, 0.0, 1.0) < 0.5) type = Bonus::Ammo;
    return type;
  }

  void spawnPickup(Bonus type) {
    const Vec3 center = player_ ? player_->position() : levelSpec().playerStart;
    Vec3 p = center;
    for (int attempt = 0; attempt < 40; ++attempt) {
      const double a = uniform(random_, 0.0, 2 * kPi), r = uniform(random_, 12.0, 38.0);
      p = center + Vec3(r * std::cos(a), r * std::sin(a), 0.0);
      p.x() = std::clamp(p.x(), -kArena + 4.0, kArena - 4.0);
      p.y() = std::clamp(p.y(), -kArena + 4.0, kArena - 4.0);
      if (!blocked(p, 2.0)) break;
    }
    spawnPickupAt(type, Vec3(p.x(), p.y(), height(p.x(), p.y())));
  }

  void spawnPickupAt(Bonus type, const Vec3& p) {
    Pickup pickup;
    pickup.type = type;
    pickup.position = p;
    pickup.expires = now() + 40.0;
    pickup.name = "supply_" + std::to_string(nextName_++);
    const BonusStyle& style = kBonusStyles[size_t(type)];
    pickup.crate = viewer_.addVisualMesh(pickup.name, assets_ + "/tank_survival/props/" + style.model + "/model.gltf",
                                         glm::dvec3(style.scale));
    const double yaw = uniform(random_, 0.0, 2 * kPi);
    pickup.crate->setPosition(p.x(), p.y(), p.z() - 0.03);
    pickup.crate->setOrientation(std::cos(0.5 * yaw), 0.0, 0.0, std::sin(0.5 * yaw));
    pickup.beam = viewer_.addVisualCylinder(pickup.name + "_beam", 0.09, 14.0,
                                            glm::vec4(style.color.r, style.color.g, style.color.b, 0.2f));
    pickup.beam->setPosition(p.x(), p.y(), p.z() + 7.0);
    pickups_.push_back(pickup);
  }

  void removeVisuals(Pickup& pickup) {
    if (!pickup.crate) return;
    viewer_.removeVisualObject(pickup.name);
    viewer_.removeVisualObject(pickup.name + "_beam");
    pickup.crate.reset();
    pickup.beam.reset();
  }

  void addMarker(const Vec3& p, double until, const glm::vec4& color) {
    Marker marker{"marker_" + std::to_string(nextName_++), until};
    auto disk = viewer_.addVisualCylinder(marker.name, 4.5, 0.06, color);
    disk->setPosition(p.x(), p.y(), height(p.x(), p.y()) + 0.12);
    markers_.push_back(marker);
  }

  void collect(Bonus type) {
    std::string text;
    switch (type) {
      case Bonus::Ammo:
        ammo_[size_t(WeaponId::AP)] = std::min(60, ammo_[size_t(WeaponId::AP)] + 12);
        ammo_[size_t(WeaponId::HE)] = std::min(30, ammo_[size_t(WeaponId::HE)] + 8);
        ammo_[size_t(WeaponId::MG)] = std::min(1200, ammo_[size_t(WeaponId::MG)] + 250);
        smokeCharges_ = std::min(5, smokeCharges_ + 1);
        text = "+12 AP  +8 HE  +250 MG  +1 SMOKE";
        break;
      case Bonus::Repair:
        player_->health = std::min(player_->spec->health, player_->health + 70.0);
        player_->tracksBrokenUntil = player_->gunDamagedUntil = -1.0;
        text = "REPAIRED +70";
        break;
      case Bonus::Reactive:
        reactive_ = std::min(6, reactive_ + 3);
        text = "REACTIVE ARMOR: 3 BLOCKS";
        break;
      case Bonus::Overdrive:
        overdriveUntil_ = now() + 12.0;
        text = "ENGINE OVERDRIVE 12 s";
        break;
      case Bonus::Missiles:
        ammo_[size_t(WeaponId::Missile)] = std::min(8, ammo_[size_t(WeaponId::Missile)] + 2);
        text = "+2 MISSILES";
        break;
      case Bonus::Radio:
        ammo_[size_t(WeaponId::Artillery)] = std::min(3, ammo_[size_t(WeaponId::Artillery)] + 1);
        text = "+1 ARTILLERY STRIKE";
        break;
    }
    score_ += 10;
    notify(text, rgba(kBonusStyles[size_t(type)].color));
  }

  void launchShell(Tank& shooter, WeaponId id, bool fromPlayer, double power, double spread) {
    const WeaponSpec& round = weapon(id);
    Vec3 position, direction;
    roundOrigin(shooter, id, position, direction);
    if (spread > 0.0) {
      std::normal_distribution<double> noise(0.0, spread);
      direction = (direction + Vec3(noise(random_), noise(random_), 0.5 * noise(random_))).normalized();
    }
    raisim::ProjectileProperties shell;
    shell.mass = round.mass;
    shell.radius = round.radius;
    shell.dragCoefficient = round.dragCoefficient;
    shell.maxLifetime = 6.0;
    shell.explosion.chargeMass = round.charge;
    shell.explosion.craterRadius = 0.9 * std::cbrt(round.charge);
    shell.explosion.craterDepth = 0.32 * std::cbrt(round.charge);
    shell.ignoredObject = shooter.system->getIndexInWorld();
    shells_.push_back({world_->addProjectile(vec(position), vec(direction * round.speed), shell), fromPlayer, id, power});
    if (id != WeaponId::MG)
      shooter.system->setExternalForce(shooter.gunBody, vec(Vec3::Zero()), vec(-round.mass * round.speed * direction / kTimeStep));
  }

  // A mortar round on a 55 degree arc at target; the ground there is marked until it lands.
  void launchMortar(Tank& shooter, const Vec3& target) {
    const Vec3 origin = shooter.turretPivot() + Vec3(0, 0, 0.8);
    const Vec3 offset = target - origin;
    const double distance = std::max(5.0, offset.head<2>().norm());
    const double angle = 55.0 * kPi / 180.0;
    const double denominator = 2.0 * std::cos(angle) * std::cos(angle) * (distance * std::tan(angle) - offset.z());
    if (denominator <= 0.0) return;
    const double speed = std::sqrt(kGravity * distance * distance / denominator);
    const Vec3 horizontal = Vec3(offset.x(), offset.y(), 0.0).normalized();
    const Vec3 velocity = speed * (std::cos(angle) * horizontal + std::sin(angle) * Vec3::UnitZ());
    raisim::ProjectileProperties shell;
    shell.mass = 12.0;
    shell.radius = 0.06;
    shell.dragCoefficient = 0.05;
    shell.maxLifetime = 15.0;
    shell.explosion.chargeMass = 3.0;
    shell.explosion.craterRadius = 1.4;
    shell.explosion.craterDepth = 0.5;
    shell.ignoredObject = shooter.system->getIndexInWorld();
    shells_.push_back({world_->addProjectile(vec(origin), vec(velocity), shell), false, WeaponId::HE, 1.0});
    addMarker(target, now() + distance / (speed * std::cos(angle)) + 0.4, {1.0f, 0.15f, 0.1f, 0.35f});
    shooter.system->setExternalForce(shooter.hullBody, vec(Vec3::Zero()), vec(-12.0 * velocity / kTimeStep));
  }

  void launchMissile(Tank& shooter) {
    Vec3 position, direction;
    shooter.muzzle(position, direction);
    Missile missile;
    missile.body = world_->addCylinder(0.07, 0.9, weapon(WeaponId::Missile).mass);
    const Vec3 start = position + direction * 1.2 + Vec3(0, 0, 0.6);
    missile.body->setPosition(start.x(), start.y(), start.z());
    missile.body->setOrientation(Eigen::Quaterniond::FromTwoVectors(Vec3::UnitZ(), direction));
    missile.body->setVelocity(direction * 60.0, Vec3::Zero());
    missile.body->setAppearance("0.35,0.38,0.3,1.0");
    missile.shooter = shooter.system->getIndexInWorld();
    missile.launched = now();
    missiles_.push_back(missile);
  }

  // Missiles fly at their speed toward the aim, against gravity, turning at up to 9 g.
  void steerMissiles() {
    const double speed = weapon(WeaponId::Missile).speed, mass = weapon(WeaponId::Missile).mass;
    for (auto& missile : missiles_) {
      if (!missile.body) continue;
      const double age = now() - missile.launched;
      size_t struck = size_t(-1);
      bool detonate = age > 9.0;
      for (const auto& contact : missile.body->getContacts()) {
        if (age < 0.4 && contact.getPairObjectIndex() == missile.shooter) continue;
        struck = contact.getPairObjectIndex();
        detonate = true;
      }
      const Vec3 p = missile.body->getPosition(), v = missile.body->getLinearVelocity();
      if (!detonate && (aim_ - p).norm() < 1.2) detonate = true;
      if (detonate) {
        explodeMissile(missile, struck, p, v);
        continue;
      }
      const Vec3 desired = (aim_ - p).normalized() * speed;
      Vec3 acceleration = 5.0 * (desired - v);
      if (acceleration.norm() > 90.0) acceleration *= 90.0 / acceleration.norm();
      missile.body->setExternalForce(0, vec(mass * (acceleration + Vec3(0, 0, kGravity))));
      if (v.norm() > 1.0) {
        missile.body->setOrientation(Eigen::Quaterniond::FromTwoVectors(Vec3::UnitZ(), v.normalized()));
        missile.body->setAngularVelocity(vec(Vec3::Zero()));
      }
    }
  }

  void explodeMissile(Missile& missile, size_t struck, const Vec3& p, const Vec3& v) {
    world_->removeObject(missile.body);
    missile.body = nullptr;
    if (Tank* target = tankAt(struck)) {
      raisim::ProjectileHit hit;
      hit.objectIndex = struck;
      hit.localIdx = target->hullBody;
      hit.position = vec(p);
      hit.velocity = vec(v.norm() > 1.0 ? v : Vec3(weapon(WeaponId::Missile).speed, 0, 0));
      hit.normal = vec(-hit.velocity.e().normalized());
      applyHit(*target, resolveHit(*target, hit, WeaponId::Missile, 1.0, random_), true, p, -v);
    }
    raisim::ExplosionProperties blast;
    blast.chargeMass = weapon(WeaponId::Missile).charge;
    blast.craterRadius = 1.0;
    blast.craterDepth = 0.3;
    world_->addExplosion(vec(p), blast);
  }

  void callArtillery(const Vec3& target) {
    const double incoming = now() + 3.0;
    for (int k = 0; k < 6; ++k) {
      const Vec3 point = target + Vec3(uniform(random_, -9.0, 9.0), uniform(random_, -9.0, 9.0), 0.0);
      const Vec3 landing(point.x(), point.y(), height(point.x(), point.y()));
      constexpr double kFlight = 2.2;
      const Vec3 origin = landing + Vec3(-120.0, 40.0, 280.0);
      const Vec3 velocity = (landing - origin) / kFlight + Vec3(0, 0, 0.5 * kGravity * kFlight);
      strikes_.push_back({incoming - kFlight + 0.35 * k, origin, velocity});
      addMarker(landing, incoming + 0.35 * k + 0.3, {0.4f, 1.0f, 0.4f, 0.28f});
    }
    notify("ARTILLERY STRIKE INBOUND", rgba(weapon(WeaponId::Artillery).color));
  }

  bool smokeBlocks(const Vec3& from, const Vec3& to) const {
    for (const auto& zone : smoke_) {
      const Vec3 d = to - from;
      const double s = std::clamp((zone.center - from).dot(d) / std::max(1e-6, d.squaredNorm()), 0.0, 1.0);
      if ((from + s * d - zone.center).head<2>().norm() < 6.5) return true;
    }
    return false;
  }

  void drive(Tank& enemy, Tank* hero, double frameTime) {
    if (!enemy.system) return;
    const double t = now();
    const VehicleSpec& spec = *enemy.spec;
    const LevelSpec& level = levelSpec();
    const Vec3 position = enemy.position();
    Vec3 goal = position;
    double throttleLimit = 1.0;
    bool faceHero = false;
    if (hero) {
      const Vec3 toHero = hero->position() - position;
      const double distance = toHero.head<2>().norm();
      const Vec3 radial = distance > 1e-3 ? Vec3(toHero.x(), toHero.y(), 0).normalized() : Vec3::UnitX();
      const Vec3 tangent(-radial.y(), radial.x(), 0.0);
      if (t > enemy.nextStrafeSwitch) {
        enemy.strafe = -enemy.strafe;
        enemy.nextStrafeSwitch = t + uniform(random_, 4.0, 9.0);
      }
      const double approach = std::clamp((distance - spec.range) / 8.0, -1.0, 1.0);
      switch (spec.role) {
        case Role::Heavy:
        case Role::Boss:
        case Role::Destroyer:
          goal = position + 10.0 * approach * radial;
          faceHero = std::abs(approach) < 0.6;
          break;
        case Role::Artillery:
          goal = position + 10.0 * approach * radial;
          throttleLimit = t - enemy.lastShot < 2.0 ? 0.0 : 1.0;
          break;
        default:
          goal = position + 10.0 * (approach * radial + (1.0 - std::abs(approach)) * enemy.strafe * tangent);
          break;
      }
      const bool moving = enemy.velocity().norm() > 2.0;
      const double spreadScale = level.enemyAccuracy * (1.0 + distance / 50.0) * (moving ? 1.4 : 1.0);
      const double reload = spec.reload * level.enemyReload * (t < enemy.gunDamagedUntil ? 2.0 : 1.0);
      const bool clear = !smokeBlocks(position, hero->position()) && t - enemy.born > 1.5 && phase_ == Phase::Playing;
      if (spec.role == Role::Artillery) {
        const double flight = distance / 25.0;
        const Vec3 predicted = hero->position() + hero->velocity() * flight * 0.6;
        enemy.aimAt(predicted, 600.0);
        enemy.gunAim = -0.35;
        if (clear && distance < spec.range + 10.0 && t - enemy.lastShot > reload) {
          const double a = uniform(random_, 0.0, 2 * kPi), r = spec.spread * spreadScale * std::sqrt(uniform(random_, 0.0, 1.0));
          launchMortar(enemy, predicted + Vec3(r * std::cos(a), r * std::sin(a), 0.0));
          enemy.lastShot = t;
        }
      } else {
        const WeaponSpec& gun = weapon(spec.gun);
        const double flight = distance / gun.speed;
        const Vec3 lead = hero->position() + hero->velocity() * flight * 0.7 + Vec3(0, 0, 0.6);
        const double error = enemy.aimAt(lead, gun.speed);
        const double interval = spec.role == Role::Scout ? (enemy.salvo % 24 < 14 ? reload : 0.6) : reload;
        const double tolerance = spec.fixedTurret ? 0.03 : spec.role == Role::Scout ? 0.12 : 0.06;
        if (clear && error < tolerance && distance < spec.range * 2.2 && t - enemy.lastShot > interval) {
          launchShell(enemy, spec.gun, false, spec.gunPower, spec.spread * spreadScale);
          enemy.lastShot = t;
          ++enemy.salvo;
          if (spec.role == Role::Boss && enemy.salvo % 2 == 1) enemy.lastShot = t - reload + 0.45;
        }
      }
      if (faceHero || spec.fixedTurret) goal = position + 10.0 * radial + 10.0 * approach * radial;
    } else {
      goal = position + 10.0 * Vec3(std::cos(enemy.heading() + 0.3), std::sin(enemy.heading() + 0.3), 0.0);
      throttleLimit = 0.4;
    }
    goal.x() = std::clamp(goal.x(), -kArena + 5.0, kArena - 5.0);
    goal.y() = std::clamp(goal.y(), -kArena + 5.0, kArena - 5.0);
    const double desired = std::atan2(goal.y() - position.y(), goal.x() - position.x());
    double turnError = wrapAngle(desired - enemy.heading());
    double throttle = std::max(0.0, std::cos(turnError)) * throttleLimit;
    if ((goal - position).head<2>().norm() < 2.0) throttle = 0.0;
    if (throttle > 0.3 && enemy.velocity().head<2>().norm() < 0.4 && t > enemy.tracksBrokenUntil)
      enemy.stuckTime += frameTime;
    else
      enemy.stuckTime = 0.0;
    if (enemy.stuckTime > 2.5) {
      enemy.reverseUntil = t + 1.5;
      enemy.stuckTime = 0.0;
    }
    if (t < enemy.reverseUntil) {
      throttle = -0.8;
      turnError = enemy.strafe;
    }
    enemy.command(throttle, 1.8 * turnError, 1.0, t);
  }

  void applyHit(Tank& target, const HitResult& result, bool fromPlayer, const Vec3& at, const Vec3& from) {
    const double t = now();
    if (result.tracks) target.tracksBrokenUntil = t + 5.0;
    if (result.gun) target.gunDamagedUntil = t + 8.0;
    applyDamage(target, result.damage, fromPlayer, from, result.penetrated);
    if (result.penetrated) viewer_.addExplosionEffect({glm::vec3(at.x(), at.y(), at.z()), 0.06f, false, {0.2f, 0.2f, 0.2f}});
    const bool isPlayer = &target == player_.get();
    if (fromPlayer && !isPlayer) {
      ++stats_.hits;
      if (result.penetrated) ++stats_.penetrations;
      hitMarker_ = t;
      hitMarkerColor_ = result.color;
    }
    if ((fromPlayer || isPlayer) && result.label[0]) {
      std::string text = result.label;
      if (result.damage >= 1.0) text += "  -" + std::to_string(int(std::round(result.damage)));
      floating_.push_back({text, isPlayer ? IM_COL32(255, 90, 80, 255) : result.color, t, at});
    }
  }

  void applyDamage(Tank& target, double amount, bool fromPlayer, const Vec3& from, bool penetrating) {
    if (amount <= 0.0 || target.health <= 0.0) return;
    if (&target == player_.get()) {
      if (phase_ != Phase::Playing) return;
      if (penetrating && reactive_ > 0) {
        --reactive_;
        amount *= 0.3;
        notify("REACTIVE ARMOR BLOCKED THE HIT", rgba(kBonusStyles[size_t(Bonus::Reactive)].color));
      }
      stats_.damageTaken += amount;
      hurtFrom_ = from;
    } else if (fromPlayer) {
      score_ += int(amount * 0.5);
    }
    target.health -= amount;
    target.hurtTime = now();
  }

  void destroy(Tank& tank) {
    const auto& gc = tank.system->getGeneralizedCoordinate().e();
    const Vec3 position = gc.head<3>();
    const Eigen::Quaterniond orientation(gc[3], gc[4], gc[5], gc[6]);
    const Vec3 velocity = tank.velocity();
    raisim::Vec<3> turretPosition;
    raisim::Mat<3, 3> turretRotation;
    tank.system->getLink("turret_link").getPose(turretPosition, turretRotation);
    const bool isPlayer = &tank == player_.get();
    if (!isPlayer) {
      score_ += tank.spec->score;
      ++stats_.kills;
      killFeed_.push_back({std::string(tank.spec->name) + " destroyed  +" + std::to_string(tank.spec->score),
                           rgba(tank.spec->marker), now()});
      if (killFeed_.size() > 5) killFeed_.erase(killFeed_.begin());
    }
    const VehicleModel& model = tank.model;
    const Vec3 hullPosition = position;
    const Eigen::Matrix3d turretRot = turretRotation.e();
    world_->removeObject(tank.system);
    tank.system = nullptr;
    tank.health = 0.0;
    Wreck wreck;
    const Vec3 hullSize = model.hullMax - model.hullMin, turretSize(model.turretMax.x() - model.turretMin.x(),
        model.turretMax.y() - model.turretMin.y(), std::min(model.turretMax.z(), model.turretPivot.z() + 1.1) - model.turretPivot.z());
    const Vec3 hullCenter = 0.5 * (model.hullMin + model.hullMax) - Vec3(0, 0, model.axleHeight);
    auto* hull = world_->addBox(hullSize.x(), hullSize.y(), hullSize.z(), 0.7 * tank.spec->mass);
    hull->setPosition(hullPosition + orientation * hullCenter);
    hull->setOrientation(orientation);
    hull->setVelocity(velocity, Vec3::Zero());
    const Vec3 turretCenter(0.5 * (model.turretMin.x() + model.turretMax.x()) - model.turretPivot.x(),
                            0.5 * (model.turretMin.y() + model.turretMax.y()) - model.turretPivot.y(), 0.5 * turretSize.z());
    auto* top = world_->addBox(turretSize.x(), turretSize.y(), turretSize.z(), 0.24 * tank.spec->mass);
    raisim::FluidCoefficients still;
    still.enabled = false;
    world_->setFluidCoefficients(hull, still);
    world_->setFluidCoefficients(top, still);
    top->setPosition(turretPosition.e() + turretRot * turretCenter + Vec3(0, 0, 0.15));
    top->setOrientation(Eigen::Matrix3d(turretRot));
    top->setVelocity(velocity + Vec3(uniform(random_, -2, 2), uniform(random_, -2, 2), uniform(random_, 6, 9)),
                     Vec3(uniform(random_, -2.5, 2.5), uniform(random_, -2.5, 2.5), uniform(random_, -2, 2)));
    const std::string folder = assets_ + "/tank_survival/vehicles/" + tank.spec->model;
    const glm::vec4 burnt(0.22f, 0.2f, 0.19f, 1.0f);
    for (auto [body, mesh, offset] : {std::tuple<raisim::SingleBodyObject*, const char*, Vec3>{hull, "hull", -hullCenter - Vec3(0, 0, model.axleHeight)},
                                      {top, "turret", -turretCenter - model.turretPivot}}) {
      body->setAppearance("hidden");
      const std::string name = "wreck_" + std::to_string(nextName_++);
      viewer_.addVisualMesh(name, folder + "/" + mesh + ".gltf", glm::dvec3(1.0), burnt);
      wreck.parts.push_back(body);
      wreck.visuals.push_back(name);
      wreck.offsets.push_back(offset);
    }
    wreck.removeAt = now() + 30.0;
    wrecks_.push_back(std::move(wreck));
    const double s = tank.scale();
    // The wreck burns on with black smoke until it is cleared away.
    const Vec3 fire = hullPosition + orientation * Vec3(model.hullMin.x() * 0.3, 0.0, hullSize.z() - model.axleHeight);
    viewer_.addFireEffect({glm::vec3(fire.x(), fire.y(), fire.z()), float(0.9 * s), 28.0f});
    raisim::ExplosionProperties blast;
    blast.chargeMass = 2.5 * s;
    blast.craterRadius = 1.5 * s;
    blast.craterDepth = 0.5;
    world_->addExplosion(vec(position + Vec3(0, 0, 0.8 * s)), blast);
    for (double delay : {0.35, 0.8, 1.3})
      pending_.push_back({now() + delay, position + Vec3(uniform(random_, -1.5, 1.5), uniform(random_, -1.5, 1.5), 0.6), 0.7});
    if (!isPlayer && uniform(random_, 0.0, 1.0) < 0.4) {
      const Vec3 drop(position.x() + uniform(random_, -3, 3), position.y() + uniform(random_, -3, 3), 0.0);
      spawnPickupAt(randomBonus(), Vec3(drop.x(), drop.y(), height(drop.x(), drop.y())));
    }
  }

  void undress(raisim::SingleBodyObject* body) {
    for (const auto& dressing : dressings_)
      if (dressing.body == body) viewer_.removeVisualObject(dressing.visual);
    std::erase_if(dressings_, [&](const Dressing& dressing) { return dressing.body == body; });
  }

  void removeWreck(Wreck& wreck) {
    for (size_t i = 0; i < wreck.parts.size(); ++i) {
      if (wreck.parts[i]) world_->removeObject(wreck.parts[i]);
      viewer_.removeVisualObject(wreck.visuals[i]);
    }
    wreck.parts.clear();
    wreck.visuals.clear();
  }

  std::vector<Tank*> tanks() {
    std::vector<Tank*> list;
    if (player_ && player_->system) list.push_back(player_.get());
    for (auto& enemy : enemies_)
      if (enemy->system) list.push_back(enemy.get());
    return list;
  }

  Tank* tankAt(size_t index) {
    if (index == size_t(-1)) return nullptr;
    for (Tank* tank : tanks())
      if (tank->system->getIndexInWorld() == index) return tank;
    return nullptr;
  }

  const Shell* findShell(size_t id) const {
    for (auto it = shells_.rbegin(); it != shells_.rend(); ++it)
      if (it->id == id) return &*it;
    return nullptr;
  }

  std::shared_ptr<raisim::World> world_;
  Viewer& viewer_;
  std::mt19937 random_;
  std::string assets_;
  std::vector<std::string> urdfs_;
  std::vector<VehicleModel> models_;
  int level_ = 0;
  Phase phase_ = Phase::Menu;
  raisim::HeightMap* terrain_ = nullptr;
  TerrainVisuals terrainVisuals_;
  Town town_;
  Scenery scenery_;
  std::vector<Dressing> dressings_;
  std::vector<std::string> visuals_;
  raisin::Material concrete_;
  int quality_ = 2;
  std::vector<raisim::Object*> props_;
  std::vector<Building> buildings_;
  std::unique_ptr<Tank> player_;
  std::vector<std::unique_ptr<Tank>> enemies_;
  std::vector<Barrel> barrels_;
  std::vector<Wreck> wrecks_;
  std::vector<Pickup> pickups_;
  std::vector<Shell> shells_;
  std::vector<Missile> missiles_;
  std::vector<Strike> strikes_;
  std::vector<Marker> markers_;
  std::vector<PendingBlast> pending_;
  std::vector<SmokeZone> smoke_;
  std::vector<Notice> notices_, floating_, killFeed_;
  std::vector<Role> queue_;
  std::array<int, kWeaponCount> ammo_{};
  std::array<double, kWeaponCount> lastFire_{};
  WeaponId selected_ = WeaponId::AP;
  Vec3 aim_ = Vec3::Zero(), hurtFrom_ = Vec3::UnitX();
  Stats stats_;
  size_t nextHit_ = 0, nextExplosion_ = 0, nextName_ = 0;
  int score_ = 0, wave_ = 0, smokeCharges_ = 0, reactive_ = 0;
  double nextSpawn_ = 0.0, nextWave_ = 0.0, nextPickup_ = 0.0, started_ = 0.0, endedAt_ = -1.0;
  double overdriveUntil_ = -1.0, artilleryRecharge_ = 0.0, lastSmoke_ = -100.0, lastEmpty_ = -100.0;
  double shake_ = 0.0, hitMarker_ = -100.0;
  ImU32 hitMarkerColor_ = IM_COL32(255, 255, 255, 255);
  bool waveCleared_ = false, reinforced_ = false;
};

}  // namespace tank
