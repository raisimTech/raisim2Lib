// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// Checks of Tank Survival's arrows to enemies out of view and of its hit notes, which need no window.

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

}  // namespace

int main() {
  arrowsPointAtEnemiesOutOfView();
  burstsOfHitsGatherPerVehicle();
  qualityStepsDownWhileFramesAreSlow();
  if (failures == 0) std::printf("tank_survival_test: all checks passed\n");
  return failures == 0 ? 0 : 1;
}
