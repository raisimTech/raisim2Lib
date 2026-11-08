// A photoreal city block grid loaded from rayrai_city.rscene, with an ANYmal C
// quadruped standing on the street next to roadworks. RaiSim builds the
// streets, buildings, parked cars, street furniture and traffic cones from the
// file; the .rscene reader creates no articulated systems, so the robot is
// added here (photoreal_scene.hpp, shared with rayrai_warehouse).
#include "photoreal_scene.hpp"

int main(int argc, char* argv[]) {
  // The robot stands at ROBOT_XY of examples/tools/generate_city_rscene.py, which keeps parked
  // cars and street furniture away from this spot.
  return photoreal_scene::run(argc, argv, {"rayrai_city", "RaiSim city", "city/rayrai_city.rscene", "city",
                                           -36.0, -1.6, 2.6});
}
