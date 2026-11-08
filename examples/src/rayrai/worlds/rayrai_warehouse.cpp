// A photoreal warehouse loaded from rayrai_warehouse.rscene, with an ANYmal C
// quadruped standing in an aisle between pallet racks. RaiSim builds the
// building, rack and load colliders, the props and a few loose cartons and
// crates from the file; the .rscene reader creates no articulated systems, so
// the robot is added here (photoreal_scene.hpp, shared with rayrai_city).
#include "photoreal_scene.hpp"

int main(int argc, char* argv[]) {
  // The robot stands at ROBOT_XY of examples/tools/generate_warehouse_rscene.py, which keeps this
  // spot of the middle aisle free.
  return photoreal_scene::run(argc, argv, {"rayrai_warehouse", "RaiSim warehouse",
                                           "warehouse/rayrai_warehouse.rscene", "warehouse", 4.4, 0.3, -0.6});
}
