#pragma once

#include <iostream>

namespace raisim_examples {

inline void printRaisimUnrealMapHint(const char* mapName = nullptr) {
  std::cerr
    << "\n"
    << "[raisim] Note: this is a RaisimUnreal map example. Use RaisimUnreal to see the full map visualization.\n";
  if (mapName)
    std::cerr << "        Map: " << mapName << "\n";
  std::cerr << std::endl;
}

}  // namespace raisim_examples

