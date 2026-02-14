#pragma once

#include <string>

#include "raisim/Path.hpp"

inline std::string rayraiRscPath(char* argv0, const std::string& relative) {
  auto binaryPath = raisim::Path::setFromArgv(argv0);
  const std::string sep = raisim::Path::separator();
  std::string path =
    binaryPath.getDirectory().getString() + sep + "rsc" + sep + relative;
  raisim::Path::replaceAntiSeparatorWithSeparator(path);
  return path;
}
