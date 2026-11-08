// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace raisin::tcp_viewer {

// A process the viewer launched. Its standard output and error go to a log file, and its
// standard input is a pipe the viewer holds open, so the child can notice the viewer is gone.
// On Windows the child also ends with the viewer (a kill-on-close job object). Use it from one
// thread.
class ChildProcess {
 public:
  ChildProcess();
  ~ChildProcess();
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  // Launches executable with arguments (paths, so that file names keep their encoding on
  // Windows). Stops a process this object launched before.
  bool start(const std::filesystem::path& executable,
             const std::vector<std::filesystem::path>& arguments,
             const std::filesystem::path& log, std::string& error);
  // True from start() until the process exits or is stopped.
  bool running();
  // The exit status of a process that ended by itself, -1 otherwise.
  int exitCode() const;
  int64_t pid() const;
  // Asks the process to end, then kills it if it is still running 200 ms later.
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The last maxBytes of a file, empty when it cannot be read.
std::string fileTail(const std::filesystem::path& path, size_t maxBytes);

}  // namespace raisin::tcp_viewer
