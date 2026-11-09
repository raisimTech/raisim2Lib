// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "TcpViewerDiscovery.hpp"

namespace raisin::tcp_viewer {

// The file, next to the example executables, that lists the RaisimServer examples among them.
inline constexpr const char* kServerExampleManifest = "raisim_server_examples.tsv";
// How long a started example may run before it tells its port (with its discovery beacon).
inline constexpr std::chrono::seconds kExampleStartTimeout{60};

struct ServerExample {
  std::string name;      // executable name without extension
  std::string category;  // e.g. "deformable"
  std::string title;     // e.g. "Cloth on a Slope"
};

// Parses a manifest: one "name<TAB>category<TAB>title" line per example, '#' starts a comment.
std::vector<ServerExample> parseServerExampleManifest(const std::string& text);

// The text without the terminal control sequences (such as colors) a program writes to its output.
std::string stripTerminalEscapes(const std::string& text);

// Runs one server example at a time, so the viewer can connect to it. Natively the examples are
// the executables listed by the manifest in a directory and run as child processes; in the
// browser they are compiled into the viewer and run on a worker thread.
class ExampleLauncher {
 public:
  // Stopped is an example that terminate() ended: like Exited, it keeps its output. TimedOut is
  // one that poll() ended because it did not tell its port within the start timeout.
  enum class State { Idle, Starting, Running, Exited, Stopped, TimedOut };

  explicit ExampleLauncher(const std::filesystem::path& directory);
  ~ExampleLauncher();
  ExampleLauncher(const ExampleLauncher&) = delete;
  ExampleLauncher& operator=(const ExampleLauncher&) = delete;

  // The examples that can be started, in manifest order.
  const std::vector<ServerExample>& examples() const;
  // Where the examples come from, why none are listed when the list is empty, or empty when
  // there is no manifest: a viewer installed without examples.
  const std::string& source() const;

  bool start(const std::string& name, std::string& error);
  // Ends the started example, if any, and forgets it and its output.
  void stop();
  // Ends the example if it still runs, but keeps it and its output to read: the state becomes
  // Stopped, or Exited if the example had ended by itself.
  void terminate();
  // Follows the running example. beacons are the discovered servers; the one from this computer
  // with the example's process id tells its port. An example that has not told its port within
  // the start timeout is ended and becomes TimedOut.
  void poll(const std::vector<DiscoveredServer>& beacons);
  // How long a started example may take to tell its port; kExampleStartTimeout by default.
  void setStartTimeout(std::chrono::steady_clock::duration timeout);
  std::chrono::steady_clock::duration startTimeout() const;

  State state() const;
  // Whether the started example is Starting or Running.
  bool active() const;
  // The started example, or nullptr when none was started.
  const ServerExample* example() const;
  // The process (or, in the browser, thread) id of the started example, 0 when none runs.
  int64_t pid() const;
  // The host and port to connect to once the example is Running.
  const std::string& host() const;
  int port() const;
  // Whether host:port reaches the running example: its own endpoint, or an address of this
  // computer that one of the beacons carrying the example's process id came from (process ids
  // name processes only on their own computer). Name loopback hosts as 127.0.0.1.
  bool serves(const std::string& host, int port, const std::vector<DiscoveredServer>& beacons) const;
  // Whether a connection to host:port has to end the example first: it is active and does not
  // serve that endpoint, so its exit would end the new session. An example that has ended keeps
  // its output until a connection is made.
  bool conflictsWith(const std::string& host, int port, const std::vector<DiscoveredServer>& beacons) const;
  // The end of the example's output, and its exit code once it has Exited (-1 if unknown or
  // Stopped).
  std::string output() const;
  int exitCode() const;
  std::chrono::steady_clock::duration age() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace raisin::tcp_viewer
