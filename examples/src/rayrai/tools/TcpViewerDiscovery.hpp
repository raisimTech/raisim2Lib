// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace raisin::tcp_viewer
{

inline constexpr int kDiscoveryPort = 59312;
// 239.255.59.12, joined on 127.0.0.1: a loopback RaisimServer also sends its beacon to this group,
// because its datagram to 127.0.0.1 reaches only one of the listeners sharing kDiscoveryPort.
inline constexpr uint32_t kDiscoveryLoopbackGroup = 0xEFFF3B0Cu;
inline constexpr const char* kDiscoveryMagic = "RAISIM_TCP_DISCOVERY_V1";
inline constexpr std::chrono::seconds kDiscoveryBeaconTimeout{8};

struct DiscoveredServer {
  std::string endpointHost;
  int endpointPort = 0;
  std::string bindHost;
  std::string process;
  std::string protocol;
  std::unordered_map<std::string, std::string> metadata;
  bool remoteBeacon = false;
  std::chrono::steady_clock::time_point lastSeen;
};

bool isCompatibleDiscoveryVersion(int version);
bool parseDiscoveryBeacon(const std::string& payload, int& port, int& version,
                          std::string& hostname,
                          std::unordered_map<std::string, std::string>& metadata);

class DiscoveryBeaconReceiver {
 public:
  DiscoveryBeaconReceiver();
  ~DiscoveryBeaconReceiver();

  DiscoveryBeaconReceiver(const DiscoveryBeaconReceiver&) = delete;
  DiscoveryBeaconReceiver& operator=(const DiscoveryBeaconReceiver&) = delete;

  bool start(std::string& status);
  bool poll();
  std::vector<DiscoveredServer> servers() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace raisin::tcp_viewer
