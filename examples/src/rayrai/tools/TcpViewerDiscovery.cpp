// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "TcpViewerDiscovery.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <system_error>

#include "rayrai/RaisimTcpCommon.hpp"

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace raisin::tcp_viewer
{
namespace
{

bool parseLongStrict(const char* value, int base, long& out) {
  if (!value) {
    return false;
  }
  while (std::isspace(static_cast<unsigned char>(*value))) {
    ++value;
  }
  if (*value == '\0') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  out = std::strtol(value, &end, base);
  if (end == value || errno == ERANGE) {
    return false;
  }
  while (std::isspace(static_cast<unsigned char>(*end))) {
    ++end;
  }
  return *end == '\0';
}

bool parsePortStrict(const std::string& value, int& port) {
  long parsed = 0;
  if (!parseLongStrict(value.c_str(), 10, parsed) || parsed <= 0 || parsed > 65535) {
    return false;
  }
  port = static_cast<int>(parsed);
  return true;
}

#if defined(_WIN32)
bool ensureWinsockInitialized(std::string* error = nullptr) {
  static const int startupResult = []() {
    WSADATA data{};
    return WSAStartup(MAKEWORD(2, 2), &data);
  }();
  if (startupResult != 0) {
    if (error) {
      *error = "WSAStartup failed with error: " + std::to_string(startupResult);
    }
    return false;
  }
  return true;
}
#endif

void sortDiscoveredServers(std::vector<DiscoveredServer>& servers) {
  std::sort(servers.begin(), servers.end(), [](const DiscoveredServer& lhs,
                                               const DiscoveredServer& rhs) {
    if (lhs.remoteBeacon != rhs.remoteBeacon) {
      return !lhs.remoteBeacon;
    }
    if (lhs.bindHost != rhs.bindHost) {
      return lhs.bindHost < rhs.bindHost;
    }
    if (lhs.endpointPort != rhs.endpointPort) {
      return lhs.endpointPort < rhs.endpointPort;
    }
    return lhs.protocol < rhs.protocol;
  });
}

#if defined(__EMSCRIPTEN__)
EM_JS_DEPS(rayrai_discovery, "$stringToNewUTF8");
// Lines of "source<TAB>beacon payload": the page's in-browser examples and, when the page comes
// from serve.py, the beacons that server hears on the LAN.
EM_JS(char*, pageBeacons, (), {
  const discovery = Module['rayraiDiscovery'];
  return stringToNewUTF8(discovery ? discovery['beacons']() : '');
});
EM_JS(int, pageListsNetworkServers, (), {
  const discovery = Module['rayraiDiscovery'];
  return discovery && discovery['network']() ? 1 : 0;
});
#endif
} // namespace

bool isCompatibleDiscoveryVersion(int version) {
  return version == kProtocolVersion;
}

bool parseDiscoveryBeacon(const std::string& payload, int& port, int& version,
                          std::string& hostname,
                          std::unordered_map<std::string, std::string>& metadata) {
  std::istringstream input(payload);
  std::string magic;
  std::string portToken;
  if (!(input >> magic >> portToken)) {
    return false;
  }
  int parsedPort = 0;
  if (magic != kDiscoveryMagic || !parsePortStrict(portToken, parsedPort)) {
    return false;
  }

  metadata.clear();
  std::vector<std::string> legacyHostTokens;
  std::vector<std::string> tokens;
  std::string token;
  while (input >> token) {
    tokens.push_back(std::move(token));
  }

  version = 0;
  size_t firstMetadataToken = 0;
  if (!tokens.empty()) {
    long parsedVersion = 0;
    if (parseLongStrict(tokens.front().c_str(), 10, parsedVersion)) {
      if (parsedVersion < 0 || parsedVersion > std::numeric_limits<int>::max()) {
        return false;
      }
      version = static_cast<int>(parsedVersion);
      firstMetadataToken = 1;
    }
  }

  for (size_t i = firstMetadataToken; i < tokens.size(); ++i) {
    const std::string& part = tokens[i];
    const auto eq = part.find('=');
    if (eq == std::string::npos || eq == 0) {
      legacyHostTokens.push_back(part);
      continue;
    }
    metadata[part.substr(0, eq)] = part.substr(eq + 1);
  }

  const auto hostIt = metadata.find("hostname");
  if (hostIt != metadata.end() && !hostIt->second.empty()) {
    hostname = hostIt->second;
  } else if (!legacyHostTokens.empty()) {
    hostname.clear();
    for (const auto& part : legacyHostTokens) {
      if (!hostname.empty()) hostname += " ";
      hostname += part;
    }
  } else {
    hostname.clear();
  }

  port = parsedPort;
  return true;
}

class DiscoveryBeaconReceiver::Impl {
 public:
  ~Impl() { closeSocket(); }

  bool start(std::string& status) {
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
    if (socketFd_ != kInvalidSocket) {
      status = "LAN beacon listener active";
      return true;
    }

#if defined(_WIN32)
    std::string startupError;
    if (!ensureWinsockInitialized(&startupError)) {
      status = startupError;
      return false;
    }
    socketFd_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socketFd_ == INVALID_SOCKET) {
      status = "LAN beacon listener unavailable: WSA error " + std::to_string(WSAGetLastError());
      return false;
    }
#else
    socketFd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socketFd_ < 0) {
      status = std::string("LAN beacon listener unavailable: ") + std::strerror(errno);
      return false;
    }
#endif

    int opt = 1;
    setsockopt(socketFd_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&opt), sizeof(opt));
#if defined(SO_REUSEPORT)
    setsockopt(socketFd_, SOL_SOCKET, SO_REUSEPORT, reinterpret_cast<char*>(&opt), sizeof(opt));
#endif

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(kDiscoveryPort);
    if (bind(socketFd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
#if defined(_WIN32)
      status = "LAN beacon listener bind failed: WSA error " + std::to_string(WSAGetLastError());
#else
      status = std::string("LAN beacon listener bind failed: ") + std::strerror(errno);
#endif
      closeSocket();
      return false;
    }

#if defined(_WIN32)
    u_long nonBlocking = 1;
    if (ioctlsocket(socketFd_, FIONBIO, &nonBlocking) == SOCKET_ERROR) {
      status = "LAN beacon listener nonblocking setup failed: WSA error " +
               std::to_string(WSAGetLastError());
      closeSocket();
      return false;
    }
#else
    const int flags = fcntl(socketFd_, F_GETFL, 0);
    if (flags >= 0) {
      fcntl(socketFd_, F_SETFL, flags | O_NONBLOCK);
    }
#endif

    status = "LAN beacon listener active";
    return true;
#elif defined(__EMSCRIPTEN__)
    status = pageListsNetworkServers() ? "Listing servers from this page's local web server"
                                       : "A browser cannot hear LAN beacons; type a host and port above";
    return true;
#else
    status = "LAN beacon listener unavailable on this platform";
    return false;
#endif
  }

  bool poll() {
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
    if (socketFd_ == kInvalidSocket) {
      return false;
    }

    bool changed = false;
    const auto now = std::chrono::steady_clock::now();
    while (true) {
      char buffer[1024];
      sockaddr_in from{};
#if defined(_WIN32)
      int fromLen = sizeof(from);
      const int count = recvfrom(socketFd_, buffer, static_cast<int>(sizeof(buffer) - 1), 0,
                                 reinterpret_cast<sockaddr*>(&from), &fromLen);
      if (count == SOCKET_ERROR) {
        const int error = WSAGetLastError();
        if (error == WSAEINTR) {
          continue;
        }
        break;
      }
#else
      socklen_t fromLen = sizeof(from);
      const ssize_t count = recvfrom(socketFd_, buffer, sizeof(buffer) - 1, 0,
                                     reinterpret_cast<sockaddr*>(&from), &fromLen);
      if (count < 0) {
        if (errno == EINTR) {
          continue;
        }
        break;
      }
#endif
      if (count == 0) {
        break;
      }
      buffer[count] = '\0';

      char sourceHost[INET_ADDRSTRLEN]{};
      if (!inet_ntop(AF_INET, &from.sin_addr, sourceHost, sizeof(sourceHost))) {
        continue;
      }

      changed = record(sourceHost, buffer, now) || changed;
    }

    return expire(now) || changed;
#elif defined(__EMSCRIPTEN__)
    bool changed = false;
    const auto now = std::chrono::steady_clock::now();
    char* text = pageBeacons();
    std::istringstream lines(text ? text : "");
    std::free(text);
    std::string line;
    while (std::getline(lines, line)) {
      const size_t tab = line.find('\t');
      if (tab != std::string::npos) changed = record(line.substr(0, tab), line.substr(tab + 1), now) || changed;
    }
    return expire(now) || changed;
#else
    return false;
#endif
  }

  std::vector<DiscoveredServer> servers() const {
    std::vector<DiscoveredServer> entries;
    entries.reserve(beacons_.size());
    for (const auto& item : beacons_) {
      entries.push_back(item.second.server);
    }
    sortDiscoveredServers(entries);
    return entries;
  }

 private:
  struct BeaconRecord {
    DiscoveredServer server;
    std::chrono::steady_clock::time_point lastSeen;
  };

  // Notes a beacon from source; returns true when it changed the list.
  bool record(const std::string& source, const std::string& payload,
              std::chrono::steady_clock::time_point now) {
    int port = 0;
    int version = 0;
    std::string hostname;
    std::unordered_map<std::string, std::string> metadata;
    if (!parseDiscoveryBeacon(payload, port, version, hostname, metadata) ||
        !isCompatibleDiscoveryVersion(version)) {
      return false;
    }
    auto& entry = beacons_[source + ":" + std::to_string(port)];
    const bool isNew = entry.server.endpointHost.empty();
    DiscoveredServer updated;
    updated.endpointHost = source;
    updated.endpointPort = port;
    updated.bindHost = source;
    updated.protocol = "raisim beacon";
    updated.metadata = metadata;
    updated.remoteBeacon = true;
    updated.lastSeen = now;
    const auto exeIt = metadata.find("exe");
    const std::string serverName =
      exeIt != metadata.end() && !exeIt->second.empty() ? exeIt->second : "RaisimServer";
    updated.process = hostname.empty() ? serverName : serverName + " on " + hostname;
    if (version > 0) {
      updated.process += " (protocol " + std::to_string(version) + ")";
    }
    const bool changed = isNew || entry.server.endpointHost != updated.endpointHost ||
                         entry.server.endpointPort != updated.endpointPort ||
                         entry.server.process != updated.process ||
                         entry.server.metadata != updated.metadata;
    entry.server = updated;
    entry.lastSeen = now;
    return changed;
  }

  bool expire(std::chrono::steady_clock::time_point now) {
    bool changed = false;
    for (auto it = beacons_.begin(); it != beacons_.end();) {
      if (now - it->second.lastSeen > kDiscoveryBeaconTimeout) {
        it = beacons_.erase(it);
        changed = true;
      } else {
        ++it;
      }
    }
    return changed;
  }

  void closeSocket() {
#if defined(_WIN32)
    if (socketFd_ != kInvalidSocket) {
      closesocket(socketFd_);
      socketFd_ = kInvalidSocket;
    }
#elif defined(__linux__) || defined(__APPLE__)
    if (socketFd_ != kInvalidSocket) {
      close(socketFd_);
      socketFd_ = kInvalidSocket;
    }
#endif
  }

#if defined(_WIN32)
  using SocketHandle = SOCKET;
  static constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
  using SocketHandle = int;
  static constexpr SocketHandle kInvalidSocket = -1;
#endif
  SocketHandle socketFd_ = kInvalidSocket;
  std::unordered_map<std::string, BeaconRecord> beacons_;
};

DiscoveryBeaconReceiver::DiscoveryBeaconReceiver()
  : impl_(std::make_unique<Impl>()) {
}

DiscoveryBeaconReceiver::~DiscoveryBeaconReceiver() = default;

bool DiscoveryBeaconReceiver::start(std::string& status) {
  return impl_->start(status);
}

bool DiscoveryBeaconReceiver::poll() {
  return impl_->poll();
}

std::vector<DiscoveredServer> DiscoveryBeaconReceiver::servers() const {
  return impl_->servers();
}

} // namespace raisin::tcp_viewer
