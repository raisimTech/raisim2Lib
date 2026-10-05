// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerSession.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <system_error>

#include "rayrai/RaisimTcpCommon.hpp"

namespace raisin::tcp_viewer
{
namespace
{

template <typename T>
bool writeBinaryPod(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
  return static_cast<bool>(output);
}

} // namespace

bool SessionRecorder::open(const std::filesystem::path& path, std::string& status) {
  close();
  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      status = "session record failed: cannot create directory";
      return false;
    }
  }
  output_.open(path, std::ios::binary);
  if (!output_) {
    status = "session record failed: cannot open " + path.string();
    return false;
  }
  output_.write(kSessionMagic, std::strlen(kSessionMagic));
  if (!output_) {
    status = "session record failed: cannot write header";
    close();
    return false;
  }
  path_ = path;
  frames_ = 0;
  bytes_ = 0;
  start_ = std::chrono::steady_clock::now();
  status = "recording " + path.string();
  return true;
}

void SessionRecorder::close() {
  if (output_.is_open()) {
    output_.close();
  }
}

bool SessionRecorder::active() const {
  return output_.is_open();
}

size_t SessionRecorder::frameCount() const {
  return frames_;
}

size_t SessionRecorder::byteCount() const {
  return bytes_;
}

std::string SessionRecorder::pathString() const {
  return path_.string();
}

bool SessionRecorder::record(const std::vector<char>& payload,
                             std::chrono::steady_clock::time_point now,
                             std::string& status) {
  if (!active()) {
    return false;
  }
  const auto micros = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count());
  if (payload.size() > std::numeric_limits<uint32_t>::max()) {
    status = "session record failed: payload too large";
    close();
    return false;
  }
  const uint32_t size = static_cast<uint32_t>(payload.size());
  if (!writeBinaryPod(output_, micros) || !writeBinaryPod(output_, size)) {
    status = "session record failed: write error";
    close();
    return false;
  }
  if (!payload.empty()) {
    output_.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }
  if (!output_) {
    status = "session record failed: write error";
    close();
    return false;
  }
  ++frames_;
  bytes_ += payload.size();
  return true;
}

bool SessionRecorder::recordReset(std::chrono::steady_clock::time_point now,
                                  std::string& status) {
  return record({}, now, status);
}

bool loadSessionFile(const std::filesystem::path& path, std::vector<RecordedFrame>& frames,
                     std::string& status) {
  frames.clear();
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    status = "replay failed: cannot open " + path.string();
    return false;
  }
  std::string magic(std::strlen(kSessionMagic), '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != kSessionMagic) {
    status = "replay failed: unsupported session file";
    return false;
  }
  const auto fail = [&](const char* message) {
    status = message;
    frames.clear();
    return false;
  };
  while (true) {
    RecordedFrame frame;
    uint32_t size = 0;
    input.read(reinterpret_cast<char*>(&frame.timeMicros), sizeof(frame.timeMicros));
    if (!input) {
      if (input.gcount() == 0 && input.eof()) {
        break;
      }
      return fail("replay failed: truncated frame header");
    }
    input.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!input) {
      return fail("replay failed: truncated frame header");
    }
    if (size > static_cast<uint32_t>(kMaxMessageBytes)) {
      return fail("replay failed: frame exceeds max message size");
    }
    frame.payload.resize(size);
    if (size > 0) {
      input.read(frame.payload.data(), size);
      if (!input) {
        return fail("replay failed: truncated frame");
      }
    }
    frames.push_back(std::move(frame));
  }
  status = "loaded replay " + path.string() + " (" + std::to_string(frames.size()) + " frames)";
  return !frames.empty();
}

size_t findSessionFrameAtOrAfter(const std::vector<RecordedFrame>& frames, uint64_t timeMicros) {
  return static_cast<size_t>(std::lower_bound(frames.begin(), frames.end(), timeMicros,
    [](const RecordedFrame& frame, uint64_t value) {
      return frame.timeMicros < value;
    }) - frames.begin());
}

} // namespace raisin::tcp_viewer
