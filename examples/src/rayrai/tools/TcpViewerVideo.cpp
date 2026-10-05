// Copyright (c) 2026 Raion Robotics Inc.
// All rights reserved.

#include "TcpViewerVideo.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>

#if defined(_WIN32)
#include <cstdio>
#define RAYRAI_POPEN _popen
#define RAYRAI_PCLOSE _pclose
#define RAYRAI_PIPE_MODE "wb"
#else
#include <sys/wait.h>

#include <csignal>
#define RAYRAI_POPEN popen
#define RAYRAI_PCLOSE pclose
#define RAYRAI_PIPE_MODE "w"
#endif

namespace raisin::tcp_viewer
{

namespace
{

#if defined(_WIN32)
constexpr const char* kFfmpegExecutableName = "ffmpeg.exe";
#else
constexpr const char* kFfmpegExecutableName = "ffmpeg";
#endif

std::string environmentValue(const char* name) {
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

bool isExecutableFile(const std::filesystem::path& path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    return false;
  }
#if defined(_WIN32)
  return true;
#else
  const auto permissions = std::filesystem::status(path, ec).permissions();
  if (ec) {
    return false;
  }
  using std::filesystem::perms;
  return (permissions & (perms::owner_exec | perms::group_exec | perms::others_exec)) !=
         perms::none;
#endif
}

std::vector<std::filesystem::path> executableSearchDirectories() {
  std::vector<std::filesystem::path> directories;
  const std::string pathVariable = environmentValue("PATH");
#if defined(_WIN32)
  constexpr char kSeparator = ';';
#else
  constexpr char kSeparator = ':';
#endif
  std::istringstream stream(pathVariable);
  std::string entry;
  while (std::getline(stream, entry, kSeparator)) {
    if (!entry.empty()) {
      directories.emplace_back(entry);
    }
  }
#if !defined(_WIN32)
  // A GUI process launched from a desktop environment often inherits a minimal
  // PATH that omits the package-manager prefixes, so check them explicitly.
  for (const char* prefix : {"/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
                             "/snap/bin"}) {
    directories.emplace_back(prefix);
  }
#endif
  return directories;
}

/** The encoder options and output path shared by every ffmpeg encode. */
void appendEncodeOutputArguments(std::vector<std::string>& arguments,
                                 const VideoEncoderSettings& sanitized,
                                 const std::filesystem::path& output) {
  arguments.insert(arguments.end(), {
    // ffmpeg's yuv420p encoders need even dimensions; crop rather than stretch.
    "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
    "-c:v", sanitized.codec,
    "-crf", std::to_string(sanitized.quality),
    "-pix_fmt", sanitized.pixelFormat,
    // Fast-start metadata so the file plays while it is still being copied.
    "-movflags", "+faststart",
    output.string(),
  });
}

/**
 * @brief Turn a pclose()/system() return value into a process exit code.
 *
 * On POSIX both return a wait status, so a plain exit code 3 arrives as 768.
 * Reporting the raw value would make every encoder error message wrong.
 */
int processExitCode(int waitStatus) {
  if (waitStatus == -1) {
    return -1;
  }
#if defined(_WIN32)
  return waitStatus;
#else
  if (WIFEXITED(waitStatus)) {
    return WEXITSTATUS(waitStatus);
  }
  if (WIFSIGNALED(waitStatus)) {
    // Report a signal the way a shell does, so "killed by SIGKILL" is legible.
    return 128 + WTERMSIG(waitStatus);
  }
  return waitStatus;
#endif
}

std::string formatFrameRate(double framesPerSecond) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6) << framesPerSecond;
  return stream.str();
}

} // namespace

VideoEncoderSettings sanitizeVideoEncoderSettings(const VideoEncoderSettings& settings) {
  VideoEncoderSettings sanitized = settings;
  if (!std::isfinite(sanitized.framesPerSecond)) {
    sanitized.framesPerSecond = 30.0;
  }
  sanitized.framesPerSecond = std::clamp(sanitized.framesPerSecond, kMinVideoFramesPerSecond,
                                         kMaxVideoFramesPerSecond);
  sanitized.quality = std::clamp(sanitized.quality, kMinVideoQuality, kMaxVideoQuality);
  if (sanitized.codec.empty()) {
    sanitized.codec = "libx264";
  }
  if (sanitized.pixelFormat.empty()) {
    sanitized.pixelFormat = "yuv420p";
  }
  sanitized.width = std::max(0, sanitized.width);
  sanitized.height = std::max(0, sanitized.height);
  return sanitized;
}

std::vector<std::string> buildRawVideoEncodeArguments(
    const VideoEncoderSettings& settings, const std::filesystem::path& output) {
  const VideoEncoderSettings sanitized = sanitizeVideoEncoderSettings(settings);
  std::vector<std::string> arguments{
    "-hide_banner",
    "-loglevel", "error",
    "-y",
    "-f", "rawvideo",
    "-pixel_format", "rgba",
    "-video_size",
      std::to_string(sanitized.width) + "x" + std::to_string(sanitized.height),
    "-framerate", formatFrameRate(sanitized.framesPerSecond),
    "-i", "-",
  };
  appendEncodeOutputArguments(arguments, sanitized, output);
  return arguments;
}

std::vector<std::string> buildPngSequenceEncodeArguments(
    const std::filesystem::path& inputPattern, const VideoEncoderSettings& settings,
    const std::filesystem::path& output) {
  const VideoEncoderSettings sanitized = sanitizeVideoEncoderSettings(settings);
  std::vector<std::string> arguments{
    "-hide_banner",
    "-loglevel", "error",
    "-y",
    "-framerate", formatFrameRate(sanitized.framesPerSecond),
    "-start_number", "0",
    "-i", inputPattern.string(),
  };
  appendEncodeOutputArguments(arguments, sanitized, output);
  return arguments;
}

std::string buildShellCommand(
    const std::string& executable, const std::vector<std::string>& arguments) {
  const auto quote = [](const std::string& value) {
    std::string quoted;
#if defined(_WIN32)
    // cmd.exe: wrap in double quotes and escape embedded ones.
    quoted.push_back('"');
    for (const char c : value) {
      if (c == '"') {
        quoted += "\\\"";
      } else {
        quoted.push_back(c);
      }
    }
    quoted.push_back('"');
#else
    // POSIX sh: single quotes disable every expansion; close/escape/reopen to
    // embed a literal quote.
    quoted.push_back('\'');
    for (const char c : value) {
      if (c == '\'') {
        quoted += "'\\''";
      } else {
        quoted.push_back(c);
      }
    }
    quoted.push_back('\'');
#endif
    return quoted;
  };

  std::string command = quote(executable);
  for (const std::string& argument : arguments) {
    command.push_back(' ');
    command += quote(argument);
  }
  return command;
}

std::string findFfmpegExecutable() {
  const std::string overridePath = environmentValue("RAYRAI_FFMPEG");
  if (!overridePath.empty()) {
    return isExecutableFile(overridePath) ? overridePath : std::string();
  }
  for (const std::filesystem::path& directory : executableSearchDirectories()) {
    const std::filesystem::path candidate = directory / kFfmpegExecutableName;
    if (isExecutableFile(candidate)) {
      return candidate.string();
    }
  }
  return {};
}

bool videoEncodingAvailable() {
  return !findFfmpegExecutable().empty();
}

size_t videoFramesDue(double elapsedSeconds, double framesPerSecond, size_t framesWritten) {
  if (!std::isfinite(elapsedSeconds) || elapsedSeconds < 0.0 ||
      !std::isfinite(framesPerSecond) || framesPerSecond <= 0.0) return 0;
  // A frame covers one interval. Avoid gaining a frame at an exact boundary
  // merely because the clock's conversion to seconds rounded upwards.
  const long double desired = std::max(1.0L,
    std::ceil(static_cast<long double>(elapsedSeconds) * framesPerSecond - 1e-9L));
  const size_t target = desired >= static_cast<long double>(std::numeric_limits<size_t>::max())
    ? std::numeric_limits<size_t>::max() : static_cast<size_t>(desired);
  return target > framesWritten ? target - framesWritten : 0;
}

std::string pngSequenceFrameName(const std::string& prefix, int index) {
  std::ostringstream name;
  name << prefix << "_" << std::setw(6) << std::setfill('0') << index << ".png";
  return name.str();
}

std::string pngSequenceFramePattern(const std::string& prefix) {
  return prefix + "_%06d.png";
}

VideoEncoder::~VideoEncoder() {
  std::string ignored;
  close(ignored);
}

bool VideoEncoder::open(const std::filesystem::path& output,
                        const VideoEncoderSettings& settings, std::string& status) {
  std::string closeStatus;
  close(closeStatus);

  const VideoEncoderSettings sanitized = sanitizeVideoEncoderSettings(settings);
  if (sanitized.width <= 0 || sanitized.height <= 0) {
    status = "video recording failed: invalid frame size";
    return false;
  }
  if (output.empty()) {
    status = "video recording failed: no output path";
    return false;
  }
  const std::string ffmpeg = findFfmpegExecutable();
  if (ffmpeg.empty()) {
    status = "video recording needs ffmpeg on PATH (or $RAYRAI_FFMPEG)";
    return false;
  }

  std::error_code ec;
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path(), ec);
    if (ec) {
      status = "video recording failed: cannot create " + output.parent_path().string();
      return false;
    }
  }

#if !defined(_WIN32)
  // If ffmpeg exits early, writing to the dead pipe raises SIGPIPE, whose
  // default action would kill the viewer. Ignoring it turns that into an EPIPE
  // return from fwrite, which writeFrameRgba() reports as a failed recording.
  std::signal(SIGPIPE, SIG_IGN);
#endif

  const std::string command =
    buildShellCommand(ffmpeg, buildRawVideoEncodeArguments(sanitized, output));
  pipe_ = RAYRAI_POPEN(command.c_str(), RAYRAI_PIPE_MODE);
  if (pipe_ == nullptr) {
    status = "video recording failed: could not start ffmpeg";
    return false;
  }

  path_ = output;
  settings_ = sanitized;
  frameCount_ = 0;
  bytesWritten_ = 0;
  frameBytes_ = static_cast<size_t>(sanitized.width) * static_cast<size_t>(sanitized.height) * 4u;
  started_ = clock_();
  lastFrame_.clear();
  timed_ = false;
  status = "recording video to " + output.string();
  return true;
}

bool VideoEncoder::writeFrameRgba(const unsigned char* rgba, size_t byteCount,
                                  std::string& status) {
  if (pipe_ == nullptr) {
    status = "video recording failed: encoder is not open";
    return false;
  }
  if (rgba == nullptr || byteCount != frameBytes_) {
    std::string closeStatus;
    close(closeStatus);
    status = "video recording stopped: frame size changed mid-recording";
    return false;
  }
  const size_t written = std::fwrite(rgba, 1, byteCount, pipe_);
  if (written != byteCount) {
    timed_ = false; // Do not retry the failed pipe while closing it.
    std::string closeStatus;
    close(closeStatus);
    status = "video recording stopped: ffmpeg pipe closed";
    return false;
  }
  ++frameCount_;
  bytesWritten_ += written;
  return true;
}

size_t VideoEncoder::framesDue() const {
  if (!isOpen()) return 0;
  const double elapsed = std::chrono::duration<double>(clock_() - started_).count();
  return videoFramesDue(elapsed, settings_.framesPerSecond, frameCount_);
}

bool VideoEncoder::writeTimedFrameRgba(const unsigned char* rgba, size_t byteCount, std::string& status) {
  if (!isOpen() || !rgba || byteCount != frameBytes_) return writeFrameRgba(rgba, byteCount, status);
  const size_t due = framesDue();
  if (due == 0) return true;
  timed_ = true;
  // Missing intervals hold the previous image; the newest gets this capture.
  // The first capture also fills any delay since the recording was started.
  for (size_t i = 1; i < due; ++i) {
    const auto* held = lastFrame_.empty() ? rgba : lastFrame_.data();
    if (!writeFrameRgba(held, byteCount, status)) return false;
  }
  if (!writeFrameRgba(rgba, byteCount, status)) return false;
  lastFrame_.assign(rgba, rgba + byteCount);
  return true;
}

bool VideoEncoder::close(std::string& status) {
  if (pipe_ == nullptr) {
    return true;
  }
  const size_t finalFrames = timed_ && !lastFrame_.empty() ? framesDue() : 0;
  timed_ = false;
  for (size_t i = 0; i < finalFrames; ++i)
    if (!writeFrameRgba(lastFrame_.data(), lastFrame_.size(), status)) return false;
  const int closeStatus = RAYRAI_PCLOSE(pipe_);
  pipe_ = nullptr;
  lastFrame_.clear();
  const size_t frames = frameCount_;
  const std::filesystem::path finished = path_;
  frameBytes_ = 0;
  const int exitCode = processExitCode(closeStatus);
  if (exitCode != 0) {
    status = "video recording failed: ffmpeg exited with code " + std::to_string(exitCode);
    return false;
  }
  status = "wrote " + std::to_string(frames) + " frame(s) to " + finished.string();
  return true;
}

bool encodePngSequenceToVideo(
    const std::filesystem::path& frameDirectory, const std::string& framePrefix,
    const std::filesystem::path& output, const VideoEncoderSettings& settings,
    std::string& status) {
  const std::string ffmpeg = findFfmpegExecutable();
  if (ffmpeg.empty()) {
    status = "PNG sequence encode needs ffmpeg on PATH (or $RAYRAI_FFMPEG)";
    return false;
  }
  std::error_code ec;
  if (!std::filesystem::is_directory(frameDirectory, ec)) {
    status = "PNG sequence encode failed: no such directory " + frameDirectory.string();
    return false;
  }
  if (!std::filesystem::is_regular_file(frameDirectory / pngSequenceFrameName(framePrefix, 0),
                                        ec)) {
    status = "PNG sequence encode failed: " + frameDirectory.string() +
             " has no frame 0 for prefix " + framePrefix;
    return false;
  }
  if (output.empty()) {
    status = "PNG sequence encode failed: no output path";
    return false;
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path(), ec);
    if (ec) {
      status = "PNG sequence encode failed: cannot create " + output.parent_path().string();
      return false;
    }
  }

  const std::filesystem::path pattern = frameDirectory / pngSequenceFramePattern(framePrefix);
  const std::string command = buildShellCommand(
    ffmpeg, buildPngSequenceEncodeArguments(pattern, settings, output));
  const int exitCode = processExitCode(std::system(command.c_str()));
  if (exitCode != 0) {
    status = "PNG sequence encode failed: ffmpeg exited with code " + std::to_string(exitCode);
    return false;
  }
  status = "encoded " + output.string();
  return true;
}

} // namespace raisin::tcp_viewer
