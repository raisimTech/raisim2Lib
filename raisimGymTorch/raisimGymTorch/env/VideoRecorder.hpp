//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYM_VIDEORECORDER_HPP
#define SRC_RAISIMGYM_VIDEORECORDER_HPP

#include <cstddef>
#include <memory>
#include <string>

namespace raisim {

class World;

/// Shell command that pipes raw RGBA frames from stdin into an H.264 MP4 at `path`. For cmd.exe
/// (`windowsShell`) the whole command is quoted once more, because cmd strips the first and the
/// last quote of a command line that holds more than two. Returns an empty string for a path that
/// a shell cannot quote (a double quote under cmd.exe).
inline std::string ffmpegPipeCommand(const std::string& ffmpeg, const std::string& path, int width,
                                     int height, int framesPerSecond, bool windowsShell) {
  const auto quoted = [windowsShell](const std::string& text) {
    if (windowsShell) return "\"" + text + "\"";
    std::string result = "'";
    for (char c : text) result += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return result + "'";
  };
  if (windowsShell &&
      (ffmpeg.find('"') != std::string::npos || path.find('"') != std::string::npos))
    return {};
  const std::string command = quoted(ffmpeg) + " -loglevel error -y -f rawvideo -pix_fmt rgba -s " +
      std::to_string(width) + "x" + std::to_string(height) + " -r " +
      std::to_string(framesPerSecond) +
      " -i - -an -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart " +
      quoted(path);
  return windowsShell ? "\"" + command + "\"" : command;
}

/// Schedules video frames on the simulated clock: one frame per 1/framesPerSecond s. A control
/// step that spans several frame periods gets the same image repeated, and a clock that goes back
/// (a reset) restarts the schedule at the new time.
class VideoFramePacer {
 public:
  explicit VideoFramePacer(double framesPerSecond) : period_(1. / framesPerSecond) {}
  void restart(double time) { next_ = last_ = time; }
  /// Number of frames due at `time`. The schedule moves on only in advance().
  [[nodiscard]] std::size_t framesDue(double time) const {
    double next = time < last_ ? time : next_;
    std::size_t frames = 0;
    for (; next <= time + 1e-9; next += period_) ++frames;
    return frames;
  }
  /// Moves the schedule past the frames due at `time`, once they are written.
  void advance(double time) {
    if (time < last_) next_ = time;
    last_ = time;
    while (next_ <= time + 1e-9) next_ += period_;
  }

 private:
  double period_, next_ = 0., last_ = 0.;
};

/// Records a World into an MP4 file with an in-process rayrai renderer and ffmpeg.
///
/// The renderer draws into a hidden window's OpenGL context, so no viewer has to be connected.
/// The camera follows the first articulated system in the world. One frame is written per 1/30 s
/// of simulated time, so the video plays back in real time however fast the simulation runs.
///
/// Call every method from the thread that first started a recording. The recorder makes its own
/// OpenGL context current while it renders and then restores the SDL window and context that were
/// current before.
class VideoRecorder {
 public:
  VideoRecorder();
  ~VideoRecorder();
  VideoRecorder(const VideoRecorder&) = delete;
  VideoRecorder& operator=(const VideoRecorder&) = delete;

  /// Start recording `world` into `path`. Returns false, with the reason in status(), when no
  /// OpenGL context can be created, ffmpeg is not on PATH or the path cannot be passed to the
  /// shell (a double quote on Windows). The world must outlive the recorder.
  bool start(World& world, const std::string& path);
  /// Write the frames that are due at the world's current time. Cheap when not recording.
  void capture();
  /// Finish the file. Returns false, with the reason in status(), if nothing usable was written.
  bool stop();

  [[nodiscard]] bool isRecording() const;
  /// What happened to the last recording, or why it could not start.
  [[nodiscard]] const std::string& status() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}

#endif //SRC_RAISIMGYM_VIDEORECORDER_HPP
