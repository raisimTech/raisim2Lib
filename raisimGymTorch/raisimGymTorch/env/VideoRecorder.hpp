//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYM_VIDEORECORDER_HPP
#define SRC_RAISIMGYM_VIDEORECORDER_HPP

#include <memory>
#include <string>

namespace raisim {

class World;

/// Records a World into an MP4 file with an in-process rayrai renderer and ffmpeg.
///
/// The renderer draws into a hidden window's OpenGL context, so no viewer has to be connected.
/// The camera follows the first articulated system in the world. One frame is written per 1/30 s
/// of simulated time, so the video plays back in real time however fast the simulation runs.
///
/// Call every method from the same thread: the OpenGL context stays current on the thread that
/// first started a recording.
class VideoRecorder {
 public:
  VideoRecorder();
  ~VideoRecorder();
  VideoRecorder(const VideoRecorder&) = delete;
  VideoRecorder& operator=(const VideoRecorder&) = delete;

  /// Start recording `world` into `path`. Returns false, with the reason in status(), when no
  /// OpenGL context can be created or ffmpeg is not on PATH. The world must outlive the recorder.
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
