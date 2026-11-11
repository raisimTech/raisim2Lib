// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#endif

#include "fem_workbench_executable.hpp"
#include "fem_workbench_geometry.hpp"
#include "raisim/PathUtils.hpp"
#include "raisim/fem/LinearElasticity.hpp"

namespace fem_workbench {
class MeshJob {
 public:
  std::filesystem::path directory;
  std::string error;
  bool running = false;

  MeshJob() {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 100; ++attempt) {
      directory = std::filesystem::temp_directory_path() /
                  ("rayrai-fem-" + std::to_string(stamp) + "-" +
                   std::to_string(attempt));
      if (std::filesystem::create_directory(directory)) return;
    }
    throw std::runtime_error("Cannot create temporary meshing directory");
  }
  ~MeshJob() {
    cancel();
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
  }
  MeshJob(const MeshJob&) = delete;
  MeshJob& operator=(const MeshJob&) = delete;

  void start(const std::string& executable, const std::string& extension,
             double elementSize, double surfaceTolerance = 0.,
             bool surfaceOnly = false) {
    if (running) throw std::logic_error("Meshing is already running");
    if (!std::isfinite(elementSize) || elementSize <= 0.)
      throw std::invalid_argument("Element size must be positive");
    if (surfaceTolerance < 0. || !std::isfinite(surfaceTolerance))
      throw std::invalid_argument("Surface tolerance must be finite and nonnegative");
    surfaceOnly_ = surfaceOnly; surfaceTolerance_ = surfaceTolerance;
    error.clear();
    for (const auto* stale : {"launch-error", "mesher.log", "volume.mesh",
                              "surface.rfs", "done", "error.txt"})
      std::filesystem::remove(directory / stale);
    elementSize_ = elementSize;
    extension_ = extension;
#if defined(__EMSCRIPTEN__)
    launch(executable);
#else
    executable_ = findExecutable(executable, environmentValue("PATH"));
    if (executable_.empty())
      throw std::runtime_error(
          mesherName() + " executable '" + executable +
          "' was not found or is not executable. Build/install the RaiSim "
          "native FEM mesher or set RAISIM_FEM_MESHER.");
    launch(executable_);
#endif
  }
  bool poll() {
    if (!running) return true;
    int exitCode = 0, terminatedSignal = 0;
#if defined(__EMSCRIPTEN__)
    if (!std::filesystem::exists(directory / "done")) return false;
    if (std::filesystem::exists(directory / "error.txt")) {
      std::ifstream log(directory / "error.txt");
      error.assign(std::istreambuf_iterator<char>(log), {});
      exitCode = 1;
    }
#elif defined(_WIN32)
    DWORD status = 0;
    if (!GetExitCodeProcess(process_, &status)) {
      error = "Cannot read " + mesherName() + " process status";
      cancel();
      return true;
    }
    if (status == STILL_ACTIVE) return false;
    exitCode = static_cast<int>(status);
    CloseHandle(process_);
    process_ = nullptr;
#else
    int status = 0;
    const pid_t result = waitpid(process_, &status, WNOHANG);
    if (result == 0 || (result < 0 && errno == EINTR)) return false;
    exitCode = result < 0 || !WIFEXITED(status) ? -1 : WEXITSTATUS(status);
    if (result > 0 && WIFSIGNALED(status)) terminatedSignal = WTERMSIG(status);
    process_ = -1;
#endif
    running = false;
#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
    std::ifstream launchError(directory / "launch-error", std::ios::binary);
    int failure[2]{};
    if (launchError.read(reinterpret_cast<char*>(failure), sizeof(failure))) {
      const char* stages[] = {"open the meshing log",
                              "enter the meshing directory",
                              "redirect the meshing log"};
      const auto stage = static_cast<unsigned>(failure[0]);
      const std::string operation =
          stage < 3 ? stages[stage]
                    : (stage == 3 ? "execute " : "launch ") + mesherName();
      error = "Cannot " + operation + " ('" + executable_ + "'): " +
              std::error_code(failure[1], std::generic_category()).message();
      if (stage == 3 && failure[1] == ENOENT)
        error +=
            ". The executable exists; check its script interpreter or dynamic "
            "loader.";
      return true;
    }
#endif
    if (exitCode != 0 || !std::filesystem::exists(directory / outputName())) {
      const auto contents = logTail();
      if (terminatedSignal != 0)
        error = mesherName() + " crashed (signal " +
                std::to_string(terminatedSignal) + ").\n" + contents;
      if (error.empty())
        error = mesherName() + " started but meshing failed (exit " +
                std::to_string(exitCode) +
                (exitCode == 127
                     ? "). Check its launcher/interpreter and dependencies.\n"
                     : "). Check the geometry and meshing settings.\n") +
                contents;
    }
    return true;
  }
  raisim::fem::TetMesh result(double scale) const {
    if (running || !error.empty())
      throw std::runtime_error(error.empty() ? "Meshing has not finished"
                                             : error);
    const auto path = directory / outputName();
    try {
      return raisim::fem::TetMesh::load(raisim::pathUtf8(path), scale);
    } catch (const std::exception& failure) {
      throw std::runtime_error(mesherName() +
                               " produced an invalid tetrahedral volume: " +
                               failure.what() + "\n" + logTail());
    }
  }
  struct Progress {
    double fraction = -1.;
    std::string stage;
  };
  Progress progress() const {
    Progress result;
    std::ifstream log(directory / "mesher.log", std::ios::binary);
    if (!log) return result;
    log.seekg(0, std::ios::end);
    const std::streamoff size = log.tellg();
    if (size <= 0) return result;
    log.seekg(std::max<std::streamoff>(0, size - 4096));
    const std::string tail{std::istreambuf_iterator<char>(log), {}};
    constexpr std::string_view kMarker = "RAISIM_FEM_PROGRESS ";
    for (size_t end = tail.rfind('\n'); end != std::string::npos && end > 0;) {
      const size_t start = tail.rfind('\n', end - 1) + 1;
      if (tail.compare(start, kMarker.size(), kMarker) == 0) {
        std::istringstream line(
            tail.substr(start + kMarker.size(), end - start - kMarker.size()));
        line.imbue(std::locale::classic());
        double fraction = -1.;
        if (line >> fraction && fraction >= 0. && fraction <= 1.)
          result.fraction = fraction;
        std::getline(line >> std::ws, result.stage);
        if (!result.stage.empty() && result.stage.back() == '\r')
          result.stage.pop_back();
        return result;
      }
      if (start == 0) break;
      end = start - 1;
    }
    return result;
  }
  SurfaceGeometry surfaceResult(double scale) const {
    if (running || !error.empty()) throw std::runtime_error(error.empty() ? "Surface import is still running" : error);
    std::ifstream input(directory / "surface.rfs", std::ios::binary);
    if (!input) throw std::runtime_error("CAD import produced no surface");
    return SurfaceGeometry::decode(std::string(std::istreambuf_iterator<char>(input), {}), scale);
  }
  void cancel() {
    if (!running) return;
#if defined(__EMSCRIPTEN__)
    EM_ASM({ globalThis['raisimFem'] ?.['cancelMesh'](); });
#elif defined(_WIN32)
    TerminateProcess(process_, 1);
    WaitForSingleObject(process_, INFINITE);
    CloseHandle(process_);
    process_ = nullptr;
#else
    kill(process_, SIGKILL);
    while (waitpid(process_, nullptr, 0) < 0 && errno == EINTR) {
    }
    process_ = -1;
#endif
    running = false;
  }

 private:
  bool surfaceOnly_ = false;
  double surfaceTolerance_ = 0.;
  std::string mesherName() const {
#if defined(__EMSCRIPTEN__)
    return "Browser mesher";
#else
    return "RaiSim FEM mesher";
#endif
  }
  const char* outputName() const {
    return surfaceOnly_ ? "surface.rfs" : "volume.mesh";
  }
  std::string logTail() const {
    std::ifstream log(directory / "mesher.log");
    std::string contents((std::istreambuf_iterator<char>(log)), {});
    if (contents.size() > 3000) contents.erase(0, contents.size() - 3000);
    return contents;
  }
  double elementSize_ = 0.;
  std::string extension_, executable_;
#if defined(__EMSCRIPTEN__)
#elif defined(_WIN32)
  HANDLE process_ = nullptr;
  static std::wstring quote(const std::wstring& argument) {
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (const wchar_t c : argument) {
      if (c == L'\\') {
        ++slashes;
        continue;
      }
      if (c == L'\"')
        result.append(2 * slashes + 1, L'\\');
      else
        result.append(slashes, L'\\');
      slashes = 0;
      result.push_back(c);
    }
    result.append(2 * slashes, L'\\');
    result.push_back(L'\"');
    return result;
  }
#else
  pid_t process_ = -1;
#endif
  void launch(const std::string& executable) {
    if (executable.empty())
      throw std::runtime_error("Set the path to the " + mesherName() +
                               " executable");
#if defined(__EMSCRIPTEN__)
    const auto path = directory.string();
    const double size = elementSize_;
    EM_ASM(
        {
          const directory = UTF8ToString($0);
          try {
            if (!globalThis['raisimFem'])
              throw Error('Browser file integration is unavailable.');
            const extension = UTF8ToString($2);
            const bytes = FS.readFile(directory + '/source' + extension);
            const files = {};
            files['writeFile'] = function(path, bytes) {
              FS.writeFile(path, bytes);
            };
            globalThis['raisimFem']['startMesh'](directory, bytes, $1, files,
                                                 extension, $3, !!$4);
          } catch (error) {
            FS.writeFile(
                directory + '/error.txt',
                new TextEncoder().encode(error.message || String(error)));
            FS.writeFile(directory + '/done', new Uint8Array([1]));
          }
        },
        path.c_str(), size, extension_.c_str(), surfaceTolerance_, int(surfaceOnly_));
#else
    std::ostringstream size;
    size.imbue(std::locale::classic());
    size << std::setprecision(17) << elementSize_;
    std::vector<std::string> args{executable, "--input", "source" + extension_,
                                  "--output", "volume.mesh", "--edge-length",
                                  size.str()};
    if (surfaceOnly_) {
      args[4] = "surface.rfs";
      args.push_back("--surface-only");
    } else {
      args.push_back("--threads");
      args.push_back("auto");
    }
    if (surfaceTolerance_ > 0.) {
      std::ostringstream tolerance; tolerance.imbue(std::locale::classic());
      tolerance << std::setprecision(17) << surfaceTolerance_;
      args.push_back("--surface-tolerance"); args.push_back(tolerance.str());
    }
#if defined(_WIN32)
    const auto logPath = (directory / "mesher.log").wstring();
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE log =
        CreateFileW(logPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &security,
                    CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE)
      throw std::runtime_error("Cannot open meshing log");
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    HANDLE input =
        CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (input == INVALID_HANDLE_VALUE) {
      CloseHandle(log);
      throw std::runtime_error("Cannot open meshing input handle");
    }
    startup.hStdInput = input;
    startup.hStdOutput = log;
    startup.hStdError = log;
    PROCESS_INFORMATION process{};
    std::wstring command;
    for (const auto& argument : args) {
      if (!command.empty()) command += L' ';
      command += quote(raisim::pathFromUtf8(argument).wstring());
    }
    const auto cwd = directory.wstring();
    const BOOL ok = CreateProcessW(nullptr, command.data(), nullptr, nullptr,
                                   TRUE, CREATE_NO_WINDOW, nullptr, cwd.c_str(),
                                   &startup, &process);
    const DWORD launchError = ok ? ERROR_SUCCESS : GetLastError();
    CloseHandle(input);
    CloseHandle(log);
    if (!ok)
      throw std::runtime_error(
          "Cannot execute " + mesherName() + " ('" + executable + "'): " +
          std::error_code(static_cast<int>(launchError), std::system_category())
              .message());
    CloseHandle(process.hThread);
    process_ = process.hProcess;
#else
    const auto logPath = (directory / "mesher.log").string();
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    const auto cwd = directory.string();
    const auto failurePath = (directory / "launch-error").string();
    process_ = fork();
    if (process_ < 0)
      throw std::runtime_error("Cannot start " + mesherName() + " process");
    if (process_ == 0) {
      const auto fail = [&](int stage) {
        const int failure[2]{stage, errno};
        const int descriptor =
            open(failurePath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (descriptor >= 0) {
          const char* data = reinterpret_cast<const char*>(failure);
          size_t remaining = sizeof(failure);
          while (remaining != 0) {
            const auto written = write(descriptor, data, remaining);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) break;
            data += written;
            remaining -= static_cast<size_t>(written);
          }
          close(descriptor);
        }
        _exit(stage == 3 ? 127 : 126);
      };
      const int log = open(logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
      if (log < 0) fail(0);
      if (chdir(cwd.c_str()) != 0) fail(1);
      if (dup2(log, STDOUT_FILENO) < 0 || dup2(log, STDERR_FILENO) < 0) fail(2);
      if (log != STDOUT_FILENO && log != STDERR_FILENO) close(log);
      execv(argv[0], argv.data());
      fail(3);
    }
#endif
#endif
    running = true;
  }
};
}  // namespace fem_workbench
