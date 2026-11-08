// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
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
             double elementSize) {
    if (running) throw std::logic_error("Meshing is already running");
    if (!std::isfinite(elementSize) || elementSize <= 0.)
      throw std::invalid_argument("Element size must be positive");
    elementSize_ = elementSize;
    extension_ = extension;
    std::ofstream geo(directory / "model.geo");
    geo.imbue(std::locale::classic());
    geo.precision(17);
    geo << "General.NumThreads = 1;\nMesh.MaxNumThreads1D = "
           "1;\nMesh.MaxNumThreads2D = 1;\nMesh.MaxNumThreads3D = "
           "1;\nMesh.ElementOrder = 1;\nMesh.MshFileVersion = "
           "4.1;\nMesh.Binary = 0;\n";
    if (extension == ".stl") {
      geo << "Merge \"source.stl\";\nClassifySurfaces{40*Pi/180, 1, 1, "
             "Pi};\nCreateGeometry;\ns() = "
             "Surface{:};\nSurface Loop(1) = {s()};\nVolume(1) = {1};\n";
    } else {
      geo << "SetFactory(\"OpenCASCADE\");\nGeometry.OCCTargetUnit = "
             "\"M\";\nMerge \"source"
          << extension << "\";\n";
    }
    geo << "Mesh.MeshSizeMin = " << elementSize * .35
        << ";\nMesh.MeshSizeMax = " << elementSize << ";\nMesh.Optimize = 1;\n";
    geo.close();
    if (!geo) throw std::runtime_error("Cannot write meshing input");
    const auto exePath = raisim::pathFromUtf8(executable);
    launch(exePath.has_parent_path()
               ? raisim::pathUtf8(std::filesystem::absolute(exePath))
               : executable);
  }
  bool poll() {
    if (!running) return true;
    int exitCode = 0;
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
      error = "Cannot read Gmsh process status";
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
    if (result == 0 || result < 0 && errno == EINTR) return false;
    exitCode = result < 0 || !WIFEXITED(status) ? -1 : WEXITSTATUS(status);
    process_ = -1;
#endif
    running = false;
    if (exitCode != 0 || !std::filesystem::exists(directory /
#if defined(__EMSCRIPTEN__)
                                                  "volume.mesh"
#else
                                                  "volume.msh"
#endif
                                                  )) {
      std::ifstream log(directory / "gmsh.log");
      std::string contents((std::istreambuf_iterator<char>(log)), {});
      if (contents.size() > 3000) contents.erase(0, contents.size() - 3000);
      if (error.empty())
        error = "Gmsh meshing failed (exit " + std::to_string(exitCode) +
                "). Install Gmsh or set its executable path.\n" + contents;
    }
    return true;
  }
  raisim::fem::TetMesh result(double scale) const {
    if (running || !error.empty())
      throw std::runtime_error(error.empty() ? "Meshing has not finished"
                                             : error);
    const auto path = directory /
#if defined(__EMSCRIPTEN__)
                      "volume.mesh"
#else
                      "volume.msh"
#endif
        ;
    return raisim::fem::TetMesh::load(raisim::pathUtf8(path), scale);
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
  double elementSize_ = 0.;
  std::string extension_;
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
      throw std::runtime_error("Set the path to the Gmsh executable");
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
                                                 extension);
          } catch (error) {
            FS.writeFile(
                directory + '/error.txt',
                new TextEncoder().encode(error.message || String(error)));
            FS.writeFile(directory + '/done', new Uint8Array([1]));
          }
        },
        path.c_str(), size, extension_.c_str());
#elif defined(_WIN32)
    const auto logPath = (directory / "gmsh.log").wstring();
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE log =
        CreateFileW(logPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &security,
                    CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE)
      throw std::runtime_error("Cannot open Gmsh log");
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = log;
    startup.hStdError = log;
    PROCESS_INFORMATION process{};
    std::wstring command =
        quote(raisim::pathFromUtf8(executable).wstring()) +
        L" model.geo -3 -format msh4 -o volume.msh -nt 1 -v 3";
    const auto cwd = directory.wstring();
    const BOOL ok = CreateProcessW(nullptr, command.data(), nullptr, nullptr,
                                   TRUE, CREATE_NO_WINDOW, nullptr, cwd.c_str(),
                                   &startup, &process);
    CloseHandle(log);
    if (!ok)
      throw std::runtime_error("Cannot start Gmsh; check the executable path");
    CloseHandle(process.hThread);
    process_ = process.hProcess;
#else
    const auto logPath = (directory / "gmsh.log").string();
    std::vector<std::string> args{
        executable,   "model.geo", "-3", "-format", "msh4", "-o",
        "volume.msh", "-nt",       "1",  "-v",      "3"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    const auto cwd = directory.string();
    process_ = fork();
    if (process_ < 0) throw std::runtime_error("Cannot start Gmsh process");
    if (process_ == 0) {
      const int log = open(logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
      if (log < 0 || chdir(cwd.c_str()) != 0) _exit(126);
      if (dup2(log, STDOUT_FILENO) < 0 || dup2(log, STDERR_FILENO) < 0)
        _exit(126);
      close(log);
      execvp(argv[0], argv.data());
      _exit(127);
    }
#endif
    running = true;
  }
};
}  // namespace fem_workbench
