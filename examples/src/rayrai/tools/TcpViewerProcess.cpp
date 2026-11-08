// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#include "TcpViewerProcess.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <thread>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace raisin::tcp_viewer {
namespace fs = std::filesystem;
namespace {
[[maybe_unused]] constexpr auto kStopGrace = std::chrono::milliseconds(200);
#if defined(_WIN32)
std::wstring quoteArgument(const std::wstring& value) {
  std::wstring out = L"\"";
  size_t slashes = 0;
  for (wchar_t c : value) {
    if (c == L'\\') {
      ++slashes;
      continue;
    }
    out.append(c == L'\"' ? slashes * 2 + 1 : slashes, L'\\');
    slashes = 0;
    out += c;
  }
  out.append(slashes * 2, L'\\');
  out += L'\"';
  return out;
}
#endif
}  // namespace

#if defined(__EMSCRIPTEN__)
struct ChildProcess::Impl {};
ChildProcess::ChildProcess() : impl_(std::make_unique<Impl>()) {}
ChildProcess::~ChildProcess() = default;
int ChildProcess::exitCode() const { return -1; }
bool ChildProcess::start(const fs::path& executable, const std::vector<fs::path>&, const fs::path&,
                         std::string& error) {
  error = "A browser cannot start " + executable.filename().string() + "; use the desktop viewer";
  return false;
}
bool ChildProcess::running() { return false; }
int64_t ChildProcess::pid() const { return 0; }
void ChildProcess::stop() {}
#else
struct ChildProcess::Impl {
  int exitCode = -1;
#if defined(_WIN32)
  HANDLE process = nullptr, job = nullptr, input = nullptr;
  DWORD id = 0;
#else
  pid_t process = -1;
  int input = -1;
#endif
};

ChildProcess::ChildProcess() : impl_(std::make_unique<Impl>()) {}
ChildProcess::~ChildProcess() { stop(); }
int ChildProcess::exitCode() const { return impl_->exitCode; }

bool ChildProcess::start(const fs::path& executable, const std::vector<fs::path>& arguments,
                         const fs::path& logPath, std::string& error) {
  stop();
  auto& p = *impl_;
  p.exitCode = -1;
  try {
#if defined(_WIN32)
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE readPipe = nullptr;
    if (!CreatePipe(&readPipe, &p.input, &security, 0)) throw std::runtime_error("cannot create process pipe");
    SetHandleInformation(p.input, HANDLE_FLAG_INHERIT, 0);
    HANDLE log = CreateFileW(logPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE) {
      CloseHandle(readPipe);
      throw std::runtime_error("cannot open process log");
    }
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<unsigned char> attributes(bytes);
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    const bool initialized = InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &bytes) != 0;
    HANDLE inherited[]{readPipe, log};
    bool prepared = initialized && UpdateProcThreadAttribute(startup.lpAttributeList, 0,
        PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited, sizeof(inherited), nullptr, nullptr);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = readPipe;
    startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = log;
    p.job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    prepared = prepared && p.job &&
               SetInformationJobObject(p.job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
    std::wstring command = quoteArgument(executable.wstring());
    for (const auto& argument : arguments) command += L" " + quoteArgument(argument.wstring());
    PROCESS_INFORMATION process{};
    const bool launched = prepared && CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
        &startup.StartupInfo, &process);
    if (initialized) DeleteProcThreadAttributeList(startup.lpAttributeList);
    CloseHandle(readPipe);
    CloseHandle(log);
    if (!launched)
      throw std::runtime_error("cannot launch " + executable.filename().string() + " (Windows error " +
                               std::to_string(GetLastError()) + ")");
    p.process = process.hProcess;
    p.id = process.dwProcessId;
    if (!AssignProcessToJobObject(p.job, p.process)) {
      CloseHandle(process.hThread);
      throw std::runtime_error("cannot attach " + executable.filename().string() + " to the viewer's lifetime");
    }
    ResumeThread(process.hThread);
    CloseHandle(process.hThread);
#else
    int pipeFds[2];
    if (pipe(pipeFds) != 0) throw std::runtime_error("cannot create process pipe");
    p.input = pipeFds[1];
    fcntl(pipeFds[0], F_SETFD, FD_CLOEXEC);
    fcntl(pipeFds[1], F_SETFD, FD_CLOEXEC);
    const int log = open(logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (log < 0) {
      close(pipeFds[0]);
      throw std::runtime_error("cannot open process log");
    }
    std::vector<std::string> args{executable.string()};
    for (const auto& argument : arguments) args.push_back(argument.string());
    std::vector<char*> pointers;
    for (auto& arg : args) pointers.push_back(arg.data());
    pointers.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipeFds[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, log, STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, log, STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipeFds[0]);
    posix_spawn_file_actions_addclose(&actions, pipeFds[1]);
    posix_spawn_file_actions_addclose(&actions, log);
    const int result = posix_spawn(&p.process, executable.c_str(), &actions, nullptr, pointers.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipeFds[0]);
    close(log);
    if (result != 0) {
      p.process = -1;
      throw std::runtime_error("cannot launch " + executable.filename().string() + ": " + std::strerror(result));
    }
#endif
    error.clear();
    return true;
  } catch (const std::exception& e) {
    error = e.what();
    stop();
    return false;
  }
}

bool ChildProcess::running() {
  auto& p = *impl_;
#if defined(_WIN32)
  if (!p.process) return false;
  if (WaitForSingleObject(p.process, 0) != WAIT_OBJECT_0) return true;
  DWORD status = 0;
  const int code = GetExitCodeProcess(p.process, &status) ? int(status) : -1;
  stop();
  p.exitCode = code;
#else
  if (p.process <= 0) return false;
  int status = 0;
  const pid_t result = waitpid(p.process, &status, WNOHANG);
  if (result == 0 || (result < 0 && errno == EINTR)) return true;
  const int code = result == p.process && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  p.process = -1;
  stop();
  p.exitCode = code;
#endif
  return false;
}

int64_t ChildProcess::pid() const {
#if defined(_WIN32)
  return impl_->process ? int64_t(impl_->id) : 0;
#else
  return impl_->process > 0 ? int64_t(impl_->process) : 0;
#endif
}

void ChildProcess::stop() {
  auto& p = *impl_;
#if defined(_WIN32)
  if (p.input) {
    CloseHandle(p.input);
    p.input = nullptr;
  }
  if (p.process) {
    if (WaitForSingleObject(p.process, 200) == WAIT_TIMEOUT) TerminateProcess(p.process, 0);
    WaitForSingleObject(p.process, 1000);
    CloseHandle(p.process);
    p.process = nullptr;
    p.id = 0;
  }
  if (p.job) {
    CloseHandle(p.job);
    p.job = nullptr;
  }
#else
  if (p.input >= 0) {
    close(p.input);
    p.input = -1;
  }
  if (p.process > 0) {
    kill(p.process, SIGTERM);
    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() + kStopGrace;
    pid_t result;
    do {
      result = waitpid(p.process, &status, WNOHANG);
      if (result == p.process || (result < 0 && errno == ECHILD)) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (std::chrono::steady_clock::now() < deadline);
    if (result == 0 || (result < 0 && errno == EINTR)) {
      kill(p.process, SIGKILL);
      while (waitpid(p.process, &status, 0) < 0 && errno == EINTR) {
      }
    }
    p.process = -1;
  }
#endif
}

#endif

std::string fileTail(const fs::path& path, size_t maxBytes) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return {};
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size <= 0) return {};
  file.seekg(size > std::streamoff(maxBytes) ? size - std::streamoff(maxBytes) : std::streamoff(0));
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

}  // namespace raisin::tcp_viewer
