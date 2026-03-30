#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced: source /path/to/raisimLib/raisim_env.sh"
  exit 1
fi

__raisim_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RAISIM_DIR="${__raisim_env_dir}"

if [[ -z "${RAISIM_OS:-}" ]]; then
  case "$(uname -s)" in
    Linux)
      if [[ "$(uname -m)" == arm* || "$(uname -m)" == aarch64 ]]; then
        if [[ -d "${RAISIM_DIR}/raisim/linux-arm" ]]; then
          export RAISIM_OS="linux-arm"
        else
          export RAISIM_OS="linux"
        fi
      else
        export RAISIM_OS="linux"
      fi
      ;;
    Darwin)
      if [[ "$(uname -m)" == arm64 && -d "${RAISIM_DIR}/raisim/m1" ]]; then
        export RAISIM_OS="m1"
      else
        export RAISIM_OS="mac"
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      export RAISIM_OS="win32"
      ;;
    *)
      export RAISIM_OS="linux"
      ;;
  esac
fi

__raisim_lib_var="LD_LIBRARY_PATH"
if [[ "${RAISIM_OS}" == "mac" || "${RAISIM_OS}" == "m1" ]]; then
  __raisim_lib_var="DYLD_LIBRARY_PATH"
fi

__raisim_prepend_path() {
  local entry="$1"
  if [[ -z "${entry}" || ! -d "${entry}" ]]; then
    return
  fi
  local current_value="${!__raisim_lib_var}"
  case ":${current_value}:" in
    *":${entry}:"*) return ;;
  esac
  if [[ -n "${current_value}" ]]; then
    export "${__raisim_lib_var}=${entry}:${current_value}"
  else
    export "${__raisim_lib_var}=${entry}"
  fi
  echo "Added to ${__raisim_lib_var}: ${entry}"
}

__raisim_prepend_path "${RAISIM_DIR}/raisim/lib"
__raisim_prepend_path "${RAISIM_DIR}/rayrai/lib"

unset -f __raisim_prepend_path
unset __raisim_lib_var
unset __raisim_env_dir
