#!/usr/bin/env bash

if [[ -n "${BASH_VERSION:-}" ]]; then
  __raisim_env_file="${BASH_SOURCE[0]}"
  if [[ "${__raisim_env_file}" == "$0" ]]; then
    echo "This script must be sourced: source /path/to/raisimLib/raisim_env.sh"
    exit 1
  fi
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  __raisim_env_file="${(%):-%x}"
else
  echo "This script must be sourced from Bash or Zsh."
  return 1 2>/dev/null || exit 1
fi

__raisim_env_dir="$(cd "$(dirname "${__raisim_env_file}")" && pwd)"
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
  local var_name="$1"
  local entry="$2"
  if [[ -z "${entry}" ]]; then
    return
  fi
  local current_value
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    current_value="${(P)var_name}"
  else
    current_value="${!var_name}"
  fi
  case ":${current_value}:" in
    *":${entry}:"*) return ;;
  esac
  if [[ -n "${current_value}" ]]; then
    export "${var_name}=${entry}:${current_value}"
  else
    export "${var_name}=${entry}"
  fi
  echo "Added to ${var_name}: ${entry}"
}

__raisim_prepend_package_lib() {
  local package_name="$1"
  local legacy_lib="${RAISIM_DIR}/${package_name}/${RAISIM_OS}/lib"
  local flat_lib="${RAISIM_DIR}/${package_name}/lib"

  if [[ -d "${legacy_lib}" ]]; then
    __raisim_prepend_path "${__raisim_lib_var}" "${legacy_lib}"
  elif [[ -d "${flat_lib}" ]]; then
    __raisim_prepend_path "${__raisim_lib_var}" "${flat_lib}"
  fi
}

__raisim_prepend_package_lib "raisim"
__raisim_prepend_package_lib "rayrai"

# raisimpy built from this checkout. Added even before the first build so the
# same shell can import it afterwards; set RAISIMPY_BUILD_DIR for other trees.
__raisim_py_dir="${RAISIMPY_BUILD_DIR:-${RAISIM_DIR}/build-py/raisimPy}"
__raisim_prepend_path "PYTHONPATH" "${__raisim_py_dir}"

# Python silently skips extensions built for another version; name the mismatch.
__raisim_python="$(command -v python3 || command -v python)"
if [[ -n "${__raisim_python}" && -d "${__raisim_py_dir}" ]]; then
  "${__raisim_python}" - "${__raisim_py_dir}" <<'EOF'
import importlib.machinery, pathlib, sys
build = pathlib.Path(sys.argv[1])
built = sorted(p.name for p in build.glob("raisimpy.*") if p.suffix in (".so", ".pyd"))
if built and not any("raisimpy" + s in built for s in importlib.machinery.EXTENSION_SUFFIXES):
    caches = [c for c in (build / "CMakeCache.txt", build.parent / "CMakeCache.txt") if c.is_file()]
    exe = next((l.split("=", 1)[1] for c in caches for l in c.read_text().splitlines()
                if l.startswith("Python_EXECUTABLE:")), "the Python it was built with")
    print(f"Warning: {built[0]} cannot be imported by {sys.executable} (Python {sys.version.split()[0]}).")
    print(f"  Use {exe}, or rebuild with -DPython_EXECUTABLE={sys.executable}.")
EOF
fi

unset -f __raisim_prepend_package_lib
unset -f __raisim_prepend_path
unset __raisim_lib_var
unset __raisim_py_dir
unset __raisim_python
unset __raisim_env_dir
unset __raisim_env_file
