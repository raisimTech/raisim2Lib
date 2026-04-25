#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./raisim_upgrade.sh [-y|--yes] [version]

Downloads and installs a raisim2Lib release asset into this checkout.
If version is omitted, the script asks whether to install the latest GitHub release.
If you decline, it lists available releases and asks for a version.

Options:
  -y, --yes     Do not ask for confirmation; if no version is given, use latest.
  -h, --help    Show this help.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
version=""
assume_yes=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -y|--yes)
      assume_yes=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "${version}" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 1
      fi
      version="$1"
      shift
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

detect_asset_platform() {
  case "$(uname -s)" in
    Linux)
      case "$(uname -m)" in
        aarch64|arm64) echo "linux-arm64" ;;
        *) echo "linux-x86" ;;
      esac
      ;;
    *)
      echo "raisim_upgrade.sh only supports Linux release assets. Use raisim_upgrade.ps1 on Windows." >&2
      exit 1
      ;;
  esac
}

fetch_latest_version() {
  local latest_json
  latest_json="$(curl -fsSL \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/raisimTech/raisim2Lib/releases/latest")"
  printf '%s\n' "${latest_json}" \
    | sed -nE 's/.*"tag_name"[[:space:]]*:[[:space:]]*"v?([^"]+)".*/\1/p' \
    | head -n 1
}

fetch_available_versions() {
  local releases_json
  releases_json="$(curl -fsSL \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/raisimTech/raisim2Lib/releases?per_page=100")"
  printf '%s\n' "${releases_json}" \
    | sed -nE 's/.*"tag_name"[[:space:]]*:[[:space:]]*"v?([^"]+)".*/\1/p'
}

download_asset() {
  local asset_file="$1"
  local output_path="$2"
  local urls=(
    "https://github.com/raisimTech/raisim2Lib/releases/download/v${version}/${asset_file}"
    "https://github.com/raisimTech/raisim2Lib/releases/download/${version}/${asset_file}"
  )

  for url in "${urls[@]}"; do
    echo "Downloading ${url}"
    if curl -fL --progress-bar "${url}" -o "${output_path}"; then
      return 0
    fi
  done

  echo "Failed to download ${asset_file} from raisim releases." >&2
  return 1
}

resolve_release_dir() {
  local extract_dir="$1"

  if [[ -d "${extract_dir}/raisim" ]]; then
    echo "${extract_dir}"
    return 0
  fi

  local child
  for child in "${extract_dir}"/*; do
    if [[ -d "${child}/raisim" ]]; then
      echo "${child}"
      return 0
    fi
  done

  return 1
}

require_command curl
require_command cmake

asset_platform="$(detect_asset_platform)"
green=""
reset=""
if [[ -t 1 ]]; then
  green="$(printf '\033[32m')"
  reset="$(printf '\033[0m')"
fi

if [[ -z "${version}" ]]; then
  echo "Checking latest raisim release..."
  latest_version="$(fetch_latest_version)"
  if [[ -z "${latest_version}" ]]; then
    echo "Could not determine latest raisim release." >&2
    exit 1
  fi

  if [[ "${assume_yes}" -eq 1 ]]; then
    version="${latest_version}"
  else
    printf "Latest raisim release is %s. Install this version? [Y/n] " "${latest_version}"
    read -r answer
    case "${answer}" in
      ""|y|Y|yes|YES)
        version="${latest_version}"
        ;;
      *)
        echo "Fetching available raisim releases..."
        available_versions="$(fetch_available_versions)"
        if [[ -z "${available_versions}" ]]; then
          echo "Could not determine available raisim releases." >&2
          exit 1
        fi

        echo "Available versions:"
        for available_version in ${available_versions}; do
          printf '  %s%s%s\n' "${green}" "${available_version}" "${reset}"
        done
        printf "Enter version to install: "
        read -r version
        if [[ -z "${version}" ]]; then
          echo "No version selected." >&2
          exit 1
        fi
        if ! printf '%s\n' "${available_versions}" | grep -Fx -- "${version}" >/dev/null; then
          echo "Version '${version}' is not in the available release list." >&2
          exit 1
        fi
        ;;
    esac
  fi
fi

asset_file="${asset_platform}-${version}.zip"

echo "This will replace:"
echo "  ${script_dir}/raisim"
echo "  ${script_dir}/rayrai"
if [[ "${assume_yes}" -ne 1 ]]; then
  printf "Upgrade to raisim %s? [y/N] " "${version}"
  read -r answer
  case "${answer}" in
    y|Y|yes|YES) ;;
    *) echo "Upgrade cancelled."; exit 0 ;;
  esac
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

archive_path="${tmp_dir}/${asset_file}"
extract_dir="${tmp_dir}/extract"
mkdir -p "${extract_dir}"

download_asset "${asset_file}" "${archive_path}"
(cd "${extract_dir}" && cmake -E tar xf "${archive_path}")

release_dir="$(resolve_release_dir "${extract_dir}")" || {
  echo "Archive extraction completed but no raisim directory was found." >&2
  exit 1
}

rm -rf "${script_dir}/raisim" "${script_dir}/rayrai"
cp -R "${release_dir}/raisim" "${script_dir}/"
if [[ -d "${release_dir}/rayrai" ]]; then
  cp -R "${release_dir}/rayrai" "${script_dir}/"
fi

echo "Installed raisim ${version}."
