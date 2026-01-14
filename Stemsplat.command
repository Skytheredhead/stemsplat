#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/Stemsplat.command"
ICON_SOURCE="${SCRIPT_DIR}/web/favicon.ico"
ICON_CACHE="${SCRIPT_DIR}/.stemsplat_icon.icns"

if [[ "$(uname -s)" == "Darwin" && -f "${ICON_SOURCE}" ]]; then
  if [[ ! -f "${ICON_CACHE}" && -x "$(command -v sips)" ]]; then
    sips -s format icns "${ICON_SOURCE}" --out "${ICON_CACHE}" >/dev/null 2>&1 || true
  fi
  if [[ -f "${ICON_CACHE}" && -x "$(command -v osascript)" ]]; then
    osascript <<EOF >/dev/null 2>&1 || true
set iconFile to POSIX file "${ICON_CACHE}"
set targetFile to POSIX file "${SCRIPT_PATH}"
set icon of targetFile to (read iconFile as picture)
EOF
  fi
fi

cd "${SCRIPT_DIR}"
python3 install.py
