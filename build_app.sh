#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${SCRIPT_DIR}/venv/bin/python" ]]; then
  PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

ARGS=(
  --noconfirm
  --clean
  --windowed
  --name "Stemsplat"
  --icon "${SCRIPT_DIR}/.stemsplat_icon.icns"
  --osx-bundle-identifier "com.stemsplat.app"
  --collect-submodules uvicorn
  --collect-submodules webview
  --exclude-module librosa
  --exclude-module scipy
  --exclude-module numba
  --exclude-module llvmlite
  --exclude-module sklearn
  --add-data "${SCRIPT_DIR}/configs:configs"
  --add-data "${SCRIPT_DIR}/web:web"
)

if [[ "${BUNDLE_MODELS:-0}" == "1" ]] && find "${SCRIPT_DIR}/models" -maxdepth 1 -type f ! -name ".gitkeep" | grep -q .; then
  echo "Including local model files in app bundle..."
  ARGS+=(--add-data "${SCRIPT_DIR}/models:models")
fi

"${PYTHON_BIN}" -m PyInstaller "${ARGS[@]}" "${SCRIPT_DIR}/launcher.py"

APP_BUNDLE="${SCRIPT_DIR}/dist/Stemsplat.app"
INFO_PLIST="${APP_BUNDLE}/Contents/Info.plist"

if [[ -f "${INFO_PLIST}" ]]; then
  /usr/libexec/PlistBuddy -c "Delete :LSUIElement" "${INFO_PLIST}" >/dev/null 2>&1 || true
  /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString 0.3" "${INFO_PLIST}" >/dev/null 2>&1 || \
    /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string 0.3" "${INFO_PLIST}" >/dev/null
  /usr/libexec/PlistBuddy -c "Set :CFBundleVersion 0.3" "${INFO_PLIST}" >/dev/null 2>&1 || \
    /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string 0.3" "${INFO_PLIST}" >/dev/null
  codesign --force --deep --sign - "${APP_BUNDLE}" >/dev/null
fi

echo "Created ${APP_BUNDLE}"
