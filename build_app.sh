#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "v0.4.3 builds only on Apple Silicon macOS." >&2
  exit 2
fi

PYTHON_EXE="${PYTHON_BIN:-${SCRIPT_DIR}/venv/bin/python}"
if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="$(command -v python3)"
fi
APP_VERSION="$(${PYTHON_EXE} -c 'from stemsplat.version import __version__; print(__version__)')"
RELEASE_MODE="${STEMSPLAT_RELEASE_MODE:-development}"

npm ci --ignore-scripts
npm run build:css
npm run check:offline

if [[ "${RELEASE_MODE}" == "production" ]]; then
  "${PYTHON_EXE}" tools/pip_audit_gate.py --requirements requirements-macos-arm64.lock
  "${PYTHON_EXE}" tools/pip_audit_gate.py --requirements requirements-dev-macos-arm64.lock
  "${PYTHON_EXE}" tools/pip_audit_gate.py --requirements requirements-quality-macos-arm64.lock
  "${PYTHON_EXE}" tools/release_gate.py
fi

ARGS=(
  --noconfirm
  --clean
  --windowed
  --name "Stemsplat"
  --distpath "${SCRIPT_DIR}/dist"
  --workpath "${SCRIPT_DIR}/build/pyinstaller"
  --specpath "${SCRIPT_DIR}/build/spec"
  --icon "${SCRIPT_DIR}/.stemsplat_icon.icns"
  --osx-bundle-identifier "com.stemsplat.app"
  --target-architecture arm64
  --collect-submodules uvicorn
  --collect-submodules webview
  --collect-submodules imageio_ffmpeg
  --collect-submodules demucs
  --collect-submodules openunmix
  --hidden-import numpy.core.multiarray
  --hidden-import numpy.core.numeric
  --hidden-import numpy._core.multiarray
  --hidden-import numpy._core.numeric
  --collect-data imageio_ffmpeg
  --exclude-module librosa
  --exclude-module scipy
  --exclude-module numba
  --exclude-module llvmlite
  --exclude-module sklearn
  --add-data "${SCRIPT_DIR}/configs:configs"
  --add-data "${SCRIPT_DIR}/models/manifest.json:models"
  --add-data "${SCRIPT_DIR}/web:web"
  --add-data "${SCRIPT_DIR}/updates:updates"
)

export PYINSTALLER_CONFIG_DIR="${SCRIPT_DIR}/build/.pyinstaller"
"${PYTHON_EXE}" -m PyInstaller "${ARGS[@]}" "${SCRIPT_DIR}/launcher.py"

APP_BUNDLE="${SCRIPT_DIR}/dist/Stemsplat.app"
INFO_PLIST="${APP_BUNDLE}/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Delete :LSUIElement" "${INFO_PLIST}" >/dev/null 2>&1 || true
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString ${APP_VERSION}" "${INFO_PLIST}" >/dev/null 2>&1 || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string ${APP_VERSION}" "${INFO_PLIST}"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion ${APP_VERSION}" "${INFO_PLIST}" >/dev/null 2>&1 || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string ${APP_VERSION}" "${INFO_PLIST}"

if [[ "${RELEASE_MODE}" == "production" ]]; then
  : "${STEMSPLAT_DEVELOPER_ID_APPLICATION:?Set STEMSPLAT_DEVELOPER_ID_APPLICATION}"
  : "${STEMSPLAT_NOTARY_PROFILE:?Set STEMSPLAT_NOTARY_PROFILE to a notarytool keychain profile}"
  : "${STEMSPLAT_UPDATE_SIGNING_KEY:?Set STEMSPLAT_UPDATE_SIGNING_KEY to the Ed25519 private-key path}"
  codesign --force --deep --options runtime --timestamp \
    --entitlements "${SCRIPT_DIR}/packaging/stemsplat.entitlements" \
    --sign "${STEMSPLAT_DEVELOPER_ID_APPLICATION}" "${APP_BUNDLE}"
  codesign --verify --deep --strict --verbose=2 "${APP_BUNDLE}"
  NOTARY_ZIP="${SCRIPT_DIR}/dist/.Stemsplat-${APP_VERSION}-notary.zip"
  RELEASE_ZIP="${SCRIPT_DIR}/dist/Stemsplat-${APP_VERSION}-arm64.zip"
  RELEASE_DMG="${SCRIPT_DIR}/dist/Stemsplat-${APP_VERSION}-arm64.dmg"
  ditto -c -k --keepParent "${APP_BUNDLE}" "${NOTARY_ZIP}"
  xcrun notarytool submit "${NOTARY_ZIP}" \
    --keychain-profile "${STEMSPLAT_NOTARY_PROFILE}" --wait
  xcrun stapler staple "${APP_BUNDLE}"
  xcrun stapler validate "${APP_BUNDLE}"
  ditto -c -k --keepParent "${APP_BUNDLE}" "${RELEASE_ZIP}"
  hdiutil create -volname "Stemsplat ${APP_VERSION}" -srcfolder "${APP_BUNDLE}" -ov -format UDZO "${RELEASE_DMG}"
  xcrun notarytool submit "${RELEASE_DMG}" --keychain-profile "${STEMSPLAT_NOTARY_PROFILE}" --wait
  xcrun stapler staple "${RELEASE_DMG}"
  xcrun stapler validate "${RELEASE_DMG}"
  "${PYTHON_EXE}" tools/sign_update.py \
    --asset "${RELEASE_ZIP}" \
    --download-url "https://github.com/Skytheredhead/stemsplat/releases/download/v${APP_VERSION}/$(basename "${RELEASE_ZIP}")" \
    --private-key "${STEMSPLAT_UPDATE_SIGNING_KEY}" \
    --public-key "${SCRIPT_DIR}/updates/public-key.pem" \
    --output-dir "${SCRIPT_DIR}/dist"
  rm -f "${NOTARY_ZIP}"
else
  codesign --force --deep --sign - "${APP_BUNDLE}"
  printf '%s\n' "DEVELOPMENT-ONLY: ad-hoc signed; not notarized" > "${SCRIPT_DIR}/dist/DEVELOPMENT-ONLY.txt"
fi

find "${SCRIPT_DIR}/dist" -maxdepth 1 -type f \( -name '*.zip' -o -name '*.dmg' \) -print0 | sort -z | xargs -0 shasum -a 256 > "${SCRIPT_DIR}/dist/SHA256SUMS" 2>/dev/null || true
echo "Created ${APP_BUNDLE} (${RELEASE_MODE}, v${APP_VERSION})"
