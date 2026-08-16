# macOS release packaging

The hardened runtime entitlement file is intentionally limited to JIT and
unsigned executable memory, the two capabilities that must be revalidated for
the packaged PyTorch/PyInstaller runtime on every clean release build. A
production release is not accepted until the clean-user MPS smoke test proves
they are required and `codesign -d --entitlements :-` confirms no additional
entitlements were introduced.

`build_app.sh` creates a Developer ID signed application, submits a temporary
ZIP for notarization, staples the app, then creates the final ZIP and DMG. The
DMG is separately notarized and stapled. Final release hashes and Ed25519-signed
update metadata are emitted under `dist/`.
