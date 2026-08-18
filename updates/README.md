# Signed update metadata

A production release must provide these GitHub release assets:

- `stemsplat-update.json`, canonical UTF-8 JSON containing schema version, release version, asset name, GitHub asset URL, byte length, SHA-256, and issue time.
- `stemsplat-update.json.sig`, an Ed25519 signature over the exact JSON bytes.
- the signed application ZIP or DMG.

The matching public key must be installed as `updates/public-key.pem` before a production build. It is intentionally absent until the project owner provisions and safeguards the signing key. Without it, update download is fail-closed.

`tools/sign_update.py` derives the version from `stemsplat/version.py`, verifies
that the private key matches the packaged public key, hashes the final release
asset, and writes both metadata files atomically. The private key is never
copied into the application or repository.
