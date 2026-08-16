#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stemsplat.model_manifest import ModelManifest  # noqa: E402
from stemsplat.version import __version__  # noqa: E402


def _quality_report_blockers(quality: object) -> list[str]:
    if not isinstance(quality, dict):
        return ["quality report root is invalid"]
    blockers: list[str] = []
    provenance = quality.get("provenance")
    if quality.get("schema_version") != 1 or quality.get("command") != "report":
        blockers.append("quality report schema or command is invalid")
    if not isinstance(provenance, dict) or provenance.get("lane") != "reference_safe":
        blockers.append("quality report is not from the reference_safe lane")
    elif provenance.get("stemsplat_version") != __version__:
        blockers.append("quality report version does not match the application")
    if not quality.get("passed") or quality.get("blockers"):
        blockers.append("quality report did not pass")
    listening = quality.get("manual_blind_listening_signoff")
    public = quality.get("public_quality")
    performance = quality.get("performance")
    if not isinstance(listening, dict) or not listening.get("passed") or len(listening.get("excerpts") or []) < 5:
        blockers.append("quality report lacks passing blind-listening evidence")
    if not isinstance(public, dict) or not public.get("passed"):
        blockers.append("quality report lacks passing public-corpus evidence")
    modes = performance.get("modes") if isinstance(performance, dict) else None
    if not isinstance(modes, dict) or not modes:
        blockers.append("quality report lacks paired performance evidence")
    elif any(
        not isinstance(evidence, dict)
        or not isinstance(evidence.get("trials"), dict)
        or not all(bool(evidence["trials"].get(kind, {}).get("passed")) for kind in ("cold", "warm"))
        for evidence in modes.values()
    ):
        blockers.append("quality report contains incomplete performance gates")
    reports = quality.get("reports")
    commands = [str(report.get("command") or "") for report in reports if isinstance(report, dict)] if isinstance(reports, list) else []
    if not {"prepare", "baseline", "compare"}.issubset(commands) or commands.count("benchmark") < 2:
        blockers.append("quality report lacks required source reports")
    return blockers


def main() -> int:
    blockers: list[str] = []
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        blockers.append("production v0.4.3 must be built on Apple Silicon macOS")
    manifest = ModelManifest(ROOT / "models" / "manifest.json")
    ineligible = [item.tag for item in manifest.by_tag.values() if not item.release_eligible]
    if ineligible:
        blockers.append("models lack immutable hashes/licenses: " + ", ".join(ineligible))
    if not (ROOT / "updates" / "public-key.pem").is_file():
        blockers.append("release update signing public key is missing")
    quality_path = ROOT / "reports" / "quality" / "release-report.json"
    if not quality_path.is_file():
        blockers.append("full public/private quality report is missing")
    else:
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("quality report is invalid")
        else:
            blockers.extend(_quality_report_blockers(quality))
    if blockers:
        print("Production release blocked:", file=sys.stderr)
        for blocker in blockers:
            print(f"- {blocker}", file=sys.stderr)
        return 2
    print("Production release prerequisites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
