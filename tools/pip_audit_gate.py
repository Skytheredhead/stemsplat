#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEPTIONS = ROOT / "docs" / "security" / "vulnerability-exceptions.json"
DEFAULT_REQUIREMENTS = ROOT / "requirements-macos-arm64.lock"


@dataclass(frozen=True)
class AuditException:
    advisory_id: str
    package: str
    versions: frozenset[str]
    unused_surface: str
    owner: str
    expires: date


def load_exceptions(path: Path) -> dict[tuple[str, str], AuditException]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"vulnerability exception file is invalid: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("vulnerability exception schema is unsupported")
    records = payload.get("exceptions")
    if not isinstance(records, list):
        raise ValueError("vulnerability exception records are missing")
    result: dict[tuple[str, str], AuditException] = {}
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("vulnerability exception record is invalid")
        advisory_id = str(raw.get("id") or "").strip()
        package = str(raw.get("package") or "").strip().lower()
        versions_raw = raw.get("versions")
        versions = frozenset(str(value) for value in versions_raw) if isinstance(versions_raw, list) else frozenset()
        unused_surface = str(raw.get("unused_surface") or "").strip()
        owner = str(raw.get("owner") or "").strip()
        try:
            expires = date.fromisoformat(str(raw.get("expires") or ""))
        except ValueError as exc:
            raise ValueError(f"exception {advisory_id or '<unknown>'} has an invalid expiration") from exc
        if not advisory_id or not package or not versions or not unused_surface or not owner:
            raise ValueError("vulnerability exception record is incomplete")
        key = (package, advisory_id)
        if key in result:
            raise ValueError(f"duplicate vulnerability exception: {package} {advisory_id}")
        result[key] = AuditException(advisory_id, package, versions, unused_surface, owner, expires)
    return result


def evaluate_audit(
    audit_payload: dict[str, Any],
    exceptions: dict[tuple[str, str], AuditException],
    *,
    today: date | None = None,
) -> tuple[list[str], list[str]]:
    current_date = today or date.today()
    blockers: list[str] = []
    accepted: list[str] = []
    dependencies = audit_payload.get("dependencies")
    if not isinstance(dependencies, list):
        return ["pip-audit output did not contain a dependency list"], accepted
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        package = str(dependency.get("name") or "").lower()
        version = str(dependency.get("version") or "")
        vulnerabilities = dependency.get("vulns")
        for vulnerability in vulnerabilities if isinstance(vulnerabilities, list) else []:
            if not isinstance(vulnerability, dict):
                continue
            advisory_id = str(vulnerability.get("id") or "")
            exception = exceptions.get((package, advisory_id))
            label = f"{package}=={version} {advisory_id}"
            if exception is None:
                blockers.append(f"unapproved vulnerability: {label}")
            elif version not in exception.versions:
                blockers.append(f"exception does not cover installed version: {label}")
            elif current_date >= exception.expires:
                blockers.append(f"expired vulnerability exception: {label} (expired {exception.expires.isoformat()})")
            else:
                accepted.append(f"{label} accepted until {exception.expires.isoformat()} by {exception.owner}")
    return sorted(set(blockers)), sorted(set(accepted))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pip-audit with expiring, version-scoped exceptions")
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS)
    args = parser.parse_args(argv)
    try:
        exceptions = load_exceptions(args.exceptions)
    except ValueError as exc:
        print(f"dependency audit gate error: {exc}", file=sys.stderr)
        return 2
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "-r",
            str(args.requirements),
            "--format",
            "json",
            "--progress-spinner",
            "off",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        audit_payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stderr, file=sys.stderr)
        print("dependency audit gate error: pip-audit did not return JSON", file=sys.stderr)
        return 2
    blockers, accepted = evaluate_audit(audit_payload, exceptions)
    for message in accepted:
        print(message)
    if blockers:
        for message in blockers:
            print(message, file=sys.stderr)
        return 1
    print("dependency audit passed with no unapproved or expired advisories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
