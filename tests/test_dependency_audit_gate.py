from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from tools.pip_audit_gate import evaluate_audit, load_exceptions


class DependencyAuditGateTests(unittest.TestCase):
    def test_exception_is_version_scoped_and_expires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exceptions.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "exceptions": [
                            {
                                "id": "ADVISORY-1",
                                "package": "example",
                                "versions": ["1.0"],
                                "unused_surface": "test-only surface",
                                "owner": "release owner",
                                "expires": "2026-09-01",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            exceptions = load_exceptions(path)
            payload = {
                "dependencies": [
                    {"name": "example", "version": "1.0", "vulns": [{"id": "ADVISORY-1"}]}
                ]
            }
            blockers, accepted = evaluate_audit(payload, exceptions, today=date(2026, 8, 13))
            self.assertEqual(blockers, [])
            self.assertEqual(len(accepted), 1)
            blockers, _ = evaluate_audit(payload, exceptions, today=date(2026, 9, 1))
            self.assertIn("expired vulnerability exception", blockers[0])
            payload["dependencies"][0]["version"] = "1.1"
            blockers, _ = evaluate_audit(payload, exceptions, today=date(2026, 8, 13))
            self.assertIn("does not cover installed version", blockers[0])

    def test_unknown_advisory_fails_closed(self) -> None:
        payload = {"dependencies": [{"name": "example", "version": "1.0", "vulns": [{"id": "NEW"}]}]}
        blockers, accepted = evaluate_audit(payload, {}, today=date(2026, 8, 13))
        self.assertEqual(accepted, [])
        self.assertIn("unapproved vulnerability", blockers[0])


if __name__ == "__main__":
    unittest.main()
