from __future__ import annotations

import unittest

from stemsplat.version import __version__
from tools.release_gate import _quality_report_blockers


class ReleaseGateTests(unittest.TestCase):
    def _passing_report(self) -> dict:
        return {
            "schema_version": 1,
            "command": "report",
            "provenance": {"lane": "reference_safe", "stemsplat_version": __version__},
            "passed": True,
            "blockers": [],
            "manual_blind_listening_signoff": {
                "passed": True,
                "excerpts": [{"sha256": str(index) * 64, "result": "pass"} for index in range(1, 6)],
            },
            "public_quality": {"passed": True},
            "performance": {
                "modes": {
                    "vocals@device": {
                        "trials": {"cold": {"passed": True}, "warm": {"passed": True}}
                    }
                }
            },
            "reports": [
                {"command": "prepare"},
                {"command": "baseline"},
                {"command": "compare"},
                {"command": "benchmark"},
                {"command": "benchmark"},
            ],
        }

    def test_complete_quality_report_is_accepted(self) -> None:
        self.assertEqual(_quality_report_blockers(self._passing_report()), [])

    def test_hand_written_pass_boolean_cannot_bypass_evidence(self) -> None:
        report = self._passing_report()
        report["performance"] = {"modes": {}}
        report["reports"] = []
        blockers = _quality_report_blockers(report)
        self.assertIn("quality report lacks paired performance evidence", blockers)
        self.assertIn("quality report lacks required source reports", blockers)


if __name__ == "__main__":
    unittest.main()
