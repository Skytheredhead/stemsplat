from __future__ import annotations

import re
import unittest
from pathlib import Path

from stemsplat.version import __version__

ROOT = Path(__file__).resolve().parents[1]


class PackagedResourceTests(unittest.TestCase):
    def test_ui_assets_are_present_and_offline(self) -> None:
        for name in ("index.html", "mobile.html", "lan_login.html", "app.css", "app.js", "mobile.js", "lan-login.js"):
            self.assertTrue((ROOT / "web" / name).is_file(), name)
        for name in ("index.html", "mobile.html", "lan_login.html", "install.html"):
            source = (ROOT / "web" / name).read_text(encoding="utf-8")
            self.assertNotRegex(source, r"fonts\.(?:googleapis|gstatic)\.com|cdn\.tailwindcss\.com")
            self.assertIsNone(re.search(r"<script(?!\s+src=)[^>]*>", source, re.IGNORECASE))
            self.assertIsNone(re.search(r"\son[a-z]+\s*=", source, re.IGNORECASE))

    def test_single_version_source_is_used(self) -> None:
        self.assertEqual(__version__, "0.4.3")
        build_script = (ROOT / "build_app.sh").read_text(encoding="utf-8")
        self.assertIn("from stemsplat.version import __version__", build_script)
        self.assertNotIn("CFBundleShortVersionString 0.4.3", build_script)


if __name__ == "__main__":
    unittest.main()
