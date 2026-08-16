from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stemsplat.security import (
    FileSecretStore,
    LanAuthManager,
    LanCertificateManager,
    LanSessionStore,
    LoginRateLimiter,
    hash_passcode,
    origin_matches,
    verify_passcode,
)


class MutableClock:
    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class SecurityHardeningTests(unittest.TestCase):
    def test_scrypt_passcode_hash_is_salted_and_verifiable(self) -> None:
        first = hash_passcode("correct horse battery staple")
        second = hash_passcode("correct horse battery staple")
        self.assertNotEqual(first, second)
        self.assertTrue(verify_passcode("correct horse battery staple", first))
        self.assertFalse(verify_passcode("incorrect", first))
        self.assertNotIn("correct horse", first)

    def test_sessions_are_256_bit_random_expiring_and_client_bound(self) -> None:
        clock = MutableClock()
        sessions = LanSessionStore(clock=clock)
        token, ttl = sessions.issue("192.168.1.20", "15m")
        self.assertGreaterEqual(len(token), 43)
        self.assertEqual(ttl, 900)
        self.assertTrue(sessions.valid(token, "192.168.1.20"))
        self.assertFalse(sessions.valid(token, "192.168.1.21"))
        sessions.revoke(token)
        self.assertFalse(sessions.valid(token, "192.168.1.20"))
        token, _ = sessions.issue("192.168.1.20", "15m")
        clock.value += 901
        self.assertFalse(sessions.valid(token, "192.168.1.20"))
        with self.assertRaises(ValueError):
            sessions.issue("192.168.1.20", "never")

    def test_five_failures_lock_client_for_fifteen_minutes(self) -> None:
        clock = MutableClock()
        limiter = LoginRateLimiter(clock=clock)
        for _ in range(4):
            self.assertTrue(limiter.record_failure("client").allowed)
        locked = limiter.record_failure("client")
        self.assertFalse(locked.allowed)
        self.assertEqual(locked.retry_after, 900)
        self.assertFalse(limiter.check("client").allowed)
        clock.value += 901
        self.assertTrue(limiter.check("client").allowed)

    def test_auth_manager_rate_limits_and_issues_session(self) -> None:
        manager = LanAuthManager()
        encoded = hash_passcode("12345678")
        with self.assertRaises(ValueError):
            manager.authenticate("client", "nope", encoded, "1d")
        token, ttl = manager.authenticate("client", "12345678", encoded, "1d")
        self.assertEqual(ttl, 86400)
        self.assertTrue(manager.sessions.valid(token, "client"))

    def test_origin_match_is_exact(self) -> None:
        self.assertTrue(origin_matches("https://music.local:9877", scheme="https", host_header="music.local:9877"))
        self.assertFalse(origin_matches("https://evil.example", scheme="https", host_header="music.local:9877"))

    @unittest.skipUnless(Path("/usr/bin/openssl").is_file(), "macOS openssl is required")
    def test_certificate_manager_reuses_ca_and_renews_for_san_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = FileSecretStore(root / "secrets")
            clock = MutableClock(1_800_000_000.0)
            manager = LanCertificateManager(root / "lan", secret_store=store, clock=clock)
            first = manager.ensure("stemsplat.local", ["192.168.1.10"])
            self.assertTrue(first.renewed)
            self.assertEqual(first.private_key_path.stat().st_mode & 0o777, 0o600)
            ca_before = first.ca_certificate_path.read_bytes()
            second = manager.ensure("stemsplat.local", ["192.168.1.10"])
            self.assertFalse(second.renewed)
            third = manager.ensure("stemsplat.local", ["192.168.1.11"])
            self.assertTrue(third.renewed)
            self.assertEqual(ca_before, third.ca_certificate_path.read_bytes())
            self.assertIn("192.168.1.11", third.san_names)
            clock.value += 24 * 24 * 60 * 60
            expiring = manager.ensure("stemsplat.local", ["192.168.1.11"])
            self.assertTrue(expiring.renewed)
            self.assertEqual(ca_before, expiring.ca_certificate_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
