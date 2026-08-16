from __future__ import annotations

import contextlib
import os
import unittest

from fastapi.testclient import TestClient

os.environ.setdefault("STEMSPLAT_DISABLE_BACKGROUND_THREADS", "1")

import main
from stemsplat.security import LanAuthManager, hash_passcode


class LanSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = contextlib.ExitStack()
        with main.compat_settings_lock:
            self.original_settings = dict(main._compat_settings)
            main._compat_settings.clear()
            main._compat_settings.update(main._normalize_settings_payload({}))
        self.original_auth_manager = main.lan_auth_manager
        self.original_runtime_status_provider = main.app.state.runtime_status_provider
        main.set_runtime_status_provider(
            lambda: {
                "lan_display": "192.168.1.20:9877",
                "lan_local_display": "stemsplat.local:9877",
                "client_url": "http://127.0.0.1:9876/",
                "kill_command": "kill 123",
            }
        )
        main.lan_auth_manager = LanAuthManager()
        self.lan_app = main.create_lan_app()
        self.client = TestClient(self.lan_app, base_url="https://192.168.1.20:9877")

    def tearDown(self) -> None:
        self.client.close()
        main.lan_auth_manager = self.original_auth_manager
        main.set_runtime_status_provider(self.original_runtime_status_provider)
        with main.compat_settings_lock:
            main._compat_settings.clear()
            main._compat_settings.update(self.original_settings)
        self.stack.close()

    def _enable(self) -> None:
        encoded = hash_passcode("correct-passcode")
        with main.compat_settings_lock:
            main._compat_settings.update(
                {
                    "lan_access_enabled": True,
                    "lan_passcode_enabled": True,
                    "lan_passcode_hash": encoded,
                    "lan_passcode_ttl": "15m",
                }
            )

    @staticmethod
    def _origin() -> dict[str, str]:
        return {"Origin": "https://192.168.1.20:9877"}

    def test_lan_is_disabled_by_default(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"], "LAN access is disabled")

    def test_direct_main_cli_rejects_non_loopback_bind(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main.cli_main(["--host", "0.0.0.0", "--port", "9988"])  # noqa: S104 - intentional rejection case
        self.assertEqual(caught.exception.code, 2)

    def test_unauthenticated_surface_only_serves_login_resources(self) -> None:
        self._enable()
        login = self.client.get("/")
        self.assertEqual(login.status_code, 200)
        self.assertIn("lan-login.js", login.text)
        self.assertEqual(login.headers["strict-transport-security"], "max-age=31536000")
        self.assertEqual(login.headers["x-frame-options"], "DENY")
        self.assertIn("default-src 'self'", login.headers["content-security-policy"])

        self.assertEqual(self.client.get("/assets/lan-login.js").status_code, 200)
        self.assertEqual(self.client.get("/assets/app.js").status_code, 401)
        self.assertEqual(self.client.get("/api/tasks").status_code, 401)

    def test_auth_requires_same_origin_and_sets_secure_cookie(self) -> None:
        self._enable()
        missing_origin = self.client.post("/api/lan/auth", json={"passcode": "correct-passcode"})
        self.assertEqual(missing_origin.status_code, 403)

        bad = self.client.post(
            "/api/lan/auth",
            json={"passcode": "incorrect"},
            headers=self._origin(),
        )
        self.assertEqual(bad.status_code, 401)

        authenticated = self.client.post(
            "/api/lan/auth",
            json={"passcode": "correct-passcode"},
            headers=self._origin(),
        )
        self.assertEqual(authenticated.status_code, 200)
        cookie = authenticated.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("Secure", cookie)
        self.assertIn("SameSite=strict", cookie)
        self.assertEqual(self.client.get("/api/tasks?limit=1").status_code, 200)

    def test_local_only_routes_are_not_registered(self) -> None:
        self._enable()
        authenticated = self.client.post(
            "/api/lan/auth",
            json={"passcode": "correct-passcode"},
            headers=self._origin(),
        )
        self.assertEqual(authenticated.status_code, 200)
        denied = self.client.post(
            "/api/lan/config",
            json={"enabled": False},
            headers=self._origin(),
        )
        self.assertEqual(denied.status_code, 404)
        route_names = {getattr(route, "name", "") for route in self.lan_app.router.routes}
        self.assertTrue(main.LAN_EXCLUDED_ROUTE_NAMES.isdisjoint(route_names))

    def test_host_origin_and_forwarded_headers_are_rejected(self) -> None:
        self._enable()
        hostile_host = self.client.get("/", headers={"Host": "evil.example"})
        self.assertEqual(hostile_host.status_code, 400)
        forwarded = self.client.get("/", headers={"X-Forwarded-For": "127.0.0.1"})
        self.assertEqual(forwarded.status_code, 400)
        hostile_origin = self.client.post(
            "/api/lan/auth",
            json={"passcode": "correct-passcode"},
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(hostile_origin.status_code, 403)

    def test_private_api_responses_are_not_cached(self) -> None:
        self._enable()
        response = self.client.get("/api/lan/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_lan_settings_redact_host_only_values(self) -> None:
        self._enable()
        authenticated = self.client.post(
            "/api/lan/auth",
            json={"passcode": "correct-passcode"},
            headers=self._origin(),
        )
        self.assertEqual(authenticated.status_code, 200)
        response = self.client.get("/api/settings")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["output_root"], "")
        self.assertEqual(payload["runtime"]["client_url"], "")
        self.assertEqual(payload["runtime"]["kill_command"], "")

    def test_unlisted_private_host_is_rejected(self) -> None:
        self._enable()
        response = self.client.get("/", headers={"Host": "192.168.1.99:9877"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
