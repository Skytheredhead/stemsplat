from __future__ import annotations

import base64
import contextlib
import ctypes
import hashlib
import ipaddress
import json
import secrets
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .atomic import atomic_write_bytes, atomic_write_json

SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SESSION_TTLS = {"15m": 15 * 60, "1d": 24 * 60 * 60, "1w": 7 * 24 * 60 * 60}
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
FORWARDED_HEADERS = frozenset(
    {
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-proto",
        "x-real-ip",
    }
)


class SecurityConfigurationError(RuntimeError):
    pass


def hash_passcode(passcode: str, *, salt: bytes | None = None) -> str:
    if not isinstance(passcode, str) or not passcode:
        raise ValueError("passcode is required")
    if len(passcode) > 128:
        raise ValueError("passcode is too long")
    actual_salt = salt or secrets.token_bytes(16)
    derived = hashlib.scrypt(
        passcode.encode("utf-8"),
        salt=actual_salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
        maxmem=64 * 1024 * 1024,
    )
    salt_text = base64.urlsafe_b64encode(actual_salt).decode("ascii").rstrip("=")
    digest_text = base64.urlsafe_b64encode(derived).decode("ascii").rstrip("=")
    return "scrypt$" + "$".join((str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P), salt_text, digest_text))


def _decode_urlsafe(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def verify_passcode(passcode: str, encoded: str) -> bool:
    try:
        algorithm, n_text, r_text, p_text, salt_text, digest_text = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        n, r, p = int(n_text), int(r_text), int(p_text)
        if (n, r, p) != (SCRYPT_N, SCRYPT_R, SCRYPT_P):
            return False
        salt = _decode_urlsafe(salt_text)
        expected = _decode_urlsafe(digest_text)
        if len(salt) != 16 or len(expected) != SCRYPT_DKLEN:
            return False
        actual = hashlib.scrypt(
            passcode.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=64 * 1024 * 1024,
        )
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(actual, expected)


@dataclass(frozen=True)
class AuthDecision:
    allowed: bool
    retry_after: int = 0


class LoginRateLimiter:
    def __init__(
        self,
        *,
        max_failures: int = 5,
        window_seconds: int = 5 * 60,
        lockout_seconds: int = 15 * 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self.clock = clock
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._locked_until: dict[str, float] = {}
        self._lock = threading.RLock()

    def check(self, client_id: str) -> AuthDecision:
        now = self.clock()
        with self._lock:
            locked_until = self._locked_until.get(client_id, 0.0)
            if locked_until > now:
                return AuthDecision(False, max(1, int(locked_until - now + 0.999)))
            self._locked_until.pop(client_id, None)
            failures = self._failures[client_id]
            cutoff = now - self.window_seconds
            while failures and failures[0] <= cutoff:
                failures.popleft()
            if not failures:
                self._failures.pop(client_id, None)
            return AuthDecision(True)

    def record_failure(self, client_id: str) -> AuthDecision:
        now = self.clock()
        with self._lock:
            current = self.check(client_id)
            if not current.allowed:
                return current
            failures = self._failures[client_id]
            failures.append(now)
            if len(failures) >= self.max_failures:
                self._locked_until[client_id] = now + self.lockout_seconds
                failures.clear()
                return AuthDecision(False, self.lockout_seconds)
            return AuthDecision(True)

    def record_success(self, client_id: str) -> None:
        with self._lock:
            self._failures.pop(client_id, None)
            self._locked_until.pop(client_id, None)


class LanSessionStore:
    """Restart-volatile, opaque server-side LAN sessions."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._sessions: dict[str, tuple[str, float]] = {}
        self._lock = threading.RLock()

    def issue(self, client_id: str, ttl_name: str) -> tuple[str, int]:
        ttl = SESSION_TTLS.get(ttl_name)
        if ttl is None:
            raise ValueError("unsupported session lifetime")
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._prune_locked()
            self._sessions[token] = (client_id, self.clock() + ttl)
        return token, ttl

    def valid(self, token: str | None, client_id: str) -> bool:
        if not token:
            return False
        with self._lock:
            self._prune_locked()
            session = self._sessions.get(token)
            if session is None or session[0] != client_id:
                self._sessions.pop(token, None)
                return False
            return True

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def revoke_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def _prune_locked(self) -> None:
        now = self.clock()
        for token, (_, expires_at) in list(self._sessions.items()):
            if expires_at <= now:
                self._sessions.pop(token, None)


class LanAuthManager:
    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self.sessions = LanSessionStore(clock=clock)
        self.rate_limiter = LoginRateLimiter(clock=clock)

    def authenticate(self, client_id: str, passcode: str, encoded_hash: str, ttl_name: str) -> tuple[str, int]:
        decision = self.rate_limiter.check(client_id)
        if not decision.allowed:
            raise PermissionError(str(decision.retry_after))
        if not verify_passcode(passcode, encoded_hash):
            decision = self.rate_limiter.record_failure(client_id)
            if not decision.allowed:
                raise PermissionError(str(decision.retry_after))
            raise ValueError("incorrect passcode")
        self.rate_limiter.record_success(client_id)
        return self.sessions.issue(client_id, ttl_name)


class SecretStore(Protocol):
    def get(self, account: str) -> bytes | None: ...

    def set(self, account: str, value: bytes) -> None: ...


class FileSecretStore:
    """Test/non-macOS secret store. Production macOS uses KeychainSecretStore."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, account: str) -> Path:
        return self.root / hashlib.sha256(account.encode("utf-8")).hexdigest()

    def get(self, account: str) -> bytes | None:
        path = self._path(account)
        return path.read_bytes() if path.is_file() else None

    def set(self, account: str, value: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self._path(account), value, mode=0o600)


class KeychainSecretStore:
    """Store CA key material in the user's macOS Keychain (Security.framework)."""

    ITEM_NOT_FOUND = -25300

    def __init__(self, service: str = "com.skylarenns.stemsplat.lan-ca") -> None:
        self.service = service
        try:
            self.framework = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
            self.core_foundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        except OSError as exc:
            raise SecurityConfigurationError("macOS Security.framework is unavailable") from exc
        self.framework.SecKeychainFindGenericPassword.restype = ctypes.c_int32
        self.framework.SecKeychainFindGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.framework.SecKeychainAddGenericPassword.restype = ctypes.c_int32
        self.framework.SecKeychainAddGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.framework.SecKeychainItemModifyAttributesAndData.restype = ctypes.c_int32
        self.framework.SecKeychainItemModifyAttributesAndData.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self.framework.SecKeychainItemFreeContent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core_foundation.CFRelease.argtypes = [ctypes.c_void_p]

    def _find(self, account: str) -> tuple[int, bytes | None, ctypes.c_void_p]:
        service = self.service.encode("utf-8")
        account_bytes = account.encode("utf-8")
        length = ctypes.c_uint32()
        data = ctypes.c_void_p()
        item = ctypes.c_void_p()
        status = int(
            self.framework.SecKeychainFindGenericPassword(
                None,
                len(service),
                service,
                len(account_bytes),
                account_bytes,
                ctypes.byref(length),
                ctypes.byref(data),
                ctypes.byref(item),
            )
        )
        value = ctypes.string_at(data, length.value) if status == 0 and data.value else None
        if data.value:
            self.framework.SecKeychainItemFreeContent(None, data)
        return status, value, item

    def get(self, account: str) -> bytes | None:
        status, value, item = self._find(account)
        if item.value:
            self.core_foundation.CFRelease(item)
        if status == self.ITEM_NOT_FOUND:
            return None
        if status != 0:
            raise SecurityConfigurationError("could not read the Stemsplat CA key from macOS Keychain")
        return value

    def set(self, account: str, value: bytes) -> None:
        encoded = base64.b64encode(value)
        status, _existing, item = self._find(account)
        buffer = ctypes.create_string_buffer(encoded)
        try:
            if status == 0 and item.value:
                result = int(
                    self.framework.SecKeychainItemModifyAttributesAndData(
                        item,
                        None,
                        len(encoded),
                        ctypes.cast(buffer, ctypes.c_void_p),
                    )
                )
            elif status == self.ITEM_NOT_FOUND:
                service = self.service.encode("utf-8")
                account_bytes = account.encode("utf-8")
                result = int(
                    self.framework.SecKeychainAddGenericPassword(
                        None,
                        len(service),
                        service,
                        len(account_bytes),
                        account_bytes,
                        len(encoded),
                        ctypes.cast(buffer, ctypes.c_void_p),
                        None,
                    )
                )
            else:
                result = status
        finally:
            if item.value:
                self.core_foundation.CFRelease(item)
        if result != 0:
            raise SecurityConfigurationError("could not store the Stemsplat CA key in macOS Keychain")

    def get_pem(self, account: str) -> bytes | None:
        encoded = self.get(account)
        if encoded is None:
            return None
        try:
            return base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise SecurityConfigurationError("the Stemsplat Keychain CA record is invalid") from exc

    def set_pem(self, account: str, value: bytes) -> None:
        self.set(account, value)


@dataclass(frozen=True)
class LanCertificate:
    certificate_path: Path
    private_key_path: Path
    ca_certificate_path: Path
    fingerprint_sha256: str
    san_names: tuple[str, ...]
    renewed: bool


class LanCertificateManager:
    CA_ACCOUNT = "persistent-p256-ca-private-key"

    def __init__(
        self,
        root: Path,
        *,
        secret_store: SecretStore | None = None,
        openssl: str = "/usr/bin/openssl",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = root
        self.openssl = openssl
        self.clock = clock
        self.secret_store = secret_store or KeychainSecretStore()
        self.ca_cert_path = root / "stemsplat-local-ca.pem"
        self.leaf_cert_path = root / "lan-server.pem"
        self.leaf_key_path = root / "lan-server-key.pem"
        self.metadata_path = root / "certificate.json"
        self._lock = threading.RLock()

    def ensure(self, hostname: str, addresses: Iterable[str]) -> LanCertificate:
        sans = self._normalize_sans(hostname, addresses)
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            ca_key = self._ensure_ca()
            renewed = self._needs_renewal(sans)
            if renewed:
                self._issue_leaf(ca_key, sans)
            self.leaf_key_path.chmod(0o600)
            return LanCertificate(
                certificate_path=self.leaf_cert_path,
                private_key_path=self.leaf_key_path,
                ca_certificate_path=self.ca_cert_path,
                fingerprint_sha256=self.fingerprint(),
                san_names=sans,
                renewed=renewed,
            )

    def fingerprint(self) -> str:
        pem = self.ca_cert_path.read_text(encoding="ascii")
        der = ssl.PEM_cert_to_DER_cert(pem)
        digest = hashlib.sha256(der).hexdigest().upper()
        return ":".join(digest[index : index + 2] for index in range(0, len(digest), 2))

    def public_status(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        with contextlib.suppress(Exception):
            loaded = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        raw_sans = metadata.get("san_names")
        san_names = [str(value) for value in raw_sans] if isinstance(raw_sans, list) else []
        return {
            "available": self.ca_cert_path.is_file() and self.leaf_cert_path.is_file(),
            "fingerprint_sha256": self.fingerprint() if self.ca_cert_path.is_file() else "",
            "san_names": san_names,
            "not_after": str(metadata.get("not_after") or ""),
            "install_instructions": (
                "Export the CA from the local desktop settings, AirDrop it to the device, install the profile, "
                "then enable full trust under Certificate Trust Settings."
            ),
        }

    def _secret_get_pem(self) -> bytes | None:
        getter = getattr(self.secret_store, "get_pem", None)
        return getter(self.CA_ACCOUNT) if callable(getter) else self.secret_store.get(self.CA_ACCOUNT)

    def _secret_set_pem(self, value: bytes) -> None:
        setter = getattr(self.secret_store, "set_pem", None)
        if callable(setter):
            setter(self.CA_ACCOUNT, value)
        else:
            self.secret_store.set(self.CA_ACCOUNT, value)

    def _ensure_ca(self) -> bytes:
        existing_key = self._secret_get_pem()
        if existing_key is not None and self.ca_cert_path.is_file():
            return existing_key
        with tempfile.TemporaryDirectory(prefix="stemsplat-ca-") as temporary:
            key_path = Path(temporary) / "ca-key.pem"
            cert_path = Path(temporary) / "ca.pem"
            self._run("ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(key_path))
            self._run(
                "req",
                "-x509",
                "-new",
                "-key",
                str(key_path),
                "-sha256",
                "-days",
                "3650",
                "-subj",
                "/CN=Stemsplat Local CA/O=Stemsplat",
                "-addext",
                "basicConstraints=critical,CA:TRUE,pathlen:0",
                "-addext",
                "keyUsage=critical,keyCertSign,cRLSign",
                "-out",
                str(cert_path),
            )
            ca_key = key_path.read_bytes()
            self._secret_set_pem(ca_key)
            atomic_write_bytes(self.ca_cert_path, cert_path.read_bytes(), mode=0o644)
            return ca_key

    def _needs_renewal(self, sans: tuple[str, ...]) -> bool:
        if not self.leaf_cert_path.is_file() or not self.leaf_key_path.is_file() or not self.metadata_path.is_file():
            return True
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if tuple(metadata.get("san_names") or ()) != sans:
                return True
            not_after = datetime.fromisoformat(str(metadata["not_after"]))
            return not_after.timestamp() - self.clock() <= 7 * 24 * 60 * 60
        except Exception:
            return True

    def _issue_leaf(self, ca_key: bytes, sans: tuple[str, ...]) -> None:
        with tempfile.TemporaryDirectory(prefix="stemsplat-leaf-") as temporary:
            temp = Path(temporary)
            ca_key_path = temp / "ca-key.pem"
            leaf_key_path = temp / "leaf-key.pem"
            leaf_cert_path = temp / "leaf.pem"
            csr_path = temp / "leaf.csr"
            config_path = temp / "extensions.cnf"
            ca_key_path.write_bytes(ca_key)
            ca_key_path.chmod(0o600)
            self._run("ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(leaf_key_path))
            common_name = next((value for value in sans if not _is_ip(value)), sans[0])
            self._run(
                "req",
                "-new",
                "-key",
                str(leaf_key_path),
                "-subj",
                f"/CN={common_name}",
                "-out",
                str(csr_path),
            )
            entries: list[str] = []
            dns_index = 0
            ip_index = 0
            for value in sans:
                if _is_ip(value):
                    ip_index += 1
                    entries.append(f"IP.{ip_index} = {value}")
                else:
                    dns_index += 1
                    entries.append(f"DNS.{dns_index} = {value}")
            config_path.write_text(
                "[v3_req]\n"
                "basicConstraints = critical,CA:FALSE\n"
                "keyUsage = critical,digitalSignature,keyAgreement\n"
                "extendedKeyUsage = serverAuth\n"
                "subjectAltName = @alt_names\n"
                "[alt_names]\n"
                + "\n".join(entries)
                + "\n",
                encoding="ascii",
            )
            self._run(
                "x509",
                "-req",
                "-in",
                str(csr_path),
                "-CA",
                str(self.ca_cert_path),
                "-CAkey",
                str(ca_key_path),
                "-set_serial",
                f"0x{secrets.token_hex(16)}",
                "-days",
                "30",
                "-sha256",
                "-extfile",
                str(config_path),
                "-extensions",
                "v3_req",
                "-out",
                str(leaf_cert_path),
            )
            atomic_write_bytes(self.leaf_key_path, leaf_key_path.read_bytes(), mode=0o600)
            atomic_write_bytes(self.leaf_cert_path, leaf_cert_path.read_bytes(), mode=0o644)
        not_after = datetime.fromtimestamp(self.clock() + 30 * 24 * 60 * 60, tz=timezone.utc)
        atomic_write_json(
            self.metadata_path,
            {
                "san_names": list(sans),
                "not_after": not_after.isoformat(),
                "issued_at": datetime.fromtimestamp(self.clock(), tz=timezone.utc).isoformat(),
            },
            mode=0o600,
        )

    def _run(self, *arguments: str) -> None:
        result = subprocess.run([self.openssl, *arguments], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SecurityConfigurationError("local TLS certificate generation failed")

    @staticmethod
    def _normalize_sans(hostname: str, addresses: Iterable[str]) -> tuple[str, ...]:
        values: set[str] = set()
        normalized_hostname = hostname.strip().lower().rstrip(".")
        if normalized_hostname:
            values.add(normalized_hostname)
        for value in addresses:
            candidate = str(value or "").strip()
            with contextlib.suppress(ValueError):
                address = ipaddress.ip_address(candidate)
                if not address.is_loopback and not address.is_unspecified:
                    values.add(str(address))
        if not values:
            fallback = socket.gethostname().split(".", 1)[0].strip() or "stemsplat"
            values.add(f"{fallback}.local")
        return tuple(sorted(values, key=lambda value: (_is_ip(value), value)))


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def normalized_host(host_header: str) -> str:
    raw = str(host_header or "").strip().lower()
    if raw.startswith("[") and "]" in raw:
        return raw[1 : raw.index("]")]
    return raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw


def request_has_forwarded_identity(headers: Iterable[tuple[str, str]]) -> bool:
    return any(str(name).lower() in FORWARDED_HEADERS for name, _ in headers)


def origin_matches(origin: str, *, scheme: str, host_header: str) -> bool:
    expected = f"{scheme.lower()}://{host_header.lower()}"
    return secrets.compare_digest(str(origin or "").rstrip("/").lower(), expected.rstrip("/"))


SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}
