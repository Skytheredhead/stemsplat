from __future__ import annotations

import argparse
import contextlib
import logging
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn

from app_paths import RUNTIME_DIR, ensure_app_dirs

ensure_app_dirs()

from main import app, set_runtime_status_provider

logger = logging.getLogger("stemsplat.launcher")
PREFERRED_PORT = 8000
FALLBACK_SCAN = 32

try:
    import webview
except Exception:  # pragma: no cover - fallback kept for non-desktop usage
    webview = None  # type: ignore[assignment]


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred: int) -> int:
    for candidate in range(preferred, preferred + FALLBACK_SCAN):
        if _port_available(host, candidate):
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with contextlib.suppress(Exception):
            with urllib.request.urlopen(url, timeout=1):
                return True
        time.sleep(0.2)
    return False


def _open_url(url: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", url])
        return
    webbrowser.open(url, new=1, autoraise=True)


def _local_lan_ip() -> str:
    with contextlib.suppress(OSError):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = str(sock.getsockname()[0] or "").strip()
            if ip and not ip.startswith("127."):
                return ip
    with contextlib.suppress(OSError):
        hostname = socket.gethostname()
        for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM):
            if family != socket.AF_INET:
                continue
            ip = str(sockaddr[0] or "").strip()
            if ip and not ip.startswith("127."):
                return ip
    return ""


class _ThreadedServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # pragma: no cover - GUI thread owns lifecycle
        return


class ServerController:
    def __init__(self, bind_host: str, client_host: str, preferred_port: int) -> None:
        self.bind_host = bind_host
        self.client_host = client_host
        self.preferred_port = preferred_port
        self.lan_ip = _local_lan_ip()
        self._lock = threading.RLock()
        self._server: _ThreadedServer | None = None
        self._thread: threading.Thread | None = None
        self._current_port: int | None = None
        self._port_conflict = False
        self._port_notice_acknowledged = False
        self._windowed = False

    def client_url(self, port: int | None = None) -> str:
        current = port if port is not None else (self._current_port or self.preferred_port)
        return f"http://{self.client_host}:{current}/"

    def lan_url(self, port: int | None = None) -> str:
        if not self.lan_ip:
            return ""
        current = port if port is not None else (self._current_port or self.preferred_port)
        return f"http://{self.lan_ip}:{current}/"

    def runtime_status(self) -> dict[str, Any]:
        with self._lock:
            current_port = self._current_port or self.preferred_port
            port_conflict = self._port_conflict
            acknowledged = self._port_notice_acknowledged
            windowed = self._windowed
        lan_url = self.lan_url(current_port)
        payload = {
            "windowed": windowed,
            "preferred_port": self.preferred_port,
            "current_port": current_port,
            "client_url": self.client_url(current_port),
            "lan_url": lan_url,
            "lan_display": f"{self.lan_ip}:{current_port}" if self.lan_ip else "",
            "port_conflict": port_conflict,
            "show_port_notice": port_conflict and not acknowledged,
            "kill_command": f"kill -9 $(lsof -ti tcp:{self.preferred_port})",
        }
        if port_conflict:
            payload["port_notice"] = (
                f"port {self.preferred_port} is already in use. stemsplat is running on port {current_port}."
            )
        else:
            payload["port_notice"] = ""
        return payload

    def set_windowed(self, enabled: bool) -> None:
        with self._lock:
            self._windowed = enabled

    def start_initial(self) -> dict[str, Any]:
        port = self.preferred_port if _port_available(self.bind_host, self.preferred_port) else _pick_port(
            self.bind_host, self.preferred_port + 1
        )
        self._start_server(port)
        with self._lock:
            self._port_conflict = port != self.preferred_port
            self._port_notice_acknowledged = port == self.preferred_port
        return self.runtime_status()

    def stop(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
            self._current_port = None
        if server is None or thread is None:
            return
        server.should_exit = True
        thread.join(timeout=6)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=2)

    def wait(self) -> None:
        thread: threading.Thread | None
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join()

    def acknowledge_port_conflict(self) -> dict[str, Any]:
        with self._lock:
            self._port_notice_acknowledged = True
        return self.runtime_status()

    def retry_preferred_port(self) -> dict[str, Any]:
        with self._lock:
            current_port = self._current_port or self.preferred_port
        if current_port == self.preferred_port:
            with self._lock:
                self._port_conflict = False
                self._port_notice_acknowledged = True
            payload = self.runtime_status()
            payload["switched"] = False
            return payload
        if not _port_available(self.bind_host, self.preferred_port):
            with self._lock:
                self._port_conflict = True
                self._port_notice_acknowledged = False
            payload = self.runtime_status()
            payload["switched"] = False
            payload["error"] = f"port {self.preferred_port} is still in use."
            return payload

        self.stop()
        try:
            self._start_server(self.preferred_port)
        except Exception as exc:
            logger.exception("failed to switch back to preferred port")
            try:
                self._start_server(current_port)
            except Exception:
                logger.exception("failed to restore previous port %s", current_port)
            with self._lock:
                self._port_conflict = self._current_port != self.preferred_port
                self._port_notice_acknowledged = False
            payload = self.runtime_status()
            payload["switched"] = False
            payload["error"] = f"could not switch to port {self.preferred_port}: {exc}"
            return payload

        with self._lock:
            self._port_conflict = False
            self._port_notice_acknowledged = True
        payload = self.runtime_status()
        payload["switched"] = True
        return payload

    def _start_server(self, port: int) -> None:
        config = uvicorn.Config(app, host=self.bind_host, port=port, reload=False, log_level="info")
        server = _ThreadedServer(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        if not _wait_until_ready(self.client_url(port)):
            server.should_exit = True
            server.force_exit = True
            thread.join(timeout=2)
            raise RuntimeError(f"server did not become ready on port {port}")
        with self._lock:
            self._server = server
            self._thread = thread
            self._current_port = port


class DesktopApi:
    def __init__(self, controller: ServerController) -> None:
        self.controller = controller

    def get_runtime_status(self) -> dict[str, Any]:
        return self.controller.runtime_status()

    def retry_preferred_port(self) -> dict[str, Any]:
        return self.controller.retry_preferred_port()

    def acknowledge_port_conflict(self) -> dict[str, Any]:
        return self.controller.acknowledge_port_conflict()


def _open_browser_when_ready(url: str) -> None:
    if _wait_until_ready(url):
        logger.info("opening browser at %s", url)
        _open_url(url)
    else:
        logger.error("server did not become ready in time for %s", url)


def _run_windowed_app(controller: ServerController) -> int:
    if webview is None:
        raise RuntimeError("pywebview is not available")
    controller.set_windowed(True)
    storage_path = RUNTIME_DIR / "webview"
    storage_path.mkdir(parents=True, exist_ok=True)
    api = DesktopApi(controller)
    window = webview.create_window(
        "stemsplat",
        controller.client_url(),
        js_api=api,
        width=1280,
        height=900,
        min_size=(1100, 760),
        text_select=False,
        background_color="#0B1A1F",
    )

    def _on_closed(*_args: Any) -> None:
        controller.stop()

    if window is not None:
        window.events.closed += _on_closed
    webview.start(
        gui="cocoa",
        debug=False,
        private_mode=False,
        storage_path=str(storage_path),
        icon=None,
    )
    controller.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the stemsplat desktop app.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--client-host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    preferred_port = args.port if args.port is not None else PREFERRED_PORT
    controller = ServerController(args.host, args.client_host, preferred_port)
    set_runtime_status_provider(controller.runtime_status)
    controller.start_initial()

    logger.info("starting stemsplat on %s", controller.client_url())

    if args.no_browser:
        try:
            controller.wait()
        except KeyboardInterrupt:
            logger.info("interrupt received; shutting down")
        finally:
            controller.stop()
        return 0

    if sys.platform == "darwin" and webview is not None:
        return _run_windowed_app(controller)

    threading.Thread(target=_open_browser_when_ready, args=(controller.client_url(),), daemon=True).start()
    try:
        controller.wait()
    except KeyboardInterrupt:
        logger.info("interrupt received; shutting down")
    finally:
        controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
