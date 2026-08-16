from __future__ import annotations

import argparse
import contextlib
import logging
import os
import socket
import ssl
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

from main import (
    _compat_settings_payload,
    _get_certificate_manager,
    _reset_lan_auth_sessions,
    _task_runner_main,
    app,
    create_lan_app,
    set_lan_runtime_controller,
    set_runtime_status_provider,
)

logger = logging.getLogger("stemsplat.launcher")
PREFERRED_PORT = 9876
LAN_PORT = 9877
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


def _wait_until_ready(url: str, timeout: float = 30.0, *, ca_file: Path | None = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with contextlib.suppress(Exception):
            kwargs: dict[str, Any] = {"timeout": 1}
            if url.lower().startswith("https://"):
                if ca_file is None:
                    raise ValueError("an HTTPS readiness probe requires its issuing CA")
                kwargs["context"] = ssl.create_default_context(cafile=str(ca_file))
            with urllib.request.urlopen(url, **kwargs):
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


def _current_network_name() -> str:
    if sys.platform != "darwin":
        return ""
    try:
        hardware = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        wifi_device = ""
        current_port = ""
        for line in hardware.splitlines():
            stripped = line.strip()
            if stripped.startswith("Hardware Port:"):
                current_port = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Device:") and current_port in {"Wi-Fi", "AirPort"}:
                wifi_device = stripped.split(":", 1)[1].strip()
                break
        if not wifi_device:
            return ""
        current = subprocess.run(
            ["networksetup", "-getairportnetwork", wifi_device],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if ":" in current:
            return current.split(":", 1)[1].strip()
    except Exception:
        logger.debug("failed to determine network name", exc_info=True)
    return ""


def _local_hostname() -> str:
    name = ""
    if sys.platform == "darwin":
        with contextlib.suppress(Exception):
            name = subprocess.run(
                ["scutil", "--get", "LocalHostName"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
    if not name:
        with contextlib.suppress(Exception):
            name = socket.gethostname().strip()
    name = str(name or "").strip().rstrip(".")
    if not name:
        return ""
    if "." in name and name.lower().endswith(".local"):
        return name
    if "." in name:
        name = name.split(".", 1)[0]
    return f"{name}.local"


class _ThreadedServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # pragma: no cover - GUI thread owns lifecycle
        return


class ServerController:
    def __init__(self, bind_host: str, client_host: str, preferred_port: int) -> None:
        self.bind_host = bind_host
        self.client_host = client_host
        self.preferred_port = preferred_port
        self.lan_ip = _local_lan_ip()
        self.local_hostname = _local_hostname()
        self.network_name = _current_network_name()
        self._lock = threading.RLock()
        self._server: _ThreadedServer | None = None
        self._thread: threading.Thread | None = None
        self._current_port: int | None = None
        self._port_conflict = False
        self._port_notice_acknowledged = False
        self._windowed = False
        self.lan_controller: LanServerController | None = None

    def client_url(self, port: int | None = None) -> str:
        current = port if port is not None else (self._current_port or self.preferred_port)
        return f"http://{self.client_host}:{current}/"

    def lan_url(self, port: int | None = None) -> str:
        if not self.lan_ip or self.lan_controller is None or not self.lan_controller.running:
            return ""
        return f"https://{self.lan_ip}:{self.lan_controller.port}/"

    def lan_local_url(self, port: int | None = None) -> str:
        if not self.local_hostname or self.lan_controller is None or not self.lan_controller.running:
            return ""
        return f"https://{self.local_hostname}:{self.lan_controller.port}/"

    def runtime_status(self) -> dict[str, Any]:
        with self._lock:
            current_port = self._current_port or self.preferred_port
            port_conflict = self._port_conflict
            acknowledged = self._port_notice_acknowledged
            windowed = self._windowed
        lan_url = self.lan_url(current_port)
        lan_local_url = self.lan_local_url(current_port)
        payload = {
            "windowed": windowed,
            "preferred_port": self.preferred_port,
            "current_port": current_port,
            "client_url": self.client_url(current_port),
            "lan_url": lan_url,
            "lan_local_url": lan_local_url,
            "lan_display": f"{self.lan_ip}:{LAN_PORT}" if lan_url else "",
            "lan_local_display": f"{self.local_hostname}:{LAN_PORT}" if lan_local_url else "",
            "lan_enabled": bool(self.lan_controller and self.lan_controller.running),
            "network_name": self.network_name,
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
        if self.lan_controller is not None:
            self.lan_controller.stop()
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

    def request_shutdown(self, grace_timeout: float = 0.2) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
        if server is None or thread is None:
            return
        server.should_exit = True
        thread.join(timeout=max(0.0, grace_timeout))
        if thread.is_alive():
            server.force_exit = True

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
        if self.lan_controller is not None:
            with contextlib.suppress(Exception):
                self.lan_controller.sync()
        payload = self.runtime_status()
        payload["switched"] = True
        return payload

    def _start_server(self, port: int) -> None:
        config = uvicorn.Config(
            app,
            host=self.bind_host,
            port=port,
            reload=False,
            log_level="info",
            proxy_headers=False,
            forwarded_allow_ips="",
        )
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


class LanServerController:
    def __init__(self, desktop: ServerController, port: int = LAN_PORT) -> None:
        self.desktop = desktop
        self.port = port
        self._lock = threading.RLock()
        self._server: _ThreadedServer | None = None
        self._thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        desktop.lan_controller = self

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive() and self._server)

    def sync(self) -> None:
        enabled = bool(_compat_settings_payload().get("lan_access_enabled"))
        if not enabled:
            self.stop()
            return
        certificate = _get_certificate_manager().ensure(
            self.desktop.local_hostname,
            [self.desktop.lan_ip],
        )
        if self.running:
            self.stop(stop_monitor=False)
        if not _port_available("0.0.0.0", self.port):
            raise RuntimeError(f"LAN port {self.port} is already in use")
        lan_app = create_lan_app()
        config = uvicorn.Config(
            lan_app,
            host="0.0.0.0",
            port=self.port,
            reload=False,
            log_level="info",
            ssl_certfile=str(certificate.certificate_path),
            ssl_keyfile=str(certificate.private_key_path),
            proxy_headers=False,
            forwarded_allow_ips="",
        )
        server = _ThreadedServer(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        ready_host = self.desktop.lan_ip or self.desktop.local_hostname
        if not ready_host or not _wait_until_ready(
            f"https://{ready_host}:{self.port}/",
            ca_file=certificate.ca_certificate_path,
        ):
            server.should_exit = True
            server.force_exit = True
            thread.join(timeout=2)
            raise RuntimeError("LAN HTTPS listener did not become ready")
        with self._lock:
            self._server = server
            self._thread = thread
        self._ensure_monitor()
        logger.info("LAN HTTPS listener started on port %s", self.port)

    def _ensure_monitor(self) -> None:
        with self._lock:
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                return
            self._monitor_stop.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(6 * 60 * 60):
            if not bool(_compat_settings_payload().get("lan_access_enabled")):
                continue
            try:
                self.desktop.lan_ip = _local_lan_ip()
                self.desktop.local_hostname = _local_hostname()
                certificate = _get_certificate_manager().ensure(
                    self.desktop.local_hostname,
                    [self.desktop.lan_ip],
                )
                if certificate.renewed:
                    _reset_lan_auth_sessions()
                    self.sync()
            except Exception:
                logger.exception("LAN certificate renewal check failed; existing listener remains active")

    def stop(self, *, stop_monitor: bool = True) -> None:
        if stop_monitor:
            self._monitor_stop.set()
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
        if server is None or thread is None:
            return
        server.should_exit = True
        thread.join(timeout=6)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=2)
        if stop_monitor:
            with self._lock:
                monitor = self._monitor_thread
                self._monitor_thread = None
            if monitor is not None and monitor is not threading.current_thread():
                monitor.join(timeout=1)


class DesktopApi:
    def __init__(self, controller: ServerController) -> None:
        self.controller = controller
        self.window: Any | None = None
        self._quitting = threading.Event()

    def get_runtime_status(self) -> dict[str, Any]:
        return self.controller.runtime_status()

    def retry_preferred_port(self) -> dict[str, Any]:
        return self.controller.retry_preferred_port()

    def acknowledge_port_conflict(self) -> dict[str, Any]:
        return self.controller.acknowledge_port_conflict()

    def close_window(self) -> None:
        if self._quitting.is_set():
            return
        self._quitting.set()
        if self.window is not None:
            with contextlib.suppress(Exception):
                self.window.hide()

        def _cleanup_and_exit() -> None:
            with contextlib.suppress(Exception):
                self.controller.request_shutdown(0.2)
            with contextlib.suppress(Exception):
                if self.window is not None:
                    self.window.destroy()
            os._exit(0)

        threading.Thread(target=_cleanup_and_exit, daemon=True).start()

    def minimize_window(self) -> None:
        if self.window is not None:
            self.window.minimize()

    def toggle_fullscreen_window(self) -> None:
        if self.window is not None:
            self.window.toggle_fullscreen()

    def pick_media_files(self) -> list[str]:
        if self.window is None or webview is None:
            return []
        try:
            result = self.window.create_file_dialog(
                webview.FileDialog.OPEN,
                allow_multiple=True,
                file_types=(
                    "Media (*.wav;*.wave;*.mp3;*.m4a;*.aac;*.flac;*.ogg;*.oga;*.aif;*.aiff;*.alac;*.opus;*.mp4;*.m4v;*.mov;*.webm;*.mkv;*.avi)",
                ),
            )
        except Exception:
            logger.exception("media picker failed")
            return []
        return [str(path) for path in (result or []) if path]

    def pick_output_folder(self) -> str:
        if self.window is None or webview is None:
            return ""
        try:
            result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        except Exception:
            logger.exception("output folder picker failed")
            return ""
        if not result:
            return ""
        chosen = result[0] if isinstance(result, (list, tuple)) else result
        return str(chosen or "")

    def open_path(self, path: str) -> bool:
        target = str(path or "").strip()
        if not target:
            return False
        try:
            expanded = str(Path(target).expanduser())
            if sys.platform == "darwin":
                subprocess.Popen(["open", expanded])
            elif os.name == "nt":
                subprocess.Popen(["explorer", expanded])
            else:
                subprocess.Popen(["xdg-open", expanded])
            return True
        except Exception:
            logger.exception("open_path failed for %s", target)
            return False


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
    with contextlib.suppress(Exception):
        webview.settings["DRAG_REGION_DIRECT_TARGET_ONLY"] = True
    storage_path = RUNTIME_DIR / "webview"
    storage_path.mkdir(parents=True, exist_ok=True)
    api = DesktopApi(controller)
    window = webview.create_window(
        "stemsplat",
        controller.client_url(),
        js_api=api,
        width=1440,
        height=900,
        min_size=(1100, 760),
        frameless=True,
        easy_drag=False,
        text_select=False,
        background_color="#0B1A1F",
        transparent=True,
        hidden=True,
    )

    def _on_closed(*_args: Any) -> None:
        controller.stop()

    if window is not None:
        api.window = window
        window.events.closed += _on_closed

    def _show_when_ready() -> None:
        if window is None:
            return
        if not _wait_until_ready(controller.client_url()):
            logger.error("server did not become ready in time for %s", controller.client_url())
            with contextlib.suppress(Exception):
                window.show()
            return
        time.sleep(0.12)
        with contextlib.suppress(Exception):
            window.show()

    webview.start(
        func=_show_when_ready,
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--client-host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--task-runner-input", default="")
    args = parser.parse_args(argv)

    if args.task_runner_input:
        _task_runner_main(Path(args.task_runner_input).expanduser())
        return 0

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the desktop listener is local-only; enable LAN HTTPS in settings")

    preferred_port = args.port if args.port is not None else PREFERRED_PORT
    controller = ServerController(args.host, args.client_host, preferred_port)
    lan_controller = LanServerController(controller)
    set_runtime_status_provider(controller.runtime_status)
    set_lan_runtime_controller(lan_controller)
    controller.start_initial()
    try:
        lan_controller.sync()
    except Exception:
        logger.exception("LAN HTTPS listener could not be started; desktop remains available")

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
