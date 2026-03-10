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

import uvicorn

from app_paths import ensure_app_dirs

ensure_app_dirs()

from main import app

logger = logging.getLogger("stemsplat.launcher")


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred: int) -> int:
    if _port_available(host, preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with contextlib.suppress(Exception):
            with urllib.request.urlopen(url, timeout=1):
                return True
        time.sleep(0.25)
    return False


def _open_url(url: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", url])
        return
    webbrowser.open(url, new=1, autoraise=True)


def _open_browser_when_ready(url: str) -> None:
    if _wait_until_ready(url):
        logger.info("opening browser at %s", url)
        _open_url(url)
    else:
        logger.error("server did not become ready in time for %s", url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the stemsplat desktop app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    port = args.port if args.port is not None else _pick_port(args.host, 8000)
    url = f"http://{args.host}:{port}/"

    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    logger.info("starting stemsplat on %s", url)
    config = uvicorn.Config(app, host=args.host, port=port, reload=False, log_level="info")
    server = uvicorn.Server(config)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
