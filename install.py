#!/usr/bin/env python3
import http.server
import json
import logging
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import List, Optional

try:
    import certifi  # noqa: F401  # ensure dependency present for downloader context
except Exception:
    certifi = None

from downloader import FILES as DL_FILES, download_to

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "install_stemsplat.log"
LEGACY_LOG = BASE_DIR / "stemsplat.log"
if LEGACY_LOG.exists():
    try:
        LEGACY_LOG.unlink()
    except Exception:
        pass
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("stemsplat.install")

progress = {"pct": 0, "step": "starting"}
choice_event = threading.Event()
shutdown_event = threading.Event()
PORT = 6060
MAIN_PORT = 8000
MODEL_URLS = [(item["filename"], item["url"]) for item in DL_FILES]
TOTAL_BYTES = int(3.68 * 1024**3)

ALIAS_MAP = {
    "models": {
        "mel_band_roformer_vocals_becruily.ckpt": ["Mel Band Roformer Vocals.ckpt"],
        "mel_band_roformer_instrumental_becruily.ckpt": ["Mel Band Roformer Instrumental.ckpt"],
    },
    "configs": {
        "config_vocals_becruily.yaml": ["Mel Band Roformer Vocals Config.yaml"],
        "config_instrumental_becruily.yaml": ["Mel Band Roformer Instrumental Config.yaml"],
        "config_deux_becruily.yaml": ["config_deux_becruily.yaml"],
    },
}

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.web_dir = Path(__file__).resolve().parent / "web"
        super().__init__(*args, directory=str(self.web_dir), **kwargs)

    def log_message(self, format, *args):  # noqa: A003 - match base signature
        logger.info("HTTP %s", format % args)

    def do_GET(self):
        if self.path == "/progress":
            logger.debug("progress requested: %s", progress)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(progress).encode())
            return

        if self.path in {"/", "/index.html", "/install.html"}:
            logger.debug("serving installer ui %s", self.path)
            self.path = "/install.html"
            return super().do_GET()

        if self.path == "/installer_shutdown":
            logger.info("installer shutdown requested via http")
            shutdown_event.set()
            self.send_response(200)
            self.end_headers()
            return

        # Serve any other static assets from web directory
        return super().do_GET()

    def do_POST(self):
        if self.path == "/download_models":
            logger.info("user requested model download")
            length = int(self.headers.get("Content-Length", 0))
            selection: Optional[List[str]] = None
            if length:
                try:
                    body = self.rfile.read(length)
                    parsed = json.loads(body.decode("utf-8"))
                    if isinstance(parsed, dict) and isinstance(parsed.get("models"), list):
                        selection = [str(x) for x in parsed["models"]]
                except Exception:
                    logger.debug("failed to parse selection body", exc_info=True)
            progress["choice"] = "download"
            progress["selection"] = selection
            choice_event.set()
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/skip_models":
            logger.info("user skipped model download")
            progress["choice"] = "skip"
            choice_event.set()
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/installer_shutdown":
            logger.info("installer shutdown requested via post")
            shutdown_event.set()
            self.send_response(200)
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()

def _installed():
    return Path("venv").exists()

def run_server():
    handler = Handler
    worker = threading.Thread(target=install, daemon=True)
    worker.start()
    socketserver.TCPServer.allow_reuse_address = True

    if not _port_available(PORT):
        msg = f"Port {PORT} is already in use. Please close the other process or change PORT."
        logger.error(msg)
        print(msg)
        sys.exit(1)

    try:
        with socketserver.TCPServer(("localhost", PORT), handler) as httpd:
            logger.info("installer ui listening on http://localhost:%s", PORT)
            webbrowser.open(f"http://localhost:{PORT}/")
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()
            logger.debug("waiting for installer routine to finish before shutting ui")
            try:
                while not shutdown_event.wait(0.5):
                    continue
                logger.info("installer routine finished; shutting down installer ui")
                httpd.shutdown()
                httpd.server_close()
                server_thread.join(timeout=2)
            except KeyboardInterrupt:
                logger.info("installer interrupted; shutting down")
                httpd.shutdown()
                httpd.server_close()
    except OSError as exc:
        logger.error("failed to bind install server on port %s: %s", PORT, exc)
        print(f"Port {PORT} is already in use. Please close the other process or change PORT.")
        sys.exit(1)

def pip_path():
    if os.name == 'nt':
        return Path('venv') / 'Scripts' / 'pip'
    else:
        return Path('venv') / 'bin' / 'pip'

def python_path():
    if os.name == 'nt':
        return Path('venv') / 'Scripts' / 'python'
    else:
        return Path('venv') / 'bin' / 'python'


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", port))
            return True
        except OSError:
            return False


def _models_missing():
    base = Path(".")
    for item in DL_FILES:
        dest = base / item["subdir"] / item["filename"]
        if dest.exists():
            continue
        aliases = ALIAS_MAP.get(item["subdir"], {}).get(item["filename"], [])
        if any((base / item["subdir"] / alt).exists() for alt in aliases):
            continue
        return True
    return False


def _start_server():
    logger.info("starting main server with uvicorn on port %s", MAIN_PORT)
    if not _port_available(MAIN_PORT):
        msg = f"Port {MAIN_PORT} is already in use. Please close the other process or change MAIN_PORT."
        logger.error(msg)
        print(msg)
        shutdown_event.set()
        return
    subprocess.Popen(
        [str(python_path()), "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(MAIN_PORT)]
    )


def _download_models(selection: Optional[list[str]] = None):
    try:
        base_dir = Path(__file__).resolve().parent
        progress["step"] = "downloading models"
        progress["pct"] = 5
        download_to(base_dir, selection or [])
        progress["pct"] = 100
        progress["step"] = "done"
        _start_server()
    except Exception as exc:
        logger.exception("model download failed")
        progress["step"] = f"download failed: {exc}"
        progress["error"] = str(exc)
        progress["pct"] = -1

def install():
    logger.info("install routine starting")
    try:
        progress['step'] = 'checking installation'
        progress['pct'] = 1
        if _installed():
            logger.info("virtual environment already present")
        steps = []
        if not _installed():
            steps.append(('creating virtual environment', [sys.executable, '-m', 'venv', 'venv']))
        steps.append(('upgrading pip', [str(pip_path()), 'install', '--upgrade', 'pip']))
        reqs = []
        req_file = Path('requirements.txt')
        if req_file.exists():
            for line in req_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    reqs.append(line)
        steps.extend([
            (f'installing {pkg}', [str(pip_path()), 'install', pkg]) for pkg in reqs
        ])

        total = max(1, len(steps))
        for i, (msg, cmd) in enumerate(steps, start=1):
            progress['step'] = msg
            progress['pct'] = int((i-1)/total*100)
            logger.info("running step %s/%s: %s", i, total, msg)
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as exc:
                logger.exception("error during %s", msg)
                progress['step'] = f'error during {msg}: {exc}'
                progress['pct'] = -1
                return

        if _models_missing():
            logger.info("models missing; waiting for user choice")
            progress['step'] = 'waiting for model choice'
            progress['pct'] = 99
            choice_event.wait()
            choice_event.clear()
            if progress.get('choice') == 'download':
                _download_models(progress.get("selection"))
                return
            else:
                logger.info("user skipped downloads")
                progress['pct'] = 100
                progress['step'] = 'done'
                _start_server()
                return

        progress['pct'] = 100
        progress['step'] = 'done'
        logger.info("prerequisites satisfied; launching server")
        _start_server()
    except Exception:
        logger.exception("installer crashed")
        progress['step'] = 'installation failed'
        progress['pct'] = -1
    finally:
        shutdown_event.set()

if __name__ == '__main__':
    run_server()
