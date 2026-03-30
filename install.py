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

progress = {"pct": 0, "step": "starting", "models_missing": []}
choice_event = threading.Event()
shutdown_event = threading.Event()
PORT = 6060
MAIN_PORT = 8000
MODEL_URLS = [(item["filename"], item["url"]) for item in DL_FILES]
TOTAL_BYTES = int(2.60 * 1024**3)

ALIAS_MAP = {
    "models": {
        "mel_band_roformer_vocals_becruily.ckpt": ["Mel Band Roformer Vocals.ckpt"],
        "mel_band_roformer_instrumental_becruily.ckpt": ["Mel Band Roformer Instrumental.ckpt"],
        "becruily_deux.ckpt": ["Mel Band Roformer Deux.ckpt"],
        "becruily_guitar.ckpt": ["Mel Band Roformer Guitar.ckpt"],
        "mel_band_roformer_karaoke_becruily.ckpt": ["Mel Band Roformer Karaoke.ckpt"],
        "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt": ["Mel Band Roformer Denoise.ckpt"],
        "BS-Rofo-SW-Fixed.ckpt": ["BS-Rofo-SW-Fixed-v1.ckpt", "BS Rofo SW Fixed.ckpt"],
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
    return python_path().exists()

def pip_path():
    if os.name == 'nt':
        return BASE_DIR / 'venv' / 'Scripts' / 'pip'
    else:
        return BASE_DIR / 'venv' / 'bin' / 'pip'

def python_path():
    if os.name == 'nt':
        return BASE_DIR / 'venv' / 'Scripts' / 'python'
    else:
        return BASE_DIR / 'venv' / 'bin' / 'python'


def pip_cmd(*args: str) -> list[str]:
    return [str(python_path()), "-m", "pip", *args]


def _ensure_venv() -> None:
    if not _installed():
        logger.info("creating virtual environment at %s", BASE_DIR / "venv")
        subprocess.check_call([sys.executable, "-m", "venv", str(BASE_DIR / "venv")], cwd=BASE_DIR)
    if not python_path().exists():
        raise FileNotFoundError(f"Virtual environment python missing at {python_path()}")
    logger.info("bootstrapping pip in virtual environment")
    subprocess.check_call([str(python_path()), "-m", "ensurepip", "--upgrade"], cwd=BASE_DIR)


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
            webbrowser.open(f"http://localhost:{PORT}/", new=0)
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
                logger.debug("installer ui closed; main app launch handled by installer page")
            except KeyboardInterrupt:
                logger.info("installer interrupted; shutting down")
                httpd.shutdown()
                httpd.server_close()
    except OSError as exc:
        logger.error("failed to bind install server on port %s: %s", PORT, exc)
        print(f"Port {PORT} is already in use. Please close the other process or change PORT.")
        sys.exit(1)


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", port))
            return True
        except OSError:
            return False


def _missing_required_models() -> list[str]:
    model_roots = [
        BASE_DIR / "models",
        BASE_DIR / "Models",
        Path("models"),
        Path("Models"),
        Path.home() / "Library/Application Support" / "stems",
    ]
    required_files = [item["filename"] for item in DL_FILES]
    missing: list[str] = []
    for filename in required_files:
        search_names = [filename, *ALIAS_MAP.get("models", {}).get(filename, [])]
        found = False
        for models_dir in model_roots:
            if not models_dir.exists():
                continue
            for path in models_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path.name in search_names:
                    found = True
                    break
            if found:
                break
        if not found:
            missing.append(filename)
    return missing


def _models_missing() -> bool:
    return len(_missing_required_models()) > 0


def _start_server():
    logger.info("starting main server with uvicorn on port %s", MAIN_PORT)
    if not _port_available(MAIN_PORT):
        logger.info("main server already running on port %s; opening browser", MAIN_PORT)
        progress["main_running"] = True
        logger.debug("main server already running; installer page will navigate")
        shutdown_event.set()
        return
    subprocess.Popen(
        [str(python_path()), "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(MAIN_PORT)],
        cwd=BASE_DIR,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    logger.debug("main server started; installer page will navigate")


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
        progress['step'] = 'installing prerequisites'
        progress['pct'] = 1
        progress["models_missing"] = _missing_required_models()
        if _installed():
            logger.info("virtual environment already present")
        steps = []
        steps.append(('preparing virtual environment', None))
        steps.append(('upgrading pip', pip_cmd('install', '--upgrade', 'pip')))
        reqs = []
        req_file = BASE_DIR / 'requirements.txt'
        if req_file.exists():
            for line in req_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    reqs.append(line)
        steps.extend([
            (f'installing {pkg}', pip_cmd('install', pkg)) for pkg in reqs
        ])

        total = max(1, len(steps))
        for i, (msg, cmd) in enumerate(steps, start=1):
            progress['step'] = msg
            progress['pct'] = int((i-1)/total*100)
            logger.info("running step %s/%s: %s", i, total, msg)
            try:
                if msg == 'preparing virtual environment':
                    _ensure_venv()
                else:
                    subprocess.check_call(cmd, cwd=BASE_DIR)
            except subprocess.CalledProcessError as exc:
                logger.exception("error during %s", msg)
                progress['step'] = f'error during {msg}: {exc}'
                progress['pct'] = -1
                return
            except FileNotFoundError as exc:
                logger.exception("missing executable during %s", msg)
                progress['step'] = f'error during {msg}: {exc}'
                progress['error'] = str(exc)
                progress['pct'] = -1
                return

        if _models_missing():
            logger.info("models missing; skipping downloads per v0.1 flow")
            progress['step'] = 'models missing; download them in the app or add them to the models folder'
            progress['pct'] = 100
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
