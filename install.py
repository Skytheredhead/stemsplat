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
MODEL_URLS = [
    ("Mel Band Roformer Vocals.ckpt",
     "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true"),
    ("Mel Band Roformer Instrumental.ckpt",
     "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true"),
    ("mel_band_roformer_karaoke_becruily.ckpt",
     "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt?download=true"),
    ("becruily_guitar.ckpt",
     "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt?download=true"),
    ("kuielab_a_bass.onnx",
     "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_bass.onnx?download=true"),
    ("kuielab_a_drums.onnx",
     "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_drums.onnx?download=true"),
    ("kuielab_a_other.onnx",
     "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_other.onnx?download=true"),
]
TOTAL_BYTES = int(3.68 * 1024**3)

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
            progress["choice"] = "download"
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
    models_dir = Path('models')
    for name, _ in MODEL_URLS:
        if not (models_dir / name).exists():
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


def _download_models():
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    downloaded = 0
    for name, url in MODEL_URLS:
        logger.info("downloading model %s from %s", name, url)
        progress['step'] = f'downloading {name}'
        dest = models_dir / name
        with urllib.request.urlopen(url) as resp, open(dest, 'wb') as out:
            t0 = time.time()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                speed = len(chunk) / 1024 / 1024 / max(now - t0, 1e-6)
                progress['pct'] = int(downloaded / TOTAL_BYTES * 100)
                progress['step'] = f'downloading {name} ({speed:.1f} MB/s)'
                t0 = now
        logger.info("finished %s", name)
    progress['pct'] = 100
    progress['step'] = 'done'
    _start_server()

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
                _download_models()
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
