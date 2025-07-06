#!/usr/bin/env python3
import http.server
import socketserver
import threading
import subprocess
import sys
import os
import json
import webbrowser
from pathlib import Path
import time
import urllib.request

progress = {"pct": 0, "step": "starting"}
choice_event = threading.Event()
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
    def do_GET(self):
        if self.path == '/progress':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(progress).encode())
        else:
            if self.path == '/' or self.path == '/index.html':
                html_file = 'install.html'
                if progress.get('pct', 0) >= 100:
                    html_file = 'index.html'
                html_path = Path(__file__).resolve().parent / 'web' / html_file
                html = html_path.read_text()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            else:
                self.send_response(404)
                self.end_headers()

    def do_POST(self):
        if self.path == '/download_models':
            progress['choice'] = 'download'
            choice_event.set()
            self.send_response(200)
            self.end_headers()
        elif self.path == '/skip_models':
            progress['choice'] = 'skip'
            choice_event.set()
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

class LaunchHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            html = (Path(__file__).resolve().parent / 'web' / 'launch.html').read_text()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

def _installed():
    return Path('venv').exists()

def run_server():
    if _installed():
        handler = LaunchHandler
        thread = threading.Thread(target=_start_server, daemon=True)
        thread.start()
    else:
        handler = Handler
        thread = threading.Thread(target=install, daemon=True)
        thread.start()
    with socketserver.TCPServer(('localhost', 6060), handler) as httpd:
        webbrowser.open('http://localhost:6060/')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

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


def _models_missing():
    models_dir = Path('models')
    for name, _ in MODEL_URLS:
        if not (models_dir / name).exists():
            return True
    return False


def _start_server():
    subprocess.Popen([str(python_path()), '-m', 'uvicorn', 'stemrunner.server:app'])


def _download_models():
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    downloaded = 0
    for name, url in MODEL_URLS:
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
    progress['pct'] = 100
    progress['step'] = 'done'
    _start_server()

def install():
    steps = [
        ('creating virtual environment', [sys.executable, '-m', 'venv', 'venv']),
        ('upgrading pip', [str(pip_path()), 'install', '--upgrade', 'pip'])
    ]
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
    total = len(steps)
    for i, (msg, cmd) in enumerate(steps, start=1):
        progress['step'] = msg
        progress['pct'] = int((i-1)/total*100)
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            progress['step'] = f'error during {msg}'
            progress['pct'] = -1
            return
    if _models_missing():
        progress['step'] = 'waiting for model choice'
        progress['pct'] = 99
        choice_event.wait()
        choice_event.clear()
        if progress.get('choice') == 'download':
            _download_models()
            return
        else:
            progress['pct'] = 100
            progress['step'] = 'done'
            _start_server()
            return
    progress['pct'] = 100
    progress['step'] = 'done'
    _start_server()

if __name__ == '__main__':
    run_server()
