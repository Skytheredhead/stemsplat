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

progress = {"pct": 0, "step": "starting"}

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

def run_server():
    with socketserver.TCPServer(('localhost', 6060), Handler) as httpd:
        webbrowser.open('http://localhost:6060/')
        install_thread = threading.Thread(target=install, daemon=True)
        install_thread.start()
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
    progress['pct'] = 100
    progress['step'] = 'done'
    # launch the web application once installation finishes
    subprocess.Popen([str(python_path()), '-m', 'uvicorn', 'stemrunner.server:app'])

if __name__ == '__main__':
    run_server()
