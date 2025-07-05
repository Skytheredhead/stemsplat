import os
import subprocess
import sys
import webbrowser
from pathlib import Path


# Determine the python executable in the virtual environment

def venv_python() -> Path:
    if os.name == 'nt':
        return Path('venv') / 'Scripts' / 'python.exe'
    return Path('venv') / 'bin' / 'python'


# Install dependencies if virtual environment is missing

def ensure_installed():
    if not venv_python().exists():
        # Run the interactive installer
        subprocess.check_call([sys.executable, 'install.py'])


def run_server():
    python = venv_python()
    cmd = [str(python), '-m', 'uvicorn', 'stemrunner.server:app']
    proc = subprocess.Popen(cmd)
    return proc


if __name__ == '__main__':
    ensure_installed()
    server_proc = run_server()
    try:
        webbrowser.open('http://localhost:8000')
        server_proc.wait()
    finally:
        server_proc.terminate()
