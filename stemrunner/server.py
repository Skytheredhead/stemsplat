try:
    from packaging import version  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - packaging optional
    from distutils.version import LooseVersion as _LooseVersion  # type: ignore

    class _CompatVersion:
        @staticmethod
        def parse(v):
            return _LooseVersion(v)

    version = _CompatVersion()  # type: ignore

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pathlib import Path
import uuid
import asyncio
import logging
import json
from urllib.parse import quote

from converter import convert_to_wav  # local utility for audio conversion

from .pipeline import process_file
import threading
import urllib.request
import time

app = FastAPI()

progress = {}
errors = {}
tasks = {}

download_lock = threading.Lock()
downloading = False

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


def _download_models():
    global downloading
    with download_lock:
        if downloading:
            return
        downloading = True
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    downloaded = 0
    for name, url in MODEL_URLS:
        dest = models_dir / name
        try:
            with urllib.request.urlopen(url) as resp, open(dest, 'wb') as out:
                t0 = time.time()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    t1 = time.time()
                    _ = len(chunk) / 1024 / 1024 / max(t1 - t0, 1e-6)
                    t0 = t1
        except Exception:
            pass
    downloading = False


@app.post('/download_models')
async def download_models():
    thread = threading.Thread(target=_download_models, daemon=True)
    thread.start()
    return {'status': 'started'}


@app.post('/upload')
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    path = Path('uploads') / file.filename
    path.parent.mkdir(exist_ok=True)
    with path.open('wb') as f:
        f.write(await file.read())

    # Convert to WAV so the splitter can consume the file.
    conv_dir = Path('uploads_converted')
    conv_dir.mkdir(exist_ok=True)
    conv_path = conv_dir / f"{path.stem}.wav"

    if path.suffix.lower() == '.wav':
        # Simple copy when already WAV
        import shutil
        shutil.copy2(path, conv_path)
    else:
        # Use FFmpeg via converter.py to handle formats like m4a
        try:
            convert_to_wav(path, conv_path)
        except Exception as exc:
            logging.exception('conversion failed')
            return JSONResponse({'detail': f'conversion failed: {exc}'}, status_code=400)

    ckpt_path = Path('models') / 'Mel Band Roformer Vocals.ckpt'
    if not ckpt_path.exists():
        ckpt_path = Path.home() / 'Library/Application Support/stems/Mel Band Roformer Vocals.ckpt'

    if not ckpt_path.exists():
        return JSONResponse({'detail': 'checkpoint not found'}, status_code=400)

    def cb(stage: str, pct: int):
        scaled = pct
        if stage == 'vocals':
            # map 0-100 -> 10-100 to match UI expectations
            scaled = 10 + int(pct * 0.9)
        progress[task_id] = {'stage': stage, 'pct': scaled}

    def run():
        try:
            process_file(conv_path, ckpt_path, progress_cb=cb)
        except Exception as exc:
            logging.exception('processing failed')
            progress[task_id] = {'stage': 'error', 'pct': -1}
            errors[task_id] = str(exc)

    background_tasks.add_task(run)
    progress[task_id] = {'stage': 'queued', 'pct': 0}
    stems = ['vocals.wav']
    out_dir = conv_dir / f"{Path(file.filename).stem}—stems"
    tasks[task_id] = {'dir': out_dir, 'stems': stems}
    return {'task_id': task_id, 'stems': stems}


@app.get('/progress/{task_id}')
async def progress_stream(task_id: str):
    async def event_generator():
        last = None
        while True:
            info = progress.get(task_id, {'stage': 'queued', 'pct': 0})
            current = (info['stage'], info['pct'])
            if current != last:
                if info['pct'] < 0:
                    yield {'event': 'error', 'data': errors.get(task_id, 'processing failed')}
                else:
                    yield {
                        'event': 'message',
                        'data': json.dumps({'stage': info['stage'], 'pct': info['pct']})
                    }
                last = current
            if info['pct'] >= 100 or info['pct'] < 0:
                break
            await asyncio.sleep(0.5)
    return EventSourceResponse(event_generator())


@app.get('/download/{task_id}')
async def download(task_id: str):
    info = tasks.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail='invalid task id')
    if not info['dir'].exists():
        raise HTTPException(status_code=404, detail='files not ready')
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name in info['stems']:
            fp = info['dir'] / name
            if fp.exists():
                zf.write(fp, arcname=name)
    buf.seek(0)
    fname = f"{info['dir'].name}.zip"
    header = f"attachment; filename*=UTF-8''{quote(fname)}"
    return StreamingResponse(buf, media_type='application/zip', headers={
        'Content-Disposition': header
    })


@app.get('/', response_class=HTMLResponse)
async def index():
    html = Path(Path(__file__).resolve().parent.parent / 'web' / 'index.html').read_text()
    return HTMLResponse(html)
