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

from .pipeline import process_file
import threading
import queue
import urllib.request
import time

app = FastAPI()

progress = {}
errors = {}
tasks = {}
controls = {}

process_queue: queue.Queue[callable] = queue.Queue()
def _worker():
    while True:
        fn = process_queue.get()
        try:
            fn()
        finally:
            process_queue.task_done()

threading.Thread(target=_worker, daemon=True).start()

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
    pause_evt = threading.Event()
    stop_evt = threading.Event()
    controls[task_id] = {'pause': pause_evt, 'stop': stop_evt}
    path = Path('uploads') / file.filename
    path.parent.mkdir(exist_ok=True)
    with path.open('wb') as f:
        f.write(await file.read())

    # For now we assume uploads are already WAV.  Just copy the file into
    # uploads_converted/ (keeping the original name) so later steps find it.
    conv_dir = Path('uploads_converted')
    conv_dir.mkdir(exist_ok=True)
    conv_path = conv_dir / path.name
    import shutil
    shutil.copy2(path, conv_path)

    ckpt_path = Path('models') / 'Mel Band Roformer Vocals.ckpt'
    if not ckpt_path.exists():
        ckpt_path = Path.home() / 'Library/Application Support/stems/Mel Band Roformer Vocals.ckpt'

    if not ckpt_path.exists():
        return JSONResponse({'detail': 'checkpoint not found'}, status_code=400)

    config_path = Path('configs') / 'Mel Band Roformer Vocals Config.yaml'
    if not config_path.exists():
        return JSONResponse({'detail': 'config file not found'}, status_code=400)

    def cb(stage: str, pct: int):
        while pause_evt.is_set():
            progress[task_id] = {'stage': 'paused', 'pct': pct}
            time.sleep(0.5)
        if stop_evt.is_set():
            progress[task_id] = {'stage': 'stopped', 'pct': -1}
            raise RuntimeError('stopped')
        progress[task_id] = {'stage': stage, 'pct': pct}

    def run():
        try:
            audio_path = conv_path

            # ── Convert to WAV if the upload isn't already WAV ─────────────────
            if audio_path.suffix.lower() not in (".wav", ".wave"):
                cb("converting", 0)

                # ffmpeg command: re‑encode to 44.1 kHz, stereo PCM WAV
                wav_path = audio_path.with_suffix(".wav")
                import subprocess, os  # local import to keep globals clean

                cmd = [
                    "ffmpeg",
                    "-y",               # overwrite without prompt
                    "-i", str(audio_path),
                    "-ar", "44100",     # 44.1 kHz
                    "-ac", "2",         # stereo
                    "-vn",              # drop any video streams
                    str(wav_path),
                ]

                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    raise RuntimeError("ffmpeg not found – please install ffmpeg and ensure it is on your PATH")

                cb("converting", 100)
                audio_path = wav_path
            # ────────────────────────────────────────────────────────────────────

            # Now run the splitter on the (possibly converted) WAV
            process_file(audio_path, ckpt_path, progress_cb=cb)

            stem_file = out_dir / 'vocals.wav'
            new_name = f"{audio_path.stem} - vocals.wav"
            if stem_file.exists():
                stem_file.rename(out_dir / new_name)
                stems[0] = new_name

            # ── Pre‑compress stems so /download is instant ────────────────
            import zipfile
            zip_path = conv_dir / f"{audio_path.stem}—stems.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for name in stems:
                    fp = out_dir / name
                    if fp.exists():
                        zf.write(fp, arcname=name)
            tasks[task_id]["zip"] = zip_path
            # ───────────────────────────────────────────────────────────────

        except Exception as exc:
            logging.exception("processing failed")
            if stop_evt.is_set():
                progress[task_id] = {"stage": "stopped", "pct": -1}
                errors[task_id] = 'stopped'
            else:
                progress[task_id] = {"stage": "error", "pct": -1}
                errors[task_id] = str(exc)

    process_queue.put(run)
    progress[task_id] = {'stage': 'queued', 'pct': 0}
    stems = [f"{Path(file.filename).stem} - vocals.wav"]
    out_dir = conv_dir / f"{Path(file.filename).stem}—stems"
    tasks[task_id] = {'dir': out_dir, 'stems': stems, 'controls': controls[task_id]}
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
    zip_path = info.get("zip") if info else None
    if zip_path and Path(zip_path).exists():
        header = f"attachment; filename*=UTF-8''{quote(Path(zip_path).name)}"
        return StreamingResponse(open(zip_path, "rb"),
                                 media_type="application/zip",
                                 headers={"Content-Disposition": header})
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


@app.post('/pause/{task_id}')
async def pause_task(task_id: str):
    ctrl = controls.get(task_id)
    if not ctrl:
        raise HTTPException(status_code=404, detail='invalid task')
    ctrl['pause'].set()
    return {'status': 'paused'}


@app.post('/resume/{task_id}')
async def resume_task(task_id: str):
    ctrl = controls.get(task_id)
    if not ctrl:
        raise HTTPException(status_code=404, detail='invalid task')
    ctrl['pause'].clear()
    return {'status': 'resumed'}


@app.post('/stop/{task_id}')
async def stop_task(task_id: str):
    ctrl = controls.get(task_id)
    if not ctrl:
        raise HTTPException(status_code=404, detail='invalid task')
    ctrl['stop'].set()
    return {'status': 'stopped'}


@app.get('/', response_class=HTMLResponse)
async def index():
    html = Path(Path(__file__).resolve().parent.parent / 'web' / 'index.html').read_text()
    return HTMLResponse(html)
