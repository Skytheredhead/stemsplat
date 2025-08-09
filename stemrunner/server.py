from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"

try:
    from packaging import version  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - packaging optional
    from distutils.version import LooseVersion as _LooseVersion  # type: ignore

    class _CompatVersion:
        @staticmethod
        def parse(v):
            return _LooseVersion(v)

    version = _CompatVersion()  # type: ignore

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from sse_starlette.sse import EventSourceResponse
from pathlib import Path
import uuid
import asyncio
import logging
import json
from urllib.parse import quote
import shutil

from .models import ModelManager
import torchaudio
import soundfile as sf
import torch
import tempfile
import os
import threading
import queue
import urllib.request
import time
import warnings

# Suppress torchaudio load deprecation notice
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
    category=UserWarning,
    module="torchaudio",
)

app = FastAPI()

progress = {}
errors = {}
tasks = {}
controls = {}

process_queue: queue.Queue[callable] = queue.Queue()


def _worker() -> None:
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

SEGMENT = 352_800
OVERLAP = 12


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
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    stems: str = Form('vocals'),
):
    task_id = str(uuid.uuid4())
    pause_evt = threading.Event()
    stop_evt = threading.Event()
    controls[task_id] = {'pause': pause_evt, 'stop': stop_evt}
    stem_list = [s for s in stems.split(',') if s]
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

    root_dir = Path(__file__).resolve().parent.parent
    ckpt_path = root_dir / 'models' / 'Mel Band Roformer Vocals.ckpt'
    if not ckpt_path.exists():
        ckpt_path = Path.home() / 'Library/Application Support/stems/Mel Band Roformer Vocals.ckpt'

    if not ckpt_path.exists():
        return JSONResponse({'detail': 'checkpoint not found'}, status_code=400)

    config_path = root_dir / 'configs' / 'Mel Band Roformer Vocals Config.yaml'
    if not config_path.exists():
        return JSONResponse({'detail': 'config file not found'}, status_code=400)

    # Ensure required models are present for the requested stems
    manager = ModelManager()
    missing_all = manager.missing_models()
    missing = []
    for s in stem_list:
        info = manager.model_info.get(s)
        if not info:
            continue
        if info[0] in missing_all:
            missing.append(info[0])
        cfg = info[1]
        if cfg and cfg in missing_all:
            missing.append(cfg)
    if missing:
        msg = f"models missing: {', '.join(missing)}"
        return JSONResponse({'detail': msg}, status_code=400)

    # define output locations before starting background work to avoid race conditions
    out_dir = conv_dir / f"{Path(file.filename).stem}—stems"
    expected = [f"{Path(file.filename).stem} - {s}.wav" for s in stem_list]

    def cb(stage: str, pct: int):
        while pause_evt.is_set():
            progress[task_id] = {'stage': 'paused', 'pct': pct}
            time.sleep(0.5)
        if stop_evt.is_set():
            progress[task_id] = {'stage': 'stopped', 'pct': 0}
            raise RuntimeError('stopped')
        progress[task_id] = {'stage': stage, 'pct': pct}

    def run():
        import torch
        audio_path = conv_path
        print(f"[checkpoint] separate() called with audio_path={audio_path}")
        try:
            cb("preparing", 0)
            cb("prepare.complete", 1)

            # ── Convert to WAV if needed ───────────────────────────────────
            if audio_path.suffix.lower() not in (".wav", ".wave"):
                cb("converting", 0)
                wav_path = audio_path.with_suffix(".wav")
                import subprocess, os
                cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "44100", "-ac", "2", "-vn", str(wav_path)]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    raise RuntimeError("ffmpeg not found – please install ffmpeg and ensure it is on your PATH")
                cb("converting", 100)
                audio_path = wav_path
                tasks[task_id]["conv_src"] = str(wav_path)

            # ── Load audio ─────────────────────────────────────────────────
            cb("load_audio.start", 1)
            print("[checkpoint] Starting separation task for", audio_path)
            print("[checkpoint] Loading model from", MODEL_DIR)
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 44100:
                waveform = torchaudio.functional.resample(waveform, sr, 44100)
                sr = 44100
            cb("load_audio.done", 2)
            print("[checkpoint] Audio loaded successfully")

            out_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(tempfile.gettempdir())
            stems_out: list[str] = []

            need_vocal_pass = ("vocals" in stem_list) or any(s in stem_list for s in ["drums","bass","other","guitar"])
            need_instrumental_model = ("instrumental" in stem_list)

            if need_vocal_pass:
                def _cb(frac: float) -> None:
                    cb("vocals", 1 + int(frac * 99))

                cb("split_vocals.start", 2)

                # Set device for separation
                import torch
                device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                print(f"[checkpoint] Using device: {device}")
                # Print before separation process
                import os
                model_path = str(MODEL_DIR / "Mel Band Roformer Vocals.ckpt")
                cfg_path = str(CONFIG_DIR / "Mel Band Roformer Vocals Config.yaml")
                print(f"[checkpoint] Model path exists? {os.path.exists(model_path)} -> {model_path}")
                print(f"[checkpoint] Config path exists? {os.path.exists(cfg_path)} -> {cfg_path}")
                print(f"[checkpoint] About to call separate() with audio_path={audio_path}")
                print("[checkpoint] Starting separation process")
                try:
                    print(f"[checkpoint] Calling separation with model={MODEL_DIR}, config={CONFIG_DIR}")
                    # Pass device to split_vocals if supported
                    voc, inst = manager.split_vocals(waveform, SEGMENT, OVERLAP, progress_cb=_cb)
                    print("[checkpoint] Separation finished successfully")
                except Exception as e:
                    print(f"[error] Separation failed: {e}")
                    raise
                cb("split_vocals.done", 50)

                if "vocals" in stem_list:
                    fname = f"{audio_path.stem} - vocals.wav"
                    sf.write(out_dir / fname, voc.T.numpy(), sr)
                    stems_out.append(fname)
                    cb("write.vocals", 52)

                need_inst = any(s in stem_list for s in ["drums", "bass", "other", "guitar"])  # inst needed for downstream MDX only
                inst_path = None
                if need_inst:
                    inst_path = temp_dir / f"{uuid.uuid4()}_inst.wav"
                    sf.write(inst_path, inst.T.numpy(), sr)
                    # instrumental will be produced by a dedicated second pass

                del voc, inst
                if hasattr(torch, "cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            else:
                inst_path = None

            # ── Optional second pass: Instrumental model (separate ckpt/yaml) ──
            if need_instrumental_model:
                def _cb_inst_model(frac: float) -> None:
                    cb("instrumental_model", 1 + int(frac * 99))

                cb("split_inst_model.start", 2)
                try:
                    inst0, inst1 = manager.split_pair_with_model(
                        manager.instrumental,
                        waveform,
                        SEGMENT,
                        OVERLAP,
                        progress_cb=_cb_inst_model,
                    )
                except Exception as e:
                    print(f"[error] Instrumental model separation failed: {e}")
                    raise
                cb("split_inst_model.done", 70)

                # By convention, the instrumental model's first output is instrumental
                fname = f"{audio_path.stem} - instrumental.wav"
                cb("write.instrumental", 72)
                sf.write(out_dir / fname, inst0.T.numpy(), sr)
                stems_out.append(fname)

                # free memory
                del inst0, inst1
                if hasattr(torch, "cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            if inst_path and any(s in stem_list for s in ["drums", "bass", "other", "guitar"]):
                inst_wave, sr2 = torchaudio.load(str(inst_path))
                if sr2 != 44100:
                    inst_wave = torchaudio.functional.resample(inst_wave, sr2, 44100)

                def _cb2(frac: float) -> None:
                    cb("instrumental", 1 + int(frac * 99))

                cb("split_inst.start", 60)
                import os
                model_path = str(MODEL_DIR / "kuielab_a_drums.onnx")
                cfg_path = "None"
                print(f"[checkpoint] Model path exists? {os.path.exists(model_path)} -> {model_path}")
                print(f"[checkpoint] Config path exists? {os.path.exists(cfg_path)} -> {cfg_path}")
                print(f"[checkpoint] About to call separate() with audio_path={audio_path}")
                print("[checkpoint] Starting separation process")
                try:
                    d, b, o, k, g = manager.split_instrumental(inst_wave, SEGMENT, OVERLAP, progress_cb=_cb2)
                    print("[checkpoint] Separation finished successfully")
                except Exception as e:
                    print(f"[error] Separation failed: {e}")
                    raise
                mapping = {"drums": d, "bass": b, "other": o, "guitar": g}
                cb("split_inst.masks_ready", 90)

                for name, tensor in mapping.items():
                    if name in stem_list:
                        cb('write.' + name, 92)
                        fname = f"{audio_path.stem} - {name}.wav"
                        sf.write(out_dir / fname, tensor.T.numpy(), sr)
                        stems_out.append(fname)

                del d, b, o, k, g, inst_wave
                if hasattr(torch, "cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                try:
                    os.remove(inst_path)
                except OSError:
                    pass

            import zipfile
            zip_path = conv_dir / f"{audio_path.stem}—stems.zip"
            cb("zip.start", 95)
            with zipfile.ZipFile(zip_path, "w") as zf:
                for name in stems_out:
                    fp = out_dir / name
                    if fp.exists():
                        zf.write(fp, arcname=name)
            cb("zip.done", 98)
            tasks[task_id]["zip"] = zip_path
            tasks[task_id]["stems"] = stems_out

            cb("finalizing", 99)
            # mark complete for the SSE stream
            progress[task_id] = {"stage": "done", "pct": 100}

        except Exception as exc:
            if stop_evt.is_set() or str(exc) == 'stopped':
                progress[task_id] = {"stage": "stopped", "pct": 0}
                errors[task_id] = 'stopped'
            else:
                logging.exception("processing failed")
                progress[task_id] = {"stage": "error", "pct": -1}
                errors[task_id] = str(exc)

    # queue the work *after* outputs are set
    process_queue.put(run)
    progress[task_id] = {'stage': 'queued', 'pct': 0}
    tasks[task_id] = {
        'dir': out_dir,
        'stems': expected,
        'controls': controls[task_id],
        'conv_src': str(conv_path),
        'orig_src': str(path),
        'stem_list': stem_list,
    }
    return {'task_id': task_id, 'stems': expected}


@app.get('/progress/{task_id}')
async def progress_stream(task_id: str):
    async def event_generator():
        last = None
        while True:
            info = progress.get(task_id, {'stage': 'queued', 'pct': 0})
            current = (info['stage'], info['pct'])
            if current != last:
                if info.get('stage') == 'stopped':
                    yield {'event': 'message', 'data': json.dumps({'stage': 'stopped', 'pct': 0})}
                elif info['pct'] < 0:
                    yield {'event': 'error', 'data': errors.get(task_id, 'processing failed')}
                else:
                    yield {'event': 'message', 'data': json.dumps({'stage': info['stage'], 'pct': info['pct']})}
                ...
                if info.get('stage') == 'stopped' or info['pct'] >= 100 or info['pct'] < 0:
                    break
            await asyncio.sleep(0.5)
    return EventSourceResponse(event_generator())

@app.post('/rerun/{task_id}')
async def rerun(task_id: str):
    # Re-run a previous task using the stored converted source and stem list
    old = tasks.get(task_id)
    if not old:
        raise HTTPException(status_code=404, detail='invalid task id')

    conv_src = Path(old.get('conv_src', ''))
    stem_list = old.get('stem_list') or []
    if not conv_src.exists() or not stem_list:
        raise HTTPException(status_code=409, detail='missing source or stems for rerun')

    new_id = str(uuid.uuid4())
    pause_evt = threading.Event()
    stop_evt = threading.Event()
    controls[new_id] = {'pause': pause_evt, 'stop': stop_evt}

    conv_dir = conv_src.parent
    out_dir = conv_dir / f"{conv_src.stem}—stems"
    expected = [f"{conv_src.stem} - {s}.wav" for s in stem_list]

    def cb(stage: str, pct: int):
        while pause_evt.is_set():
            progress[new_id] = {'stage': 'paused', 'pct': pct}
            time.sleep(0.5)
        if stop_evt.is_set():
            progress[new_id] = {'stage': 'stopped', 'pct': 0}
            raise RuntimeError('stopped')
        progress[new_id] = {'stage': stage, 'pct': pct}

    def run_again():
        try:
            cb('preparing', 0)
            cb('prepare.complete', 1)

            audio_path = conv_src
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 44100:
                waveform = torchaudio.functional.resample(waveform, sr, 44100)
                sr = 44100
            cb('load_audio.done', 2)

            manager = ModelManager()
            out_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(tempfile.gettempdir())
            stems_out: list[str] = []

            need_vocal_pass = ('vocals' in stem_list) or any(s in stem_list for s in ['drums','bass','other','guitar'])
            need_instrumental_model = ('instrumental' in stem_list)

            if need_vocal_pass:
                def _cb(frac: float): cb('vocals', 1 + int(frac * 99))
                cb('split_vocals.start', 2)
                voc, inst = manager.split_vocals(waveform, SEGMENT, OVERLAP, progress_cb=_cb)
                cb('split_vocals.done', 50)
                if 'vocals' in stem_list:
                    fname = f"{audio_path.stem} - vocals.wav"
                    sf.write(out_dir / fname, voc.T.numpy(), sr)
                    stems_out.append(fname)
                    cb('write.vocals', 52)
                need_inst = any(s in stem_list for s in ['drums','bass','other','guitar'])
                inst_path = None
                if need_inst:
                    inst_path = temp_dir / f"{uuid.uuid4()}_inst.wav"
                    sf.write(inst_path, inst.T.numpy(), sr)
                del voc, inst
                if hasattr(torch, 'cuda'):
                    try: torch.cuda.empty_cache()
                    except Exception: pass
            else:
                inst_path = None

            if need_instrumental_model:
                def _cb_im(frac: float): cb('instrumental_model', 1 + int(frac * 99))
                cb('split_inst_model.start', 2)
                inst0, inst1 = manager.split_pair_with_model(
                    manager.instrumental, waveform, SEGMENT, OVERLAP, progress_cb=_cb_im)
                cb('split_inst_model.done', 70)
                fname = f"{audio_path.stem} - instrumental.wav"
                cb('write.instrumental', 72)
                sf.write(out_dir / fname, inst0.T.numpy(), sr)
                stems_out.append(fname)
                del inst0, inst1
                if hasattr(torch, 'cuda'):
                    try: torch.cuda.empty_cache()
                    except Exception: pass

            if inst_path and any(s in stem_list for s in ['drums','bass','other','guitar']):
                inst_wave, sr2 = torchaudio.load(str(inst_path))
                if sr2 != 44100:
                    inst_wave = torchaudio.functional.resample(inst_wave, sr2, 44100)
                def _cb2(frac: float): cb('instrumental', 1 + int(frac * 99))
                cb('split_inst.start', 60)
                d,b,o,k,g = manager.split_instrumental(inst_wave, SEGMENT, OVERLAP, progress_cb=_cb2)
                mapping = {'drums': d, 'bass': b, 'other': o, 'guitar': g}
                cb('split_inst.masks_ready', 90)
                for name, tensor in mapping.items():
                    if name in stem_list:
                        cb('write.'+name, 92)
                        fname = f"{audio_path.stem} - {name}.wav"
                        sf.write(out_dir / fname, tensor.T.numpy(), sr)
                        stems_out.append(fname)
                del d,b,o,k,g, inst_wave
                if hasattr(torch, 'cuda'):
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                try: os.remove(inst_path)
                except OSError: pass

            import zipfile
            zip_path = conv_dir / f"{audio_path.stem}—stems.zip"
            cb('zip.start', 95)
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for name in stems_out:
                    fp = out_dir / name
                    if fp.exists():
                        zf.write(fp, arcname=name)
            cb('zip.done', 98)
            tasks[new_id] = { **old, 'zip': zip_path, 'controls': controls[new_id] }
            cb('finalizing', 99)
            progress[new_id] = {'stage': 'done', 'pct': 100}
        except Exception as exc:
            if stop_evt.is_set() or str(exc) == 'stopped':
                progress[new_id] = {'stage': 'stopped', 'pct': 0}
                errors[new_id] = 'stopped'
            else:
                logging.exception('rerun failed')
                progress[new_id] = {'stage': 'error', 'pct': -1}
                errors[new_id] = str(exc)

    process_queue.put(run_again)
    progress[new_id] = {'stage': 'queued', 'pct': 0}
    # store basics so future reruns still work from the new id
    tasks[new_id] = {
        'dir': out_dir,
        'stems': expected,
        'controls': controls[new_id],
        'conv_src': str(conv_src),
        'orig_src': old.get('orig_src'),
        'stem_list': stem_list,
    }
    return {'task_id': new_id, 'stems': expected}

@app.get('/download/{task_id}')
async def download(task_id: str):
    info = tasks.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail='invalid task id')
    zip_path = info.get("zip")
    if zip_path and Path(zip_path).exists():
        return FileResponse(path=zip_path, media_type="application/zip", filename=Path(zip_path).name)
    raise HTTPException(status_code=409, detail='files not ready')


# Clear all uploads endpoint
@app.post('/clear_all_uploads')
async def clear_all_uploads():
    dirs = ['uploads', 'uploads_converted']
    root = Path(__file__).resolve().parent.parent
    for d in dirs:
        dir_path = root / d
        if dir_path.exists():
            for entry in dir_path.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
    return {"status": "cleared"}


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
