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

app = FastAPI()
# map task_id -> {'stage': str, 'pct': int}
progress = {}
errors = {}
tasks = {}


@app.post('/upload')
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    path = Path('uploads') / file.filename
    path.parent.mkdir(exist_ok=True)
    with path.open('wb') as f:
        f.write(await file.read())

    ckpt_path = Path('models') / 'Mel Band Roformer Vocals.ckpt'
    if not ckpt_path.exists():
        ckpt_path = Path.home() / 'Library/Application Support/stems/Mel Band Roformer Vocals.ckpt'

    if not ckpt_path.exists():
        return JSONResponse({'detail': 'checkpoint not found'}, status_code=400)


    def cb(stage: str, pct: int):
        progress[task_id] = {'stage': stage, 'pct': pct}

    def run():
        try:
            process_file(path, ckpt_path, progress_cb=cb)
        except Exception as exc:
            logging.exception('processing failed')
            progress[task_id] = {'stage': 'error', 'pct': -1}
            errors[task_id] = str(exc)

    background_tasks.add_task(run)
    progress[task_id] = {'stage': 'queued', 'pct': 0}
    stems = [f"{Path(file.filename).stem}—Vocals.wav"]
    out_dir = Path('uploads') / f"{Path(file.filename).stem}—stems"
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
