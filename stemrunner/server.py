from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pathlib import Path
import uuid
import asyncio
import logging

from .pipeline import process_file
from .models import ModelManager

app = FastAPI()
manager = ModelManager()
progress = {}


@app.post('/upload')
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    path = Path('uploads') / file.filename
    path.parent.mkdir(exist_ok=True)
    with path.open('wb') as f:
        f.write(await file.read())

    missing = manager.missing_models()
    if missing:
        return JSONResponse({'detail': 'models not uploaded', 'missing': missing}, status_code=400)

    def cb(pct: int):
        progress[task_id] = pct

    def run():
        try:
            process_file(path, manager, progress_cb=cb)
        except Exception:
            logging.exception('processing failed')
            progress[task_id] = -1

    background_tasks.add_task(run)
    progress[task_id] = 0
    stems = [f"{Path(file.filename).stem}—{name}.wav" for name in [
        'Vocals', 'Instrumental', 'Drums', 'Bass', 'Other', 'Karaoke', 'Guitar']]
    return {'task_id': task_id, 'stems': stems}


@app.get('/progress/{task_id}')
async def progress_stream(task_id: str):
    async def event_generator():
        last = -1
        while True:
            pct = progress.get(task_id, 0)
            if pct != last:
                yield {'event': 'message', 'data': str(pct)}
                last = pct
            if pct >= 100 or pct < 0:
                break
            await asyncio.sleep(0.5)
    return EventSourceResponse(event_generator())


@app.get('/', response_class=HTMLResponse)
async def index():
    html = Path(Path(__file__).resolve().parent.parent / 'web' / 'index.html').read_text()
    return HTMLResponse(html)
