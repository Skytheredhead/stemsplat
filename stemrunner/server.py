from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from pathlib import Path

from .pipeline import process_file
from .models import ModelManager

app = FastAPI()
manager = ModelManager()


@app.post('/upload')
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    path = Path('uploads') / file.filename
    path.parent.mkdir(exist_ok=True)
    with path.open('wb') as f:
        f.write(await file.read())
    background_tasks.add_task(process_file, path, manager)
    stems = [f"{Path(file.filename).stem}—{name}.wav" for name in [
        'Vocals', 'Instrumental', 'Drums', 'Bass', 'Other', 'Karaoke', 'Guitar']]
    return {'queued': file.filename, 'stems': stems}


@app.get('/', response_class=HTMLResponse)
async def index():
    html = Path(Path(__file__).resolve().parent.parent / 'web' / 'index.html').read_text()
    return HTMLResponse(html)
