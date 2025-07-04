# Stemrunner

Self-hosted stem-separation service with real-time progress updates. The
project now auto-detects NVIDIA GPUs as well as Apple Silicon machines using
Metal (MPS).

## Prerequisites

- Python 3.10 or higher **or** Docker
- Working NVIDIA drivers for GPU usage (if using a PC GPU)
- Internet access on first install to download torch and torchaudio

## Getting the code

**Option A: with Git**

1. [Install Git](https://git-scm.com/) if you don't have it.
2. Open a terminal or command prompt.
3. Clone the repository:

   ```bash
   git clone https://github.com/yourname/stemsplat.git
   cd stemsplat
   ```

**Option B: without Git**

1. Visit the project page on GitHub.
2. Press **Code** → **Download ZIP**.
3. Extract the archive and open a terminal in the folder.

## Installing dependencies

```bash
python -m venv venv
source venv/bin/activate   # on Windows use "venv\Scripts\activate"
pip install -r requirements.txt
```

## Folder layout

```
stemrunner/
  models/   # put .ckpt and .onnx files here (ignored by Git)
  configs/  # architecture .yaml files (tracked)
  music/    # optional watch folder
```

## Add the models

Create a `models/` directory at the repo root and drop all checkpoint files there. Place every accompanying `.yaml` file inside `configs/`.

## Running from the command line

```bash
python -m stemrunner path/to/*.wav   # auto selects CUDA or Metal if available
```

The CLI will print progress for each file to the terminal.

## Run the web server locally

After installing the requirements you can start the FastAPI server directly:

```bash
uvicorn stemrunner.server:app --reload
```

Then open `http://localhost:8000` in your browser.

## Quick-start (Docker CPU)

```bash
docker build -f Dockerfile.cpu -t stemrunner .
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/music:/music" \
  stemrunner
```

After running the container point your browser to `http://localhost:8000`.

## Quick-start (Docker GPU)

```bash
docker build -f Dockerfile.gpu -t stemrunner-gpu .
docker run --gpus all --rm -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/music:/music" \
  stemrunner-gpu
```

This image uses NVIDIA CUDA and requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) to be installed.

## Browser GUI

1. Open `http://localhost:8000` in your browser.
2. Drag-and-drop audio files onto the card or click it to pick files.
3. Progress bars update in real time as stems are created using server-sent
   events.

When processing finishes the interface lists the output stem files (vocals,
instrumental, drums, bass, other, karaoke and guitar).

## Troubleshooting

- Check CUDA drivers if GPU inference fails.
- Ensure checkpoint files are present in `models/`.
- Verify `.yaml` configs are in `configs/`.

