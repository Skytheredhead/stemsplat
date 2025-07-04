# Stemrunner

Self-hosted stem-separation service.

## Prerequisites

- Python 3.10 or higher, or Docker
- Working NVIDIA drivers for GPU usage
- Internet access on first install to download torch and torchaudio

## Folder layout

```
stemrunner/
  models/   # put .ckpt and .onnx files here (ignored by Git)
  configs/  # architecture .yaml files (tracked)
  music/    # optional watch folder
```

## Model setup

Create a `models/` directory at the repo root and drop all checkpoint files there. Place every accompanying `.yaml` file inside `configs/`.

## Quick-start (CLI)

```bash
git clone <repo>
cd stemrunner
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # downloads torch + torchaudio
python -m stemrunner path/to/*.wav --gpu 0
```

## Quick-start (Docker CPU)

```bash
docker build -f Dockerfile.cpu -t stemrunner .
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/music:/music" \
  stemrunner
```

Point your browser to `http://localhost:8000` and drag files onto the page.

## Browser GUI usage

Visit `http://localhost:8000` after running the server. Drag-and-drop audio files onto the page to queue them for processing. The page lists the names of the seven expected stem files.

## Troubleshooting

- Check CUDA drivers if GPU inference fails.
- Ensure checkpoint files are present in `models/`.
- Verify `.yaml` configs are in `configs/`.

