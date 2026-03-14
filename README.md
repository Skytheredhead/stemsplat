# stemsplat - v0.3.0 release

**MAC ONLY!** This is a free, high quality, no bs stem splitter. No weird numbers to fiddle with and the best quality.

## Prerequisites

- Python (I'm using 3.13.5, not sure about other versions)
- APPLE SILICON Mac (M-series)

## Quickstart
1. Download the .zip from the latest release
2. Extract the .zip
3. Open the .app
4. Download the models in the app
5. Done 

## Current Models (Thank you Becruily)

**Becruily's Huggingface:**
- Vocals - https://huggingface.co/becruily/mel-band-roformer-vocals/tree/main
- Instrumental - https://huggingface.co/becruily/mel-band-roformer-instrumental/tree/main
- Deux - https://huggingface.co/becruily/mel-band-roformer-deux/tree/main

## v0.4.0 plans
- Change port to something that’s not :8000
- Having an “edit” button that has a popup card with a waveform of the whole song the user can select in and out points on (and preview) instead of splitting an entire song.
- ^ Mute/solo toggles for each selected stem 
- Presets: Fast, Balanced, Best Quality
- Settings/nerd stuff/device (mps, cpu toggle)
- In settings, 
- History (with file size limits, etc.) use to cache outputs, and other stuff. Default user warning to 10GB, able to change this in settings to higher (or lower) amount.
- Model Manager: Installed models list (already there), estimated disk usage
- Update available badge
- Checksum verification for model downloads
- When a lan device tries to process but another device is already processing, say that somewhere
- Bpm/key detector
- Bpm/key shifter

Models to add:
- Becruily Guitar: https://huggingface.co/becruily/mel-band-roformer-guitar/tree/main
- 6-stem model: https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/tree/main (588800)
- Demucs for fast (but crap) stem splits
