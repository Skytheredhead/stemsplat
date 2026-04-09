# stemsplat

**Mac-only stem splitter for Apple Silicon.**  
The best and easiest to use ap for stem splitting.

<img width="1430" height="892" alt="Screenshot 2026-04-09 at 9 35 16 AM" src="https://github.com/user-attachments/assets/20aac5ad-c93f-4ab5-b3e3-78d664b7214d" />

## Quickstart

1. Download the latest `.zip` from the Releases page
2. Extract it
3. Open the `.app`

## What It Does

- Runs fully locally on your Mac
- No cloud uploads
- Batch queue processing
- LAN/mobile access from other devices on your local network
- Multiple export formats
- Previous-files history so recent outputs are easy to reopen or reuse

## Why This Exists

Most online stem splitters either suck or are behind paywalls, and UVR was a pain to do anything.
I made Stemsplat to fill in the gaps UVR left. 

email stemsplat@gmail.com for bugs/feature requests

## Stems

- Vocals
- Instrumental
- Both (deux)
- Both (separate)
- Guitar
- Background vocals
- Full mix
- Full mix faster
- Drum split - 4
- Drum split - 6
- All stems
- Boost harmonies
- Denoise

## Requirements

### For the app
- Apple Silicon Mac (M-series)
- macOS
- Enough free disk space for models and exports (a couple gigs)
- Python 3.10+

## Model Credits

### Becruily
- Vocals  
  https://huggingface.co/becruily/mel-band-roformer-vocals
- Instrumental  
  https://huggingface.co/becruily/mel-band-roformer-instrumental
- Deux  
  https://huggingface.co/becruily/mel-band-roformer-deux
- Guitar  
  https://huggingface.co/becruily/mel-band-roformer-guitar
- Karaoke / Background vocals  
  https://huggingface.co/becruily/mel-band-roformer-karaoke

### Jarredou
- Denoise  
  https://huggingface.co/jarredou/aufr33_MelBand_Denoise
- BS-Roformer 6-stem  
  https://huggingface.co/jarredou/BS-ROFO-SW-Fixed
- DrumSep 6-stem  
  https://github.com/jarredou/models/releases

### Other bundled/downloadable sources
- Demucs drums / bass / other / 6-stem variants
- ZFTurbo DrumSep 4-stem


## Future release plans (subject to change):

### v0.5.0:
- [ ]  Having an “edit” button that opens a popup card with a full-song waveform, in/out selection, and preview
- [ ]  Mute/solo toggles for each selected stem inside that edit flow
- [ ]  For a large upload all at once, keep it as one stack with a unified progress bar, then expand into individual cards on click and collapse back up
- [ ]  Make the stop button a pause button, but only if it behaves like a real queue pause and not fake task resume
- [ ]  Add estimated processing times to songs in queue and total estimated time for the queue
- [ ]  Add user-created presets
- [ ]  Export album covers with instrument pictures so splits are visually differentiated
- [ ]  Custom right click
- [ ]  When I right click a model and hit “remove from queue,” make the scroll animation cleaner/smoother
- [ ]  Instead of “- vocals,” include the model name like “- vocals (single)” or “- vocals (full)”
- [ ]  Settings/nerd stuff/device toggle (mps / cpu)
- [ ]  Settings + quit + other stuff in the top toolbar
- [ ]  Add uninstall
- [ ]  Add delete models
- [ ]  Right click models to uninstall one by one in model manager
- [ ]  Add models resuming if partially downloaded and the network fails, if you feel good about the reliability work

### v0.6.0:
- [ ]  BPM/key detector
- [ ]  BPM/key shifter
- [ ]  De-reverb model
- [ ]  Wind model
- [ ]  Strings model
- [ ]  Single-channel compute optimization
- [ ]  In-app updates instead of redownloading from GitHub
