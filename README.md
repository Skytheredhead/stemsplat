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
- Optional passcode-protected HTTPS LAN/mobile access (off by default)
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



## v0.4.3 hardening status

The v0.4.3 lane is `reference_safe`: model selection, overlap, precision, output semantics, and MPS behavior remain unchanged unless the quality gates approve a later promotion.

Automatic model downloads are intentionally blocked until every checkpoint has an immutable revision, exact SHA-256, and documented distribution rights in `models/manifest.json`. Existing unsupported user-provided state-dict checkpoints may be used at the owner’s risk; legacy pickle-based models require an exact approved manifest hash.

A production build additionally requires the full public/private quality report, Developer ID signing/notarization credentials, and signed update metadata. Without those prerequisites, `build_app.sh` produces a development-only ad-hoc build.

See `docs/qa/reconciliation-matrix.md` and `quality/README.md` for the release evidence workflow.
