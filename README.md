Status: 1/14/26 11:40a PST
- working. Adding finishing touches before v0.1 release.

# stemsplat - DEVELOPER PREVIEW

**MAC ONLY!** This is a free, high quality, no bs stem splitter. No weird numbers in UVR you need to ask ChatGPT what they mean. Back when I used UVR, there were too many settings
and the default models provided super mediocre results. I've found [Becruily's models on Huggingface](https://huggingface.co/becruily) to be consistently good, so I'm using
them for this project.

**This is currently in ALPHA!!**
*BUGS/SUGGESTIONS: https://forms.gle/wSpe2nyFgcmuxSr28*

## Prerequisites

- Python (I'm using 3.13.5, not sure about other versions)
- APPLE SILICON Mac (M-series)

## Dev Quickstart

1. Download the zip of the project from the v0.1 release.
3. Open the app "Stemsplat" which will open a terminal and a page in your web browser.
6. You'll then be prompted to download the models, downlad them and put them in the Models folder.

## Current Models (Thank you Becruily)

**Becruily's Huggingface:**
- Vocals - https://huggingface.co/becruily/mel-band-roformer-vocals/tree/main
- Instrumental - https://huggingface.co/becruily/mel-band-roformer-instrumental/tree/main
- Deux - https://huggingface.co/becruily/mel-band-roformer-deux/tree/main

## Future Plans
- Add a “estimated time remaining” to the processing of the stack of songs based on song length and how long it’s taken to process the previous length of the song.
- adding single instrument splits
- Add different models to optimize for speed, like a low/med/high quality toggle that will use different models.
- Video support?
- Cloud Compute

## Way further down the line:
- Model Training (maybe as diff app)
- Running your own models
