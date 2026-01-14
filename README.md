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

## Quickstart

1. Download the zip of the project from the v0.1 release.
2. Unzip the folder.
3. Download these 3 models [Vocals](https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true) [Instrumental](https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true) [Deux](https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt?download=true)
4. Put those models into the Models folder which is inside the stemsplat folder you unzipped in step 2.
5. Open the app "Stemsplat" (will get blocked by Apple, just go through the steps to allow it) which will open a terminal and a page in your web browser. The UI is in your browser, ignore the terminal. To close the app, hit the x in the top left corner, everything else should be self explanatory.


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
