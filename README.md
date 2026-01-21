# stemsplat - V0.1.0

**MAC ONLY!** This is a free, high quality, no bs stem splitter. This app will provide better results than the free online stem splitters, and this happens to run locally on your computer! Also, there's no weird numbers that you need to mess around wth, it's all pre-decided for super high quality. I've found [Becruily's models on Huggingface](https://huggingface.co/becruily) to be consistently good, so I'm currently using them for this project.
*BUGS/SUGGESTIONS (that aren't listed below): https://forms.gle/wSpe2nyFgcmuxSr28*

**System Requirements**
A non-air apple silicon mac.

## Quickstart
0. Make sure you have python installed (https://www.python.org/downloads/macos/)
1. Download the zip of the project from the v0.1.0 release on the right side of the page.
2. Unzip the folder.
3. Download these 3 models (~2.5G alltogether) [Vocals](https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true) [Instrumental](https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true) [Deux](https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt?download=true)
4. Put those models into the Models folder which is inside the stemsplat folder you unzipped in step 2.
5. Open the app "Stemsplat" (will get blocked by Apple, just go through the steps to allow it) which will open a terminal and a page in your web browser. The UI is in your browser, ignore the terminal. To close the app, hit the x in the top left corner, everything else should be self explanatory.


## Current Models (Thank you Becruily)
- Vocals - https://huggingface.co/becruily/mel-band-roformer-vocals/tree/main
- Instrumental - https://huggingface.co/becruily/mel-band-roformer-instrumental/tree/main
- Deux - https://huggingface.co/becruily/mel-band-roformer-deux/tree/main

## v0.1.1 plans:
- Python file consolidation into one main.py file
- Progress bar updates for smoothness
- Est time remaining
- UI changes

## Future Plans
- adding single instrument splits
- Add different models to optimize for speed, like a low/med/high quality toggle that will use different models.
- Video support?
- Cloud Compute (maybe)
- Higher quality splits

## In a galaxy far far away:
- Model Training (maybe as different app)
- Running your own models
