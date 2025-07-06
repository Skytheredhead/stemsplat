# stemsplat - ❗❗ONLY VOCALS WORKING❗❗

A lil stem splitter than runs locally on your machine. Basically just a easier-to-use version of UVR. 
I made this just to make my own "ensemble mode" layout where it splits vocals and instrumentals, then uses the instrumental
to derive other instruments.

**This is currently in ALPHA!!**
*BUGS/SUGGESTIONS: https://forms.gle/wSpe2nyFgcmuxSr28*

## Prerequisites

- Python (I'm using 3.13, not sure about other versions)
- Decently powerful machine (haven't tested on lesser machines or on Windows yet)

## Quickstart

1. Download the zip of the project (green code button at the top right ish -> download zip) or use git. 
2. Open a terminal (mac) or command prompt (pc).
3. Type "python3 install.py" This will open localhost:6060 and start the setup, then redirect you to 8000 (the main page) after setup is completed.
6. You'll then be prompted to download the models or add them yourself (downloads below). [currently you need to download them]

## Current Models (thx Becruily and KUIELab)

**Becruily:**
- Vocals - https://huggingface.co/becruily/mel-band-roformer-vocals/tree/main
- Instrumental - https://huggingface.co/becruily/mel-band-roformer-instrumental/tree/main
- Guitar - https://huggingface.co/becruily/mel-band-roformer-guitar/tree/main
- Karaoke - https://huggingface.co/becruily/mel-band-roformer-karaoke/tree/main

**KUIELab — https://huggingface.co/Politrees/UVR_resources/tree/main/models/MDXNet:**
- kuielab_a_bass.onnx
- kuielab_a_drums.onnx
- kuielab_a_other.onnx

## Future Plans

- Add different models for pc specs, like a low/med/high quality toggle that will use different models.
- Improve the UI
- video support??

