# Quality evidence

The quality lane is `reference_safe`. Dataset and personal audio must remain outside Git.

Install the reproducible local-release tooling with `python -m pip install --require-hashes -r requirements-quality-macos-arm64.lock`.

1. Copy `corpus.example.json` to ignored `private-corpus.json`, set external paths, and replace all private excerpt hash/tag placeholders.
2. Run `python quality.py prepare`.
3. Capture untouched v0.4.2 output with `quality.py baseline`, passing every exact `--checkpoint-sha256`, `--config-sha256`, and an exact `--device-id`. The baseline records both artifact and decoded-waveform SHA-256 values plus BS.1770 integrated loudness from FFmpeg.
4. Run the candidate on the same audio, then run `quality.py compare` with the same hashes and device identifier. A provenance mismatch fails the comparison.
5. Record three cold and three warm MPS trials per representative mode with `quality.py benchmark`. Pass separate `--cold-command` and `--warm-command`, `--role baseline|candidate`, `--mode`, `--device-id`, `--audio-seconds`, and the worker-observed cold/warm peak-MPS byte counts. The combined report pairs baseline and candidate only when mode and device match.
6. Export per-track baseline/candidate `museval` BSSEval v4 scores using `public-metrics.example.json` as the normalized format, including the exact museval version and both MUSDB18-HQ and Slakh2100-tiny provenance.
7. Record at least five hash-only blind comparisons using `listening-signoff.example.json`.
8. Generate the combined report with both benchmark reports, the compare report, `--public-metrics`, and `--listening-signoff`.

`quality.py report` fails closed if a public target is missing, a track failed,
median degradation exceeds 0.10 dB, fifth-percentile degradation exceeds
0.25 dB, either benchmark lacks MPS evidence, or manual signoff is incomplete.
