# Voice Builder Experiment

This directory contains an experimental MP3/WAV to Supertonic voice-style JSON
builder.

It does not use the official Supertonic Voice Builder encoder. The public
runtime can load and synthesize from `style_ttl` / `style_dp` JSON files, but
the official reference-audio encoder is not included in the local package.

This experiment derives a compatible JSON directly from the input audio. The
current `spectral_projection_v4` method projects reference-audio spectral
features into `style_ttl`, then extrapolates along a Supertonic latent direction
(`F2 + alpha * (F1 - M2)`) inferred from reference pitch/brightness. It adjusts
both `style_ttl` and `style_dp`, because testing showed that `style_dp` carries
duration/prosody information needed to match the reference startup clip.

This is a local MP3/WAV-to-JSON path, not a built-in voice search. Quality is
experimental because the mapping is not a trained Supertonic speaker encoder.

Example:

```bash
uv run python -m experiments.voice_builder.mp3_to_json \
  --input /path/to/reference.mp3 \
  --text "안녕하세요 작업을 시작할까요 무엇을 명령하실겁니까 휴먼" \
  --out-dir .cache/agent-voice/voice-builder/mp3-to-json/example \
  --speed 0.7
```

Outputs:

- `voice-style.json`: generated Supertonic-compatible voice style
- `preview.wav`: synthesized preview using that generated JSON

Structure:

- `features.py`: extracts reference duration, pitch, brightness, rolloff, and RMS.
- `latent_basis.py`: builds the Supertonic latent direction used by v4.
- `evaluate.py`: scores generated audio against a reference.
- `optimize.py`: defines coefficient search primitives.
- `mp3_to_json.py`: CLI and JSON assembly.
