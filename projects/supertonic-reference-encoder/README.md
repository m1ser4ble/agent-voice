# Supertonic Reference Encoder Experiment

This is a standalone training project inside the agent-voice monorepo. It is
kept separate so `torch` and `torchaudio` do not become dependencies of the
runtime `agent-voice` package.

Goal:

```text
reference audio -> style_ttl / style_dp
```

The target tensors match the public Supertonic voice-style JSON shapes:

- `style_ttl`: `[batch, 50, 256]`
- `style_dp`: `[batch, 8, 16]`

This project does not include Supertonic's private reference encoder weights.
The model is a PyTorch scaffold based on the architecture details published in
the SupertonicTTS paper, intended for later supervised or reconstruction-based
training.

Model shape alignment with the paper:

- 228-band log-mel input
- latent encoder: `228 -> 24` channels while preserving time
- temporal compressor: concatenate 6 adjacent latent frames, `24 * 6 = 144`
- TTL reference encoder: `144 -> 128`, 6 ConvNeXt blocks, two attention passes
- TTL JSON: concatenate separately predicted 50x128 key and 50x128 value tokens
  into Supertonic's public `[50, 256]` format
- DP reference encoder: `144 -> 64`, 4 ConvNeXt blocks, 8 query tokens with
  16-d attention projection, exported as the public `[8, 16]` JSON format

## Training data

The current runnable training path is supervised style prediction:

```text
reference audio -> predicted style_ttl/style_dp
target          -> Supertonic voice-style JSON
```

Create a JSONL manifest with one record per training sample:

```jsonl
{"audio":"audio/speaker_001.wav","style_json":"styles/speaker_001.json","speaker_id":"speaker_001"}
{"audio":"audio/speaker_002.wav","style_json":"styles/speaker_002.json","speaker_id":"speaker_002"}
```

Relative paths are resolved from the manifest directory. The `style_json` file
must use Supertonic's public voice-style format:

```json
{
  "style_ttl": {"dims": [1, 50, 256], "data": [[[...]]]},
  "style_dp": {"dims": [1, 8, 16], "data": [[[...]]]}
}
```

Run training:

```bash
cd projects/supertonic-reference-encoder
uv run supertonic-reference-train \
  --manifest /path/to/manifest.jsonl \
  --output-dir runs/first \
  --epochs 50 \
  --batch-size 8 \
  --device cuda
```

## Speech Autoencoder Pretraining

For raw audio without `voice-style.json` labels, start with the paper-style
speech autoencoder path:

```text
waveform -> 228-band mel -> 24-d latent -> causal latent decoder -> waveform
```

Manifest:

```jsonl
{"audio":"audio/my_voice_0001.wav"}
{"audio":"audio/my_voice_0002.wav"}
```

Train:

```bash
uv run supertonic-autoencoder-train \
  --manifest /path/to/audio_manifest.jsonl \
  --output-dir runs/autoencoder \
  --epochs 10 \
  --batch-size 2 \
  --device auto
```

The reconstruction loss follows the paper's multi-resolution mel setup:

- FFT 1024, 64 mel bands, hop 256
- FFT 2048, 128 mel bands, hop 512
- FFT 4096, 128 mel bands, hop 1024

This stage does not produce `style_ttl/style_dp` yet. It learns the 24-d speech
latent space that the reference encoders should consume.

Recommended public sample-corpus mix for this stage:

- VCTK: multi-speaker English TTS / voice cloning corpus.
- LibriTTS `dev-clean` first, then larger train splits if storage allows.
- Zeroth-Korean: Korean transcribed multi-speaker corpus.
- FLEURS Korean: smaller Korean speech set, useful as a clean supplement.
- Common Voice Korean validated clips, if already downloaded from Mozilla Data
  Collective.

Keep the original corpora outside git, then build one audio-only manifest:

```bash
scripts/prepare_public_autoencoder_sample.sh
```

By default, the script takes up to 2,000 files from each corpus path:

```text
/datasets/LibriTTS/dev-clean
/datasets/VCTK-Corpus/wav48_silence_trimmed
/datasets/zeroth_korean
/datasets/fleurs/ko_kr
/datasets/common_voice_ko
```

Override paths or caps with environment variables:

```bash
SAMPLES_PER_SOURCE=5000 \
LIBRITTS_ROOT=/mnt/data/LibriTTS/dev-clean \
ZEROTH_ROOT=/mnt/data/zeroth_korean \
scripts/prepare_public_autoencoder_sample.sh
```

Use `COPY_MODE=copy` when transferring the prepared dataset to another machine.
The default `symlink` mode avoids duplicating large local corpora.

Train on CUDA with batch size 16:

```bash
scripts/train_autoencoder_cuda.sh
```

Resume from a previous checkpoint:

```bash
RESUME=runs/autoencoder-public-cuda/best.pt \
OUTPUT_DIR=runs/autoencoder-public-cuda-ft1 \
LEARNING_RATE=0.00005 \
scripts/train_autoencoder_cuda.sh
```

Outputs:

- `config.json`: resolved training config
- `metrics.jsonl`: per-epoch loss metrics
- `latest.pt`: latest checkpoint
- `best.pt`: best checkpoint by total loss

## Using the bundled Supertonic presets

The public `M1..M5` and `F1..F5` JSON files can seed a tiny supervised dataset:

1. Synthesize many text prompts with each preset.
2. Save each synthesized WAV.
3. Use the preset JSON that generated the WAV as the sample's `style_json`.
4. Train this encoder on the synthetic `(wav, preset_json)` pairs.

This is cheap and runnable, but it only teaches the encoder the public preset
space. It is useful as a smoke test and warm start; it is not enough by itself
to reproduce arbitrary unseen voices like a private Voice Builder encoder.

Generate a small preset dataset:

```bash
cat > /tmp/supertonic-texts.txt <<'EOF'
Allow me to introduce myself.
Systems are now fully operational.
What would you like me to do next?
EOF

uv run supertonic-preset-dataset \
  --texts /tmp/supertonic-texts.txt \
  --output-dir data/preset-smoke \
  --presets M2 F1 \
  --lang en
```

Then train from the generated manifest:

```bash
uv run supertonic-reference-train \
  --manifest data/preset-smoke/manifest.jsonl \
  --output-dir runs/preset-smoke \
  --epochs 2 \
  --batch-size 2 \
  --device auto
```
