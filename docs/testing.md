# Testing Strategy

`agent-voice` has three different test layers. They should not be mixed up.

## 1. Unit / Contract Tests

Command:

```bash
uv run pytest
```

This runs on GitHub-hosted CI. It uses fake agents, fake transcript sources,
and fake speakers to verify code contracts:

- CLI target passthrough for Codex/Pi
- `agent-voice doctor` diagnostic behavior with fake probes
- `PexpectAgent` behavior
- `VoicePresenter` summaries
- `VoiceLoop` state transitions
- interrupt loop behavior
- `KokoroSpeaker` and managed-loop resource contracts

This does not require a microphone, speaker, Codex auth, Pi auth, or downloaded
model files beyond normal Python package dependencies.

Local readiness command:

```bash
uv run agent-voice doctor --list-devices
uv run agent-voice doctor --agent codex --companion-codex
```

`doctor` is not a replacement for product E2E. It checks package imports,
agent command availability, audio device discovery, Codex companion capability
flags, and Kokoro cache status on the current machine so runtime failures are
easier to diagnose. Kokoro cache checks include minimum file sizes so
interrupted downloads and HTML error responses do not look like valid model
assets.

## 2. Provider Smoke

Command:

```bash
uv run python scripts/provider_smoke.py
```

This validates that the local provider libraries can work together:

```text
Kokoro ONNX -> WAV file -> faster-whisper -> Pipecat Smart Turn v3
```

It proves that:

- Kokoro ONNX can synthesize an audio file.
- faster-whisper can transcribe that generated file.
- Pipecat Smart Turn v3 can classify the generated utterance as complete.
- This stack works without Torch/CUDA.

This is still not a product E2E test. It does not use the microphone, does not
play through the speaker, does not spawn Codex/Pi, and does not test barge-in
against live audio playback.

The GitHub workflow exposes this as an opt-in `workflow_dispatch` job rather
than a required push/PR check. The script downloads external model files from
Hugging Face, so normal CI should not fail just because an upstream service is
rate-limiting anonymous runners.

## 3. Hardware E2E

Hardware E2E means testing the real product loop on a real machine:

```text
real microphone
  -> live speech capture
  -> Smart Turn / VAD
  -> Whisper transcription
  -> Codex or Pi process with real auth/session
  -> VoicePresenter
  -> Kokoro synthesis
  -> real speaker playback
  -> user says "잠깐"
  -> speaker stops while runtime keeps listening
```

This is the test that answers: "Can I sit at a machine, say commands out loud,
hear useful spoken responses, interrupt them, and keep working without touching
the keyboard?"

Minimum manual checklist:

1. Run `uv run agent-voice codex`.
2. Say a short command such as `auth 버그 고쳐`.
3. Verify Codex receives the transcript and responds.
4. Verify `VoicePresenter` speaks a short summary through Kokoro.
5. While speech is playing, say `잠깐`.
6. Verify playback stops promptly.
7. Say another command and verify the same Codex session is still alive.
8. Say `종료` and verify speaker and agent process stop.

Why this cannot run on normal GitHub-hosted CI:

- GitHub-hosted runners do not provide the user's microphone and speaker.
- They do not have the user's authenticated Codex/Pi session.
- Audio device timing, echo, permissions, and PortAudio behavior are
  machine-specific.
- Barge-in latency must be measured against real playback and real mic capture.

Future automation path:

- Add a self-hosted runner labeled for audio hardware.
- Add a hardware E2E script that can play a known command audio fixture into
  the mic path or use a loopback audio device.
- Record latency metrics for transcript completion, agent response, TTS start,
  and interrupt-to-speaker-stop.
- Keep this opt-in. It should not block normal pull requests until the hardware
  lab is stable.
