# agent-voice

Local-first voice layer for terminal coding agents.

`agent-voice` is not another hosted voice assistant. It is a thin local layer
that sits on top of terminal coding agents like Codex, Pi, and Claude Code
without modifying them.

Part of the value is assembly: choosing compatible local components, wiring
them together, and providing sane defaults for Smart Turn/VAD, Whisper, Kokoro,
and terminal-agent adapters so users do not have to hand-build a voice stack.

The long-term target is a voice operating layer for coding agents:

```text
Mic
  -> Smart Turn
  -> Whisper
  -> Agent Adapter
  -> Codex / Pi / Claude Code
  -> Voice Presenter
  -> Kokoro
  -> Speaker
```

The first working target is Codex.

## Current Status

This repo currently contains the local voice MVP core:

- `PexpectAgent`: spawns a terminal agent and submits transcript text as input
- `VoicePresenter`: turns noisy agent output into short speech summaries
- `InterruptManager`: detects stop phrases while the system is speaking
- `VoiceSession`: tracks `LISTENING -> THINKING -> SPEAKING -> INTERRUPTED`
- `VoiceLoop`: coordinates transcript, agent, presenter, speaker, and interrupt
  contracts
- `MicrophoneWhisperTranscriptSource`: captures mic audio, segments speech with
  a simple energy gate plus Smart Turn, and transcribes with faster-whisper
- `KokoroSpeaker`: synthesizes presenter output with Kokoro ONNX and plays it
  through the local speaker
- `agent-voice codex ...`: starts voice mode by default and passes all target
  args through to Codex
- `agent-voice pi ...`: uses the same pexpect boundary for Pi
- `agent-voice doctor`: checks Python packages, agent command availability,
  audio input/output devices, and Kokoro model cache status

## Install

```bash
uv sync
```

## Run

Check local runtime readiness:

```bash
uv run agent-voice doctor
uv run agent-voice doctor --agent pi --list-devices
uv run agent-voice doctor --agent none
```

`doctor` verifies the current machine. It does not install dependencies or
download model files; `uv sync` handles Python dependencies, and runtime model
assets are downloaded by the provider code on first use.

Voice mode entrypoint:

```bash
uv run agent-voice codex
```

Pass Codex options after the `codex` target. `agent-voice` does not parse these
options, so Codex version changes should not require `agent-voice` CLI changes:

```bash
uv run agent-voice codex resume
uv run agent-voice codex --model <model>
uv run agent-voice codex resume --model <model>
```

Put `agent-voice` options before the target:

```bash
uv run agent-voice --language ko --whisper-model tiny --stt-language ko codex
uv run agent-voice --tts-voice af_sarah codex --model <model>
```

Start the current persistent text-mode Codex session:

```bash
uv run agent-voice --text codex
```

Text mode also passes Codex options through:

```bash
uv run agent-voice --text codex resume --model <model>
```

### Connect to Pi

Pi uses the same target passthrough model:

```bash
uv run agent-voice pi
uv run agent-voice pi -c
```

Text-mode Pi:

```bash
uv run agent-voice --text pi -c
```

Send one text command for smoke tests or automation:

```bash
uv run agent-voice --text --once "auth 버그 고쳐" codex
uv run agent-voice --text --once "테스트는?" pi -c
```

### Current Voice Caveats

The default voice path is wired, but it still needs real-device tuning. It uses
local mic/speaker access through `sounddevice`, downloads Kokoro model assets
into `.cache/agent-voice/kokoro/`, and uses CPU faster-whisper by default.
System PortAudio/microphone permissions must be available.

## Why This Exists

Existing projects mostly solve adjacent problems:

- remote/mobile control of coding agents
- STT or TTS wrappers
- Claude Code-specific voice plugins
- output narration companions

`agent-voice` is scoped around the missing local layer:

- local-first by default
- no Realtime API dependency
- no agent internals patching
- assembled local voice stack with sane defaults
- shared adapters for Codex, Pi, and Claude Code
- Voice Presenter as a first-class module
- interrupt/barge-in as part of the core state machine

## Development

```bash
uv run pytest
```

GitHub Actions runs the same test suite on Python 3.12 and 3.13, builds the
package, and runs provider smoke on `main` pushes or manual dispatch.

Check the current machine before a manual voice run:

```bash
uv run agent-voice doctor --list-devices
```

Run the real local provider smoke:

```bash
uv run python scripts/provider_smoke.py
```

This runs:

```text
Kokoro ONNX -> WAV -> faster-whisper -> Pipecat Smart Turn v3
```

The smoke downloads Kokoro model assets into `.cache/provider-smoke/` and uses
`faster-whisper` `tiny.en` on CPU. It is not part of the default test suite
because it downloads model files.

Project notes:

- Architecture: `docs/architecture.md`
- Testing strategy: `docs/testing.md`
- Roadmap: `docs/roadmap.md`
- Design: `docs/superpowers/specs/2026-06-08-agent-voice-design.md`
- Implementation plan: `docs/superpowers/plans/2026-06-08-agent-voice-mvp.md`
- Handoff: `docs/handoff.md`
