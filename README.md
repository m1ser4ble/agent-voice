# agent-voice

Local-first voice layer for terminal coding agents.

`agent-voice` is not another hosted voice assistant. It is a thin local layer
that sits on top of terminal coding agents like Codex, Pi, and Claude Code
without modifying them.

Part of the value is assembly: choosing compatible local components, wiring
them together, and providing sane defaults for Smart Turn/VAD, Whisper,
Supertonic/Kokoro, and terminal-agent adapters so users do not have to
hand-build a voice stack.

The long-term target is a voice operating layer for coding agents:

```text
Mic
  -> Smart Turn
  -> Whisper
  -> Agent Adapter
  -> Codex / Pi / Claude Code
  -> Voice Presenter
  -> Supertonic / Kokoro
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
- `KeyboardTranscriptSource`: lets you type a line and press Enter while voice
  mode is still running
- terminal I/O visibility: prints completed transcripts, submitted agent input,
  raw agent output, and spoken summaries during voice mode
- `SupertonicSpeaker`: synthesizes Korean presenter output with Supertonic and
  plays it through the local speaker
- `KokoroSpeaker`: fallback local TTS backend for Kokoro ONNX voices
- LiveKit/WebRTC AEC: feeds TTS playback into `AudioProcessingModule` as a
  reverse stream and sends microphone capture through the same processor before
  VAD/Whisper
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
assets are downloaded by the provider code on first use. If a cached Kokoro
asset is missing or too small because a download was interrupted/rate-limited,
runtime will replace it on the next start and print download progress to
stderr.

Voice mode entrypoint:

```bash
uv run agent-voice codex
```

Voice mode keeps the terminal transparent. You can speak commands through the
mic, or type a line and press Enter in the same session. The terminal prints the
completed transcript, the exact input submitted to Codex/Pi, the raw agent
output collected from the child process, and the shorter voice summary. Say or
type `종료` / `exit` to stop the session.

When speaker audio leaks back into the microphone, voice mode may print
`[ignored while speaking]` for transcripts caught during playback or
`[ignored self echo]` for delayed transcripts that match text it just spoke.

The default voice preset is `jarvis_style`. With `--tts-backend auto`, Korean
speech uses Supertonic by default. The bundled `jarvis_style` preset maps to
Supertonic `M2`, a stock male assistant-style voice; it is not a celebrity or
movie-character voice clone. Force a backend when needed:

```bash
uv run agent-voice --tts-backend supertonic --supertonic-voice F2 codex
uv run agent-voice --tts-backend macos-say --macos-say-voice Yuna codex
uv run agent-voice --tts-backend kokoro codex
```

Kokoro is still available as an explicit backend, but it is not treated as the
default Korean TTS provider.

Pass Codex options after the `codex` target. `agent-voice` does not parse these
options, so Codex version changes should not require `agent-voice` CLI changes:

```bash
uv run agent-voice codex resume
uv run agent-voice codex --model <model>
uv run agent-voice codex resume --model <model>
```

Put `agent-voice` options before the target:

```bash
uv run agent-voice --language ko --whisper-model tiny codex
uv run agent-voice --voice-preset high_quality codex
uv run agent-voice --supertonic-voice F2 --tts-speed 1.0 codex --model <model>
uv run agent-voice --tts-backend kokoro --tts-voice af_sarah codex
```

`--language ko` controls the speech summary language and is the default.
`--stt-language ko` controls the Whisper transcription language and is also the
default. Override it, for example with `--stt-language en`, when you want to
speak another language.
Use `--quiet-agent-io` to hide terminal transcript/raw-output events, or
`--no-keyboard` if you want mic-only input.
LiveKit/WebRTC echo cancellation is enabled by default for Supertonic/Kokoro
playback paths. TTS playback is streamed in 10 ms chunks so the AEC reverse
stream is fed in playback order, and microphone capture is processed through
the same APM before VAD/Whisper. Use `--aec-delay-ms <ms>` to tune estimated
speaker-to-microphone delay, or `--no-aec` to disable it while tuning devices.

Experimental Codex app-server backend:

```bash
uv run agent-voice --agent-backend codex-app-server codex
uv run agent-voice --agent-backend codex-app-server --text --once "OK 라고만 답해" codex
```

This backend starts `codex app-server` and reads JSON-RPC agent events instead
of scraping the terminal TUI. It currently renders assistant-message and
file-change events into the existing presenter path while ignoring command
execution lifecycle noise. It is an MVP for comparing output quality before
replacing the pexpect backend.

By default, `agent-voice` uses the OS/sounddevice default input and output
devices. Select a specific microphone or speaker by sounddevice index or name
when you want to override the system default. Use `doctor --list-devices` to
find the available device values and see `[default input]` / `[default output]`
markers:

```bash
uv run agent-voice doctor --list-devices
uv run agent-voice --input-device 2 pi
uv run agent-voice --input-device "USB Microphone" --output-device "USB Speaker" codex
```

Voice presets live in `src/agent_voice/voice_presets.toml`. Add or override
presets with your own TOML file:

```toml
[defaults]
preset = "workstation"

[presets.workstation]
voice = "bm_george"
kokoro_voice = "bm_george"
supertonic_voice = "M3"
lang = "en-gb"
speed = 0.9
description = "Local workstation voice."
```

```bash
uv run agent-voice --voice-config ./voice-presets.toml codex
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
local mic/speaker access through `sounddevice`, downloads Supertonic assets via
the Hugging Face cache on first Korean TTS use, and uses CPU faster-whisper by
default. Kokoro downloads its own assets into `.cache/agent-voice/kokoro/` when
that backend is selected. System PortAudio/microphone permissions must be
available.

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

GitHub Actions runs the same test suite on Python 3.12 and 3.13 and builds the
package. Provider smoke is available on manual workflow dispatch because it
downloads external model files and can be rate-limited by upstream services.

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
