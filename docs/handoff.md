# agent-voice Handoff

## Project Goal

Build an advanced local voice layer for terminal coding agents.

Target UX:

```text
"야 codex"
"auth 버그 고쳐"
"지금 뭐하고 있어?"
"테스트는?"
"잠깐"
"그 수정 취소"
```

No keyboard should be required for common coding-agent control.

## Positioning

This is not primarily STT or TTS. The important product surface is:

- `Agent Adapter`: voice transcript becomes terminal input without modifying Codex
- `Voice Presenter`: terminal output becomes a short spoken summary
- `Interrupt Manager`: user can stop speech and issue a new command
- `Agent State Awareness`: later, answer "what are you doing now?"

## Architecture

Detailed component architecture: `docs/architecture.md`

Testing strategy and hardware E2E definition: `docs/testing.md`

Future project candidates and issue seeds: `docs/roadmap.md`

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

Current code implements the core voice boundary:

- `src/agent_voice/adapter.py`
- `src/agent_voice/presenter.py`
- `src/agent_voice/interrupt.py`
- `src/agent_voice/loop.py`
- `src/agent_voice/providers.py`
- `src/agent_voice/cli.py`

## MVP Scope

Implemented:

- Codex adapter via `pexpect`
- transcript-to-terminal input emulation
- Korean/English rule-based speech summaries
- session state machine
- interrupt phrase detection
- `VoiceLoop` contract with `TranscriptSource` and `Speaker` protocols
- `VoiceLoop.run_once()` command and interrupt paths covered by tests
- `VoiceLoop.run_until_idle()` batch multi-turn path covered by tests
- `VoiceLoop.run_forever()` keeps polling during user silence and exits only on
  explicit exit intent
- Runtime exit intent covered by tests: `이제 그만`, `종료`, `exit`, `quit`
- Barge-in interrupt loop covered by tests: while `speaker.say()` is playing,
  `VoiceLoop` keeps polling transcripts and calls `speaker.stop()` on `잠깐`
- Non-interrupt transcripts during `SPEAKING` are ignored to avoid echoing the
  spoken summary back into the agent as a command
- `agent-voice codex` reserved as the default voice-mode entrypoint
- `agent-voice codex ...` passes all target args through to Codex
- `agent-voice pi ...` passes all target args through to Pi
- `agent-voice --text codex ...` persistent text session
- `agent-voice --text --once "..." codex ...` for smoke tests and automation
- `MicrophoneWhisperTranscriptSource` implementation with mic capture, simple
  energy VAD, Smart Turn, and faster-whisper
- `KokoroSpeaker` implementation with Kokoro ONNX and `sounddevice`
- bundled voice preset config with `jarvis_style` as the default Kokoro preset
- `--voice-config`, `--voice-preset`, `--tts-voice`, `--tts-lang`, and
  `--tts-speed` CLI controls
- default CLI voice-mode wiring through `VoiceLoop`
- `agent-voice doctor` runtime readiness checks for packages, Codex/Pi command
  lookup, audio input/output devices, and Kokoro cache status
- test coverage for all current modules

Not implemented yet:

- real hardware E2E validation for mic -> Whisper -> agent -> Kokoro speaker
- tuned mic thresholds, echo behavior, and latency budget
- setup/profile commands such as `agent-voice setup --profile local-cpu`
- Pi / Claude Code structured adapters
- agent state inspection

## Provider Smoke Result

The target local provider combination has been tested once in this repo:

```bash
uv run python scripts/provider_smoke.py
```

Verified stack:

- Kokoro ONNX generated `.cache/provider-smoke/kokoro-smoke.wav`.
- faster-whisper `tiny.en` transcribed the generated audio as:
  `Authentication bug fixed. One test passed.`
- Pipecat Smart Turn v3.2 CPU ONNX returned `EndOfTurnState.COMPLETE`.
- Observed Smart Turn probability: `0.947`.
- Observed Smart Turn latency: tens of milliseconds in the project venv.

This is provider-level smoke, not full product hardware E2E. The repo now has
`MicrophoneWhisperTranscriptSource`, `KokoroSpeaker`, and a test-backed
`VoiceLoop`, but the full mic-to-agent-to-speaker flow still needs real-device
validation.

## Connecting Agents

Codex is the current first-class path:

```bash
uv run agent-voice codex
```

This is now the voice-mode entrypoint. It currently reports that the voice loop
is wired to local providers. Put Codex options after the `codex` target:

```bash
uv run agent-voice codex resume
uv run agent-voice codex --model <model>
```

The default voice preset is `jarvis_style`, defined in
`src/agent_voice/voice_presets.toml`. It is a stock Kokoro assistant-style
preset, not a celebrity or movie-character voice clone. Override it before the
target:

```bash
uv run agent-voice --voice-preset high_quality codex
uv run agent-voice --tts-voice af_sarah --tts-speed 1.0 codex
uv run agent-voice --voice-config ./voice-presets.toml codex
```

Use text mode for the keyboard-driven debug path:

```bash
uv run agent-voice --text codex resume --model <model>
```

Pi uses the same target passthrough:

```bash
uv run agent-voice pi
uv run agent-voice pi -c
uv run agent-voice --text pi -c
```

Implement that with a Pi-specific adapter. Prefer Pi's structured RPC or JSON
event modes over TUI scraping so the voice layer can derive agent state from
events instead of raw terminal text.

## Doctor

Run a read-only local readiness check before a manual voice run:

```bash
uv run agent-voice doctor --list-devices
uv run agent-voice doctor --agent pi
uv run agent-voice doctor --agent none
```

This checks Python package imports, the selected agent command, audio input and
output device discovery, and whether the expected Kokoro model files already
exist under `.cache/agent-voice/kokoro/`. It does not install dependencies,
download models, test real playback, or run live transcription.

## Next Session Task

Recommended next task:

1. Run real hardware E2E: `uv run agent-voice codex` with mic and speaker.
2. Tune `MicrophoneWhisperTranscriptSource` thresholds and latency.
3. Listen-test and tune `jarvis_style` against real spoken summaries.
4. Add structured Pi / Claude Code adapters.

Keep the public contract local-first and avoid Realtime API dependency.

## Future Project Ideas

See `docs/roadmap.md` for roadmap candidates:

- LiteLLM one-shot agent adapter for broader API/local-model usage.
- Portable ONNX provider profiles.
- Mobile / Galaxy XR companion mode where the device is the mic/speaker/UI and
  the workstation remains the agent runtime.
